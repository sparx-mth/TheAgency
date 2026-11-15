"""Turn a run's episode records into the numbers a paper table needs, with their uncertainty.

Aggregation is where a correct run can still produce a wrong number, so this
module refuses the quiet ways that happens: two agents or two splits pooled
into one figure, an episode counted twice after a resume, a failure left out
of a mean. Every mean is over *all* of a group's episodes, as the leaderboards
count them -- a failure contributes 0 to SPL, and to SoftSPL the progress it
made (SoftSPL is not gated on success, as habitat-lab computes it). The one
exception is the final distance to goal: an episode that ended where no goal is
reachable has no finite distance, so the mean is over the others and the
unreachable ones are counted beside it rather than averaged in as infinity.

A summary does not depend on the order the records arrive in: each group is
sorted by episode id before its bootstrap, so a resumed run and a fresh run of
the same episodes print the same intervals.

Python 3.8 syntax, standard library only.
"""
from __future__ import annotations

import math
from collections import Counter
from typing import Dict, List, Sequence, Tuple

from sparx_agency.tasks.planning.objnav_benchmark.checks import is_name
from sparx_agency.tasks.planning.objnav_benchmark.errors import HarnessError
from sparx_agency.tasks.planning.objnav_benchmark.records import (
    RECORD_TERMINATIONS,
    TERMINATION_AGENT_ERROR,
    EpisodeRecord,
)
from sparx_agency.tasks.planning.objnav_benchmark.stats import (
    bootstrap_mean_interval,
    mcnemar_exact_p,
    paired_permutation_p,
    wilson_interval,
)
from sparx_agency.tasks.planning.objnav_benchmark.summaries import (
    BenchmarkSummary,
    GroupSummary,
    PairedComparison,
)

#: The key of the group that holds every episode of a run.
OVERALL_KEY = "all"


def _mean(values: Sequence[float]) -> float:
    return math.fsum(values) / len(values)


def _preview(items: Sequence[str], limit: int = 5) -> str:
    shown = ", ".join(repr(item) for item in items[:limit])
    if len(items) <= limit:
        return shown
    return "%s, ... (%d in all)" % (shown, len(items))


def _sorted_records(records: Sequence[EpisodeRecord],
                    what: str) -> List[EpisodeRecord]:
    """The records sorted by episode id, after the type and emptiness checks."""
    items = list(records)
    if not items:
        raise HarnessError("%s has no episode records to summarise" % what)
    for item in items:
        if not isinstance(item, EpisodeRecord):
            raise TypeError(
                "%s: expected EpisodeRecord, got %s"
                % (what, type(item).__name__))
    return sorted(items, key=lambda record: record.episode_id)


def _run_identity(records: Sequence[EpisodeRecord],
                  what: str) -> Tuple[str, str, str]:
    """The one ``(benchmark, split, agent)`` the records share, refusing a mixture or a repeat."""
    runs = Counter((r.benchmark, r.split, r.agent) for r in records)
    if len(runs) != 1:
        listing = ", ".join(
            "%s/%s/%s (%d episodes)" % (benchmark, split, agent, count)
            for (benchmark, split, agent), count in sorted(runs.items()))
        raise HarnessError(
            "%s mixes %d runs: %s; summarise each (benchmark, split, agent) "
            "on its own" % (what, len(runs), listing))
    repeats = sorted(episode for episode, count
                     in Counter(r.episode_id for r in records).items()
                     if count > 1)
    if repeats:
        raise HarnessError(
            "%s holds these episode ids more than once: %s; an episode counted "
            "twice (re-run after a resume, two result files concatenated) "
            "inflates n and biases every mean" % (what, _preview(repeats)))
    return next(iter(runs))


def _grouped(records: Sequence[EpisodeRecord],
             attribute: str) -> List[Tuple[str, List[EpisodeRecord]]]:
    groups: Dict[str, List[EpisodeRecord]] = {}
    for record in records:
        groups.setdefault(getattr(record, attribute), []).append(record)
    return sorted(groups.items(), key=lambda item: item[0])


def group_summary(key: str, records: Sequence[EpisodeRecord], *,
                  confidence: float = 0.95, resamples: int = 2000,
                  seed_key: str = "objnav") -> GroupSummary:
    """The metrics of one group of episodes, with a Wilson SR and a bootstrap SPL interval.

    Args:
        key: The group's name: ``"all"``, a category or a scene.
        records: The group's episodes; at least one.
        confidence: Coverage of both intervals, in ``(0, 1)``.
        resamples: Bootstrap resamples for the SPL interval.
        seed_key: Seeds the bootstrap. :func:`summarise` passes
            ``"benchmark/split/agent/group"``.

    Returns:
        The group's summary.

    Raises:
        TypeError: If an element is not an :class:`EpisodeRecord`.
        HarnessError: On a blank key, an empty group, or a bad confidence or
            resample count.
    """
    if not is_name(key):
        raise HarnessError("a group key must be a non-blank string, got %r"
                           % (key,))
    ordered = _sorted_records(records, "group %r" % key)
    n = len(ordered)
    successes = sum(1 for record in ordered if record.success)
    spls = [record.spl for record in ordered]
    reachable = [record.distance_to_goal_m for record in ordered
                 if record.distance_to_goal_m is not None]
    return GroupSummary(
        key=key,
        n_episodes=n,
        success_rate=successes / n,
        success_rate_ci=wilson_interval(successes, n, confidence),
        spl=_mean(spls),
        spl_ci=bootstrap_mean_interval(spls, confidence, resamples, seed_key),
        soft_spl=_mean([record.soft_spl for record in ordered]),
        distance_to_goal_m=_mean(reachable) if reachable else None,
        n_unreachable_end=n - len(reachable),
        mean_steps=_mean([float(record.steps) for record in ordered]),
        mean_path_length_m=_mean([record.path_length_m for record in ordered]),
    )


def summarise(records: Sequence[EpisodeRecord], *, confidence: float = 0.95,
              resamples: int = 2000) -> BenchmarkSummary:
    """One agent's result on one benchmark split: overall, per category, per scene.

    Each group's SPL bootstrap is seeded by
    ``"benchmark/split/agent/<group key>"``, so every interval is reproducible
    and no two groups of a run share a stream by accident.

    Args:
        records: Every episode of one run.
        confidence: Coverage of every interval, in ``(0, 1)``.
        resamples: Bootstrap resamples per SPL interval.

    Returns:
        The summary; ``by_category`` and ``by_scene`` sorted by key, and
        ``terminations`` counting every kind in :data:`RECORD_TERMINATIONS`,
        zeros included.

    Raises:
        TypeError: If an element is not an :class:`EpisodeRecord`.
        HarnessError: On no records, records from more than one (benchmark,
            split, agent), a repeated episode id, or a bad confidence or
            resample count.
    """
    ordered = _sorted_records(records, "summarise")
    benchmark, split, agent = _run_identity(ordered, "summarise")
    prefix = "%s/%s/%s/" % (benchmark, split, agent)

    def summary_of(key: str, members: Sequence[EpisodeRecord]) -> GroupSummary:
        return group_summary(key, members, confidence=confidence,
                             resamples=resamples, seed_key=prefix + key)

    overall = summary_of(OVERALL_KEY, ordered)
    terminations = Counter(record.termination for record in ordered)
    return BenchmarkSummary(
        benchmark=benchmark,
        split=split,
        agent=agent,
        overall=overall,
        by_category=tuple(summary_of(key, members) for key, members
                          in _grouped(ordered, "target_category")),
        by_scene=tuple(summary_of(key, members) for key, members
                       in _grouped(ordered, "scene_id")),
        terminations={kind: terminations.get(kind, 0)
                      for kind in RECORD_TERMINATIONS},
        agent_errors=terminations.get(TERMINATION_AGENT_ERROR, 0),
        confidence=float(confidence),
    )


def _check_same_episodes(left: Sequence[EpisodeRecord],
                         right: Sequence[EpisodeRecord],
                         agents: Tuple[str, str]) -> None:
    """Refuse two sides that did not run exactly the same episodes."""
    left_ids = [record.episode_id for record in left]
    right_ids = [record.episode_id for record in right]
    if left_ids != right_ids:
        only_left = sorted(set(left_ids) - set(right_ids))
        only_right = sorted(set(right_ids) - set(left_ids))
        raise HarnessError(
            "paired_comparison needs exactly the same episodes on both sides: "
            "%d only in %r (%s), %d only in %r (%s); run both agents on one "
            "episode list, or compare their summaries instead"
            % (len(only_left), agents[0], _preview(only_left),
               len(only_right), agents[1], _preview(only_right)))
    for a, b in zip(left, right):
        if (a.scene_id, a.target_category) != (b.scene_id, b.target_category):
            raise HarnessError(
                "episode %r is %s/%s for %r but %s/%s for %r; the two runs "
                "read different episode datasets"
                % (a.episode_id, a.scene_id, a.target_category, agents[0],
                   b.scene_id, b.target_category, agents[1]))


def paired_comparison(first: Sequence[EpisodeRecord],
                      second: Sequence[EpisodeRecord], *,
                      permutations: int = 10000) -> PairedComparison:
    """Two agents on the same episodes, compared episode by episode.

    The sides are joined on episode id, never on position. Success is
    compared with the exact McNemar test and SPL with the paired sign-flip
    permutation test; the permutation stream uses a fixed seed, so the
    p-value is reproducible and identical with the sides swapped.

    Args:
        first: Every episode of the first agent's run.
        second: Every episode of the second agent's run.
        permutations: Passed to :func:`paired_permutation_p`.

    Returns:
        The comparison; ``spl_difference`` is ``spl_first - spl_second``.

    Raises:
        TypeError: If an element is not an :class:`EpisodeRecord`.
        HarnessError: On an empty side, a side mixing runs or repeating an
            episode, sides from different benchmarks or splits, the same
            agent on both sides, or different episodes on the two sides.
    """
    left = _sorted_records(first, "the first side")
    right = _sorted_records(second, "the second side")
    benchmark, split, first_agent = _run_identity(left, "the first side")
    other_benchmark, other_split, second_agent = _run_identity(
        right, "the second side")
    if (benchmark, split) != (other_benchmark, other_split):
        raise HarnessError(
            "paired_comparison: the first side ran %s/%s and the second "
            "%s/%s; a paired test needs the same episodes"
            % (benchmark, split, other_benchmark, other_split))
    if first_agent == second_agent:
        raise HarnessError(
            "paired_comparison: both sides are agent %r; name the variants "
            "apart (an agent compared with itself is usually one run loaded "
            "twice)" % (first_agent,))
    _check_same_episodes(left, right, (first_agent, second_agent))
    left_spl = [record.spl for record in left]
    right_spl = [record.spl for record in right]
    wins = sum(1 for a, b in zip(left_spl, right_spl) if a > b)
    losses = sum(1 for a, b in zip(left_spl, right_spl) if a < b)
    spl_first, spl_second = _mean(left_spl), _mean(right_spl)
    return PairedComparison(
        first_agent=first_agent,
        second_agent=second_agent,
        n_episodes=len(left),
        success_rate_first=_mean([float(r.success) for r in left]),
        success_rate_second=_mean([float(r.success) for r in right]),
        spl_first=spl_first,
        spl_second=spl_second,
        spl_difference=spl_first - spl_second,
        success_p_value=mcnemar_exact_p([r.success for r in left],
                                        [r.success for r in right]),
        spl_p_value=paired_permutation_p(left_spl, right_spl,
                                         permutations=permutations),
        spl_wins=wins,
        spl_ties=len(left) - wins - losses,
        spl_losses=losses,
    )
