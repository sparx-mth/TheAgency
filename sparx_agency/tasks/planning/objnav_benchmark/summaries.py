"""A run's numbers as a paper table needs them: its summary, a paired comparison, a paper's figure.

Each type refuses the summary that could not have come from a real run --
groups that do not partition it, terminations that do not add up to it, a
difference that is not the difference of its two means -- because such a
summary prints a plausible number nobody can trace back to episodes.

Scores are fractions in ``[0, 1]`` everywhere in here. Papers print percent;
convert when a paper's number is typed into a :class:`ReportedResult`, and only
the presentation layer multiplies by 100.

Python 3.8 syntax, standard library only.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, Optional, Tuple

from sparx_agency.tasks.planning.objnav_benchmark.checks import (
    is_count,
    is_fraction,
    is_length,
    is_name,
    is_real,
)
from sparx_agency.tasks.planning.objnav_benchmark.errors import ResultsError
from sparx_agency.tasks.planning.objnav_benchmark.records import (
    RECORD_TERMINATIONS,
    TERMINATION_AGENT_ERROR,
)


@dataclass(frozen=True)
class GroupSummary:
    """The metrics of one group of episodes: the whole run, a category, a scene.

    Every mean is over *all* the group's episodes, failures included -- a
    failure contributes 0 to SPL, exactly as the leaderboards count it.

    Attributes:
        key: ``"all"``, or the category or scene the group is.
        n_episodes: Episodes in the group.
        success_rate: Mean success.
        success_rate_ci: Wilson interval of the success rate.
        spl: Mean SPL.
        spl_ci: Percentile-bootstrap interval of the mean SPL.
        soft_spl: Mean SoftSPL.
        distance_to_goal_m: Mean final distance to goal over the episodes that
            ended with a goal reachable; None when none did.
        n_unreachable_end: Episodes that ended with no goal reachable.
        mean_steps: Mean actions per episode.
        mean_path_length_m: Mean distance travelled.

    Raises:
        ResultsError: On an empty group, a rate outside ``[0, 1]``, an interval
            that is not an ordered pair inside ``[0, 1]``, or a count or mean
            that cannot be.
    """

    key: str
    n_episodes: int
    success_rate: float
    success_rate_ci: Tuple[float, float]
    spl: float
    spl_ci: Tuple[float, float]
    soft_spl: float
    distance_to_goal_m: Optional[float]
    n_unreachable_end: int
    mean_steps: float
    mean_path_length_m: float

    def __post_init__(self) -> None:
        if not is_name(self.key):
            raise ResultsError(
                "GroupSummary.key must be a non-blank string, got %r"
                % (self.key,))
        if not (is_count(self.n_episodes) and self.n_episodes > 0):
            raise ResultsError(
                "GroupSummary %r must hold at least one episode, got %r"
                % (self.key, self.n_episodes))
        for name in ("success_rate", "spl", "soft_spl"):
            if not is_fraction(getattr(self, name)):
                raise ResultsError(
                    "GroupSummary %r: %s must lie in [0, 1], got %r"
                    % (self.key, name, getattr(self, name)))
        for name in ("success_rate_ci", "spl_ci"):
            pair = tuple(getattr(self, name))
            if (len(pair) != 2 or not all(is_fraction(v) for v in pair)
                    or pair[0] > pair[1]):
                raise ResultsError(
                    "GroupSummary %r: %s must be an ordered pair inside "
                    "[0, 1], got %r" % (self.key, name, getattr(self, name)))
            object.__setattr__(self, name, pair)
        if (self.distance_to_goal_m is not None
                and not is_length(self.distance_to_goal_m)):
            raise ResultsError(
                "GroupSummary %r: distance_to_goal_m must be None or finite "
                "and non-negative, got %r" % (self.key, self.distance_to_goal_m))
        if not (is_count(self.n_unreachable_end)
                and self.n_unreachable_end <= self.n_episodes):
            raise ResultsError(
                "GroupSummary %r: n_unreachable_end must be a count no larger "
                "than n_episodes, got %r" % (self.key, self.n_unreachable_end))
        for name in ("mean_steps", "mean_path_length_m"):
            if not is_length(getattr(self, name)):
                raise ResultsError(
                    "GroupSummary %r: %s must be finite and non-negative, got "
                    "%r" % (self.key, name, getattr(self, name)))


@dataclass(frozen=True)
class BenchmarkSummary:
    """One agent's result on one benchmark split.

    Attributes:
        benchmark: Benchmark key.
        split: Dataset split.
        agent: The agent's name.
        overall: Every episode.
        by_category: One group per goal category, sorted by category; empty
            only when the breakdown was not computed.
        by_scene: One group per scene, sorted by scene; empty only when the
            breakdown was not computed.
        terminations: Episodes per termination, every one of
            :data:`RECORD_TERMINATIONS` present.
        agent_errors: Episodes the agent crashed in (already counted as
            failures in every metric).
        confidence: The confidence level of every interval.

    Raises:
        ResultsError: On a blank identifier, groups that do not partition the
            run, terminations that do not add up to it, or a confidence
            outside ``(0, 1)``.
    """

    benchmark: str
    split: str
    agent: str
    overall: GroupSummary
    by_category: Tuple[GroupSummary, ...]
    by_scene: Tuple[GroupSummary, ...]
    terminations: Dict[str, int]
    agent_errors: int
    confidence: float

    def __post_init__(self) -> None:
        for name in ("benchmark", "split", "agent"):
            if not is_name(getattr(self, name)):
                raise ResultsError(
                    "BenchmarkSummary.%s must be a non-blank string, got %r"
                    % (name, getattr(self, name)))
        if not isinstance(self.overall, GroupSummary):
            raise ResultsError(
                "BenchmarkSummary.overall must be a GroupSummary, got %r"
                % (self.overall,))
        total = self.overall.n_episodes
        for name in ("by_category", "by_scene"):
            groups = tuple(getattr(self, name))
            if not all(isinstance(group, GroupSummary) for group in groups):
                raise ResultsError(
                    "BenchmarkSummary.%s must hold GroupSummary objects" % name)
            # Empty means the breakdown was not computed; a breakdown that
            # exists must account for every episode exactly once.
            if groups and sum(group.n_episodes for group in groups) != total:
                raise ResultsError(
                    "BenchmarkSummary.%s must partition the run's %d episodes, "
                    "but its groups hold %d"
                    % (name, total, sum(g.n_episodes for g in groups)))
            object.__setattr__(self, name, groups)
        counts = self.terminations
        if (not isinstance(counts, dict)
                or set(counts) != set(RECORD_TERMINATIONS)
                or not all(is_count(value) for value in counts.values())
                or sum(counts.values()) != total):
            raise ResultsError(
                "BenchmarkSummary.terminations must count every one of %r and "
                "add up to %d episodes, got %r"
                % (RECORD_TERMINATIONS, total, counts))
        if self.agent_errors != counts[TERMINATION_AGENT_ERROR]:
            raise ResultsError(
                "BenchmarkSummary.agent_errors=%r disagrees with its %r "
                "termination count %r" % (self.agent_errors,
                                          TERMINATION_AGENT_ERROR,
                                          counts[TERMINATION_AGENT_ERROR]))
        if not (is_real(self.confidence) and 0.0 < self.confidence < 1.0):
            raise ResultsError(
                "BenchmarkSummary.confidence must lie in (0, 1), got %r"
                % (self.confidence,))

    def to_dict(self) -> Dict[str, Any]:
        """The summary as plain JSON-ready data."""
        return asdict(self)


@dataclass(frozen=True)
class PairedComparison:
    """Two agents on the same episodes, compared episode by episode.

    Paired rather than two aggregates side by side: the same episodes carry
    most of the variance, so pairing is what lets a few hundred episodes
    separate two variants of one method.

    Attributes:
        first_agent: Name of the first agent.
        second_agent: Name of the second agent.
        n_episodes: Episodes both agents ran.
        success_rate_first: The first agent's success rate on them.
        success_rate_second: The second agent's.
        spl_first: The first agent's mean SPL.
        spl_second: The second agent's.
        spl_difference: ``spl_first - spl_second``.
        success_p_value: Exact McNemar test on the discordant successes.
        spl_p_value: Paired sign-flip permutation test on SPL.
        spl_wins: Episodes where the first agent's SPL is higher.
        spl_ties: Episodes where they are equal.
        spl_losses: Episodes where the second agent's SPL is higher.

    Raises:
        ResultsError: On the same agent twice, a rate or p-value outside
            ``[0, 1]``, a difference that is not the two means' difference, or
            wins, ties and losses that do not add up to the episodes.
    """

    first_agent: str
    second_agent: str
    n_episodes: int
    success_rate_first: float
    success_rate_second: float
    spl_first: float
    spl_second: float
    spl_difference: float
    success_p_value: float
    spl_p_value: float
    spl_wins: int
    spl_ties: int
    spl_losses: int

    def __post_init__(self) -> None:
        for name in ("first_agent", "second_agent"):
            if not is_name(getattr(self, name)):
                raise ResultsError(
                    "PairedComparison.%s must be a non-blank string, got %r"
                    % (name, getattr(self, name)))
        if self.first_agent == self.second_agent:
            raise ResultsError(
                "PairedComparison compares %r with itself" % (self.first_agent,))
        if not (is_count(self.n_episodes) and self.n_episodes > 0):
            raise ResultsError(
                "PairedComparison.n_episodes must be positive, got %r"
                % (self.n_episodes,))
        for name in ("success_rate_first", "success_rate_second", "spl_first",
                     "spl_second", "success_p_value", "spl_p_value"):
            if not is_fraction(getattr(self, name)):
                raise ResultsError(
                    "PairedComparison.%s must lie in [0, 1], got %r"
                    % (name, getattr(self, name)))
        if not (is_real(self.spl_difference) and abs(
                self.spl_difference - (self.spl_first - self.spl_second))
                <= 1e-9):
            raise ResultsError(
                "PairedComparison.spl_difference=%r is not spl_first - "
                "spl_second = %r" % (self.spl_difference,
                                     self.spl_first - self.spl_second))
        tallies = (self.spl_wins, self.spl_ties, self.spl_losses)
        if not all(is_count(t) for t in tallies) or sum(tallies) != self.n_episodes:
            raise ResultsError(
                "PairedComparison wins, ties and losses %r must add up to "
                "n_episodes=%r" % (tallies, self.n_episodes))


@dataclass(frozen=True)
class ReportedResult:
    """A number a paper reported, with where it came from.

    Attributes:
        method: The method's name as the paper gives it.
        benchmark: Benchmark key, matching the harness's (``"hm3d_v1"``).
        split: Dataset split.
        success_rate: As a fraction -- divide the paper's percent by 100.
        spl: As a fraction.
        source: Where the number is printed (``"arXiv:2504.14478 Table I"``).
            Required: a figure without a source cannot be checked.
        soft_spl: As a fraction, when the paper reports it.
        distance_to_goal_m: When the paper reports it.
        notes: Anything that makes the number not directly comparable (a
            different episode subset, a different camera height, ...).

    Raises:
        ResultsError: On an empty identifier or source, a score outside
            ``[0, 1]`` (the usual sign that a percent was typed in), or notes
            that are not a string.
    """

    method: str
    benchmark: str
    split: str
    success_rate: float
    spl: float
    source: str
    soft_spl: Optional[float] = None
    distance_to_goal_m: Optional[float] = None
    notes: str = ""

    def __post_init__(self) -> None:
        for name in ("method", "benchmark", "split", "source"):
            if not is_name(getattr(self, name)):
                raise ResultsError(
                    "ReportedResult.%s must be a non-empty string, got %r"
                    % (name, getattr(self, name)))
        for name in ("success_rate", "spl", "soft_spl"):
            value = getattr(self, name)
            if value is None and name == "soft_spl":
                continue
            if not is_fraction(value):
                raise ResultsError(
                    "ReportedResult.%s of %r must be a fraction in [0, 1], "
                    "got %r -- papers print percent; divide by 100"
                    % (name, self.method, value))
        if (self.distance_to_goal_m is not None
                and not is_length(self.distance_to_goal_m)):
            raise ResultsError(
                "ReportedResult.distance_to_goal_m must be None or finite and "
                "non-negative, got %r" % (self.distance_to_goal_m,))
        if not isinstance(self.notes, str):
            raise ResultsError(
                "ReportedResult.notes must be a string, got %r" % (self.notes,))
