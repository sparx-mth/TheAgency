"""Aggregation is where a correct run can still print a wrong number; these pin the arithmetic and the refusals."""
from __future__ import annotations

import dataclasses
import json
import random

import pytest

from sparx_agency.core.planning.objnav.types.measurement import (
    TERMINATION_STEP_LIMIT,
    TERMINATION_STOP,
)
from sparx_agency.tasks.planning.objnav_benchmark import aggregate
from sparx_agency.tasks.planning.objnav_benchmark.aggregate import (
    group_summary,
    paired_comparison,
    summarise,
)
from sparx_agency.tasks.planning.objnav_benchmark.errors import (
    HarnessError,
    ResultsError,
)
from sparx_agency.tasks.planning.objnav_benchmark.records import (
    ACTION_NAMES,
    TERMINATION_AGENT_ERROR,
    EpisodeRecord,
)
from sparx_agency.tasks.planning.objnav_benchmark.stats import (
    bootstrap_mean_interval,
    mcnemar_exact_p,
    paired_permutation_p,
    wilson_interval,
)


def record(episode_id, success=False, spl=0.0, soft_spl=0.0, dtg=1.0,
           category="chair", scene="s1", agent="agent_a", split="val",
           steps=10, path=2.0, termination=TERMINATION_STOP):
    """A valid record; every field the aggregation does not read is fixed."""
    counts = dict.fromkeys(ACTION_NAMES, 0)
    counts["MOVE_FORWARD"] = steps
    error = ("RuntimeError: boom" if termination == TERMINATION_AGENT_ERROR
             else None)
    return EpisodeRecord(
        benchmark="hm3d_v2", split=split, episode_id=episode_id,
        scene_id=scene, target_category=category, agent=agent,
        success=success, spl=spl, soft_spl=soft_spl, distance_to_goal_m=dtg,
        path_length_m=path, observed_path_length_m=path, shortest_path_m=1.5,
        start_distance_to_goal_m=1.5, steps=steps,
        stop_called=termination != TERMINATION_STEP_LIMIT,
        termination=termination, action_counts=counts, wall_s=0.1,
        agent_error=error)


def run(agent="agent_a"):
    """Six episodes over three categories and two scenes, with every kind of ending."""
    return [
        record("ep1", True, 0.8, 0.9, 0.05, "tv", "s2", agent, steps=12,
               path=3.0),
        record("ep2", True, 0.5, 0.6, 0.08, "bed", "s1", agent, steps=20,
               path=4.0),
        record("ep3", False, 0.0, 0.0, None, "chair", "s1", agent, steps=500,
               path=10.0, termination=TERMINATION_STEP_LIMIT),
        record("ep4", False, 0.0, 0.2, 2.0, "chair", "s2", agent, steps=8,
               path=1.0),
        record("ep5", True, 1.0, 1.0, 0.0, "tv", "s1", agent, steps=6,
               path=1.5),
        record("ep6", False, 0.0, 0.0, 3.0, "bed", "s2", agent, steps=3,
               path=0.5, termination=TERMINATION_AGENT_ERROR),
    ]


# -- summarise -------------------------------------------------------------

def test_summarise_groups_by_category_and_scene_in_sorted_order():
    """Sorted keys make two runs' tables line up row for row."""
    summary = summarise(run(), resamples=100)
    assert (summary.benchmark, summary.split, summary.agent) == (
        "hm3d_v2", "val", "agent_a")
    assert summary.overall.key == "all"
    assert [(g.key, g.n_episodes) for g in summary.by_category] == [
        ("bed", 2), ("chair", 2), ("tv", 2)]
    assert [(g.key, g.n_episodes) for g in summary.by_scene] == [
        ("s1", 3), ("s2", 3)]


def test_every_mean_counts_failures_as_zero():
    """Leaving failures out of SPL is the classic way to inflate it."""
    overall = summarise(run(), resamples=100).overall
    assert overall.n_episodes == 6
    assert overall.success_rate == 0.5
    assert overall.spl == pytest.approx(2.3 / 6)
    assert overall.soft_spl == pytest.approx(2.7 / 6)
    assert overall.mean_steps == pytest.approx(549 / 6)
    assert overall.mean_path_length_m == pytest.approx(20.0 / 6)


def test_the_success_interval_is_wilson_and_the_spl_interval_is_the_seeded_bootstrap():
    """Each group's bootstrap is seeded by benchmark/split/agent/group, so it reproduces."""
    summary = summarise(run(), resamples=300)
    assert summary.overall.success_rate_ci == wilson_interval(3, 6)
    assert summary.overall.spl_ci == bootstrap_mean_interval(
        [0.8, 0.5, 0.0, 0.0, 1.0, 0.0], 0.95, 300, "hm3d_v2/val/agent_a/all")
    tv = summary.by_category[2]
    assert tv.spl_ci == bootstrap_mean_interval(
        [0.8, 1.0], 0.95, 300, "hm3d_v2/val/agent_a/tv")
    assert summary.confidence == 0.95


def test_unreachable_ends_are_counted_beside_the_distance_mean_not_in_it():
    """An infinite distance averaged in would make the mean infinite; dropped silently, it would flatter."""
    overall = summarise(run(), resamples=100).overall
    assert overall.distance_to_goal_m == pytest.approx(5.13 / 5)
    assert overall.n_unreachable_end == 1


def test_a_group_where_no_episode_ended_reachable_has_no_distance():
    """None, not 0 and not inf, when there is nothing to average."""
    group = group_summary("chair", [
        record("e1", dtg=None, termination=TERMINATION_STEP_LIMIT),
        record("e2", dtg=None, termination=TERMINATION_STEP_LIMIT)],
        resamples=50)
    assert group.distance_to_goal_m is None
    assert group.n_unreachable_end == 2


def test_terminations_list_every_kind_with_zeros_and_count_agent_errors():
    """A zero step-limit count is information; a missing key would read as unknown."""
    summary = summarise(run(), resamples=100)
    assert summary.terminations == {"stop": 4, "step_limit": 1,
                                    "agent_error": 1}
    assert summary.agent_errors == 1
    clean = summarise([record("e1"), record("e2")], resamples=50)
    assert clean.terminations == {"stop": 2, "step_limit": 0,
                                  "agent_error": 0}
    assert clean.agent_errors == 0


def test_summarise_refuses_to_pool_two_agents_and_lists_both():
    """Two agents in one summary produce a figure neither of them earned."""
    records = run("agent_a")[:3] + run("agent_b")[3:]
    with pytest.raises(HarnessError, match=r"agent_a.*agent_b"):
        summarise(records)


def test_summarise_refuses_to_pool_two_splits():
    """Val and test episodes in one mean is the comparison error the harness exists to stop."""
    with pytest.raises(HarnessError, match="mixes 2 runs"):
        summarise([record("e1"), record("e2", split="test")])


def test_summarise_refuses_an_episode_counted_twice():
    """A resume that re-ran an episode would weigh it double in every mean."""
    with pytest.raises(HarnessError, match="'ep2'"):
        summarise(run() + [run()[1]])


def test_summarise_refuses_an_empty_run():
    """There is no honest summary of zero episodes."""
    with pytest.raises(HarnessError, match="no episode records"):
        summarise([])


def test_the_summary_does_not_depend_on_record_order():
    """A resumed run logs in another order than a fresh one; the intervals must match."""
    shuffled = run()
    random.Random(5).shuffle(shuffled)
    assert summarise(shuffled, resamples=200) == summarise(run(),
                                                           resamples=200)


def test_the_summary_is_strict_json():
    """summary.json is written with allow_nan=False; a NaN or inf here would fail the run's last step."""
    json.dumps(summarise(run(), resamples=50).to_dict(), allow_nan=False)


def test_group_summary_refuses_a_blank_key_a_non_record_and_an_empty_group():
    """Each is a caller bug that would otherwise surface as a malformed table."""
    with pytest.raises(HarnessError, match="key"):
        group_summary("", run())
    with pytest.raises(TypeError, match="EpisodeRecord"):
        group_summary("all", [{"episode_id": "e1"}])
    with pytest.raises(HarnessError):
        group_summary("all", [])


def test_group_summary_refuses_a_key_of_spaces_itself():
    """A string of spaces is non-empty; it must be refused here, not by GroupSummary after the arithmetic."""
    with pytest.raises(HarnessError, match="^a group key must be a non-blank string"):
        group_summary("   ", run())


@pytest.mark.parametrize("field", ["benchmark", "split", "episode_id", "scene_id",
                                   "target_category", "agent"])
def test_a_record_with_a_blank_identity_is_refused_when_it_is_built(field):
    """A blank name would be logged, then refused by the summary after the whole run; refuse it at the record."""
    with pytest.raises(ResultsError, match="EpisodeRecord.%s must be a non-blank string" % field):
        dataclasses.replace(record("ep1"), **{field: "   "})


def test_a_failure_adds_its_progress_to_softspl_as_the_module_says():
    """SoftSPL is ungated in habitat-lab; a docstring saying failures add 0 invites a 'fix' that breaks it."""
    failed = record("ep1", success=False, spl=0.0, soft_spl=0.75)
    solved = record("ep2", success=True, spl=1.0, soft_spl=1.0, dtg=0.0)
    group = group_summary("all", [failed, solved], resamples=50)
    assert (group.spl, group.soft_spl) == (0.5, 0.875)
    doc = " ".join(aggregate.__doc__.split())
    assert "a failure contributes 0 to SPL, and to SoftSPL the progress it made" in doc


# -- paired_comparison -----------------------------------------------------

def second_run():
    """The same six episodes for another agent."""
    return [
        record("ep1", False, 0.0, 0.1, 1.0, "tv", "s2", "agent_b"),
        record("ep2", True, 0.7, 0.7, 0.05, "bed", "s1", "agent_b"),
        record("ep3", True, 0.4, 0.5, 0.05, "chair", "s1", "agent_b"),
        record("ep4", False, 0.0, 0.0, 2.0, "chair", "s2", "agent_b"),
        record("ep5", True, 1.0, 1.0, 0.0, "tv", "s1", "agent_b"),
        record("ep6", False, 0.0, 0.0, 3.0, "bed", "s2", "agent_b"),
    ]


def test_paired_comparison_joins_on_episode_id_not_position():
    """Joined by position, a reordered log would pair unrelated episodes."""
    comparison = paired_comparison(run(), list(reversed(second_run())))
    assert (comparison.first_agent, comparison.second_agent) == (
        "agent_a", "agent_b")
    assert comparison.n_episodes == 6
    assert (comparison.spl_wins, comparison.spl_ties,
            comparison.spl_losses) == (1, 3, 2)
    assert comparison.success_rate_first == 0.5
    assert comparison.success_rate_second == 0.5


def test_paired_comparison_reports_the_difference_and_both_p_values():
    """The p-values must be the stats module's tests on the joined episodes."""
    first_spl = [0.8, 0.5, 0.0, 0.0, 1.0, 0.0]
    second_spl = [0.0, 0.7, 0.4, 0.0, 1.0, 0.0]
    comparison = paired_comparison(run(), second_run())
    assert comparison.spl_first == pytest.approx(2.3 / 6)
    assert comparison.spl_second == pytest.approx(2.1 / 6)
    assert comparison.spl_difference == pytest.approx(0.2 / 6)
    assert comparison.success_p_value == mcnemar_exact_p(
        [True, True, False, False, True, False],
        [False, True, True, False, True, False])
    assert comparison.spl_p_value == paired_permutation_p(first_spl,
                                                          second_spl)


def test_paired_comparison_refuses_different_episode_sets():
    """Pairing is only valid on the same episodes; a missing one must be named."""
    with pytest.raises(HarnessError, match=r"1 only in 'agent_a' \('ep6'\)"):
        paired_comparison(run(), second_run()[:5])


def test_paired_comparison_refuses_an_agent_compared_with_itself():
    """Usually one run loaded twice, which would report a perfect tie."""
    with pytest.raises(HarnessError, match="both sides"):
        paired_comparison(run(), run())


def test_paired_comparison_refuses_different_splits():
    """The same episode id on two splits is two different episodes."""
    other = [record("ep%d" % i, agent="agent_b", split="test")
             for i in range(1, 7)]
    with pytest.raises(HarnessError, match="val.*test"):
        paired_comparison(run(), other)


def test_paired_comparison_refuses_an_episode_whose_scene_differs():
    """Same id, different scene means the runs read different episode datasets."""
    other = second_run()
    other[0] = record("ep1", category="tv", scene="s9", agent="agent_b")
    with pytest.raises(HarnessError, match="different episode datasets"):
        paired_comparison(run(), other)


def test_paired_comparison_refuses_a_side_that_mixes_agents():
    """A side must be one run, or its name in the result is a lie."""
    mixed = second_run()[:3] + run("agent_c")[3:]
    with pytest.raises(HarnessError, match="mixes"):
        paired_comparison(run(), mixed)
