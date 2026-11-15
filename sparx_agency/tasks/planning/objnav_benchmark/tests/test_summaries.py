"""A run's summaries refuse numbers that could not have come from a real run.

Groups that do not partition the run, terminations that do not add up to it,
a difference that is not the difference of its two means, a percent typed
where a fraction belongs: each prints a plausible figure that nobody can
trace back to episodes.
"""
from __future__ import annotations

import pytest

from sparx_agency.tasks.planning.objnav_benchmark.errors import ResultsError
from sparx_agency.tasks.planning.objnav_benchmark.summaries import (
    BenchmarkSummary,
    GroupSummary,
    PairedComparison,
    ReportedResult,
)

NAN = float("nan")
INF = float("inf")


def group(key="all", n=4, **overrides):
    """A valid group of ``n`` episodes; only what a test varies is a parameter."""
    fields = dict(key=key, n_episodes=n, success_rate=0.5,
                  success_rate_ci=(0.15, 0.85), spl=0.4, spl_ci=(0.1, 0.7),
                  soft_spl=0.45, distance_to_goal_m=1.2, n_unreachable_end=0,
                  mean_steps=120.0, mean_path_length_m=9.5)
    fields.update(overrides)
    return GroupSummary(**fields)


def summary(**overrides):
    """A valid four-episode summary: two categories, no scene breakdown, one agent error."""
    fields = dict(benchmark="hm3d_v2", split="val", agent="agent_a",
                  overall=group(),
                  by_category=(group("bed", 1), group("chair", 3)), by_scene=(),
                  terminations={"stop": 3, "step_limit": 0, "agent_error": 1},
                  agent_errors=1, confidence=0.95)
    fields.update(overrides)
    return BenchmarkSummary(**fields)


def comparison(**overrides):
    """A valid comparison of two agents on four episodes."""
    fields = dict(first_agent="agent_a", second_agent="agent_b", n_episodes=4,
                  success_rate_first=0.5, success_rate_second=0.25,
                  spl_first=0.4, spl_second=0.2, spl_difference=0.4 - 0.2,
                  success_p_value=0.625, spl_p_value=0.5, spl_wins=2,
                  spl_ties=1, spl_losses=1)
    fields.update(overrides)
    return PairedComparison(**fields)


# -- GroupSummary --------------------------------------------------------------

def test_a_valid_group_keeps_its_intervals_as_tuples():
    """Intervals read back from JSON are lists; stored as tuples they compare equal to a fresh run's."""
    built = group(success_rate_ci=[0.15, 0.85])
    assert built.success_rate_ci == (0.15, 0.85)
    assert group(distance_to_goal_m=None).distance_to_goal_m is None


@pytest.mark.parametrize("overrides", [
    {"key": " "}, {"n": 0}, {"n": True}, {"success_rate": 46.2},
    {"spl": NAN}, {"success_rate_ci": (0.9, 0.1)}, {"spl_ci": (0.1,)},
    {"distance_to_goal_m": INF}, {"n_unreachable_end": 5},
    {"mean_steps": -1.0}],
    ids=["blank_key", "empty", "bool_count", "percent", "nan_spl",
         "unordered_interval", "short_interval", "infinite_distance",
         "more_unreachable_than_episodes", "negative_mean"])
def test_a_group_that_cannot_describe_real_episodes_is_refused(overrides):
    """Each is a caller bug that would print a table row no run could have produced."""
    with pytest.raises(ResultsError, match="GroupSummary"):
        group(**overrides)


# -- BenchmarkSummary ----------------------------------------------------------

def test_a_valid_summary_builds_and_an_absent_breakdown_is_allowed():
    """Empty means the breakdown was not computed; it must not be read as zero episodes."""
    built = summary()
    assert [g.key for g in built.by_category] == ["bed", "chair"]
    assert built.by_scene == ()


@pytest.mark.parametrize("name", ["by_category", "by_scene"])
def test_a_breakdown_must_account_for_every_episode_exactly_once(name):
    """Categories that hold three of four episodes hide the fourth from every per-category number."""
    with pytest.raises(ResultsError, match="must partition the run's 4 episodes"):
        summary(**{name: (group("bed", 1), group("chair", 2))})


@pytest.mark.parametrize("terminations,agent_errors", [
    ({"stop": 4, "step_limit": 0}, 0),
    ({"stop": 2, "step_limit": 0, "agent_error": 1}, 1),
    ({"stop": 3, "step_limit": 0, "agent_error": 1}, 0)],
    ids=["kind_missing", "short_of_the_run", "agent_errors_disagree"])
def test_terminations_must_count_every_kind_and_add_up_to_the_run(terminations,
                                                                   agent_errors):
    """A termination table that does not add up hides episodes, or crashes, from the reader."""
    with pytest.raises(ResultsError, match="BenchmarkSummary"):
        summary(terminations=terminations, agent_errors=agent_errors)


@pytest.mark.parametrize("overrides", [
    {"confidence": 0.0}, {"confidence": 95}, {"confidence": True},
    {"agent": " "}, {"overall": {"key": "all"}}],
    ids=["zero_confidence", "percent_confidence", "bool_confidence",
         "blank_agent", "overall_not_a_group"])
def test_a_summary_with_a_malformed_field_is_refused(overrides):
    """A confidence of 95 or a blank agent name prints a table row that looks fine and is not."""
    with pytest.raises(ResultsError, match="BenchmarkSummary"):
        summary(**overrides)


# -- PairedComparison ----------------------------------------------------------

def test_a_valid_comparison_builds():
    """The fixture itself must be a comparison a real pair of runs could produce."""
    assert comparison().spl_difference == pytest.approx(0.2)


@pytest.mark.parametrize("overrides,message", [
    ({"second_agent": "agent_a"}, "with itself"),
    ({"spl_difference": 0.3}, "is not spl_first - spl_second"),
    ({"spl_losses": 2}, "must add up to"),
    ({"spl_p_value": 1.5}, "must lie in"),
    ({"n_episodes": 0}, "must be positive")],
    ids=["same_agent", "wrong_difference", "tallies_do_not_add_up",
         "p_value_above_one", "no_episodes"])
def test_a_comparison_that_contradicts_itself_is_refused(overrides, message):
    """A difference that is not the difference of the means would be reported as a result."""
    with pytest.raises(ResultsError, match=message):
        comparison(**overrides)


# -- ReportedResult ------------------------------------------------------------

def test_a_reported_result_without_a_soft_spl_is_allowed():
    """Most papers do not report SoftSPL; a missing value is None, never 0."""
    result = ReportedResult(method="A", benchmark="hm3d_v2", split="val",
                            success_rate=0.5, spl=0.25, source="Paper A, Table 1")
    assert result.soft_spl is None and result.notes == ""


@pytest.mark.parametrize("overrides,message", [
    ({"success_rate": 46.2}, "divide by 100"),
    ({"source": " "}, "source"),
    ({"notes": None}, "notes must be a string"),
    ({"distance_to_goal_m": -1.0}, "distance_to_goal_m")],
    ids=["percent", "blank_source", "notes_not_text", "negative_distance"])
def test_a_reported_result_that_cannot_be_checked_or_compared_is_refused(overrides,
                                                                       message):
    """A percent typed as a fraction, or a figure without its source, would sit in the table unnoticed."""
    fields = dict(method="A", benchmark="hm3d_v2", split="val",
                  success_rate=0.5, spl=0.25, source="Paper A, Table 1")
    fields.update(overrides)
    with pytest.raises(ResultsError, match=message):
        ReportedResult(**fields)
