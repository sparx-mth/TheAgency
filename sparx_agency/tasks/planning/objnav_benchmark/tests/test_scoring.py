"""SPL, SoftSPL, and the checks that stop an adapter bug from moving them.

The worked examples are Anderson et al. 2018's; the edge cases are the corners
where habitat-lab divides by zero, so the harness has to fix a convention of
its own and hold to it.
"""
from __future__ import annotations

import math

import pytest

from sparx_agency.core.planning.objnav.types.measurement import (
    NATIVE_KEYS,
    TERMINATION_STEP_LIMIT,
    TERMINATION_STOP,
    EpisodeMeasurement,
)
from sparx_agency.tasks.planning.objnav_benchmark.errors import (
    HarnessError,
    ScoringError,
)
from sparx_agency.tasks.planning.objnav_benchmark.scoring import (
    path_length_3d,
    score_episode,
    soft_spl_score,
    spl_score,
)

INF = float("inf")
NAN = float("nan")


def measurement(success=True, stop_called=True, shortest=4.0, start=4.0,
                final=0.05, path=5.0, native=None):
    """A valid measurement; the defaults are a success with SPL 0.8."""
    return EpisodeMeasurement(
        success=success,
        stop_called=stop_called,
        termination=TERMINATION_STOP if stop_called else TERMINATION_STEP_LIMIT,
        steps=20,
        shortest_path_m=shortest,
        start_distance_to_goal_m=start,
        final_distance_to_goal_m=final,
        path_length_m=path,
        native_metrics=dict(native or {}),
    )


def mean(values):
    return sum(values) / len(values)


# -- SPL -------------------------------------------------------------------

def test_half_the_episodes_succeeding_on_optimal_paths_scores_half():
    """Anderson et al.'s first example: SPL is success-weighted, so 50% success on optimal paths is 0.5."""
    assert mean([spl_score(True, 5.0, 5.0), spl_score(False, 5.0, 5.0)]) == 0.5


def test_every_episode_succeeding_on_paths_twice_optimal_scores_half():
    """Anderson et al.'s second example: 100% success at twice the shortest path is 0.5."""
    assert mean([spl_score(True, 3.0, 6.0), spl_score(True, 7.0, 14.0)]) == 0.5


def test_half_succeeding_on_paths_twice_optimal_scores_a_quarter():
    """Anderson et al.'s third example: the two penalties multiply."""
    assert mean([spl_score(True, 4.0, 8.0), spl_score(False, 4.0, 8.0)]) == 0.25


def test_a_failure_scores_zero_spl_however_short_its_path():
    """SPL is gated on success; a short path on a failure must not earn credit."""
    assert spl_score(False, 4.0, 4.0) == 0.0


def test_a_path_shorter_than_the_reference_caps_spl_at_one():
    """A coarse grid can overestimate l; the ratio must still never exceed 1."""
    assert spl_score(True, 5.0, 4.5) == 1.0


def test_starting_on_the_goal_and_stopping_in_place_scores_one():
    """l == p == 0 is where habitat-lab raises; the convention takes the ratio as 1."""
    assert spl_score(True, 0.0, 0.0) == 1.0
    assert soft_spl_score(0.0, 0.0, 0.0, 0.0) == 1.0


def test_wandering_off_a_goal_the_agent_started_on_scores_zero_spl():
    """l == 0 < p is not a special case: the formula gives 0, and so must we."""
    assert spl_score(True, 0.0, 2.0) == 0.0


@pytest.mark.parametrize("success", ["false", 1, None])
def test_spl_refuses_a_success_that_is_not_a_bool(success):
    """bool("false") is True; a string success must fail, not score as a success."""
    with pytest.raises(ScoringError, match="bool"):
        spl_score(success, 4.0, 5.0)


@pytest.mark.parametrize("shortest,path", [
    (-1.0, 1.0), (1.0, -1.0), (NAN, 1.0), (INF, 1.0), (1.0, INF), ("4", 5.0)])
def test_spl_refuses_negative_and_non_finite_lengths(shortest, path):
    """A negative or infinite length is an adapter bug that would give a plausible SPL."""
    with pytest.raises(ScoringError):
        spl_score(True, shortest, path)


# -- SoftSPL ---------------------------------------------------------------

def test_soft_spl_multiplies_progress_by_path_efficiency():
    """Ending 1 m from a 4 m goal on a path twice optimal is 0.75 * 0.5."""
    assert soft_spl_score(4.0, 1.0, 4.0, 8.0) == pytest.approx(0.375)


def test_soft_spl_credits_progress_without_success_or_stop():
    """Habitat's SoftSPL is gated on neither; a timed-out approach still scores."""
    score = score_episode(measurement(success=False, stop_called=False,
                                      final=1.0, path=4.0))
    assert score.spl == 0.0
    assert score.soft_spl == pytest.approx(0.75)


def test_ending_farther_than_the_start_gives_zero_soft_spl():
    """max(0, ...) floors negative progress at 0 instead of a negative score."""
    assert soft_spl_score(2.0, 3.0, 2.0, 2.0) == 0.0


def test_a_zero_start_distance_gives_soft_success_one_only_at_the_goal():
    """d0 == 0 divides by zero in habitat-lab; the convention is 1 at the goal, else 0."""
    assert soft_spl_score(0.0, 0.0, 2.0, 2.0) == 1.0
    assert soft_spl_score(0.0, 0.5, 2.0, 2.0) == 0.0


def test_an_unreachable_end_gives_zero_soft_spl():
    """dT == inf is NaN in habitat-lab; it must score 0, not poison the mean."""
    assert soft_spl_score(4.0, INF, 4.0, 4.0) == 0.0
    assert soft_spl_score(0.0, INF, 0.0, 0.0) == 0.0


@pytest.mark.parametrize("start,final,shortest,path", [
    (NAN, 1.0, 1.0, 1.0), (INF, 1.0, 1.0, 1.0), (1.0, NAN, 1.0, 1.0),
    (1.0, -1.0, 1.0, 1.0), (1.0, 1.0, -1.0, 1.0), (1.0, 1.0, 1.0, INF)])
def test_soft_spl_refuses_nan_negative_and_infinite_references(
        start, final, shortest, path):
    """Only dT may be infinite; any other bad value is a measurement bug."""
    with pytest.raises(ScoringError):
        soft_spl_score(start, final, shortest, path)


# -- path length -----------------------------------------------------------

def test_path_length_sums_three_dimensional_chords():
    """A climb counts: p is measured in 3-D, as habitat-lab measures it."""
    assert path_length_3d([(0.0, 0.0, 0.0), (3.0, 4.0, 0.0),
                           (3.0, 4.0, 12.0)]) == pytest.approx(17.0)


def test_turning_in_place_adds_no_path_length():
    """Turns, tilts and STOP repeat a position; they must add exactly 0."""
    assert path_length_3d([(1.0, 1.0, 0.0)] * 4) == 0.0
    assert path_length_3d([(1.0, 1.0, 0.0)]) == 0.0


def test_path_length_needs_the_reset_position():
    """An empty trace means the runner lost the reset pose, not that p is 0."""
    with pytest.raises(ScoringError, match="reset position"):
        path_length_3d([])


@pytest.mark.parametrize("positions", [
    [(0.0, 0.0)], [(0.0, 0.0, NAN)], [("a", 0.0, 0.0)],
    [(0.0, 0.0, 0.0, 0.0)], [1.0]])
def test_path_length_refuses_a_malformed_position(positions):
    """A 2-D or non-finite pose would silently shorten or poison p."""
    with pytest.raises(ScoringError, match="position 0"):
        path_length_3d(positions)


# -- score_episode ---------------------------------------------------------

def test_a_clean_episode_scores_as_the_formulas_say():
    """The score must be exactly the two formulas applied to the measurement."""
    score = score_episode(measurement(), 5.0)
    assert score.success is True
    assert score.spl == pytest.approx(0.8)
    assert score.soft_spl == pytest.approx((1.0 - 0.05 / 4.0) * 0.8)
    assert (score.distance_to_goal_m, score.path_length_m,
            score.shortest_path_m, score.observed_path_length_m) == (
                0.05, 5.0, 4.0, 5.0)
    assert score.native_checked == ()


def test_success_without_stop_is_refused_when_the_protocol_requires_stop():
    """Habitat and RoboTHOR require STOP; an adapter granting success without it inflates SR."""
    with pytest.raises(ScoringError, match="STOP"):
        score_episode(measurement(success=True, stop_called=False))


def test_success_without_stop_is_credited_when_the_protocol_allows_it():
    """SemExp's Gibson evaluator grants success on proximity; the option must honour it."""
    score = score_episode(measurement(success=True, stop_called=False),
                          require_stop_for_success=False)
    assert score.success is True
    assert score.spl == pytest.approx(0.8)


def test_a_path_length_the_poses_disagree_with_is_refused_as_an_adapter_bug():
    """A unit or accounting bug shows first as p disagreeing with the observed poses."""
    with pytest.raises(ScoringError, match="unit or accounting bug"):
        score_episode(measurement(path=5.0), 5.5)


def test_a_path_length_within_tolerance_is_accepted_and_recorded():
    """Float rounding between simulator and runner must not fail an episode."""
    score = score_episode(measurement(path=5.0), 5.0004)
    assert score.observed_path_length_m == 5.0004


def test_no_path_tolerance_records_the_observed_length_without_checking_it():
    """Turning the check off must still keep the observed value for the record."""
    score = score_episode(measurement(path=5.0), 50.0, path_tolerance_m=None)
    assert score.observed_path_length_m == 50.0


@pytest.mark.parametrize("observed", [-1.0, NAN, INF])
def test_a_malformed_observed_path_length_is_refused(observed):
    """A negative or non-finite observed p is a runner bug, not a measurement."""
    with pytest.raises(ScoringError, match="observed_path_length_m"):
        score_episode(measurement(), observed)


def test_native_metrics_that_agree_are_listed_in_canonical_order():
    """native_checked follows NATIVE_KEYS whatever order the adapter filled the dict in."""
    native = {"distance_to_goal": 0.05, "soft_spl": 0.79, "spl": 0.8,
              "success": 1.0}
    score = score_episode(measurement(native=native))
    assert score.native_checked == NATIVE_KEYS


def test_only_the_native_keys_reported_are_checked():
    """An adapter that reports only SPL must not be credited with checking the rest."""
    assert score_episode(measurement(native={"spl": 0.8})).native_checked == (
        "spl",)


def test_a_native_success_that_disagrees_is_refused():
    """The simulator and the adapter disagreeing on success is the costliest silent bug."""
    with pytest.raises(ScoringError, match="'success'"):
        score_episode(measurement(native={"success": 0.0}))


def test_a_native_success_that_is_not_zero_or_one_is_refused():
    """A fractional success means the adapter mapped the wrong simulator key."""
    with pytest.raises(ScoringError, match="exactly 0 or 1"):
        score_episode(measurement(native={"success": 0.5}))


def test_a_native_spl_beyond_tolerance_is_refused_naming_both_values():
    """The message must show both numbers, or nobody can tell which side is wrong."""
    with pytest.raises(ScoringError, match=r"'spl'.*0\.7.*0\.8"):
        score_episode(measurement(native={"spl": 0.7}))


def test_a_native_spl_within_tolerance_passes():
    """float32 simulator arithmetic differs in the fifth digit; that is not a bug."""
    assert score_episode(measurement(native={"spl": 0.80005})).native_checked == (
        "spl",)


def test_a_native_distance_of_inf_matches_an_unreachable_end():
    """inf - inf is NaN, which would pass a naive tolerance check for the wrong reason."""
    score = score_episode(measurement(success=False, final=INF,
                                      native={"distance_to_goal": INF}))
    assert score.native_checked == ("distance_to_goal",)
    assert score.soft_spl == 0.0


@pytest.mark.parametrize("final,native", [(INF, 3.0), (3.0, INF)])
def test_a_finite_native_distance_against_an_infinite_one_is_refused(final,
                                                                     native):
    """Reachable on one side and unreachable on the other is a goal-set mismatch."""
    with pytest.raises(ScoringError, match="distance_to_goal"):
        score_episode(measurement(success=False, final=final,
                                  native={"distance_to_goal": native}))


@pytest.mark.parametrize("options", [
    {"require_stop_for_success": "false"},
    {"path_tolerance_m": -1.0},
    {"path_tolerance_m": NAN},
    {"native_tolerance": -1e-4},
    {"native_tolerance": INF},
])
def test_malformed_scoring_options_are_refused(options):
    """A config string 'false' for require_stop would silently read as True."""
    with pytest.raises(HarnessError):
        score_episode(measurement(), **options)


def test_score_episode_refuses_anything_but_a_measurement():
    """A dict that looks like a measurement skipped every validation the type does."""
    with pytest.raises(TypeError, match="EpisodeMeasurement"):
        score_episode({"success": True})


def test_every_score_is_a_finite_fraction_or_a_valid_distance():
    """The score feeds strict-JSON records; NaN anywhere would break them later."""
    score = score_episode(measurement(success=False, final=INF))
    assert all(math.isfinite(v) for v in (score.spl, score.soft_spl))
    assert score.distance_to_goal_m == INF
