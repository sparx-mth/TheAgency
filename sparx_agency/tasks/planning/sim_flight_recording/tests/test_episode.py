"""The parts of an episode that can be judged without a simulator."""
from __future__ import annotations

import math

import pytest

from sparx_agency.tasks.planning.sim_flight_recording import episode
from sparx_agency.tasks.planning.sim_flight_recording.path_follower import FollowSpec


def test_only_a_landing_counts_as_a_good_episode():
    assert episode.EpisodeResult(outcome=episode.OUTCOME_LANDED).ok
    for bad in (episode.OUTCOME_CRASHED, episode.OUTCOME_ARM_TIMEOUT,
                episode.OUTCOME_STALLED, episode.OUTCOME_OFFBOARD_LOST,
                episode.OUTCOME_FLIGHT_TIMEOUT, episode.OUTCOME_LAND_TIMEOUT,
                episode.OUTCOME_MISSED_GOAL):
        assert not episode.EpisodeResult(outcome=bad).ok, bad


def test_a_landing_short_of_the_goal_is_not_a_good_episode():
    """The one that mattered: a drone that never left the pad still 'lands'."""
    assert episode.OUTCOME_MISSED_GOAL not in episode.GOOD_OUTCOMES


def test_the_flight_budget_grows_with_the_route():
    short = episode.flight_budget_s(5.0)
    long = episode.flight_budget_s(50.0)
    assert long > short > 0.0
    # Generous enough that the cruise speed plus takeoff and landing fits.
    assert short >= 5.0 / FollowSpec().cruise_speed + 30.0


def test_the_commanded_heading_is_never_derived_from_the_position_error():
    """The loop that made the aircraft orbit its waypoint must stay closed off."""
    assert not hasattr(episode, "_tracking_yaw"), (
        "a live bearing-to-waypoint heading closes a yaw/position feedback loop; "
        "the follower's rate-limited heading is the fix"
    )


def test_a_stall_is_caught_well_inside_the_flight_budget():
    """A stalled aircraft must not burn the whole route's budget standing still."""
    assert episode.STALL_WINDOW_S < episode.flight_budget_s(5.0)


def test_hold_commands_motion_back_toward_the_hold_point():
    vx, vy, vz = episode.hold_velocity((0.0, 0.0, 1.0), (3.0, 0.0, 1.5), FollowSpec())
    assert vx > 0.0 and vy == pytest.approx(0.0)
    assert vz > 0.0, "and climbs toward the target altitude"


def test_hold_is_quiet_once_it_is_there():
    vx, vy, _vz = episode.hold_velocity((2.0, 2.0, 1.5), (2.0, 2.0, 1.5), FollowSpec())
    assert (vx, vy) == pytest.approx((0.0, 0.0))


def test_hold_never_commands_a_speed_the_autopilot_ignores():
    spec = FollowSpec()
    vx, vy, _vz = episode.hold_velocity((0.0, 0.0, 1.5), (1.0, 0.0, 1.5), spec)
    assert math.hypot(vx, vy) >= spec.min_speed


def test_hold_is_clamped_to_the_cruise_speed():
    spec = FollowSpec()
    vx, vy, _vz = episode.hold_velocity((0.0, 0.0, 1.5), (100.0, 0.0, 1.5), spec)
    assert math.hypot(vx, vy) == pytest.approx(spec.cruise_speed)


def test_the_climb_rate_is_clamped_separately_from_the_cruise_speed():
    spec = FollowSpec()
    _vx, _vy, vz = episode.hold_velocity((0.0, 0.0, 0.0), (0.0, 0.0, 50.0), spec)
    assert vz == pytest.approx(spec.max_climb_rate)
    _vx, _vy, vz = episode.hold_velocity((0.0, 0.0, 50.0), (0.0, 0.0, 0.0), spec)
    assert vz == pytest.approx(-spec.max_climb_rate)


def test_the_heading_is_slewed_never_stepped():
    """The rate limit is what makes the camera pan instead of whip."""
    rate, dt = 0.45, 0.1
    yaw = episode.slew_towards(0.0, math.pi, rate, dt)
    assert yaw == pytest.approx(rate * dt)


def test_slewing_takes_the_short_way_round():
    """170 deg to -170 deg is a 20 deg turn through 180, not 340 deg back through 0."""
    # A step large enough to cover the short way but not the long way.
    yaw = episode.slew_towards(math.radians(170), math.radians(-170),
                               math.radians(30), 1.0)
    assert yaw == pytest.approx(math.radians(-170))

    # And with a small step, it moves the short way: past 180, so it wraps negative.
    yaw = episode.slew_towards(math.radians(175), math.radians(-175),
                               math.radians(6), 1.0)
    assert yaw < 0.0


def test_slewing_stops_on_arrival_rather_than_overshooting():
    assert episode.slew_towards(1.0, 1.01, 10.0, 1.0) == pytest.approx(1.01)


def test_a_flight_that_leaves_its_route_is_a_named_failure():
    assert episode.OUTCOME_OFF_ROUTE not in episode.GOOD_OUTCOMES
