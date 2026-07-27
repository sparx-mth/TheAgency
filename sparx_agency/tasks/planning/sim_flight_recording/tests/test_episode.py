"""The parts of an episode that can be judged without a simulator."""
from __future__ import annotations

import math

import pytest

from sparx_agency.tasks.planning.sim_flight_recording import episode


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
    # Generous enough that PX4's ~1 m/s cruise plus takeoff and landing fits.
    assert short >= 5.0 / episode.cruise_speed_hint() + 30.0


def test_the_commanded_heading_comes_from_the_plan_not_the_position_error():
    """The loop that made the aircraft orbit its waypoint must stay closed off."""
    assert not hasattr(episode, "_tracking_yaw"), (
        "a live bearing-to-waypoint heading closes a yaw/position feedback loop; "
        "the per-leg heading from episode_plan.waypoints_with_heading is the fix"
    )


def test_a_stall_is_detected_faster_than_a_waypoint_times_out():
    """A stalled aircraft must not spend the whole route's budget standing still."""
    from sparx_agency.tasks.planning.sim_flight_recording.waypoint_mission import TIMEOUT_S

    assert episode.STALL_WINDOW_S < TIMEOUT_S


def test_guidance_points_straight_at_the_target():
    vx, vy, vz = episode.guidance_velocity((0.0, 0.0, 1.5), (3.0, 4.0, 1.5),
                                           cruise_speed=1.0)
    assert math.atan2(vy, vx) == pytest.approx(math.atan2(4.0, 3.0))
    assert vz == pytest.approx(0.0)


def test_guidance_is_clamped_to_the_cruise_speed():
    vx, vy, _vz = episode.guidance_velocity((0.0, 0.0, 1.5), (100.0, 0.0, 1.5),
                                            cruise_speed=1.0)
    assert math.hypot(vx, vy) == pytest.approx(1.0)


def test_guidance_slows_down_as_it_arrives():
    far = episode.guidance_velocity((0.0, 0.0, 1.5), (20.0, 0.0, 1.5), 10.0)
    near = episode.guidance_velocity((15.0, 0.0, 1.5), (20.0, 0.0, 1.5), 10.0)
    assert math.hypot(*far[:2]) > math.hypot(*near[:2])
    assert math.hypot(*near[:2]) == pytest.approx(5.0 * episode.GUIDANCE_GAIN)


def test_guidance_never_commands_a_speed_the_autopilot_ignores():
    """A pure proportional taper leaves the aircraft hovering short of the goal."""
    _vx, _vy, _vz = episode.guidance_velocity((0.0, 0.0, 1.5), (0.05, 0.0, 1.5), 1.0)
    vx, vy, _vz = episode.guidance_velocity((0.0, 0.0, 1.5), (1.0, 0.0, 1.5), 1.0)
    assert math.hypot(vx, vy) >= episode.MIN_GUIDANCE_SPEED


def test_guidance_commands_no_motion_inside_the_arrival_radius():
    assert episode.guidance_velocity((2.0, 2.0, 1.5), (2.0, 2.0, 1.5), 1.0) == \
        pytest.approx((0.0, 0.0, 0.0))
    # Just inside the radius: settle, do not keep pushing past it and back.
    assert episode.guidance_velocity((2.5, 2.0, 1.5), (2.0, 2.0, 1.5), 1.0,
                                     arrival_radius_m=0.8)[:2] == pytest.approx((0.0, 0.0))


def test_the_acceptance_radius_matches_what_the_airframe_can_hold():
    """0.35 m looked right and failed three flights that had already arrived."""
    from sparx_agency.tasks.planning.sim_flight_recording import waypoint_mission

    assert waypoint_mission.FINAL_RADIUS_M >= 0.7


def test_the_climb_rate_is_clamped_separately_from_the_cruise_speed():
    _vx, _vy, vz = episode.guidance_velocity((0.0, 0.0, 0.0), (0.0, 0.0, 50.0),
                                             cruise_speed=1.0)
    assert vz == pytest.approx(episode.MAX_CLIMB_RATE)
    _vx, _vy, vz = episode.guidance_velocity((0.0, 0.0, 50.0), (0.0, 0.0, 0.0),
                                             cruise_speed=1.0)
    assert vz == pytest.approx(-episode.MAX_CLIMB_RATE)
