"""The 3D law flies where the route goes, not where the nose already points.

Velocity used to be commanded along the vehicle's own heading, scaled by the
cosine of its error against the carrot's bearing, so a rate-limited yaw gated all
motion. Four distinct failures came out of that one coupling, and each one has a
test here: corners taken wide, the last metre to a goal unreachable, a path that
wove side to side, and reversals clamped to a standstill.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from sparx_agency.core.planning.trackers.pure_pursuit import algorithm as alg


def _velocity(pos, target, yaw, speed=1.0, max_speed_z=1.0, hold=False):
    return alg.compute_velocity_3d(np.array(pos, dtype=float),
                                   np.array(target, dtype=float),
                                   yaw, speed, max_speed_z, hold_for_turn=hold)


# --- velocity goes at the carrot, whatever the heading ------------------------

def test_velocity_points_at_the_carrot_not_along_the_heading():
    """A multirotor translates in any direction; the camera is a separate command."""
    vx, vy, _, _ = _velocity((0, 0, 0), (0, 5, 0), yaw=0.0)   # carrot due north
    assert vy == pytest.approx(1.0)
    assert vx == pytest.approx(0.0)


def test_full_speed_is_commanded_even_facing_ninety_degrees_away():
    """The old law scaled by cos(error), so this was zero and the vehicle stalled
    a metre short of every goal it approached side-on."""
    vx, vy, _, _ = _velocity((0, 0, 0), (0, 5, 0), yaw=0.0, speed=1.2)
    assert math.hypot(vx, vy) == pytest.approx(1.2)


def test_a_carrot_behind_the_vehicle_is_flown_to_rather_than_clamped_out():
    """cos(180 deg) is negative and was clamped to zero, so a reversal simply
    never happened."""
    vx, vy, _, _ = _velocity((0, 0, 0), (-5, 0, 0), yaw=0.0)
    assert vx == pytest.approx(-1.0)
    assert vy == pytest.approx(0.0)


def test_the_commanded_speed_does_not_depend_on_the_heading_at_all():
    speeds = [math.hypot(*_velocity((0, 0, 0), (3, 3, 0), yaw=yaw)[:2])
              for yaw in np.linspace(-math.pi, math.pi, 12)]
    assert max(speeds) - min(speeds) < 1e-9


def test_the_bearing_to_the_carrot_is_still_reported_for_yaw():
    *_, bearing = _velocity((0, 0, 0), (0, 5, 0), yaw=0.0)
    assert bearing == pytest.approx(math.pi / 2)


def test_climb_rate_is_scaled_by_the_vertical_share_of_the_distance():
    _, _, vz, _ = _velocity((0, 0, 0), (0, 0, 5), yaw=0.0, speed=1.0, max_speed_z=2.0)
    assert vz == pytest.approx(1.0)


def test_climb_rate_is_clamped_separately():
    _, _, vz, _ = _velocity((0, 0, 0), (0, 0, 5), yaw=0.0, speed=5.0, max_speed_z=0.4)
    assert vz == pytest.approx(0.4)


def test_a_carrot_on_top_of_the_vehicle_commands_nothing():
    assert _velocity((1, 1, 1), (1, 1, 1), yaw=0.3)[:3] == (0.0, 0.0, 0.0)


# --- holding still to turn ---------------------------------------------------

def test_holding_for_a_turn_stops_horizontal_motion():
    vx, vy, _, _ = _velocity((0, 0, 0), (5, 0, 0), yaw=0.0, hold=True)
    assert (vx, vy) == (0.0, 0.0)


def test_holding_for_a_turn_still_holds_altitude():
    """Waiting to rotate is no reason to stop climbing."""
    _, _, vz, _ = _velocity((0, 0, 0), (5, 0, 3), yaw=0.0, speed=1.0,
                            max_speed_z=1.0, hold=True)
    assert vz > 0.0


def test_holding_for_a_turn_still_reports_where_to_point():
    *_, bearing = _velocity((0, 0, 0), (0, -5, 0), yaw=0.0, hold=True)
    assert bearing == pytest.approx(-math.pi / 2)


# --- pointing the heading down the route, not at the carrot -------------------

def _corner(step=0.1):
    """A path that runs 5 m east then turns 90 degrees north."""
    east = [(x, 0.0, 0.0) for x in np.arange(0.0, 5.0, step)]
    north = [(5.0, y, 0.0) for y in np.arange(0.0, 5.0, step)]
    return np.array(east + north, dtype=float)


def test_the_heading_aims_past_the_corner_rather_than_at_it():
    """Aimed at a carrot 1 m ahead the vehicle reaches the corner still pointing
    east; aimed 4 m ahead it is already turning north."""
    xyz = _corner()
    pos = np.array([4.0, 0.0, 0.0])
    near = alg.route_heading_3d(xyz, pos, 40, 1.0)
    far = alg.route_heading_3d(xyz, pos, 40, 4.0)
    assert abs(near) < math.radians(30), "1 m ahead is still down the first leg"
    assert far > math.radians(30), "4 m ahead has to account for the turn"


def test_the_heading_is_straight_down_a_straight_route():
    xyz = np.array([(x, 0.0, 0.0) for x in np.arange(0.0, 10.0, 0.1)])
    heading = alg.route_heading_3d(xyz, np.array([1.0, 0.0, 0.0]), 10, 3.0)
    assert heading == pytest.approx(0.0, abs=1e-6)


def test_no_heading_is_offered_when_the_route_ahead_is_too_short():
    """At the very end there is no meaningful direction left, and snapping the
    camera to a noisy one is worse than holding the heading."""
    xyz = np.array([(0.0, 0.0, 0.0), (0.05, 0.0, 0.0)])
    assert alg.route_heading_3d(xyz, np.array([0.05, 0.0, 0.0]), 0, 3.0) is None


def test_asking_beyond_the_end_of_the_route_aims_at_its_end():
    xyz = np.array([(x, 0.0, 0.0) for x in np.arange(0.0, 3.0, 0.1)])
    heading = alg.route_heading_3d(xyz, np.array([0.0, -2.0, 0.0]), 0, 999.0)
    assert heading == pytest.approx(math.atan2(2.0, 2.9), abs=0.05)


# --- the stop-and-turn latch --------------------------------------------------

def _tracker(stop_deg, resume_deg):
    from sparx_agency.core.planning.trackers.pure_pursuit import PurePursuitTracker3D
    from sparx_agency.core.planning.trackers.pure_pursuit.params import (
        PurePursuitParams3D,
    )

    return PurePursuitTracker3D(PurePursuitParams3D(
        stop_turn_rad=math.radians(stop_deg),
        resume_turn_rad=math.radians(resume_deg)))


def test_a_small_heading_error_never_stops_the_flight():
    """Ordinary curves must not trip it, or the flight is a series of pauses."""
    tracker = _tracker(40, 12)
    assert not tracker._update_turning(math.radians(20), tracker.params)


def test_a_large_heading_error_stops_to_turn():
    tracker = _tracker(40, 12)
    assert tracker._update_turning(math.radians(80), tracker.params)


def test_the_turn_is_held_until_the_lower_threshold():
    """Releasing at the same threshold chatters between flying and turning."""
    tracker = _tracker(40, 12)
    tracker._update_turning(math.radians(80), tracker.params)          # latched
    assert tracker._update_turning(math.radians(30), tracker.params)   # still turning
    assert not tracker._update_turning(math.radians(5), tracker.params)


def test_the_direction_of_the_error_does_not_matter():
    tracker = _tracker(40, 12)
    assert tracker._update_turning(math.radians(-80), tracker.params)


def test_zero_disables_stopping_altogether():
    tracker = _tracker(0, 0)
    assert not tracker._update_turning(math.pi, tracker.params)
