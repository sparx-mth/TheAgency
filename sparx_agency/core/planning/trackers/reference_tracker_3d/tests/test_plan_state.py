"""Tests for the finished-plan classifier and the forward-progression helpers.

These are the two things the tracker could not previously tell apart: a plan
that is still advancing, and a dead setpoint being restamped at 100 Hz. Every
case here is taken from a real failure in run ``nav_debug_20260906_235809``.
"""
import math

import pytest

from sparx_agency.core.common.types import KinematicLimits, TrajectoryPoint
from sparx_agency.core.planning.trackers.reference_tracker_3d import (
    ReferenceTracker3D, ReferenceTrackerParams,
)
from sparx_agency.core.planning.trackers.reference_tracker_3d.plan_state import (
    ENDPOINT, FOLLOW, STALE, PlanState, PlanStateParams,
)
from sparx_agency.core.planning.trackers.reference_tracker_3d.progress import (
    limit_lead, split_error,
)

DT = 0.05


def _frozen(x=0.0, y=0.0, z=1.5, yaw=0.0):
    """A traj_server frozen endpoint: a position, and no motion at all."""
    return TrajectoryPoint(t=0.0, x=x, y=y, z=z, vx=0.0, vy=0.0, vz=0.0, yaw=yaw)


def _moving(speed=0.5, yaw=0.0):
    """A live reference travelling along +x."""
    return TrajectoryPoint(t=0.0, x=0.0, y=0.0, z=1.5, vx=speed, vy=0.0, vz=0.0, yaw=yaw)


# ── the classifier ───────────────────────────────────────────────────
def test_a_moving_reference_is_followed():
    plan = PlanState()
    for _ in range(100):
        assert plan.update(_moving(), 0.01, 1.0, DT) == FOLLOW


def test_a_frozen_reference_becomes_the_endpoint_after_the_grace_period():
    """The defect: a fresh stamp on a dead setpoint is not a live plan."""
    plan = PlanState(PlanStateParams(endpoint_grace_s=0.5))
    # Fresh every tick, exactly as traj_server republishes it.
    for _ in range(10):                       # 0.50 s -- still inside the grace
        assert plan.update(_frozen(), 0.003, 1.0, DT) == FOLLOW
    assert plan.update(_frozen(), 0.003, 1.0, DT) == ENDPOINT


def test_a_brief_stop_inside_a_trajectory_is_not_the_end_of_it():
    """A slow curve passes through zero speed; that is not a finished plan."""
    plan = PlanState(PlanStateParams(endpoint_grace_s=0.5))
    for _ in range(5):
        assert plan.update(_frozen(), 0.003, 1.0, DT) == FOLLOW
    assert plan.update(_moving(), 0.003, 1.0, DT) == FOLLOW
    # The accumulator restarted, so the next freeze gets a full grace period.
    for _ in range(10):
        assert plan.update(_frozen(), 0.003, 1.0, DT) == FOLLOW


def test_a_rotating_plan_is_still_advancing():
    """An exploration planner aims its sensor: rotating on the spot is normal.

    Judging on translation alone would call every camera-aiming segment
    finished and stop the aircraft turning.
    """
    plan = PlanState(PlanStateParams(endpoint_grace_s=0.5))
    yaw = 0.0
    for _ in range(100):
        yaw += 0.5 * DT
        assert plan.update(_frozen(yaw=yaw), 0.003, 1.0, DT) == FOLLOW


def test_a_new_trajectory_does_not_inherit_the_previous_freeze():
    """Otherwise a trajectory arriving during a freeze is dead on arrival."""
    plan = PlanState(PlanStateParams(endpoint_grace_s=0.5))
    for _ in range(40):
        plan.update(_frozen(), 0.003, 1.0, DT, trajectory_id=7)
    assert plan.update(_frozen(), 0.003, 1.0, DT, trajectory_id=7) == ENDPOINT
    assert plan.update(_frozen(), 0.003, 1.0, DT, trajectory_id=8) == FOLLOW


def test_an_old_reference_is_stale_not_an_endpoint():
    """A dead planner and a parked one need different responses."""
    plan = PlanState()
    assert plan.update(_moving(), 5.0, 1.0, DT) == STALE
    assert plan.update(None, 0.0, 1.0, DT) == STALE


@pytest.mark.parametrize("kwargs", [
    {"stopped_speed_mps": -1.0},
    {"stopped_yaw_rate_rad_s": -1.0},
    {"endpoint_grace_s": -1.0},
    {"endpoint_reach_m": 0.0},
])
def test_invalid_plan_state_params_are_rejected(kwargs):
    with pytest.raises(ValueError):
        PlanStateParams(**kwargs)


# ── the tracker's response ───────────────────────────────────────────
def _tracker():
    return ReferenceTracker3D(ReferenceTrackerParams(
        plan_state=PlanStateParams(endpoint_grace_s=0.2),
        command_smoothing_alpha=1.0,
        limits=KinematicLimits(max_speed_xy=1.5, max_speed_z=0.8,
                               max_yaw_rate=math.radians(90.0))))


def _settle(tracker, reference, position, ticks=20):
    out = None
    for _ in range(ticks):
        out = tracker.update(reference, position, yaw=0.0, dt=DT, reference_age=0.003)
    return out


def _fly_to(tracker, reference, start, ticks=400):
    """Closed loop: integrate the commanded velocity, so the profile is tested.

    A fixed-position harness cannot tell a controller that decelerates onto a
    point from one that is merely pointed at it.
    """
    position = list(start)
    out = None
    peak = 0.0
    for _ in range(ticks):
        out = tracker.update(reference, tuple(position), yaw=0.0, dt=DT,
                             reference_age=0.003)
        for axis, v in enumerate(out.velocity()):
            position[axis] += v * DT
        peak = max(peak, math.hypot(out.vx, out.vy))
    return out, tuple(position), peak


def test_a_finished_plan_flies_to_its_last_waypoint_and_stops_there():
    """The requirement: follow the path only up to the last valid waypoint."""
    tracker = _tracker()
    endpoint = _frozen(x=1.0, y=0.0, z=1.5)
    out, position, _ = _fly_to(tracker, endpoint, (0.0, 0.0, 1.5))
    assert out.past_end and out.holding
    assert position[0] == pytest.approx(1.0, abs=0.05)      # arrived
    assert out.vx == pytest.approx(0.0, abs=0.01)           # and stopped


def test_a_finished_plan_is_not_chased_at_cruise():
    """Frame 642: 1.74 m from a dead endpoint, still commanding 0.92 m/s.

    The old code took the tracking branch on a frozen reference, where the
    saturating position term is a constant-magnitude pull with no arrival
    criterion. The endpoint hold must approach proportionally instead, so the
    peak command scales with the gap rather than railing.
    """
    tracker = _tracker()
    near, _, peak_near = _fly_to(tracker, _frozen(x=0.4, y=0.0, z=1.5),
                                 (0.0, 0.0, 1.5), ticks=200)
    tracker.reset()
    far, _, peak_far = _fly_to(tracker, _frozen(x=1.74, y=0.0, z=1.5),
                               (0.0, 0.0, 1.5), ticks=200)
    assert near.past_end and far.past_end
    # Proportional, not bang-bang: a 0.4 m gap must not command what a 1.74 m
    # gap does. Under the old constant-magnitude pull these were equal.
    assert peak_near < 0.6 * peak_far
    # NOTE what is deliberately NOT asserted: that the approach is slow. It is
    # not -- at 1.74 m the position loop still commands ~1 m/s, the same as
    # before. The fix is not a gentler approach, it is that the aircraft now
    # ARRIVES AND STOPS instead of chasing a dead setpoint indefinitely, and
    # that it refuses the approach at all beyond endpoint_reach_m.
    assert peak_far <= _tracker().params.limits.max_speed_xy


def test_a_finished_plan_far_away_holds_position_instead():
    """Nothing has vouched for the ground in between since the plan ended."""
    tracker = _tracker()
    out = _settle(tracker, _frozen(x=9.0, y=0.0, z=1.5), (0.0, 0.0, 1.5))
    assert out.past_end and out.holding
    assert out.vx == pytest.approx(0.0, abs=0.05)


def test_the_endpoint_hold_does_not_ratchet():
    """The hold point is latched, so a drifting aircraft returns to it."""
    tracker = _tracker()
    endpoint = _frozen(x=0.0, y=0.0, z=1.5)
    _settle(tracker, endpoint, (0.0, 0.0, 1.5))
    pushed = _settle(tracker, endpoint, (0.4, 0.0, 1.5), ticks=5)
    assert pushed.vx < 0.0          # comes back to the latched point
    # Asserted on the latch itself: a re-latching hold would have walked to the
    # pushed position, and the FOLLOW path never latches at all.
    assert tracker._hold == pytest.approx((0.0, 0.0, 1.5))
    assert pushed.past_end and pushed.holding


def test_the_endpoint_hold_is_abandoned_if_the_aircraft_ends_up_far_from_it():
    """Found by replaying the run: 93 ticks at 0.87 m/s toward a 3 m-away hold.

    The reach test used to run only against the incoming reference, so an
    aircraft pushed away from an already-latched endpoint kept flying metres
    back to a point no live plan had vouched for since.
    """
    tracker = _tracker()
    endpoint = _frozen(x=0.0, y=0.0, z=1.5)
    _settle(tracker, endpoint, (0.0, 0.0, 1.5))          # latched at the endpoint
    far = _settle(tracker, endpoint, (9.0, 0.0, 1.5), ticks=3)
    assert far.past_end
    assert far.vx == pytest.approx(0.0, abs=0.05)        # holds here, not there
    assert tracker._hold[0] == pytest.approx(9.0)


def test_a_new_trajectory_drops_the_previous_hold_point():
    """Frame 1251: the reference jumped 2.9 m backward on a handover."""
    tracker = _tracker()
    _settle(tracker, _frozen(x=5.0, y=0.0, z=1.5), (0.0, 0.0, 1.5))
    assert tracker._hold is not None
    tracker.on_new_trajectory()
    assert tracker._hold is None


def test_the_finished_plan_hold_reports_an_honest_error_split():
    """Reporting 0.0 lag here would hide the very overshoot that matters."""
    tracker = _tracker()
    tracker.update(_moving(speed=0.5), (0.0, 0.0, 1.5), yaw=0.0, dt=DT,
                   reference_age=0.003)
    out = _settle(tracker, _frozen(x=0.0, y=0.0, z=1.5), (0.6, 0.0, 1.5))
    assert out.past_end
    # The aircraft is 0.6 m PAST the endpoint along the plan's own direction.
    assert out.along_track_lag_m == pytest.approx(-0.6, abs=0.01)
    assert out.cross_track_error_m == pytest.approx(0.0, abs=0.01)


# ── forward progression ──────────────────────────────────────────────
def test_the_error_split_uses_the_direction_it_is_given():
    along, cross = split_error((1.0, 0.5, 0.0), (1.0, 0.0, 0.0))
    assert along == pytest.approx(1.0)
    assert cross == pytest.approx(0.5)


def test_with_no_direction_the_whole_gap_is_cross_track():
    """An offset from a hover point is not lateness."""
    along, cross = split_error((0.6, 0.8, 0.0), None)
    assert along == 0.0
    assert cross == pytest.approx(1.0)


def test_a_stopped_reference_keeps_the_last_direction_of_travel():
    """Otherwise the lag inverts to cross-track exactly when it matters most."""
    tracker = _tracker()
    tracker.update(_moving(speed=0.5), (0.0, 0.0, 1.5), yaw=0.0, dt=DT,
                   reference_age=0.003)
    out = tracker.update(_frozen(x=1.0, y=0.0, z=1.5), (0.0, 0.0, 1.5),
                         yaw=0.0, dt=DT, reference_age=0.003)
    assert out.along_track_lag_m == pytest.approx(1.0, abs=1e-6)
    assert out.cross_track_error_m == pytest.approx(0.0, abs=1e-6)


def test_being_ahead_of_the_reference_is_bounded_not_flown_out():
    """Frame 1320: lag -1.21 m and the correction saturated pulling backward."""
    ahead = limit_lead((-1.0, 0.0, 0.0), (1.0, 0.0, 0.0), 0.25)
    assert ahead[0] == pytest.approx(-0.25)


def test_the_lead_limit_leaves_cross_track_alone():
    """Cross-track is the component that keeps the aircraft off walls."""
    bounded = limit_lead((-1.0, 0.8, 0.0), (1.0, 0.0, 0.0), 0.25)
    assert bounded[0] == pytest.approx(-0.25)
    assert bounded[1] == pytest.approx(0.8)


def test_being_behind_the_reference_is_untouched():
    """Lag is the normal state while accelerating; only the retard is bounded."""
    behind = limit_lead((1.0, 0.0, 0.0), (1.0, 0.0, 0.0), 0.25)
    assert behind[0] == pytest.approx(1.0)


def test_a_negative_lead_limit_is_rejected():
    with pytest.raises(ValueError):
        limit_lead((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), -0.1)


def test_the_tracker_does_not_brake_backward_onto_a_passed_reference():
    """The whole point: ahead of schedule must not become reverse thrust."""
    loose = ReferenceTracker3D(ReferenceTrackerParams(
        command_smoothing_alpha=1.0, max_lead_error_m=0.25))
    tight = ReferenceTracker3D(ReferenceTrackerParams(
        command_smoothing_alpha=1.0, max_lead_error_m=1.0))
    reference = TrajectoryPoint(t=0.0, x=0.0, y=0.0, z=1.5,
                                vx=0.5, vy=0.0, vz=0.0)
    ahead = (1.5, 0.0, 1.5)                  # aircraft 1.5 m past it
    bounded = loose.update(reference, ahead, yaw=0.0, dt=DT, reference_age=0.0)
    unbounded = tight.update(reference, ahead, yaw=0.0, dt=DT, reference_age=0.0)
    assert bounded.vx > unbounded.vx         # less backward pull
    assert bounded.along_track_lag_m == pytest.approx(-1.5, abs=1e-6)
