"""How hard a displaced aircraft turns back toward its plan, and what that costs.

A tracker that closes on the reference *position* flies straight at it. When the
plan is rounding an obstacle, "straight at the reference" can be through the
obstacle -- so the correction ceiling (``PidGains.out_limit``) is, geometrically,
an obstacle-avoidance parameter as much as a control one.

The obvious conclusion from that geometry is wrong, and these tests exist to
keep it from being re-derived. Lowering the ceiling from 1.0 to 0.35 m/s against
a 0.45 m/s plan was measured on Isaac Sim: it took the mean tracking error from
0.4-1.4 m to 2.0-2.7 m and the runs ended *sooner*, because an aircraft that is
permanently two metres off the plan is in more danger than one that occasionally
turns sharply back onto it. Convergence is itself the safety property.

So what is tested here is the trade-off, not a bound: that the ceiling really is
what sets the turn-in angle, and that the tracker converges. The dial for an
aircraft cutting corners into obstacles is the planner's speed, not this one.
"""
import math

import pytest

from sparx_agency.core.common.types import TrajectoryPoint
from sparx_agency.core.planning.trackers.drift_pid.pid import PidGains
from sparx_agency.core.planning.trackers.reference_tracker_3d import (
    ReferenceTracker3D, ReferenceTrackerParams,
)

DT = 0.02
PLAN_SPEED = 0.45


def _turn_in_angle(params, offset_m):
    """Degrees between the commanded velocity and the planned one.

    The reference travels +x at cruise speed; the aircraft sits ``offset_m`` to
    one side of it. Zero degrees means the aircraft flies parallel to the plan
    and never converges; ninety means it abandons the plan and beelines.
    """
    tracker = ReferenceTracker3D(params)
    reference = TrajectoryPoint(t=0.0, x=0.0, y=0.0, z=1.4,
                                vx=PLAN_SPEED, vy=0.0, yaw=0.0)
    out = tracker.update(reference, (0.0, -offset_m, 1.4), yaw=0.0, dt=DT)
    return abs(math.degrees(math.atan2(out.vy, out.vx)))


def _with_cap(out_limit):
    return ReferenceTrackerParams(
        horizontal_pid=PidGains(kp=1.0, ki=0.10, kd=0.05, i_limit=0.20,
                                d_tau_s=0.3, deadband=0.01, out_limit=out_limit))


def test_the_correction_ceiling_is_what_sets_the_turn_in_angle():
    """A smaller ceiling means a shallower turn back toward the plan."""
    shallow = _turn_in_angle(_with_cap(0.35), offset_m=4.0)
    steep = _turn_in_angle(_with_cap(1.0), offset_m=4.0)
    assert shallow < steep
    assert shallow == pytest.approx(math.degrees(math.atan2(0.35, PLAN_SPEED)), abs=1.0)
    assert steep == pytest.approx(math.degrees(math.atan2(1.0, PLAN_SPEED)), abs=1.0)


@pytest.mark.parametrize("offset_m", [2.0, 4.0, 20.0])
def test_the_angle_saturates_rather_than_growing_without_bound(offset_m):
    """However far away the reference is, the correction is capped and so is the turn.

    This is what stops a wildly displaced aircraft from flying perpendicular to
    its plan: the correction saturates, the feed-forward does not.
    """
    assert _turn_in_angle(ReferenceTrackerParams(), offset_m) < 70.0


def test_a_nearby_reference_barely_deflects_it():
    """Near the plan, the aircraft flies the plan. The correction is a nudge."""
    assert _turn_in_angle(ReferenceTrackerParams(), offset_m=0.1) < 15.0


def test_it_converges_onto_the_plan_and_stays_there():
    """The property the ceiling exists to preserve, not to trade away."""
    tracker = ReferenceTracker3D(ReferenceTrackerParams())
    offset = 2.0
    for step in range(int(30.0 / DT)):
        t = step * DT
        reference = TrajectoryPoint(t=t, x=PLAN_SPEED * t, y=0.0, z=1.4,
                                    vx=PLAN_SPEED, vy=0.0, yaw=0.0)
        out = tracker.update(reference, (PLAN_SPEED * t, -offset, 1.4), 0.0, DT)
        offset -= out.vy * DT
    assert abs(offset) < 0.05


def test_a_lower_ceiling_converges_more_slowly():
    """The measured trade: a shallower turn takes longer to close the same gap."""
    def settle_time(params):
        tracker = ReferenceTracker3D(params)
        offset = 2.0
        for step in range(int(60.0 / DT)):
            t = step * DT
            reference = TrajectoryPoint(t=t, x=PLAN_SPEED * t, y=0.0, z=1.4,
                                        vx=PLAN_SPEED, vy=0.0, yaw=0.0)
            out = tracker.update(reference, (PLAN_SPEED * t, -offset, 1.4), 0.0, DT)
            offset -= out.vy * DT
            if abs(offset) < 0.1:
                return t
        return float("inf")

    assert settle_time(_with_cap(0.35)) > settle_time(_with_cap(1.0))
