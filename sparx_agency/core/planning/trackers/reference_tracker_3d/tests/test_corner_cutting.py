"""How hard a displaced aircraft turns back toward its plan, and what it costs.

A position loop points straight at the reference. That is right when the
aircraft is near the plan and wrong when it is far from it: a trajectory rounding
an obstacle puts its reference on the far side of that obstacle for a second or
two, and an aircraft that has fallen behind will take the direct line -- through
whatever the planner was routing around. It ended two Isaac Sim runs against
walls FALCON had already mapped and correctly planned around.

The tempting conclusion is to turn the position loop down. It has now been tried
twice and measured wrong twice:

* capping the loop's **output** at 0.35 m/s against a 0.45 m/s plan took the mean
  tracking error from 0.4-1.4 m to 2.0-2.7 m, and the runs ended sooner;
* clamping the position **error** at 0.5 m, which bounds the same pull, took the
  mean error to 3 m and ended a flight in twelve seconds.

Both fail for the same reason: an aircraft that is permanently two metres off its
plan is in more danger than one that occasionally turns hard back onto it.
Convergence is itself the safety property, and on an airframe whose inner loop
lags -- PX4 reaches a commanded velocity through tilt -- convergence needs
authority.

So the clamp is kept, at a metre, where it does not bind in the regime the
aircraft actually flies in and still stops the far field from growing without
bound. These tests hold that shape: the dial works, the pull saturates, and the
loop converges. They deliberately do NOT assert that the pull is smaller than
flight speed, because making it so is what broke the flights.
"""
import math

import pytest

from sparx_agency.core.common.types import TrajectoryPoint
from sparx_agency.core.planning.trackers.reference_tracker_3d import (
    ReferenceTracker3D, ReferenceTrackerParams,
)

DT = 0.02
PLAN_SPEED = 0.45


def _turn_in_angle(params, offset_m, velocity=(PLAN_SPEED, 0.0, 0.0)):
    """Degrees between the commanded velocity and the planned one.

    The reference travels +x at cruise speed; the aircraft sits ``offset_m`` to
    one side of it, moving along the plan. Zero degrees means the aircraft flies
    parallel to the plan and never converges; ninety means it abandons the plan
    and beelines at the reference.
    """
    tracker = ReferenceTracker3D(params)
    reference = TrajectoryPoint(t=0.0, x=0.0, y=0.0, z=1.4,
                                vx=PLAN_SPEED, vy=0.0, yaw=0.0)
    out = tracker.update(reference, (0.0, -offset_m, 1.4), yaw=0.0, dt=DT,
                         velocity=velocity)
    return abs(math.degrees(math.atan2(out.vy, out.vx)))


def _with_clamp(metres):
    return ReferenceTrackerParams(position_error_clamp_m=metres)


def test_the_clamp_is_the_dial_that_sets_the_turn_in():
    """Tighter clamp, shallower turn back toward the plan."""
    assert (_turn_in_angle(_with_clamp(0.2), 4.0)
            < _turn_in_angle(_with_clamp(0.5), 4.0)
            < _turn_in_angle(_with_clamp(4.0), 4.0))


def test_a_tight_clamp_really_does_bound_the_turn_in():
    """The mechanism works; it is the flights that say not to use it this tight."""
    assert _turn_in_angle(_with_clamp(0.2), offset_m=8.0) < 30.0


@pytest.mark.parametrize("offset_m", [2.0, 4.0, 20.0, 200.0])
def test_the_turn_in_saturates_rather_than_growing_without_bound(offset_m):
    """However far off the aircraft is, the pull stops growing.

    This is what the clamp is kept for: the near field is untouched, and a wild
    displacement cannot turn into an ever-harder turn toward a reference that
    might be behind something.
    """
    assert _turn_in_angle(ReferenceTrackerParams(), offset_m) < 70.0


def test_a_nearby_reference_barely_deflects_it():
    """Inside the clamp the loop is linear: near the plan, the aircraft flies it."""
    assert _turn_in_angle(ReferenceTrackerParams(), offset_m=0.1) < 15.0


def test_the_default_clamp_does_not_bind_in_the_normal_regime():
    """The regime the aircraft actually flies in must be untouched by the clamp.

    Tracking error on a healthy run is well under a metre; the clamp is set
    above that on purpose, so it costs nothing until something has already gone
    unusual. Measured: clamping inside this range cost the flight.
    """
    params = ReferenceTrackerParams()
    typical_error_m = 0.5
    assert params.position_error_clamp_m >= typical_error_m * 2.0


def test_it_converges_onto_the_plan():
    """The property the whole tuning is protecting."""
    tracker = ReferenceTracker3D(ReferenceTrackerParams())
    offset = 2.0
    lateral = 0.0
    for step in range(int(30.0 / DT)):
        t = step * DT
        reference = TrajectoryPoint(t=t, x=PLAN_SPEED * t, y=0.0, z=1.4,
                                    vx=PLAN_SPEED, vy=0.0, yaw=0.0)
        out = tracker.update(reference, (PLAN_SPEED * t, -offset, 1.4), 0.0, DT,
                             velocity=(PLAN_SPEED, lateral, 0.0))
        lateral = out.vy
        offset -= out.vy * DT
    assert abs(offset) < 0.05


def test_a_tighter_clamp_converges_more_slowly():
    """The measured trade, in one assertion."""
    def settle_time(params):
        tracker = ReferenceTracker3D(params)
        offset, lateral = 2.0, 0.0
        for step in range(int(60.0 / DT)):
            t = step * DT
            reference = TrajectoryPoint(t=t, x=PLAN_SPEED * t, y=0.0, z=1.4,
                                        vx=PLAN_SPEED, vy=0.0, yaw=0.0)
            out = tracker.update(reference, (PLAN_SPEED * t, -offset, 1.4), 0.0, DT,
                                 velocity=(PLAN_SPEED, lateral, 0.0))
            lateral = out.vy
            offset -= out.vy * DT
            if abs(offset) < 0.1:
                return t
        return float("inf")

    assert settle_time(_with_clamp(0.2)) > settle_time(ReferenceTrackerParams())
