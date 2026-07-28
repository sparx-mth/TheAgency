"""Tests for the two terms that make an inner loop with lag track well.

Damping on the **measured** velocity error is what closes the gap left by an
autopilot that reaches a commanded velocity through tilt. It is the term that
does most of the tracking, which is why the position loop is allowed to stay
gentle and clamped. Smoothing the output is what stops the airframe being handed
steps it can only answer with a lurch.
"""
import math

import pytest

from sparx_agency.core.common.types import TrajectoryPoint
from sparx_agency.core.planning.trackers.reference_tracker_3d import (
    ReferenceTracker3D, ReferenceTrackerParams,
)

DT = 0.02
SPEED = 0.6


class _LaggingAircraft:
    """First-order velocity lag: what a tilt-limited inner loop looks like."""

    def __init__(self, position=(0.0, 0.0, 1.4), tau=0.45):
        self.position = list(position)
        self.velocity = [0.0, 0.0, 0.0]
        self.tau = tau

    def step(self, command, dt):
        alpha = dt / (self.tau + dt)
        for i in range(3):
            self.velocity[i] += alpha * (command[i] - self.velocity[i])
            self.position[i] += self.velocity[i] * dt


def _straight(t):
    return TrajectoryPoint(t=t, x=SPEED * t, y=0.0, z=1.4,
                           vx=SPEED, vy=0.0, vz=0.0, yaw=0.0)


def _fly(params, seconds=20.0, feed_velocity=True):
    """Fly a straight reference and return the worst post-transient lag."""
    tracker = ReferenceTracker3D(params)
    aircraft = _LaggingAircraft()
    worst = 0.0
    for step in range(int(seconds / DT)):
        t = step * DT
        out = tracker.update(
            _straight(t), tuple(aircraft.position), 0.0, DT,
            velocity=tuple(aircraft.velocity) if feed_velocity else None)
        aircraft.step(out.velocity(), DT)
        if t > 4.0:
            worst = max(worst, out.position_error_m)
    return worst


def test_velocity_feedback_tracks_a_laggy_airframe_better():
    """The whole reason the term exists, measured on the same flight."""
    with_damping = _fly(ReferenceTrackerParams(), feed_velocity=True)
    without = _fly(ReferenceTrackerParams(), feed_velocity=False)
    assert with_damping < without
    assert with_damping < 0.15


def test_omitting_the_measured_velocity_still_flies():
    """A caller that cannot measure velocity degrades, it does not break."""
    assert _fly(ReferenceTrackerParams(), feed_velocity=False) < 1.0


def test_the_damping_term_opposes_a_velocity_shortfall():
    """An aircraft slower than its plan is commanded faster, in proportion."""
    params = ReferenceTrackerParams(velocity_damping_xy=0.5, command_smoothing_alpha=1.0)
    tracker = ReferenceTracker3D(params)
    reference = _straight(0.0)
    on_plan = tracker.update(reference, (0.0, 0.0, 1.4), 0.0, DT,
                             velocity=(SPEED, 0.0, 0.0))
    tracker.reset()
    slow = tracker.update(reference, (0.0, 0.0, 1.4), 0.0, DT,
                          velocity=(SPEED - 0.4, 0.0, 0.0))
    assert slow.vx - on_plan.vx == pytest.approx(0.5 * 0.4, abs=1e-6)


def test_the_damping_term_is_not_clamped_by_the_position_loop():
    """It has to out-pull the clamped position term, or the lag stays.

    The position loop is bounded on purpose (see test_corner_cutting); if the
    damping were bounded with it there would be nothing left to track with.
    """
    params = ReferenceTrackerParams(velocity_damping_xy=0.5,
                                    position_error_clamp_m=0.1,
                                    command_smoothing_alpha=1.0)
    tracker = ReferenceTracker3D(params)
    out = tracker.update(_straight(0.0), (0.0, 0.0, 1.4), 0.0, DT,
                         velocity=(0.0, 0.0, 0.0))
    # feed-forward 0.6 + damping 0.5*0.6 = 0.9, well past the clamped position
    # term's ceiling of kp * 0.1.
    assert out.vx > SPEED + 0.2


def test_the_output_is_smoothed_rather_than_stepped():
    """A step in the reference does not become a step in the command."""
    params = ReferenceTrackerParams(command_smoothing_alpha=0.3)
    tracker = ReferenceTracker3D(params)
    hover = TrajectoryPoint(t=0.0, x=0.0, y=0.0, z=1.4, yaw=0.0)
    tracker.update(hover, (0.0, 0.0, 1.4), 0.0, DT, velocity=(0.0, 0.0, 0.0))

    commands = []
    for _ in range(6):
        out = tracker.update(_straight(0.0), (0.0, 0.0, 1.4), 0.0, DT,
                             velocity=(0.0, 0.0, 0.0))
        commands.append(out.vx)
    # Rising toward the demand, not arriving at it in one tick.
    assert commands[0] < commands[-1]
    assert commands[0] < 0.6 * commands[-1]
    assert commands == sorted(commands)


def test_smoothing_can_be_switched_off():
    params = ReferenceTrackerParams(command_smoothing_alpha=1.0)
    tracker = ReferenceTracker3D(params)
    out = tracker.update(_straight(0.0), (0.0, 0.0, 1.4), 0.0, DT,
                         velocity=(SPEED, 0.0, 0.0))
    assert out.vx == pytest.approx(SPEED, abs=1e-9)


def test_smoothing_never_exceeds_the_speed_limit():
    """Smoothing is applied after the clamp, so it cannot overshoot the ceiling."""
    params = ReferenceTrackerParams(command_smoothing_alpha=0.3)
    tracker = ReferenceTracker3D(params)
    fast = TrajectoryPoint(t=0.0, x=0.0, y=0.0, z=1.4, vx=99.0, yaw=0.0)
    for _ in range(200):
        out = tracker.update(fast, (0.0, 0.0, 1.4), 0.0, DT, velocity=(0.0, 0.0, 0.0))
        assert math.hypot(out.vx, out.vy) <= params.limits.max_speed_xy + 1e-9


@pytest.mark.parametrize("kwargs", [
    {"velocity_damping_xy": -0.1},
    {"velocity_damping_z": -0.1},
    {"position_error_clamp_m": 0.0},
    {"command_smoothing_alpha": 0.0},
    {"command_smoothing_alpha": 1.5},
])
def test_invalid_params_are_rejected(kwargs):
    with pytest.raises(ValueError):
        ReferenceTrackerParams(**kwargs)
