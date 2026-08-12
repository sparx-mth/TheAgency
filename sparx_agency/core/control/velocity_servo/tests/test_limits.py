"""A speed limit must never become a steering command.

That is the whole subject of these tests. Both clamps here can be written in
three lines each with ``min``/``max`` per axis, both of those versions pass any
test that only checks magnitudes, and both of them turn a saturated command into
a heading error that grows with how saturated it is -- which is invisible in a
log of speeds and obvious in a flight that clips a doorway.

So the assertions are about **direction**, and the magnitudes are checked only
as a secondary matter. Where the per-axis version would differ, the test says by
how much, so the failure is recognisable rather than merely red.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from sparx_agency.core.control.velocity_servo.limits import (
    VelocityLimits, limit_velocity, slew_velocity,
)


def heading_of(velocity):
    # type: (object) -> float
    """Horizontal heading of a velocity, radians CCW from world +x."""
    return math.atan2(float(velocity[1]), float(velocity[0]))


def test_an_over_speed_command_keeps_its_heading():
    """The direction of travel is the part of the command that must survive.

    ``(1.4, 0.4)`` against a 1.0 m/s ceiling is the case that separates the two
    implementations: scaled as a pair it comes back at 1.0 m/s on the same
    bearing, clipped per axis it comes back as ``(1.0, 0.4)`` -- 5.9 degrees to
    the left of where the controller aimed, and further off the harder it is
    saturated. The heading is asserted to 1e-12 because scaling both components
    by one factor is exact in floating point; anything looser would let a subtly
    rotating clamp through.
    """
    limits = VelocityLimits()
    wanted = np.array([1.4, 0.4, 0.0])
    limited, saturated = limit_velocity(wanted, limits, max_speed_xy=1.0)

    assert saturated
    assert abs(heading_of(limited) - heading_of(wanted)) < 1e-12
    assert abs(math.hypot(limited[0], limited[1]) - 1.0) < 1e-12

    # The version this test exists to reject, and the size of its mistake.
    per_axis = np.clip(wanted[:2], -1.0, 1.0)
    assert math.degrees(abs(heading_of(per_axis) - heading_of(wanted))) > 5.0


def test_a_command_within_the_ceiling_is_returned_untouched():
    """``saturated`` is read by the integrator gate, so it must mean something.

    The gate freezes the integral on any axis whose command did not reach the
    airframe. A clamp that reported saturation on every tick would stop the
    servo ever learning a standing bias, which is the one thing the integral is
    there for.
    """
    limits = VelocityLimits()
    wanted = np.array([0.6, -0.3, 0.2])
    limited, saturated = limit_velocity(wanted, limits)

    assert not saturated
    assert np.allclose(limited, wanted, atol=0.0, rtol=0.0)


def test_the_vertical_axis_is_clipped_on_its_own_and_asymmetrically():
    """Climb and descent are not the same manoeuvre, so they are not one limit.

    Descending fast into ground effect is how a flight ends, so the descent
    ceiling is deliberately the smaller of the two. And the vertical axis is
    clipped independently of the horizontal pair: scaling all three together to
    respect an altitude limit would slow the aircraft along its route for a
    reason that has nothing to do with its route.
    """
    limits = VelocityLimits(max_speed_up=1.0, max_speed_down=0.4)

    climbing, saturated = limit_velocity((0.5, 0.0, 3.0), limits)
    assert saturated
    assert abs(climbing[2] - 1.0) < 1e-12
    assert abs(climbing[0] - 0.5) < 1e-12       # horizontal untouched by a vertical clip

    diving, saturated = limit_velocity((0.5, 0.0, -3.0), limits)
    assert saturated
    assert abs(diving[2] + 0.4) < 1e-12
    assert abs(diving[0] - 0.5) < 1e-12

    # Asymmetric on purpose: a symmetric clamp would have given -1.0 above.
    assert limits.max_speed_down < limits.max_speed_up


def test_the_per_tick_ceiling_tightens_but_never_loosens():
    """The plan's speed may slow the aircraft down; it may not speed it up.

    ``max_speed_xy`` is where the trajectory's own planned speed enters, and
    FALCON checks its route against the map at that speed. But the airframe
    ceiling is a property of the aircraft, not of the plan, so a trajectory
    asking for 5 m/s must not be able to raise it. The two are combined with a
    ``min`` for exactly this reason.
    """
    limits = VelocityLimits(max_speed_xy=1.5)
    fast = np.array([4.0, 0.0, 0.0])

    loosened, saturated = limit_velocity(fast, limits, max_speed_xy=5.0)
    assert saturated
    assert abs(loosened[0] - 1.5) < 1e-12

    tightened, saturated = limit_velocity(fast, limits, max_speed_xy=0.4)
    assert saturated
    assert abs(tightened[0] - 0.4) < 1e-12

    absent, _ = limit_velocity(fast, limits, max_speed_xy=None)
    assert abs(absent[0] - 1.5) < 1e-12


def test_the_first_tick_after_a_reset_is_not_slewed():
    """There is no previous command to be near, so there is no step to bound.

    ``previous=None`` is what a reset leaves behind -- a handover from takeoff,
    a planner restart. Slewing against a remembered zero would ramp the aircraft
    in from a standstill it is not actually at, which on a handover mid-flight
    is a commanded deceleration nobody asked for.
    """
    limits = VelocityLimits(max_accel_xy=2.0, max_accel_z=2.0)
    wanted = np.array([1.2, -0.9, 0.6])
    slewed, rate_limited = slew_velocity(None, wanted, limits, dt=0.02)

    assert not rate_limited
    assert np.allclose(slewed, wanted, atol=0.0, rtol=0.0)


def test_the_horizontal_step_is_bounded_as_a_vector():
    """A bounded step must still point where the change was pointing.

    Same trap as the magnitude clamp, one derivative up: clipping dvx and dvy
    separately turns a large change of direction into a change of a *different*
    direction, and it does so during exactly the manoeuvre -- a replan, a hard
    corner -- where the command is changing most and being wrong costs most.
    """
    limits = VelocityLimits(max_accel_xy=2.0)
    previous = np.array([0.2, 0.1, 0.0])
    wanted = previous + np.array([1.4, 0.4, 0.0])
    dt = 0.02

    slewed, rate_limited = slew_velocity(previous, wanted, limits, dt)
    step = slewed - previous

    assert rate_limited
    assert abs(heading_of(step) - heading_of(wanted - previous)) < 1e-12
    assert abs(math.hypot(step[0], step[1]) - limits.max_accel_xy * dt) < 1e-12


def test_the_vertical_step_is_bounded_independently():
    """Altitude has its own acceleration budget, and spends only its own.

    A hard horizontal change must not throttle the climb, and a climb must not
    throttle the turn. They are separate control authorities on a multirotor --
    one is thrust, the other is tilt -- so a single shared step limit would let
    either one starve the other.
    """
    limits = VelocityLimits(max_accel_xy=2.0, max_accel_z=1.0)
    dt = 0.05

    slewed, rate_limited = slew_velocity((0.0, 0.0, 0.0), (0.01, 0.0, 3.0), limits, dt)
    assert rate_limited
    assert abs(slewed[2] - 1.0 * dt) < 1e-12
    assert abs(slewed[0] - 0.01) < 1e-12         # inside the horizontal allowance

    dropping, rate_limited = slew_velocity((0.0, 0.0, 0.0), (0.0, 0.0, -3.0), limits, dt)
    assert rate_limited
    assert abs(dropping[2] + 1.0 * dt) < 1e-12


def test_a_step_inside_the_allowance_is_reported_as_free():
    """``rate_limited`` every tick is a diagnosis, so it must not cry wolf.

    A replan blend that trips the slew limit continuously is a blend that is not
    working, and that is worth knowing from a log. It is only worth knowing if
    the flag is quiet the rest of the time.
    """
    limits = VelocityLimits(max_accel_xy=2.0, max_accel_z=2.0)
    previous = np.array([0.5, 0.5, 0.1])
    wanted = previous + np.array([0.01, -0.005, 0.02])

    slewed, rate_limited = slew_velocity(previous, wanted, limits, dt=0.02)
    assert not rate_limited
    assert np.allclose(slewed, wanted, atol=1e-15)


def test_a_non_positive_limit_is_rejected():
    """A zero ceiling is an aircraft that cannot move, and it flies silently.

    Every field is a ceiling that something divides by or scales against, and a
    zero or negative one reads as "hold still" rather than as "misconfigured".
    Raising at construction is the only point at which anyone will notice.
    """
    for field_name in ("max_speed_xy", "max_speed_up", "max_speed_down",
                       "max_accel_xy", "max_accel_z", "max_yaw_rate", "max_yaw_accel"):
        with pytest.raises(ValueError):
            VelocityLimits(**{field_name: 0.0})
        with pytest.raises(ValueError):
            VelocityLimits(**{field_name: -1.0})
