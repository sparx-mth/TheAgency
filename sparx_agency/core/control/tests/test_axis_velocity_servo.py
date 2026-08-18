"""Tests for the remote-control axis velocity servo."""

from sparx_agency.core.control.axis_velocity_servo import (
    AxisVelocityServo,
    feedforward_axis,
)

# Rooster's measured forward axis (2026-08-17, Sphera ground truth).
DEADZONE = 620.0
V_FULL = 1.25


def test_feedforward_starts_at_the_deadzone_not_at_zero():
    assert feedforward_axis(0.0, DEADZONE, V_FULL) == 0.0
    assert feedforward_axis(0.001, DEADZONE, V_FULL) > DEADZONE
    assert feedforward_axis(V_FULL, DEADZONE, V_FULL) == 1000.0
    assert feedforward_axis(-V_FULL, DEADZONE, V_FULL) == -1000.0


def test_feedforward_saturates_rather_than_exceeding_full_deflection():
    assert feedforward_axis(10.0 * V_FULL, DEADZONE, V_FULL) == 1000.0


def test_integrator_closes_a_persistent_shortfall():
    """The measured failure: the curve asks for 0.30, the platform gives 0.11."""
    servo = AxisVelocityServo(DEADZONE, V_FULL, kp=220.0, ki=260.0)
    open_loop = feedforward_axis(0.30, DEADZONE, V_FULL)
    axis = open_loop
    for _ in range(40):                       # 2 s at 20 Hz
        axis = servo.update(0.30, 0.11, 0.05)
    assert axis > open_loop + 100.0
    assert axis <= 1000.0


def test_integrator_does_not_wind_up_against_a_saturated_output():
    servo = AxisVelocityServo(DEADZONE, V_FULL, kp=220.0, ki=260.0,
                              max_correction=300.0)
    for _ in range(400):                      # 20 s of an aircraft that never moves
        servo.update(1.25, 0.0, 0.05)
    assert abs(servo.integral) <= 300.0


def test_correction_is_bounded_even_with_a_huge_error():
    servo = AxisVelocityServo(DEADZONE, V_FULL, kp=220.0, ki=260.0,
                              max_correction=300.0)
    servo.update(0.5, -5.0, 0.05)
    assert abs(servo.last_correction) <= 300.0


def test_overspeed_pulls_the_axis_back_below_feedforward():
    servo = AxisVelocityServo(DEADZONE, V_FULL, kp=220.0, ki=260.0)
    open_loop = feedforward_axis(0.30, DEADZONE, V_FULL)
    axis = servo.update(0.30, 0.90, 0.05)     # flying far too fast
    assert axis < open_loop


def test_a_brief_zero_demand_keeps_the_integrator():
    """A momentary zero must not cost the loop what it learned.

    The follower emits exact zeros on ordinary ticks (a hold, a taper, an
    alignment gate). Resetting on the first of them destroyed the standing
    bias several times a second, which is the whole thing the integrator
    exists to find.
    """
    servo = AxisVelocityServo(DEADZONE, V_FULL, kp=220.0, ki=260.0,
                              min_command_mps=0.15, integral_hold_s=0.6)
    for _ in range(20):
        servo.update(0.30, 0.10, 0.05)
    learned = servo.integral
    assert learned != 0.0
    assert servo.update(0.0, 0.10, 0.05) == 0.0
    assert servo.integral == learned


def test_a_sustained_stop_still_resets_the_integrator():
    servo = AxisVelocityServo(DEADZONE, V_FULL, kp=220.0, ki=260.0,
                              min_command_mps=0.15, integral_hold_s=0.6)
    for _ in range(20):
        servo.update(0.30, 0.10, 0.05)
    assert servo.integral != 0.0
    for _ in range(int(0.6 / 0.05) + 1):
        assert servo.update(0.0, 0.10, 0.05) == 0.0
    assert servo.integral == 0.0


def test_a_moving_aircraft_uses_the_moving_curve():
    """The same demand must ask for less stick once the aircraft is rolling."""
    servo = AxisVelocityServo(DEADZONE, V_FULL, kp=0.0, ki=0.0,
                              v_full_moving=4.0, move_eps_mps=0.10)
    standing = servo.update(0.30, 0.0, 0.05)
    moving = servo.update(0.30, 0.50, 0.05)
    assert DEADZONE < moving < standing


def test_the_axis_is_never_muted_while_motion_is_wanted():
    """A correction may slow the aircraft but must not release the stick.

    Below the dead band the platform does not go slower, it stops -- and a
    released stick is an active brake in PX4 Position mode, so the loop then
    has to climb back through the dead band to restart. That is a limit cycle,
    not control.
    """
    servo = AxisVelocityServo(DEADZONE, V_FULL, kp=220.0, ki=260.0,
                              max_correction=350.0, brake_release_margin_mps=0.15)
    axis = servo.update(0.30, 0.38, 0.05)      # slightly fast, not braking
    assert abs(axis) >= DEADZONE


def test_a_genuine_overspeed_may_still_release_the_stick():
    servo = AxisVelocityServo(DEADZONE, V_FULL, kp=220.0, ki=260.0,
                              max_correction=350.0, brake_release_margin_mps=0.15)
    axis = servo.update(0.30, 1.20, 0.05)      # far over the demand
    assert axis < DEADZONE


def test_zero_dt_integrates_nothing():
    servo = AxisVelocityServo(DEADZONE, V_FULL, kp=0.0, ki=260.0)
    servo.update(0.30, 0.10, 0.0)
    assert servo.integral == 0.0
