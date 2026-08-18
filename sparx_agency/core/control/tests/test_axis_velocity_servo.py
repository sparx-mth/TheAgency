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


def test_stop_request_resets_the_integrator():
    servo = AxisVelocityServo(DEADZONE, V_FULL, kp=220.0, ki=260.0,
                              min_command_mps=0.15)
    for _ in range(20):
        servo.update(0.30, 0.10, 0.05)
    assert servo.integral != 0.0
    assert servo.update(0.0, 0.10, 0.05) == 0.0
    assert servo.integral == 0.0


def test_zero_dt_integrates_nothing():
    servo = AxisVelocityServo(DEADZONE, V_FULL, kp=0.0, ki=260.0)
    servo.update(0.30, 0.10, 0.0)
    assert servo.integral == 0.0
