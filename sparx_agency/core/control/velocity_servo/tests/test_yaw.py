"""Heading on a yaw-rate airframe is the axis that fails quietly.

Nothing above this loop closes it. A servo that only fed the plan's yaw rate
forward would still look correct in every log -- the commanded rate matches the
plan, tick by tick -- while the aircraft's actual heading walked away from the
plan by whatever the turns did not achieve. On an exploration aircraft that is
not a comfort issue: FALCON picks heading to aim the depth camera at the
frontier it means to observe next, so 20 degrees of accumulated heading error is
a map built of the wrong wall.

The tests are therefore about the two things the law adds over pure
feedforward -- that the proportional term is present and signed to *reduce* the
error, and that "reduce" is measured the short way round the circle -- plus the
three guards that stop the correction becoming a hazard of its own.
"""
from __future__ import annotations

import math

import pytest

from sparx_agency.core.control.velocity_servo.yaw import YawServo

SETTLED_DT = 1.0
"""A tick long enough that the slew limit cannot bind, for testing the law alone.

``max_accel * 1.0`` exceeds ``max_rate`` on every configuration used here, so a
single call at this dt returns the clamped law output directly. The slew limit
has its own test rather than being entangled with all the others.
"""


def test_a_reached_heading_is_pure_feedforward():
    """With the plan's heading achieved, the plan's rate is the whole answer.

    The proportional term is a correction, not a contribution: an aircraft that
    is exactly on the planned heading must turn at exactly the planned rate, or
    the servo is fighting the feedforward it was given.
    """
    servo = YawServo()
    rate = servo.update(reference_yaw=0.7, reference_yaw_rate=0.3,
                        measured_yaw=0.7, dt=SETTLED_DT)
    assert abs(rate - 0.3) < 1e-12
    assert abs(servo.error_rad) < 1e-12


def test_a_standing_error_is_driven_out_not_merely_reported():
    """The correction must be signed to shrink the error, from either side.

    Stepped against a heading that does not move, so the sign is unambiguous:
    a reference to the aircraft's left must produce a positive (CCW) rate and
    one to its right a negative rate. A sign error here is a servo that runs the
    heading away from the plan as fast as its limits allow, which is the single
    worst failure this module can have and is invisible in a magnitude check.
    """
    for reference, expected_sign in ((0.6, +1.0), (-0.6, -1.0)):
        servo = YawServo()
        rate = 0.0
        for _ in range(50):
            rate = servo.update(reference_yaw=reference, reference_yaw_rate=0.0,
                                measured_yaw=0.0, dt=0.02)
        assert rate * expected_sign > 0.0, (reference, rate)
        # The law itself: gain * error, once the slew ramp has finished.
        assert abs(rate - servo.gain * reference) < 1e-9
        assert abs(servo.error_rad - reference) < 1e-12

    # And with the aircraft on the far side, the correction reverses.
    servo = YawServo()
    left = servo.update(0.0, 0.0, measured_yaw=-0.5, dt=SETTLED_DT)
    servo.reset()
    right = servo.update(0.0, 0.0, measured_yaw=0.5, dt=SETTLED_DT)
    assert left > 0.0 > right


def test_the_correction_takes_the_short_way_round():
    """179 degrees left is a left turn; 181 degrees left is a right turn.

    This is the entire reason ``normalize_angle`` is in the law. Without it the
    servo answers a reference just past the wrap by turning 181 degrees the long
    way -- a second and a half of the camera pointed at nothing, and on a
    reference that keeps crossing the wrap, an aircraft that spins.

    The two cases are one degree apart in the plan and opposite in the command,
    which is exactly the discontinuity a wrapped error is supposed to have.
    """
    servo = YawServo()
    just_short = servo.update(reference_yaw=math.radians(179.0), reference_yaw_rate=0.0,
                              measured_yaw=0.0, dt=SETTLED_DT)
    assert just_short > 0.0, just_short
    assert servo.error_rad > 0.0

    servo.reset()
    just_past = servo.update(reference_yaw=math.radians(181.0), reference_yaw_rate=0.0,
                             measured_yaw=0.0, dt=SETTLED_DT)
    assert just_past < 0.0, just_past
    assert abs(servo.error_rad + math.radians(179.0)) < 1e-9

    # The same one-degree step measured about a heading away from zero, so the
    # test is about the wrap and not about the sign of the reference.
    servo.reset()
    around = servo.update(reference_yaw=math.radians(-179.0), reference_yaw_rate=0.0,
                          measured_yaw=math.radians(90.0), dt=SETTLED_DT)
    assert around > 0.0, around      # 91 degrees left, not 269 degrees right


def test_the_rate_ceiling_bites_before_the_wire():
    """A large heading error must not turn into a rate the airframe cannot fly.

    The proportional term is unbounded by construction -- a pi radian error at a
    gain of 1.5 asks for 4.7 rad/s -- and a yaw-rate airframe handed that will
    saturate its own loop, which throws away the horizontal command's frame
    correction at the same time. The clamp is what keeps the commanded heading
    rate inside the aircraft's envelope.
    """
    servo = YawServo(gain=1.5, max_rate=1.0, max_accel=100.0)
    rate = servo.update(reference_yaw=math.pi, reference_yaw_rate=0.0,
                        measured_yaw=0.0, dt=0.1)
    assert abs(rate - 1.0) < 1e-12

    servo.reset()
    rate = servo.update(reference_yaw=-math.pi + 1e-6, reference_yaw_rate=0.0,
                        measured_yaw=0.0, dt=0.1)
    assert abs(rate + 1.0) < 1e-12

    # The feedforward alone is clamped too: the plan does not get an exemption.
    servo.reset()
    rate = servo.update(reference_yaw=None, reference_yaw_rate=9.0,
                        measured_yaw=0.0, dt=0.1)
    assert abs(rate - 1.0) < 1e-12


def test_the_rate_command_ramps_and_reset_releases_the_ramp():
    """A step in commanded yaw rate is a step in commanded tilt, indirectly.

    The aircraft's own yaw loop takes ~0.5 s to reach a commanded rate, so a
    step command is spent entirely on a transient the camera sees as a jerk.
    Each tick may move the command by ``max_accel * dt`` and no further.

    ``reset`` must clear that memory, because the constraint is only meaningful
    between consecutive ticks of one flight: after a handover the previous rate
    describes a phase of flight that is over, and slewing down from it would
    command a turn nobody asked for.
    """
    servo = YawServo(gain=1.5, max_rate=2.0, max_accel=3.0)
    dt = 0.02
    allowance = 3.0 * dt

    previous = 0.0
    for _ in range(10):
        rate = servo.update(reference_yaw=1.5, reference_yaw_rate=0.0,
                            measured_yaw=0.0, dt=dt)
        assert rate - previous <= allowance + 1e-12, (previous, rate)
        previous = rate
    assert previous > 0.0
    assert previous < servo.gain * 1.5      # still ramping, not yet at the law
    assert abs(servo.commanded_rate - previous) < 1e-12

    # Without a reset the command may only step down by the same allowance.
    held = servo.update(reference_yaw=None, reference_yaw_rate=0.0,
                        measured_yaw=0.0, dt=dt)
    assert abs(held - (previous - allowance)) < 1e-12

    servo.reset()
    assert servo.commanded_rate == 0.0
    released = servo.update(reference_yaw=None, reference_yaw_rate=0.0,
                            measured_yaw=0.0, dt=dt)
    assert abs(released) < 1e-12


def test_the_deadband_silences_the_correction_and_never_the_plan():
    """Stop dithering about a heading already reached, without stopping the turn.

    A few milliradians of estimator noise multiplied by the gain is a yaw rate
    command that reverses every tick, and on a camera-carrying airframe that is
    visible as a shimmer in the map. The deadband suppresses the *correction*
    only: a plan that is genuinely turning must keep turning at its planned rate
    while the aircraft is on heading, which is the normal case throughout a
    curve and not an edge case.
    """
    servo = YawServo(gain=1.5, deadband_rad=0.01)
    inside = servo.update(reference_yaw=0.005, reference_yaw_rate=0.2,
                          measured_yaw=0.0, dt=SETTLED_DT)
    assert abs(inside - 0.2) < 1e-12         # feedforward survives untouched
    assert abs(servo.error_rad - 0.005) < 1e-12   # still reported, just not acted on

    servo.reset()
    outside = servo.update(reference_yaw=0.05, reference_yaw_rate=0.2,
                           measured_yaw=0.0, dt=SETTLED_DT)
    assert abs(outside - (0.2 + 1.5 * 0.05)) < 1e-12


def test_a_plan_with_no_heading_opinion_is_fed_forward_only():
    """No reference heading is not the same as a reference heading of zero.

    A trajectory that carries no yaw curve, or a hold with nothing to aim at,
    must not be read as "point along world +x" -- that would spin the aircraft
    to a heading the planner never asked for. It means "turn at the rate given
    and do not correct", and the reported error must be 0.0 rather than a stale
    value a log would read as a real heading error.
    """
    servo = YawServo()
    servo.update(reference_yaw=0.6, reference_yaw_rate=0.0, measured_yaw=0.0,
                 dt=SETTLED_DT)
    assert servo.error_rad > 0.0

    rate = servo.update(reference_yaw=None, reference_yaw_rate=0.25,
                        measured_yaw=2.0, dt=SETTLED_DT)
    assert abs(rate - 0.25) < 1e-12
    assert servo.error_rad == 0.0


def test_a_non_positive_dt_raises():
    """A stalled or reordered clock must not silently produce a command.

    ``dt`` scales the slew allowance, so a zero freezes the command at whatever
    it last was and a negative one inverts the bound -- both produce a plausible
    number from a nonsensical input, which is the failure mode that reaches a
    flight.
    """
    servo = YawServo()
    for bad in (0.0, -0.02):
        with pytest.raises(ValueError):
            servo.update(0.0, 0.0, 0.0, bad)


def test_construction_rejects_a_configuration_that_cannot_servo():
    """The gains are read from a launch file, where a typo is a flight.

    A negative gain is a correction that drives the error the wrong way; a
    non-positive rate or acceleration ceiling is an aircraft that can never
    turn; a negative deadband is meaningless. All are cheap to reject here and
    expensive to notice in the air.
    """
    with pytest.raises(ValueError):
        YawServo(gain=-0.1)
    with pytest.raises(ValueError):
        YawServo(max_rate=0.0)
    with pytest.raises(ValueError):
        YawServo(max_accel=-1.0)
    with pytest.raises(ValueError):
        YawServo(deadband_rad=-0.001)
