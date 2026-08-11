"""The plant model is the one thing in this backend that is measured, not tuned.

Everything the velocity servo does better than the controller it replaces comes
out of three numbers per axis, so the tests here are about the *contract* those
numbers carry rather than about arithmetic: a plant that validates nothing lets
a zero DC gain through and the lead term silently becomes a divide-by-nothing
argument; a plant that advertises a per-axis feedforward lead invites sampling
one curve at two instants, which is a quieter and worse bug than lag.

The closed-loop numbers live in ``test_servo.py``. This file only guards the
description of the airframe those numbers were produced against.
"""
from __future__ import annotations

import numpy as np
import pytest

from sparx_agency.core.control.velocity_servo.plant import AxisPlant, VelocityPlant
from sparx_agency.core.control.velocity_servo.tests.airframe import LaggingAirframe


def test_a_plant_that_cannot_be_inverted_is_rejected():
    """Each invariant exists because the inversion divides or leads by it.

    A non-positive DC gain is an axis that does not respond, or responds
    backwards, and no lead term rescues that. A negative time constant is a lead
    that pushes the command the wrong way at every corner. A negative delay is
    a plan read from the past. All three are configuration mistakes that would
    otherwise fly.
    """
    for bad in (0.0, -1.0):
        with pytest.raises(ValueError):
            AxisPlant(dc_gain=bad)
    with pytest.raises(ValueError):
        AxisPlant(time_constant_s=-0.01)
    with pytest.raises(ValueError):
        AxisPlant(delay_s=-0.01)


def test_the_gain_bound_is_the_delay_bound():
    """The transport delay, not taste, is what caps the position gain.

    ``1 / (3 * delay)`` is the crossover that keeps roughly 20 degrees of phase
    margin, so the bound must fall as the delay rises -- doubling the delay must
    halve the gain the axis can carry. This is the number a tuning decision is
    checked against, and a bound that did not move with the plant would be
    decoration.
    """
    quick = AxisPlant(delay_s=0.05)
    measured = AxisPlant(delay_s=0.18)
    sluggish = AxisPlant(delay_s=0.36)

    assert abs(measured.stable_position_gain - 1.0 / (3.0 * 0.18)) < 1e-12
    assert quick.stable_position_gain > measured.stable_position_gain
    assert measured.stable_position_gain > sluggish.stable_position_gain
    # Halving the delay doubles the bound: the relationship is the point.
    assert abs(sluggish.stable_position_gain * 2.0 - measured.stable_position_gain) < 1e-12


def test_a_delay_free_axis_is_bounded_by_the_floor_not_by_infinity():
    """No delay must not read as unlimited gain.

    An axis configured with ``delay_s=0`` is a modelling shortcut, never a real
    airframe -- something always takes a tick. Without the 1e-3 floor the bound
    is a division by zero, and an advisory number that comes back infinite is
    worse than no advice: it is an invitation to raise the gain until the loop
    rings.
    """
    floored = AxisPlant(delay_s=0.0).stable_position_gain
    assert np.isfinite(floored)
    assert abs(floored - 1.0 / (3.0 * 1e-3)) < 1e-9
    # Anything below the floor lands on the same bound rather than climbing.
    assert abs(AxisPlant(delay_s=1e-9).stable_position_gain - floored) < 1e-9


def test_the_feedforward_lead_is_horizontal_and_shared():
    """One instant of the curve, for every axis. Not the per-axis delay.

    The vertical loop is measurably faster than the horizontal one -- 0.05 s
    against 0.18 s on the reference airframe -- so leading each axis by its own
    delay looks like the more accurate thing to do. It is not, and the reason is
    that the feedforward is a *sample of a curve*: reading x and y at ``t+0.18``
    while reading z at ``t+0.05`` composes a point that lies on no part of the
    trajectory at all. On a climbing turn that composite point sits several
    centimetres off the plan, permanently, and the position loop then spends its
    authority arguing with a reference the planner never produced.

    See ``test_mixing_two_instants_of_one_curve_leaves_the_curve`` below for the
    measurement behind that claim.
    """
    plant = VelocityPlant()
    assert plant.feedforward_lead_s == plant.horizontal.delay_s
    assert plant.vertical.delay_s != plant.horizontal.delay_s

    # Still the horizontal one when the axes are re-measured on another airframe.
    odd = VelocityPlant(horizontal=AxisPlant(delay_s=0.24),
                        vertical=AxisPlant(delay_s=0.9),
                        yaw=AxisPlant(delay_s=0.01))
    assert odd.feedforward_lead_s == 0.24


def test_mixing_two_instants_of_one_curve_leaves_the_curve():
    """Quantify the bug the shared lead avoids, on a climbing turn.

    A 2 m radius turn flown at 1 m/s while climbing at 0.5 m/s, sampled with the
    horizontal lead on x and y and the vertical lead on z. The resulting point
    is ~5.8 cm from the nearest point of the curve it was supposedly read from,
    and that offset is a standing one -- it does not average out, because both
    leads are constant. Sampling once puts the point exactly on the curve.
    """
    def curve(t):
        # type: (object) -> np.ndarray
        """A 2 m radius left turn at 0.5 rad/s, climbing at 0.5 m/s."""
        return np.stack([2.0 * np.cos(0.5 * t), 2.0 * np.sin(0.5 * t), 0.5 * t], axis=-1)

    plant = VelocityPlant()
    now = 3.0
    dense = curve(np.linspace(now - 2.0, now + 2.0, 40001))

    honest = curve(now + plant.feedforward_lead_s)
    mixed = np.array([honest[0], honest[1], curve(now + plant.vertical.delay_s)[2]])

    # The grid resolves the curve to ~0.1 mm, so "on it" means well under 1 mm.
    assert float(np.min(np.linalg.norm(dense - honest, axis=1))) < 1e-3
    departure = float(np.min(np.linalg.norm(dense - mixed, axis=1)))
    assert departure > 0.05, departure


def test_the_defaults_describe_the_representative_airframe():
    """The documented defaults are part of the interface; pin them.

    They are advertised as a representative indoor quadrotor, and the ordering
    between the axes is the physical claim: horizontal is the slowest because a
    horizontal force has to be produced by rotating the whole airframe, while
    the vertical axis only has to change the thrust. A default set that lost
    that ordering would still fly and would quietly mis-lead every corner.
    """
    plant = VelocityPlant()
    assert (plant.horizontal.dc_gain, plant.horizontal.time_constant_s,
            plant.horizontal.delay_s) == (1.0, 0.5, 0.18)
    assert (plant.vertical.dc_gain, plant.vertical.time_constant_s,
            plant.vertical.delay_s) == (1.0, 0.4, 0.05)
    assert (plant.yaw.dc_gain, plant.yaw.time_constant_s,
            plant.yaw.delay_s) == (1.0, 0.5, 0.06)

    assert plant.horizontal.delay_s > plant.vertical.delay_s
    assert plant.horizontal.delay_s > plant.yaw.delay_s
    assert plant.horizontal.time_constant_s > plant.vertical.time_constant_s
    # A bare axis is the same shape of airframe, slightly optimistic on delay.
    assert (AxisPlant().dc_gain, AxisPlant().time_constant_s,
            AxisPlant().delay_s) == (1.0, 0.5, 0.15)


def test_the_defaults_match_the_airframe_the_suite_is_flown_against():
    """A lead term is only as good as the tau it was built from.

    ``LaggingAirframe`` carries the numbers measured on the reference Gazebo
    airframe (horizontal tau 0.51 s, delay 0.18 s, vertical tau 0.41 s). The
    closed-loop results in ``test_servo.py`` only mean what they claim if the
    plant the servo inverts is the plant it is flying against to within the ~10%
    the first-order fit is good for. If someone re-measures one and not the
    other, the tracking numbers degrade for a reason nobody would look for here.

    Only the horizontal delay is compared: the stand-in applies a single
    transport delay to all three axes for simplicity, which is deliberately not
    what the plant model claims.
    """
    plant = VelocityPlant()
    airframe = LaggingAirframe()
    assert abs(plant.horizontal.time_constant_s - float(airframe.tau[0])) \
        <= 0.1 * float(airframe.tau[0])
    assert abs(plant.vertical.time_constant_s - float(airframe.tau[2])) \
        <= 0.1 * float(airframe.tau[2])
    assert abs(plant.horizontal.delay_s - airframe.delay_s) <= 0.1 * airframe.delay_s
    assert abs(plant.yaw.time_constant_s - airframe.yaw_tau) <= 0.1 * airframe.yaw_tau
