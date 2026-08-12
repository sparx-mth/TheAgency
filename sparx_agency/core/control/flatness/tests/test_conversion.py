"""The flatness conversion, checked against the physics it claims to encode."""
from __future__ import annotations

import math

import numpy as np
import pytest

from sparx_agency.core.control.constants import GRAVITY_MPS2
from sparx_agency.core.control.flatness import (
    AccelerationLimits, acceleration_to_attitude,
)


def _matrix_from_quaternion(qw, qx, qy, qz):
    """Rotation matrix from a (w, x, y, z) quaternion, written independently."""
    return np.array([
        [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
        [2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)],
        [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx * qx + qy * qy)],
    ], dtype=float)


def _body_z(command):
    """The commanded thrust axis, recovered from the quaternion."""
    return _matrix_from_quaternion(*command.quaternion_wxyz())[:, 2]


def _yaw_of(command):
    """The heading of a commanded attitude, from its body x axis."""
    body_x = _matrix_from_quaternion(*command.quaternion_wxyz())[:, 0]
    return math.atan2(float(body_x[1]), float(body_x[0]))


def test_hover_is_level_and_weighs_one_g():
    """Asking for no acceleration asks for exactly enough thrust to not fall."""
    command = acceleration_to_attitude([0.0, 0.0, 0.0], yaw=0.0)
    assert command.specific_thrust_mps2 == pytest.approx(GRAVITY_MPS2)
    assert command.tilt_rad == pytest.approx(0.0, abs=1e-9)
    assert _body_z(command) == pytest.approx([0.0, 0.0, 1.0], abs=1e-9)
    assert not command.saturated


def test_the_quaternion_is_a_unit_quaternion_at_every_attitude():
    """A non-unit quaternion is silently a scaling, and PX4 will reject it."""
    for ax in (-4.0, -1.0, 0.0, 2.5):
        for ay in (-3.0, 0.0, 1.5):
            for yaw in (-2.0, 0.0, 1.1, 3.0):
                command = acceleration_to_attitude([ax, ay, 1.0], yaw)
                norm = math.sqrt(sum(v * v for v in command.quaternion_wxyz()))
                assert norm == pytest.approx(1.0, abs=1e-12)


def test_the_thrust_axis_points_along_acceleration_plus_gravity():
    """The defining property: thrust carries both the manoeuvre and the weight."""
    wanted = np.array([1.5, -0.8, 0.6])
    command = acceleration_to_attitude(wanted, yaw=0.7)
    expected = wanted + np.array([0.0, 0.0, GRAVITY_MPS2])
    assert command.specific_thrust_mps2 == pytest.approx(float(np.linalg.norm(expected)))
    assert _body_z(command) == pytest.approx(expected / np.linalg.norm(expected), abs=1e-9)


def test_heading_survives_the_conversion():
    """Yaw is the free rotation about the thrust axis, and must come out intact."""
    for yaw in (-3.0, -1.2, 0.0, 0.9, 2.7):
        command = acceleration_to_attitude([1.0, -1.0, 0.3], yaw)
        assert _yaw_of(command) == pytest.approx(yaw, abs=1e-9)


def test_tilt_is_the_arctangent_of_horizontal_over_vertical():
    """A textbook relation, and the one that says the geometry is right."""
    command = acceleration_to_attitude([GRAVITY_MPS2 * math.tan(0.3), 0.0, 0.0], yaw=0.0)
    assert command.tilt_rad == pytest.approx(0.3, abs=1e-9)


def test_heading_does_not_change_how_hard_it_accelerates():
    """Translation and heading are decoupled; spinning must not change the thrust."""
    thrusts = [acceleration_to_attitude([1.0, 0.5, 0.2], yaw).specific_thrust_mps2
               for yaw in np.linspace(-math.pi, math.pi, 12)]
    assert max(thrusts) - min(thrusts) < 1e-12


def test_the_tilt_ceiling_binds_and_is_reported():
    """An absurd horizontal demand is capped, not obeyed."""
    limits = AccelerationLimits(max_tilt_rad=math.radians(30.0))
    command = acceleration_to_attitude([50.0, 0.0, 0.0], yaw=0.0, limits=limits)
    assert command.tilt_rad == pytest.approx(math.radians(30.0), abs=1e-6)
    assert command.saturated


def test_vertical_wins_when_the_airframe_runs_out_of_thrust():
    """Saturating, the aircraft holds its climb and gives up the corner.

    The other way round is a controller that trades altitude for cornering,
    which on an indoor flight means the floor.
    """
    limits = AccelerationLimits(max_specific_thrust=1.2 * GRAVITY_MPS2)
    command = acceleration_to_attitude([30.0, 0.0, 1.0], yaw=0.0, limits=limits)
    axis = _body_z(command) * command.specific_thrust_mps2
    assert command.saturated
    assert float(axis[2]) == pytest.approx(1.0 + GRAVITY_MPS2, abs=1e-6)
    assert command.specific_thrust_mps2 <= 1.2 * GRAVITY_MPS2 + 1e-9


def test_free_fall_still_produces_a_valid_attitude():
    """Cancelling gravity exactly is a singularity; it must not become a NaN."""
    command = acceleration_to_attitude([0.0, 0.0, -GRAVITY_MPS2], yaw=0.4)
    assert command.specific_thrust_mps2 > 0.0
    assert all(math.isfinite(v) for v in command.quaternion_wxyz())
    assert command.saturated


def test_no_jerk_means_no_rate_feedforward():
    """Absent a third derivative the rates are zero -- correct, just slower."""
    command = acceleration_to_attitude([1.0, 1.0, 0.0], yaw=0.0)
    assert command.body_rates()[:2] == (0.0, 0.0)


def test_rate_feedforward_matches_the_attitude_it_will_produce():
    """The strongest check here: the feedforward equals the true angular rate.

    A smooth acceleration profile is differentiated analytically for the jerk,
    and the resulting attitude command is differentiated *numerically* over the
    same instant. If the flatness algebra is right the two agree, which is what
    makes the feedforward worth having -- it is the rate the aircraft will need,
    known before any error exists.
    """
    def acceleration(t):
        return np.array([1.2 * math.sin(0.9 * t), 0.8 * math.cos(1.3 * t), 0.3 * math.sin(t)])

    def jerk(t):
        return np.array([1.2 * 0.9 * math.cos(0.9 * t),
                         -0.8 * 1.3 * math.sin(1.3 * t),
                         0.3 * math.cos(t)])

    step = 1e-6
    for t in (0.4, 1.1, 2.6, 4.0):
        here = acceleration_to_attitude(acceleration(t), yaw=0.0, jerk=jerk(t))
        before = _matrix_from_quaternion(
            *acceleration_to_attitude(acceleration(t - step), yaw=0.0).quaternion_wxyz())
        after = _matrix_from_quaternion(
            *acceleration_to_attitude(acceleration(t + step), yaw=0.0).quaternion_wxyz())
        rotation = _matrix_from_quaternion(*here.quaternion_wxyz())
        skew = rotation.T.dot((after - before) / (2.0 * step))
        measured = (skew[2, 1], skew[0, 2], skew[1, 0])
        assert here.roll_rate == pytest.approx(measured[0], abs=1e-4)
        assert here.pitch_rate == pytest.approx(measured[1], abs=1e-4)


def test_yaw_rate_feedforward_is_projected_onto_the_body_axis():
    """A tilted aircraft turns more slowly about its own z than about the world's."""
    level = acceleration_to_attitude([0.0, 0.0, 0.0], yaw=0.0, yaw_rate=0.5)
    tilted = acceleration_to_attitude([5.0, 0.0, 0.0], yaw=0.0, yaw_rate=0.5)
    assert level.yaw_rate == pytest.approx(0.5)
    assert 0.0 < tilted.yaw_rate < 0.5
