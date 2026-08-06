"""ENU/FLU to NED/FRD, checked against the conversion the velocity path already does.

A frame mistake here does not raise. It produces an aircraft that rolls when
told to pitch, or flies a mirror image of its plan, and it reads from the
outside exactly like a badly tuned controller -- so the properties are asserted
rather than eyeballed.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from sparx_agency.core.control.flatness import (
    acceleration_to_attitude, matrix_from_quaternion, quaternion_from_matrix,
    world_attitude_to_ned_frd,
)


def _level_at(yaw):
    """The attitude of a level aircraft heading ``yaw`` in world ENU."""
    return acceleration_to_attitude([0.0, 0.0, 0.0], yaw).quaternion_wxyz()


def _ned_yaw(quaternion_wxyz):
    """Heading of an NED/FRD attitude: clockwise from north."""
    body_x = matrix_from_quaternion(quaternion_wxyz)[:, 0]
    return math.atan2(float(body_x[1]), float(body_x[0]))


def test_level_headings_match_the_velocity_path_conversion():
    """The two paths must agree, or switching control modes yaws the aircraft.

    ``send_velocity_world`` converts an ENU heading with ``yaw_ned = pi/2 -
    yaw_enu``. Whatever this module does to a full attitude has to reduce to
    exactly that when the aircraft is level.
    """
    for yaw_enu in (-2.5, -1.0, 0.0, 0.6, 1.57, 3.0):
        converted = world_attitude_to_ned_frd(_level_at(yaw_enu))
        expected = math.atan2(math.sin(math.pi / 2.0 - yaw_enu),
                              math.cos(math.pi / 2.0 - yaw_enu))
        assert _ned_yaw(converted) == pytest.approx(expected, abs=1e-9)


def test_the_result_is_a_rotation_and_not_a_reflection():
    """Determinant +1 at every attitude. The wrong basis change gives -1."""
    for ax in (-3.0, 0.0, 2.0):
        for ay in (-2.0, 1.5):
            for yaw in (-1.0, 0.4, 2.2):
                command = acceleration_to_attitude([ax, ay, 0.5], yaw)
                matrix = matrix_from_quaternion(
                    world_attitude_to_ned_frd(command.quaternion_wxyz()))
                assert float(np.linalg.det(matrix)) == pytest.approx(1.0, abs=1e-9)
                assert matrix.dot(matrix.T) == pytest.approx(np.eye(3), abs=1e-9)


def _thrust_direction_ned(world_quaternion_wxyz):
    """Which way the rotors push, in NED, given a **world ENU/FLU** attitude.

    Two steps, and both are places to get a sign wrong. The attitude is
    converted, and then the thrust is read as the *negative* of the third
    column: that column is the body's **down** axis in NED/FRD, while the thrust
    comes out of the top of the aircraft.
    """
    converted = world_attitude_to_ned_frd(world_quaternion_wxyz)
    return -matrix_from_quaternion(converted)[:, 2]


def test_a_level_aircraft_has_its_body_down_axis_along_ned_down():
    """The defining flip: body-FLU up becomes body-FRD down, and NED down is +z."""
    body_down_ned = matrix_from_quaternion(world_attitude_to_ned_frd(_level_at(0.0)))[:, 2]
    assert body_down_ned == pytest.approx([0.0, 0.0, 1.0], abs=1e-9)
    assert _thrust_direction_ned(_level_at(0.0)) == pytest.approx([0.0, 0.0, -1.0], abs=1e-9)


def test_leaning_east_stays_a_lean_east_in_ned():
    """Accelerating east must still be a lean east after the conversion.

    A swapped horizontal axis passes the level-heading test and fails this one,
    which is why both are here: heading and tilt can be wrong independently.
    """
    command = acceleration_to_attitude([4.0, 0.0, 0.0], yaw=0.0)
    thrust = _thrust_direction_ned(command.quaternion_wxyz())
    # Mostly up (NED -z), tilted toward east (+y), nothing north.
    assert float(thrust[2]) < -0.9
    assert float(thrust[1]) > 0.2
    assert abs(float(thrust[0])) < 1e-9


def test_leaning_north_stays_a_lean_north_in_ned():
    """The other horizontal axis, for the same reason."""
    command = acceleration_to_attitude([0.0, 4.0, 0.0], yaw=0.0)
    thrust = _thrust_direction_ned(command.quaternion_wxyz())
    assert float(thrust[0]) > 0.2
    assert abs(float(thrust[1])) < 1e-9


def test_the_yaw_offset_rotates_about_the_world_axis():
    """Where an autopilot's heading bias goes, and it goes in ENU, before the swap."""
    plain = world_attitude_to_ned_frd(_level_at(0.5))
    offset = world_attitude_to_ned_frd(_level_at(0.5), yaw_offset=-0.3)
    assert _ned_yaw(offset) == pytest.approx(_ned_yaw(plain) + 0.3, abs=1e-9)


def test_the_conversion_is_its_own_inverse_on_headings():
    """Applying it twice returns the original level attitude, up to sign.

    S and D are both involutions, so ``S R D`` converted again is ``R`` -- a
    cheap end-to-end check that neither matrix has drifted.
    """
    for yaw in (-2.0, 0.0, 1.1):
        once = world_attitude_to_ned_frd(_level_at(yaw))
        twice = world_attitude_to_ned_frd(once)
        assert matrix_from_quaternion(twice) == pytest.approx(
            matrix_from_quaternion(_level_at(yaw)), abs=1e-9)


def test_quaternion_and_matrix_round_trip():
    """The shared helpers agree with each other at arbitrary attitudes."""
    rng = np.random.RandomState(3)
    for _ in range(20):
        q = rng.normal(size=4)
        q /= np.linalg.norm(q)
        if q[0] < 0:
            q = -q
        assert quaternion_from_matrix(matrix_from_quaternion(q)) == pytest.approx(q, abs=1e-9)


def test_a_zero_quaternion_is_refused():
    """It names no rotation, and silently treating it as identity hides a bug."""
    with pytest.raises(ValueError, match="not a rotation"):
        matrix_from_quaternion([0.0, 0.0, 0.0, 0.0])
