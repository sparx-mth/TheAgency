"""Tests for where the Iris's camera is.

These matter more than their size suggests. A wrong camera extrinsic does not
raise anywhere downstream -- it produces a complete, self-consistent map of the
building in the wrong place -- so the only place it can be caught is here and in
the start-up cross-check against Isaac's own accessor.
"""
import math

import numpy as np
import pytest

from sparx_agency.core.common.spatial_math import quat_to_rot
from sparx_agency.robots.PEGASUS.adapters.camera_pose import (
    BODY_TO_OPTICAL, camera_pose_world, optical_axis_world,
)
from sparx_agency.robots.PEGASUS.adapters.vehicle import CAMERA_OFFSET_FLU

LEVEL = (0.0, 0.0, 0.0, 1.0)


def _yaw_quaternion(yaw):
    return (0.0, 0.0, math.sin(yaw / 2.0), math.cos(yaw / 2.0))


def test_optical_axes_are_the_ones_falcons_map_config_declares():
    """BODY_TO_OPTICAL must equal the rotation block of FALCON's T_b_c.

    FALCON's own map configs carry ``T_b_c`` as body-FLU from camera-RDF. If this
    matrix and that YAML ever disagree, the depth cloud lands somewhere else and
    nothing says so.
    """
    falcon_t_b_c = np.array([
        [0.0, 0.0, 1.0],
        [-1.0, 0.0, 0.0],
        [0.0, -1.0, 0.0],
    ])
    np.testing.assert_allclose(BODY_TO_OPTICAL, falcon_t_b_c, atol=1e-12)


def test_optical_axes_point_where_the_names_say():
    """Columns are image-right, image-down and the optical axis, in body FLU."""
    right, down, forward = (BODY_TO_OPTICAL[:, i] for i in range(3))
    np.testing.assert_allclose(right, (0.0, -1.0, 0.0), atol=1e-12)   # body -y
    np.testing.assert_allclose(down, (0.0, 0.0, -1.0), atol=1e-12)    # body -z
    np.testing.assert_allclose(forward, (1.0, 0.0, 0.0), atol=1e-12)  # body +x


def test_it_is_a_rotation():
    np.testing.assert_allclose(BODY_TO_OPTICAL.dot(BODY_TO_OPTICAL.T), np.eye(3), atol=1e-12)
    assert np.linalg.det(BODY_TO_OPTICAL) == pytest.approx(1.0)


def test_camera_sits_forward_of_the_body_when_level():
    translation, _ = camera_pose_world((1.0, 2.0, 3.0), LEVEL)
    assert translation == pytest.approx((1.0 + CAMERA_OFFSET_FLU[0], 2.0, 3.0))


def test_the_mount_offset_rotates_with_the_aircraft():
    """Facing +y, the camera is offset along +y, not along +x."""
    translation, _ = camera_pose_world((0.0, 0.0, 1.5), _yaw_quaternion(math.pi / 2.0))
    assert translation == pytest.approx((0.0, CAMERA_OFFSET_FLU[0], 1.5), abs=1e-9)


# core's shared quat_to_rot returns float32, so a direction is only good to
# ~1e-7. At the 8 m the depth camera reaches that is a tenth of a micron, against
# 10 cm voxels -- the tolerance is float32's, not this code's.
DIRECTION_TOLERANCE = 1e-6


@pytest.mark.parametrize("yaw_deg", [0.0, 37.0, 90.0, 180.0, -120.0])
def test_the_optical_axis_is_the_heading(yaw_deg):
    yaw = math.radians(yaw_deg)
    axis = optical_axis_world(_yaw_quaternion(yaw))
    assert axis == pytest.approx((math.cos(yaw), math.sin(yaw), 0.0),
                                 abs=DIRECTION_TOLERANCE)


def test_the_returned_quaternion_is_the_returned_rotation():
    """The quaternion must describe world-from-optical, scalar last."""
    body_quaternion = _yaw_quaternion(0.7)
    _translation, optical_quaternion = camera_pose_world((0.0, 0.0, 0.0), body_quaternion)
    expected = np.asarray(quat_to_rot(*body_quaternion), dtype=float).dot(BODY_TO_OPTICAL)
    np.testing.assert_allclose(np.asarray(quat_to_rot(*optical_quaternion), dtype=float),
                               expected, atol=1e-6)


def test_a_pixel_ray_back_projects_to_the_right_side_of_the_aircraft():
    """The end-to-end property FALCON depends on.

    A point to the RIGHT of image centre must land to the aircraft's right, and
    one BELOW centre must land below it. Getting either backwards is the failure
    that builds a mirrored map, and it survives every check that only looks at
    the map's shape.
    """
    yaw = math.radians(30.0)
    translation, quaternion = camera_pose_world((2.0, -1.0, 1.5), _yaw_quaternion(yaw))
    rotation = np.asarray(quat_to_rot(*quaternion), dtype=float)

    # Optical-frame rays: +x is image right, +y is image down, +z is forward.
    right_of_centre = rotation.dot(np.array([1.0, 0.0, 3.0]))
    below_centre = rotation.dot(np.array([0.0, 1.0, 3.0]))

    # The aircraft's own right is -y in body FLU, rotated into the world.
    body_right = np.array([math.sin(yaw), -math.cos(yaw), 0.0])
    assert float(right_of_centre.dot(body_right)) > 0.0
    assert float(below_centre[2]) < 0.0
    assert translation[2] == pytest.approx(1.5)   # a level camera does not change height
