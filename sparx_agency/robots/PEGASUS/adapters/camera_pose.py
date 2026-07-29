"""Where the Iris's camera actually is, in the frame a mapper wants it.

A depth mapper does not want the aircraft's pose. It wants the pose of the
**optical frame** -- the one whose x is image-right, y is image-down and z points
out of the lens -- because that is the frame a pinhole back-projection produces
points in. Feeding a body pose instead produces a complete, confident, 90-degrees-
wrong map, and nothing anywhere raises.

Two things separate the two poses on this airframe:

* the camera is bolted 20 cm forward of the body origin
  (:data:`~sparx_agency.robots.PEGASUS.adapters.vehicle.CAMERA_OFFSET_FLU`), so
  its position is the body position plus that offset **rotated into the world**;
* body FLU and optical RDF differ by a fixed rotation, :data:`BODY_TO_OPTICAL`.

There is no pitch: the camera looks straight along body +x. That is not a detail
-- FALCON's frontier-visibility model hard-codes camera boresight = body +x
(``perception_utils.cpp``), so a pitched camera would make it choose viewpoints
whose heading does not actually see the frontier it chose them for.

This module imports no Isaac Sim, so it can be unit-tested anywhere.
:func:`camera_pose_world` is also cross-checked against Isaac's own
``Camera.get_world_pose(camera_axes="ros")`` at mission start-up -- see
``tasks/planning/falcon_pegasus/isaac/sensing.py`` -- because a silent
disagreement between the two is exactly the failure this module exists to
prevent.
"""
from __future__ import annotations

from typing import Tuple

import numpy as np

from sparx_agency.core.common.spatial_math import quat_to_rot, rot_to_quat
from sparx_agency.robots.PEGASUS.adapters.vehicle import CAMERA_OFFSET_FLU

BODY_TO_OPTICAL = np.array([
    [0.0, 0.0, 1.0],
    [-1.0, 0.0, 0.0],
    [0.0, -1.0, 0.0],
], dtype=float)
"""Rotation from the optical (RDF) frame into the body (FLU) frame.

Its columns are the optical axes written in body coordinates: image-right is
body -y, image-down is body -z, and the optical axis is body +x. It is the
rotation block of FALCON's own ``map_config/T_b_c``, which is how a FALCON map
config and this airframe are known to agree.
"""


def camera_pose_world(body_position, body_quaternion_xyzw,
                      offset_flu: Tuple[float, float, float] = CAMERA_OFFSET_FLU):
    """The camera optical frame's pose in the world.

    Args:
        body_position: World-frame ``(x, y, z)`` of the vehicle body, metres.
        body_quaternion_xyzw: World-from-body rotation, scalar **last** -- which
            is what ``Multirotor.state.attitude`` gives. Isaac's own
            ``isaacsim.core.prims`` uses scalar-first for the same quantity, so
            check which one you are holding before passing it here.
        offset_flu: Camera position in the body FLU frame, metres.

    Returns:
        ``(translation, quaternion_xyzw)`` -- the world position of the optical
        centre and the world-from-optical rotation, scalar last. This pair is
        exactly FALCON's ``T_w_c``.
    """
    qx, qy, qz, qw = (float(v) for v in body_quaternion_xyzw)
    rotation_world_body = np.asarray(quat_to_rot(qx, qy, qz, qw), dtype=float)
    translation = (np.asarray(body_position, dtype=float)
                   + rotation_world_body.dot(np.asarray(offset_flu, dtype=float)))
    rotation_world_optical = rotation_world_body.dot(BODY_TO_OPTICAL)
    return tuple(float(v) for v in translation), tuple(rot_to_quat(rotation_world_optical))


def optical_axis_world(body_quaternion_xyzw):
    """Which way the camera is looking, as a world-frame unit vector.

    Args:
        body_quaternion_xyzw: World-from-body rotation, scalar last.

    Returns:
        The world-frame ``(x, y, z)`` direction of the optical axis.
    """
    qx, qy, qz, qw = (float(v) for v in body_quaternion_xyzw)
    rotation = np.asarray(quat_to_rot(qx, qy, qz, qw), dtype=float)
    return tuple(float(v) for v in rotation.dot(BODY_TO_OPTICAL[:, 2]))
