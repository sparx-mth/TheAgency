"""Where a depth pixel lies in the world, and where a world point lands in the image.

Every mapper, detection fusion and privileged fake on the ObjectNav layer needs
the same two maps between the agent's RGB-D camera and world ENU, so they are
written once, here, against the repo's frames doctrine. Each slip they guard
against yields a plausible map and a lower score, never a crash:

* **The optical frame treated as FLU.** A raw pinhole ray is optical (``x``
  right, ``y`` down, ``z`` forward). Rotating it by the body pose directly is
  the SJTU object mapper's bug: a surface dead ahead lands ``d`` metres *above*
  the camera. The ray goes through
  :data:`~sparx_agency.core.mapping.objects.geometry.R_BODY_OPTICAL` first.
* **A pitch sign flip.** REP-103: positive camera pitch looks *down*, so the
  floor seen after LOOK_DOWN lands on the floor, not on the ceiling.
* **Depth read as ray length.** It is the optical-frame ``z``, so
  ``x = (u - cx) / fx * d`` with no normalisation of the ray.
* **A half-pixel principal point.** An integer pixel index is the pixel's
  centre, so a symmetric pinhole's principal point is ``((W - 1) / 2,
  (H - 1) / 2)``; ``W / 2`` shifts every point sideways.
* **Invalid readings becoming points.** ``NaN`` (no reading), ``+inf``
  (nothing in range), readings outside the sensor's range and readings *at*
  its clip bounds -- a clipped return, not a measurement -- are dropped, never
  clamped to a surface that is not there.

The pinhole itself comes from a field of view in
:mod:`~sparx_agency.core.planning.objnav.camera_intrinsics`; its two helpers
are re-exported here, where callers first found them.

Rotations come from :func:`~sparx_agency.core.common.math.se3.quaternion_matrix`.
The base is level (``AgentPose`` has no roll or body pitch), so the camera's
attitude is the agent's yaw followed by the camera's pitch.

Python 3.8 syntax, numpy.
"""
from __future__ import annotations

import math
import numbers
from typing import Tuple

import numpy as np

from sparx_agency.core.common.math.se3 import quaternion_matrix
from sparx_agency.core.mapping.objects.geometry import R_BODY_OPTICAL
from sparx_agency.core.planning.objnav.camera_intrinsics import (
    intrinsics_from_hfov,
    intrinsics_from_vfov,
)
from sparx_agency.core.planning.objnav.errors import ObjNavError, ObservationError
from sparx_agency.core.planning.objnav.types.camera import CameraSpec
from sparx_agency.core.planning.objnav.types.pose import AgentPose

__all__ = [
    "intrinsics_from_hfov",
    "intrinsics_from_vfov",
    "camera_rotation_world",
    "camera_position_world",
    "world_T_camera_optical",
    "backproject_depth",
    "project_to_image",
]


def camera_rotation_world(pose: AgentPose) -> np.ndarray:
    """The world-from-camera-FLU rotation ``Rz(yaw) @ Ry(camera_pitch)``.

    The camera turns with the base about world ``+z``, then tilts about its
    own ``+y`` (left) axis; REP-103 makes positive pitch swing the forward
    axis toward ``-z``, i.e. look down. Columns are the camera's forward,
    left and up axes in world ENU.

    Args:
        pose: The agent's pose.

    Returns:
        ``(3, 3)`` float64 rotation matrix.

    Raises:
        TypeError: If ``pose`` is not an :class:`AgentPose`.
    """
    _require(pose, AgentPose, "pose")
    yaw = pose.yaw
    pitch = pose.camera_pitch
    r_yaw = quaternion_matrix(
        (0.0, 0.0, math.sin(yaw / 2.0), math.cos(yaw / 2.0)))[:3, :3]
    r_pitch = quaternion_matrix(
        (0.0, math.sin(pitch / 2.0), 0.0, math.cos(pitch / 2.0)))[:3, :3]
    return r_yaw.dot(r_pitch)


def camera_position_world(pose: AgentPose, camera: CameraSpec) -> np.ndarray:
    """The optical centre in world ENU: straight above the base, at mount height.

    Args:
        pose: The agent's pose; ``z`` is the floor under it.
        camera: The camera, mounted ``height_m`` above that floor.

    Returns:
        ``(3,)`` float64 ``(x, y, z + height_m)``.

    Raises:
        TypeError: If ``pose`` or ``camera`` has the wrong type.
    """
    _require(pose, AgentPose, "pose")
    _require(camera, CameraSpec, "camera")
    return np.array([pose.x, pose.y, pose.z + camera.height_m],
                    dtype=np.float64)


def world_T_camera_optical(pose: AgentPose, camera: CameraSpec) -> np.ndarray:
    """The world-from-camera-optical transform: ``p_world = T @ [p_optical, 1]``.

    The rotation is ``camera_rotation_world(pose) @ R_BODY_OPTICAL`` -- optical
    ``z`` (forward) onto the camera's forward axis, optical ``x`` (right) onto
    its ``-y``, optical ``y`` (down) onto its ``-z`` -- and the translation is
    :func:`camera_position_world`. Its ``[:3, :3]`` block is what a ray caster
    that renders in the optical frame needs.

    Args:
        pose: The agent's pose.
        camera: The camera.

    Returns:
        ``(4, 4)`` float64 homogeneous transform.

    Raises:
        TypeError: If ``pose`` or ``camera`` has the wrong type.
    """
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = camera_rotation_world(pose).dot(R_BODY_OPTICAL)
    transform[:3, 3] = camera_position_world(pose, camera)
    return transform


def backproject_depth(depth_m: np.ndarray, camera: CameraSpec,
                      pose: AgentPose, stride: int = 1) -> np.ndarray:
    """World points of the valid pixels of a depth image.

    A pixel ``(u, v)`` reading ``d`` is the optical point
    ``((u - cx) / fx * d, (v - cy) / fy * d, d)``, moved to the world by
    :func:`world_T_camera_optical`. A reading is kept only when it is finite,
    positive and strictly inside ``(min_depth_m, max_depth_m)``; ``NaN``,
    ``+inf``, out-of-range readings and readings equal to a clip bound are
    dropped, not clamped. A reading at a bound is a clipped return, not a
    measurement: Habitat's normalised 0 un-normalises to exactly
    ``min_depth_m`` (something nearer), its 1 to exactly ``max_depth_m``
    (nothing in range).

    Args:
        depth_m: ``(H, W)`` floating-point depth, metres, optical-frame ``z``,
            matching ``camera``.
        camera: The camera the image was captured with.
        pose: The pose it was captured at.
        stride: Keep every ``stride``-th pixel in ``u`` and in ``v``, starting
            at pixel ``(0, 0)``.

    Returns:
        ``(N, 3)`` float64 world points, in row-major pixel order (``v``, then
        ``u``); ``(0, 3)`` when no reading is valid.

    Raises:
        ObservationError: If ``depth_m`` is not a floating-point array of the
            camera's ``(height, width)``.
        ObjNavError: If ``stride`` is not an integer >= 1.
        TypeError: If ``pose`` or ``camera`` has the wrong type.
    """
    _require(camera, CameraSpec, "camera")
    _require(pose, AgentPose, "pose")
    if (not isinstance(stride, numbers.Integral) or isinstance(stride, bool)
            or stride < 1):
        raise ObjNavError(
            "backproject_depth: stride must be an integer >= 1 (1 keeps every "
            "pixel), got %r" % (stride,))
    _check_depth(depth_m, camera)
    step = int(stride)
    sampled = depth_m[::step, ::step]
    rows, cols = np.nonzero(_measured(sampled, camera))
    d = sampled[rows, cols].astype(np.float64)
    u = cols.astype(np.float64) * step
    v = rows.astype(np.float64) * step
    k = camera.intrinsics
    optical = np.stack([(u - k.cx) / k.fx * d, (v - k.cy) / k.fy * d, d],
                       axis=1)
    transform = world_T_camera_optical(pose, camera)
    return optical.dot(transform[:3, :3].T) + transform[:3, 3]


def project_to_image(points_world: np.ndarray, camera: CameraSpec,
                     pose: AgentPose) -> Tuple[np.ndarray, np.ndarray]:
    """Pixel coordinates and optical depth of world points, as this camera sees them.

    The inverse of :func:`backproject_depth`: a point it returned projects
    back onto its pixel, at its depth. No point is filtered: one behind the
    camera has optical ``z <= 0``, and its pixel is mirrored (``z < 0``) or
    not finite (``z == 0``). Keep ``z > 0`` -- and the pixels inside the
    image -- before using them.

    Args:
        points_world: ``(N, 3)`` finite world points, metres.
        camera: The camera.
        pose: The pose to look from.

    Returns:
        ``(pixels, z)``: ``(N, 2)`` float64 ``(u, v)`` -- ``u`` the column,
        ``v`` the row, integers at pixel centres -- and ``(N,)`` float64
        optical-frame depth, metres.

    Raises:
        ObjNavError: If ``points_world`` is not an ``(N, 3)`` array of finite
            numbers.
        TypeError: If ``pose`` or ``camera`` has the wrong type.
    """
    _require(camera, CameraSpec, "camera")
    _require(pose, AgentPose, "pose")
    points = _as_points(points_world)
    transform = world_T_camera_optical(pose, camera)
    # Row vectors: p_optical = R^T (p_world - t) is (p_world - t) @ R.
    optical = (points - transform[:3, 3]).dot(transform[:3, :3])
    z = optical[:, 2]
    k = camera.intrinsics
    # z == 0 (a point in the camera's own plane) divides by zero; its pixel is
    # documented as meaningless, so the warning would only be noise.
    with np.errstate(divide="ignore", invalid="ignore"):
        u = k.fx * optical[:, 0] / z + k.cx
        v = k.fy * optical[:, 1] / z + k.cy
    return np.stack([u, v], axis=1), z


def _measured(depth: np.ndarray, camera: CameraSpec) -> np.ndarray:
    """Mask of the readings that measured a surface: finite, positive, strictly inside the clip range.

    Deliberately not ``depth_bbox_fusion.valid_depth_mask``: its bounds are
    inclusive, so a clipped return at exactly ``min_depth_m`` or
    ``max_depth_m`` would become a point at the clip distance.
    """
    # NaN compares False either way; errstate keeps older numpy from warning.
    with np.errstate(invalid="ignore"):
        return (np.isfinite(depth) & (depth > 0.0)
                & (depth > camera.min_depth_m) & (depth < camera.max_depth_m))


def _require(value, kind, name: str) -> None:
    """Raise TypeError unless ``value`` is a ``kind``."""
    if not isinstance(value, kind):
        raise TypeError("%s must be a %s, got %r"
                        % (name, kind.__name__, value))


def _as_points(points_world) -> np.ndarray:
    """``points_world`` as a finite float64 ``(N, 3)`` array, or ObjNavError."""
    try:
        points = np.asarray(points_world, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ObjNavError(
            "project_to_image: points_world must be an (N, 3) array of world "
            "points in metres: %s" % (exc,)) from exc
    if points.ndim != 2 or points.shape[1] != 3:
        raise ObjNavError(
            "project_to_image: points_world must have shape (N, 3) -- reshape "
            "a single point with .reshape(1, 3) -- got shape %r"
            % (points.shape,))
    if not np.all(np.isfinite(points)):
        raise ObjNavError(
            "project_to_image: points_world must be finite; drop invalid "
            "points before projecting")
    return points


def _check_depth(depth_m, camera: CameraSpec) -> None:
    """Raise ObservationError unless ``depth_m`` is float metres of the camera's size."""
    size = (camera.intrinsics.height, camera.intrinsics.width)
    if (isinstance(depth_m, np.ndarray) and depth_m.shape == size
            and np.issubdtype(depth_m.dtype, np.floating)):
        return
    got = ("shape %r dtype %s" % (depth_m.shape, depth_m.dtype)
           if isinstance(depth_m, np.ndarray) else type(depth_m).__name__)
    raise ObservationError(
        "backproject_depth: depth_m must be a %r (height, width) "
        "floating-point array in metres, matching the camera; convert integer "
        "millimetres to float metres at the adapter. Got %s" % (size, got))
