"""Planar (SE(2)) frame changes: the body↔world pair every layer needs.

This is unambiguous, universal arithmetic -- rotate by a yaw, translate by a
position -- so it lives in the widest ``common`` rather than beside any one
consumer. Planners, trackers, mapping and the VLA policies all express the same
two operations, and until this module existed each wrote its own copy: a Python
loop in ``vlas/navdp/geometry``, a numpy stack in ``vlas/common/plan_commit``,
and an inline ``cos``/``sin`` pair in half a dozen followers. Identical maths,
independently maintained.

The frame is the repo's body FLU (REP-103): ``x`` forward, ``y`` left, yaw CCW
about ``+z``. A pose is ``(x, y, yaw)`` in the world frame; a body point is
``(forward, left)`` with the robot at the origin looking along ``+x``::

    world_x = ref_x + forward * cos(yaw) - left * sin(yaw)
    world_y = ref_y + forward * sin(yaw) + left * cos(yaw)

:func:`world_to_body_2d` is the exact inverse, so a point taken to the world and
back with the same pose round-trips.

Scalar and vectorised forms both exist on purpose: a follower converts one
carrot per tick and wants floats, while a policy anchors a whole trajectory at
once and wants an array. Sharing one of them and re-deriving the other is how
the copies started.

Python 3.8 compatible and numpy-1.17-API only -- the FALCON Noetic adapter
imports this through ``vlas/navdp/geometry``. No scipy.
"""
from __future__ import annotations

import math
from typing import Tuple

import numpy as np


def rotate_2d(x, y, angle_rad):
    # type: (float, float, float) -> Tuple[float, float]
    """Rotate a planar vector CCW by ``angle_rad``.

    Args:
        x, y: the vector.
        angle_rad: rotation, radians, counter-clockwise.

    Returns:
        The rotated ``(x, y)``.
    """
    c, s = math.cos(angle_rad), math.sin(angle_rad)
    return float(x * c - y * s), float(x * s + y * c)


def body_to_world_2d(forward, left, ref_x, ref_y, ref_yaw):
    # type: (float, float, float, float, float) -> Tuple[float, float]
    """One body-frame FLU point, expressed in the world frame.

    Args:
        forward, left: the body point (metres), robot at the origin facing ``+x``.
        ref_x, ref_y, ref_yaw: the robot's world pose.

    Returns:
        ``(world_x, world_y)``.
    """
    dx, dy = rotate_2d(forward, left, ref_yaw)
    return float(ref_x + dx), float(ref_y + dy)


def world_to_body_2d(wx, wy, ref_x, ref_y, ref_yaw):
    # type: (float, float, float, float, float) -> Tuple[float, float]
    """One world point, expressed in the body FLU frame at ``ref``.

    The exact inverse of :func:`body_to_world_2d`.

    Args:
        wx, wy: the world point (metres).
        ref_x, ref_y, ref_yaw: the robot's world pose (the body origin/heading).

    Returns:
        ``(forward, left)`` body coordinates (metres).
    """
    dx, dy = wx - ref_x, wy - ref_y
    c, s = math.cos(ref_yaw), math.sin(ref_yaw)
    return float(dx * c + dy * s), float(-dx * s + dy * c)


def body_to_world_xy(body_xy, ref_x, ref_y, ref_yaw):
    # type: (np.ndarray, float, float, float) -> np.ndarray
    """A whole body-frame path, expressed in the world frame.

    Args:
        body_xy: ``(N, >=2)`` body ``(forward, left)`` waypoints. Extra columns
            -- a policy's yaw channel, say -- are ignored rather than rotated,
            because a heading is not a position and rotating it here would be
            wrong twice.
        ref_x, ref_y, ref_yaw: the robot's world pose.

    Returns:
        ``(N, 2)`` float64 world waypoints.

    Raises:
        ValueError: ``body_xy`` is not ``(N, >=2)``.
    """
    pts = np.atleast_2d(np.asarray(body_xy, dtype=np.float64))
    if pts.ndim != 2 or pts.shape[1] < 2:
        raise ValueError("body_xy must be (N, >=2); got shape %r"
                         % (tuple(np.shape(body_xy)),))
    c, s = math.cos(ref_yaw), math.sin(ref_yaw)
    return np.stack([ref_x + pts[:, 0] * c - pts[:, 1] * s,
                     ref_y + pts[:, 0] * s + pts[:, 1] * c], axis=1)
