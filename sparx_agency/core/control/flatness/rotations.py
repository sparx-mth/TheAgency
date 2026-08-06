"""Quaternion and rotation-matrix conversions for the control chain.

``core.common.spatial_math`` already has a quaternion-to-matrix routine, and
this is not a duplicate of it by accident: that one takes **xyzw** and returns
**float32**, both of which are wrong here. MAVLink, and every attitude command
on this path, orders the quaternion **wxyz**, and an attitude that has been
through float32 comes back with about a milliradian of error -- harmless as a
visualisation, not something to put in a control loop and then differentiate.

So: wxyz, float64, and nothing else.
"""
from __future__ import annotations

import math

import numpy as np


def matrix_from_quaternion(quaternion_wxyz):
    # type: (object) -> np.ndarray
    """Rotation matrix from a ``(w, x, y, z)`` quaternion.

    Args:
        quaternion_wxyz: Four components, scalar first. Normalised internally,
            so a slightly stale unit quaternion is fine.

    Returns:
        A ``(3, 3)`` float64 rotation matrix whose columns are the rotated basis
        vectors.

    Raises:
        ValueError: If the quaternion has zero length and so names no rotation.
    """
    q = np.asarray(quaternion_wxyz, dtype=float).reshape(4)
    norm = float(np.linalg.norm(q))
    if norm <= 0.0:
        raise ValueError("a zero quaternion is not a rotation")
    w, x, y, z = q / norm
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ], dtype=float)


def quaternion_from_matrix(rotation):
    # type: (object) -> tuple
    """Convert a rotation matrix to a ``(w, x, y, z)`` quaternion.

    Shepperd's method: pick the branch whose divisor is largest, so the
    conversion stays well conditioned at every attitude rather than losing
    precision near the ones where a naive formula divides by nearly zero.

    Args:
        rotation: A ``(3, 3)`` rotation matrix.

    Returns:
        ``(w, x, y, z)``, with a non-negative scalar part -- the same rotation
        either way, but logs and tools read more sanely when a level hover is
        ``(1, 0, 0, 0)`` rather than ``(-1, 0, 0, 0)``.
    """
    m = np.asarray(rotation, dtype=float).reshape(3, 3)
    trace = float(m[0, 0] + m[1, 1] + m[2, 2])
    if trace > 0.0:
        scale = math.sqrt(trace + 1.0) * 2.0
        w = 0.25 * scale
        x = float(m[2, 1] - m[1, 2]) / scale
        y = float(m[0, 2] - m[2, 0]) / scale
        z = float(m[1, 0] - m[0, 1]) / scale
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        scale = math.sqrt(1.0 + float(m[0, 0] - m[1, 1] - m[2, 2])) * 2.0
        w = float(m[2, 1] - m[1, 2]) / scale
        x = 0.25 * scale
        y = float(m[0, 1] + m[1, 0]) / scale
        z = float(m[0, 2] + m[2, 0]) / scale
    elif m[1, 1] > m[2, 2]:
        scale = math.sqrt(1.0 + float(m[1, 1] - m[0, 0] - m[2, 2])) * 2.0
        w = float(m[0, 2] - m[2, 0]) / scale
        x = float(m[0, 1] + m[1, 0]) / scale
        y = 0.25 * scale
        z = float(m[1, 2] + m[2, 1]) / scale
    else:
        scale = math.sqrt(1.0 + float(m[2, 2] - m[0, 0] - m[1, 1])) * 2.0
        w = float(m[1, 0] - m[0, 1]) / scale
        x = float(m[0, 2] + m[2, 0]) / scale
        y = float(m[1, 2] + m[2, 1]) / scale
        z = 0.25 * scale
    if w < 0.0:
        w, x, y, z = -w, -x, -y, -z
    return w, x, y, z


def rotation_about_z(angle):
    # type: (float) -> np.ndarray
    """Rotation matrix for a turn of ``angle`` radians about the z axis."""
    cos, sin = math.cos(float(angle)), math.sin(float(angle))
    return np.array([[cos, -sin, 0.0], [sin, cos, 0.0], [0.0, 0.0, 1.0]], dtype=float)
