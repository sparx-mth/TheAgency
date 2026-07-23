"""
Minimal SE(3) / quaternion helpers (pure numpy, ROS-free).

Quaternions are ``[x, y, z, w]`` (same order as ROS geometry_msgs and
tf.transformations) so adapters can feed message fields straight through.
Transforms are 4x4 homogeneous matrices. These helpers are shared across the
stack (temporal pose alignment in localization, the FALCON planning adapters)
and stay importable without a ROS environment (the FALCON adapter runs in a
container that has neither tf nor scipy guaranteed on the Python path).
"""
from __future__ import annotations

import math
from typing import Sequence

import numpy as np

_EPS = 1e-12


def quaternion_matrix(q: Sequence[float]) -> np.ndarray:
    """Quaternion [x,y,z,w] -> 4x4 homogeneous rotation matrix."""
    x, y, z, w = (float(v) for v in q)
    n = x * x + y * y + z * z + w * w
    T = np.identity(4)
    if n < _EPS:
        return T
    s = 2.0 / n
    xs, ys, zs = x * s, y * s, z * s
    wx, wy, wz = w * xs, w * ys, w * zs
    xx, xy, xz = x * xs, x * ys, x * zs
    yy, yz, zz = y * ys, y * zs, z * zs
    T[0, 0], T[0, 1], T[0, 2] = 1.0 - (yy + zz), xy - wz, xz + wy
    T[1, 0], T[1, 1], T[1, 2] = xy + wz, 1.0 - (xx + zz), yz - wx
    T[2, 0], T[2, 1], T[2, 2] = xz - wy, yz + wx, 1.0 - (xx + yy)
    return T


def quaternion_from_matrix(T: np.ndarray) -> np.ndarray:
    """4x4 (or 3x3) rotation -> quaternion [x,y,z,w] (Shepperd's method)."""
    m = np.asarray(T, dtype=float)
    m00, m01, m02 = m[0, 0], m[0, 1], m[0, 2]
    m10, m11, m12 = m[1, 0], m[1, 1], m[1, 2]
    m20, m21, m22 = m[2, 0], m[2, 1], m[2, 2]
    tr = m00 + m11 + m22
    if tr > 0.0:
        s = math.sqrt(tr + 1.0) * 2.0
        w, x, y, z = 0.25 * s, (m21 - m12) / s, (m02 - m20) / s, (m10 - m01) / s
    elif m00 > m11 and m00 > m22:
        s = math.sqrt(1.0 + m00 - m11 - m22) * 2.0
        w, x, y, z = (m21 - m12) / s, 0.25 * s, (m01 + m10) / s, (m02 + m20) / s
    elif m11 > m22:
        s = math.sqrt(1.0 + m11 - m00 - m22) * 2.0
        w, x, y, z = (m02 - m20) / s, (m01 + m10) / s, 0.25 * s, (m12 + m21) / s
    else:
        s = math.sqrt(1.0 + m22 - m00 - m11) * 2.0
        w, x, y, z = (m10 - m01) / s, (m02 + m20) / s, (m12 + m21) / s, 0.25 * s
    return np.array([x, y, z, w])


def quaternion_slerp(q0: Sequence[float], q1: Sequence[float],
                     fraction: float) -> np.ndarray:
    """Shortest-path spherical interpolation between two [x,y,z,w] quaternions."""
    a = np.asarray(q0, dtype=float)
    b = np.asarray(q1, dtype=float)
    a = a / (np.linalg.norm(a) or 1.0)
    b = b / (np.linalg.norm(b) or 1.0)
    if fraction <= 0.0:
        return a
    if fraction >= 1.0:
        return b
    d = float(np.dot(a, b))
    if d < 0.0:          # take the shorter arc
        d, b = -d, -b
    if d > 1.0 - _EPS:   # nearly identical -> linear
        return a
    angle = math.acos(d)
    s = math.sin(angle)
    return (math.sin((1.0 - fraction) * angle) * a
            + math.sin(fraction * angle) * b) / s


def make_transform(position: Sequence[float], q: Sequence[float]) -> np.ndarray:
    """(position xyz, quaternion [x,y,z,w]) -> 4x4 homogeneous transform."""
    T = quaternion_matrix(q)
    T[0, 3], T[1, 3], T[2, 3] = (float(v) for v in position)
    return T


def yaw_from_quaternion(q: Sequence[float]) -> float:
    """Yaw (rotation about +z) of a [x,y,z,w] quaternion, in radians."""
    x, y, z, w = (float(v) for v in q)
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def tilt_from_vertical(q: Sequence[float]) -> float:
    """Angle (radians, [0, pi]) between the body's +z axis and world +z.

    0 = level, pi/2 = on its side, pi = upside down. Independent of yaw.
    Used to reject poses captured during a physical upset (e.g. a collision)
    rather than a normal, level flight attitude."""
    T = quaternion_matrix(q)
    return math.acos(max(-1.0, min(1.0, float(T[2, 2]))))
