# sparx_agency/core/localization/tag_triangulation.py
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np


# -------------------------
# Data models (ROS-agnostic)
# -------------------------

@dataclass(frozen=True)
class TagWorldPose:
    """
    Known tag pose in the world frame.
    xyz: (x,y,z) in meters
    rpy: (roll,pitch,yaw) in radians
    """
    xyz: Tuple[float, float, float]
    rpy: Tuple[float, float, float]


@dataclass(frozen=True)
class TagObservation:
    """
    Observation from camera to a tag.
    cam_T_tag: 4x4 homogeneous transform from camera frame to tag frame.
    """
    tag_id: int
    cam_T_tag: np.ndarray  # shape (4,4)


@dataclass(frozen=True)
class PoseEstimate:
    """
    Estimated camera pose in world frame.
    world_T_cam: 4x4 homogeneous transform
    used_tag_ids: list of tag ids used for estimation
    """
    world_T_cam: np.ndarray
    used_tag_ids: List[int]


# -------------------------
# Math utilities
# -------------------------

def euler_matrix(roll: float, pitch: float, yaw: float) -> np.ndarray:
    sr, cr = math.sin(roll), math.cos(roll)
    sp, cp = math.sin(pitch), math.cos(pitch)
    sy, cy = math.sin(yaw), math.cos(yaw)

    Rx = np.array([[1, 0, 0],
                   [0, cr, -sr],
                   [0, sr, cr]], dtype=float)

    Ry = np.array([[cp, 0, sp],
                   [0, 1, 0],
                   [-sp, 0, cp]], dtype=float)

    Rz = np.array([[cy, -sy, 0],
                   [sy, cy, 0],
                   [0, 0, 1]], dtype=float)

    R = Rz @ Ry @ Rx
    M = np.eye(4, dtype=float)
    M[:3, :3] = R
    return M


def quaternion_matrix(q: Iterable[float]) -> np.ndarray:
    x, y, z, w = [float(v) for v in q]
    norm = math.sqrt(x * x + y * y + z * z + w * w)
    if norm == 0:
        return np.eye(4, dtype=float)

    x /= norm
    y /= norm
    z /= norm
    w /= norm

    xx, yy, zz = x*x, y*y, z*z
    xy, xz, yz = x*y, x*z, y*z
    wx, wy, wz = w*x, w*y, w*z

    R = np.array([
        [1 - 2*(yy + zz*]()
