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
    tag_id: int
    cam_T_tag: np.ndarray  # shape (4,4)
    weight: float = 1.0    # pixel area or confidence — used by weighted fusion


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
        [1 - 2*(yy + zz),     2*(xy - wz),         2*(xz + wy)],
        [2*(xy + wz),         1 - 2*(xx + zz),     2*(yz - wx)],
        [2*(xz - wy),         2*(yz + wx),         1 - 2*(xx + yy)],
    ], dtype=float)

    M = np.eye(4, dtype=float)
    M[:3, :3] = R
    return M


def quaternion_from_matrix(M: np.ndarray) -> List[float]:
    R = M[:3, :3]
    trace = float(R[0, 0] + R[1, 1] + R[2, 2])

    if trace > 0:
        s = 0.5 / math.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (R[2, 1] - R[1, 2]) * s
        y = (R[0, 2] - R[2, 0]) * s
        z = (R[1, 0] - R[0, 1]) * s
    else:
        if R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
            s = 2.0 * math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
            w = (R[2, 1] - R[1, 2]) / s
            x = 0.25 * s
            y = (R[0, 1] + R[1, 0]) / s
            z = (R[0, 2] + R[2, 0]) / s
        elif R[1, 1] > R[2, 2]:
            s = 2.0 * math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
            w = (R[0, 2] - R[2, 0]) / s
            x = (R[0, 1] + R[1, 0]) / s
            y = 0.25 * s
            z = (R[1, 2] + R[2, 1]) / s
        else:
            s = 2.0 * math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
            w = (R[1, 0] - R[0, 1]) / s
            x = (R[0, 2] + R[2, 0]) / s
            y = (R[1, 2] + R[2, 1]) / s
            z = 0.25 * s

    return [float(x), float(y), float(z), float(w)]


def rpy_from_rotation_matrix(R: np.ndarray) -> Tuple[float, float, float]:
    """
    Extract roll, pitch, yaw from rotation matrix using the same convention:
    R = Rz(yaw) @ Ry(pitch) @ Rx(roll)
    Returns radians.
    """
    sy = math.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])

    singular = sy < 1e-6

    if not singular:
        roll = math.atan2(R[2, 1], R[2, 2])
        pitch = math.atan2(-R[2, 0], sy)
        yaw = math.atan2(R[1, 0], R[0, 0])
    else:
        roll = math.atan2(-R[1, 2], R[1, 1])
        pitch = math.atan2(-R[2, 0], sy)
        yaw = 0.0

    return roll, pitch, yaw


def print_transform_debug(name: str, T: np.ndarray):
    R = T[:3, :3]
    t = T[:3, 3]

    roll, pitch, yaw = rpy_from_rotation_matrix(R)

    print(f"\n========== {name} ==========")
    print("T:")
    print(np.array2string(T, precision=4, suppress_small=True))

    print(f"translation: x={t[0]:.4f}, y={t[1]:.4f}, z={t[2]:.4f}")

    print(
        "rpy rad: "
        f"roll={roll:.4f}, pitch={pitch:.4f}, yaw={yaw:.4f}"
    )
    print(
        "rpy deg: "
        f"roll={math.degrees(roll):.2f}, "
        f"pitch={math.degrees(pitch):.2f}, "
        f"yaw={math.degrees(yaw):.2f}"
    )

    # Columns of R are the local frame axes expressed in the parent frame
    print(f"{name} local X axis in parent frame: {R[:, 0]}")
    print(f"{name} local Y axis in parent frame: {R[:, 1]}")
    print(f"{name} local Z axis in parent frame: {R[:, 2]}")
    print("====================================")

def world_T_tag_from_pose(pose: TagWorldPose) -> np.ndarray:
    (x, y, z) = pose.xyz
    (roll, pitch, yaw) = pose.rpy
    M = euler_matrix(roll, pitch, yaw)
    M[0, 3] = float(x)
    M[1, 3] = float(y)
    M[2, 3] = float(z)
    return M


# -------------------------
# Core estimation functions
# -------------------------

def estimate_world_T_cam_from_single_tag(
    tag_world_pose: TagWorldPose,
    cam_T_tag: np.ndarray,
) -> np.ndarray:
    """
    Important:
    In your current pipeline, cam_T_tag is actually camera_T_tag from solvePnP:
        X_camera = camera_T_tag @ X_tag

    Therefore:
        tag_T_camera = inv(camera_T_tag)
        world_T_camera = world_T_tag @ tag_T_camera
    """
    world_T_tag = world_T_tag_from_pose(tag_world_pose)

    camera_T_tag = cam_T_tag
    tag_T_camera = np.linalg.inv(camera_T_tag)

    world_T_camera = world_T_tag @ tag_T_camera
    
    #print_transform_debug("world_T_tag", world_T_tag)
    #print_transform_debug("camera_T_tag from solvePnP", camera_T_tag)
    #print_transform_debug("tag_T_camera = inv(camera_T_tag)", tag_T_camera)
    #print_transform_debug("world_T_camera = world_T_tag @ tag_T_camera", world_T_camera)


    return world_T_camera


def fuse_world_T_cam(
    world_T_cam_list: List[np.ndarray],
    method: str = "avg_translation_keep_first_rotation",
) -> np.ndarray:
    """
    Fuse multiple world_T_cam transforms.
    Default matches your behavior:
      - Average translation
      - Keep rotation from the first estimate
    """
    if not world_T_cam_list:
        raise ValueError("world_T_cam_list is empty")

    if len(world_T_cam_list) == 1:
        return world_T_cam_list[0]

    if method != "avg_translation_keep_first_rotation":
        raise ValueError(f"Unsupported fuse method: {method}")

    translations = np.array([M[:3, 3] for M in world_T_cam_list], dtype=float)
    avg_t = np.mean(translations, axis=0)

    R = world_T_cam_list[0][:3, :3]
    out = np.eye(4, dtype=float)
    out[:3, :3] = R
    out[:3, 3] = avg_t
    return out


def estimate_camera_pose_from_tags(
    observations: List[TagObservation],
    tag_map: Dict[int, TagWorldPose],
    fuse_method: str = "avg_translation_keep_first_rotation",
) -> Optional[PoseEstimate]:
    """
    Main entry point:
    - For each observation: compute world_T_cam using known world pose of that tag
    - Fuse multiple tags
    """
    world_poses: List[np.ndarray] = []
    used_ids: List[int] = []

    for obs in observations:
        if obs.tag_id not in tag_map:
            continue
        try:
            wTc = estimate_world_T_cam_from_single_tag(tag_map[obs.tag_id], obs.cam_T_tag)
            world_poses.append(wTc)
            used_ids.append(obs.tag_id)
        except Exception:
            continue

    if not world_poses:
        return None

    fused = fuse_world_T_cam(world_poses, method=fuse_method)
    return PoseEstimate(world_T_cam=fused, used_tag_ids=used_ids)


def matrix_to_pose(world_T_cam: np.ndarray) -> Tuple[Tuple[float, float, float], Tuple[float, float, float, float]]:
    """
    Returns:
      position (x,y,z), quaternion (qx,qy,qz,qw)
    """
    x, y, z = (float(world_T_cam[0, 3]), float(world_T_cam[1, 3]), float(world_T_cam[2, 3]))
    qx, qy, qz, qw = quaternion_from_matrix(world_T_cam)
    return (x, y, z), (qx, qy, qz, qw)
