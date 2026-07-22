# sparx_agency/core/localization/tag_triangulation.py
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from sparx_agency.core.common.spatial_math import (
    rpy_to_transform,
    rot_to_quat,
    rot_to_rpy,
    transform_to_pose,
)


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
    weight: float = 1.0  # Weight for fusion (e.g., tag area in pixels)


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
# Math utilities imported from spatial_math:
#   rpy_to_transform, rot_to_quat, rot_to_rpy, transform_to_pose
# -------------------------


def print_transform_debug(name: str, T: np.ndarray):
    R = T[:3, :3]
    t = T[:3, 3]

    roll, pitch, yaw = rot_to_rpy(R)

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
    M = rpy_to_transform(roll, pitch, yaw)
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

    # print_transform_debug("world_T_tag", world_T_tag)
    # print_transform_debug("camera_T_tag from solvePnP", camera_T_tag)
    # print_transform_debug("tag_T_camera = inv(camera_T_tag)", tag_T_camera)
    # print_transform_debug("world_T_camera = world_T_tag @ tag_T_camera", world_T_camera)

    return world_T_camera


def fuse_world_T_cam(
        world_T_cam_list: List[np.ndarray],
        weights: Optional[List[float]] = None,
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

    # Fallback to equal weights if none are provided
    if weights is None or len(weights) != len(world_T_cam_list):
        weights = [1.0] * len(world_T_cam_list)

    # Normalize weights so they sum to 1.0
    total_weight = sum(weights)
    if total_weight == 0:
        normalized_weights = [1.0 / len(weights)] * len(weights)
    else:
        normalized_weights = [w / total_weight for w in weights]

    # Calculate weighted average for translation (X, Y, Z)
    avg_t = np.zeros(3)
    for T, w in zip(world_T_cam_list, normalized_weights):
        avg_t += T[:3, 3] * w

    if method != "avg_translation_keep_first_rotation":
        raise ValueError(f"Unsupported fuse method: {method}")

    # Select the rotation matrix from the tag with the highest weight
    max_weight_idx = int(np.argmax(weights))
    R = world_T_cam_list[max_weight_idx][:3, :3]

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
    - Collect weights and fuse multiple tags
    """
    world_poses: List[np.ndarray] = []
    used_ids: List[int] = []
    weights: List[float] = []

    for obs in observations:
        if obs.tag_id not in tag_map:
            continue
        try:
            wTc = estimate_world_T_cam_from_single_tag(tag_map[obs.tag_id], obs.cam_T_tag)
            world_poses.append(wTc)
            used_ids.append(obs.tag_id)
            weights.append(obs.weight)
        except Exception:
            continue

    if not world_poses:
        return None

    # Pass the collected weights to the fusion function
    fused = fuse_world_T_cam(world_poses, weights=weights, method=fuse_method)
    return PoseEstimate(world_T_cam=fused, used_tag_ids=used_ids)

