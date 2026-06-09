"""Data types for AprilTag triangulation / camera-pose estimation (ROS-free)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np


@dataclass(frozen=True)
class TagWorldPose:
    """Known tag pose in the world frame.

    Attributes:
        xyz: (x, y, z) position in meters.
        rpy: (roll, pitch, yaw) orientation in radians.
    """

    xyz: Tuple[float, float, float]
    rpy: Tuple[float, float, float]


@dataclass(frozen=True)
class TagTransformObservation:
    """A full 6-DOF observation from the camera to a tag.

    Attributes:
        tag_id: Detected tag identifier.
        cam_T_tag: 4x4 homogeneous transform from camera frame to tag frame.
        weight: Confidence weight used when fusing multiple tags (e.g. the tag's
            pixel area). Defaults to 1.0 for callers that do not weight detections.
    """

    tag_id: int
    cam_T_tag: np.ndarray  # shape (4, 4)
    weight: float = 1.0


@dataclass(frozen=True)
class PoseEstimate:
    """Estimated camera pose in the world frame.

    Attributes:
        world_T_cam: 4x4 homogeneous transform from world frame to camera frame.
        used_tag_ids: Tag ids that contributed to the estimate.
    """

    world_T_cam: np.ndarray
    used_tag_ids: List[int]
