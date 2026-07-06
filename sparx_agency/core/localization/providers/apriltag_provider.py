"""AprilTag-based localization provider (pure Python, no ROS2)."""
from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import yaml

from sparx_agency.core.common.filters import ExponentialMovingAverage
from sparx_agency.core.common.types.geometry import Pose3D
from sparx_agency.core.common.types.perception import Observation
from sparx_agency.core.localization.base import BaseLocalizationProvider, LocalizationEstimate
from sparx_agency.core.localization.tag_triangulation import TagWorldPose
from sparx_agency.tasks.localization.common.apriltag_cv_common import (
    load_camera_calib_yaml,
    make_detector,
)
from sparx_agency.tasks.localization.common.apriltag_pnp import (
    TagDetection,
    estimate_camera_pose,
    pose_confidence,
)

# OpenCV camera frame → ROS/world frame convention
_CV_TO_ROS = np.array(
    [
        [0.0, -1.0, 0.0, 0.0],
        [0.0, 0.0, -1.0, 0.0],
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ],
    dtype=float,
)


def _load_tag_world_map(path: str, default_size: float) -> Tuple[Dict[int, TagWorldPose], Dict[int, float]]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"tag_map_path does not exist: {path}")
    with p.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    tags = data.get("tags", data)
    out_poses: Dict[int, TagWorldPose] = {}
    out_sizes: Dict[int, float] = {}
    for k, v in tags.items():
        tid = int(k)
        xyz = tuple(float(x) for x in v["xyz"])
        rpy = tuple(float(x) for x in v["rpy"])
        out_poses[tid] = TagWorldPose(xyz=xyz, rpy=rpy)
        out_sizes[tid] = float(v.get("size", default_size))
    if not out_poses:
        raise ValueError(f"No tags loaded from: {path}")
    return out_poses, out_sizes


class AprilTagLocalizationProvider(BaseLocalizationProvider):
    """
    Localization from AprilTags.

    Wraps :func:`apriltag_pnp.estimate_camera_pose`: a single joint PnP over the
    pooled corners of all visible tags (>= 2), or a disambiguated IPPE-square
    solve for a single tag. Applies EMA smoothing on position and quaternion
    (same as apriltag_triangulation_node).
    """

    source_name = "apriltag"

    def __init__(
        self,
        tag_map_path: str,
        camera_calib_path: str,
        tag_size_m: float = 0.13,
        tag_family: str = "tag36h11",
        min_margin: float = 10.0,
        alpha: float = 0.1,
        nthreads: int = 2,
        fuse_method: str = "avg_translation_keep_first_rotation",
    ) -> None:
        self._tag_map, self._tag_sizes = _load_tag_world_map(tag_map_path, tag_size_m)
        self._default_tag_size = float(tag_size_m)
        self._calib = load_camera_calib_yaml(camera_calib_path)
        self._detector = make_detector(tag_family, nthreads=nthreads)
        self._min_margin = float(min_margin)
        self._alpha = float(alpha)
        self._fuse_method = fuse_method

        self._pos_ema = ExponentialMovingAverage(alpha=alpha)
        self._filtered_q: Optional[np.ndarray] = None
        # Previous camera position in world for temporal disambiguation of the
        # single-tag planar (IPPE) pose ambiguity.
        self._prev_cam_pos: Optional[np.ndarray] = None
        self._healthy = True

        import rclpy.logging
        self._log = rclpy.logging.get_logger("apriltag_provider")

    def is_healthy(self) -> bool:
        return self._healthy

    def reset(self) -> None:
        self._pos_ema = ExponentialMovingAverage(alpha=self._alpha)
        self._filtered_q = None
        self._prev_cam_pos = None

    def update(self, obs: Observation) -> Optional[LocalizationEstimate]:
        if obs.rgb is None:
            return None

        frame = obs.rgb.image
        stamp_sec = obs.rgb.stamp_sec

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
        dets = self._detector.detect(gray)

        tag_dets: List[TagDetection] = []

        for d in dets:
            tag_id = int(d.tag_id)
            margin = d.decision_margin
            if margin < self._min_margin:
                self._log.info(f"[apriltag] tag {tag_id}: margin={margin:.1f} SKIP(low < {self._min_margin:.1f})")
                continue
            if tag_id not in self._tag_map:
                self._log.info(f"[apriltag] tag {tag_id}: margin={margin:.1f} SKIP(not in map)")
                continue
            corners = np.array(d.corners, dtype=np.float64).reshape(4, 2)
            size_m = self._tag_sizes.get(tag_id, self._default_tag_size)
            self._log.info(f"[apriltag] tag {tag_id}: margin={margin:.1f} USED")
            tag_dets.append(TagDetection(tag_id=tag_id, corners=corners, size_m=size_m))

        if not tag_dets:
            return None

        # One joint PnP over all tags (>= 2), or a disambiguated IPPE-square
        # solve for a single tag, seeded with the previous position.
        est = estimate_camera_pose(
            tag_dets, self._tag_map, self._calib.K, self._calib.D,
            prev_cam_pos_world=self._prev_cam_pos,
        )
        if est is None:
            return None

        self._prev_cam_pos = est.world_T_cam[:3, 3].copy()

        world_T_ros = est.world_T_cam @ _CV_TO_ROS
        x = float(world_T_ros[0, 3])
        y = float(world_T_ros[1, 3])
        z = float(world_T_ros[2, 3])
        x, y, z = self._pos_ema.update(np.array([x, y, z], dtype=float))

        q_raw = np.array(_quat_from_matrix(world_T_ros), dtype=float)
        if self._filtered_q is None:
            self._filtered_q = q_raw
        else:
            if np.dot(self._filtered_q, q_raw) < 0.0:
                q_raw = -q_raw
            self._filtered_q = self._alpha * q_raw + (1.0 - self._alpha) * self._filtered_q
            self._filtered_q /= np.linalg.norm(self._filtered_q)

        qx, qy, qz, qw = self._filtered_q
        yaw = math.atan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz))

        confidence = pose_confidence(est)
        used_ids = sorted(est.used_tag_ids)
        self._log.info(
            f"[apriltag] pose: conf={confidence:.2f} n_tags={est.n_tags} "
            f"used={used_ids} rms={est.reproj_rms_px:.2f}px amb={est.ambiguity:.2f}")

        return LocalizationEstimate(
            pose=Pose3D(x=float(x), y=float(y), z=float(z), yaw=yaw),
            source=self.source_name,
            confidence=confidence,
            stamp_sec=stamp_sec,
            pos_std_m=max(0.01, 0.1 / max(confidence, 1e-6) * 0.05),
            yaw_std_rad=0.05,
        )


def _quat_from_matrix(M: np.ndarray) -> Tuple[float, float, float, float]:
    """Extract quaternion (x, y, z, w) from a 4×4 rotation matrix."""
    R = M[:3, :3]
    trace = R[0, 0] + R[1, 1] + R[2, 2]
    if trace > 0:
        s = 0.5 / math.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (R[2, 1] - R[1, 2]) * s
        y = (R[0, 2] - R[2, 0]) * s
        z = (R[1, 0] - R[0, 1]) * s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
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
    return float(x), float(y), float(z), float(w)