"""AprilTag-based localization provider (pure Python, no ROS2)."""
from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

_log = logging.getLogger(__name__)

import cv2
import numpy as np
import yaml

from sparx_agency.core.common.filters import ExponentialMovingAverage
from sparx_agency.core.common.types.geometry import Pose3D
from sparx_agency.core.common.types.perception import Observation
from sparx_agency.core.localization.base import BaseLocalizationProvider, LocalizationEstimate
from sparx_agency.core.localization.tag_triangulation import (
    TagObservation,
    TagWorldPose,
    estimate_camera_pose_from_tags,
)
from sparx_agency.tasks.localization.common.apriltag_cv_common import (
    load_camera_calib_yaml,
    make_detector,
    solvepnp_ippe_square,
    tag_object_points,
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


def _confidence_from_area(total_area_px: float) -> float:
    """Map total visible tag area (px²) to a 0–1 confidence score."""
    return float(min(1.0, total_area_px / 10_000.0))


class AprilTagLocalizationProvider(BaseLocalizationProvider):
    """
    Localization from AprilTag triangulation.

    Wraps the core estimate_camera_pose_from_tags() algorithm.
    Applies EMA smoothing on position and quaternion (same as apriltag_triangulation_node).
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
        self._healthy = True

    def is_healthy(self) -> bool:
        return self._healthy

    def reset(self) -> None:
        self._pos_ema = ExponentialMovingAverage(alpha=self._alpha)
        self._filtered_q = None

    def update(self, obs: Observation) -> Optional[LocalizationEstimate]:
        if obs.rgb is None:
            return None

        frame = obs.rgb.image
        stamp_sec = obs.rgb.stamp_sec

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
        dets = self._detector.detect(gray)

        observations: List[TagObservation] = []
        total_area = 0.0

        for d in dets:
            tag_id = int(d.tag_id)
            margin = d.decision_margin
            if margin < self._min_margin:
                _log.info("[apriltag] tag %d: margin=%.1f SKIP(low < %.1f)", tag_id, margin, self._min_margin)
                continue
            if tag_id not in self._tag_map:
                _log.info("[apriltag] tag %d: margin=%.1f SKIP(not in map)", tag_id, margin)
                continue
            corners = np.array(d.corners, dtype=np.float64).reshape(4, 2)
            obj_pts = tag_object_points(self._tag_sizes.get(tag_id, self._default_tag_size))
            cam_T_tag = solvepnp_ippe_square(corners, obj_pts, self._calib.K, self._calib.D)
            if cam_T_tag is None:
                _log.info("[apriltag] tag %d: margin=%.1f SKIP(solvePnP failed)", tag_id, margin)
                continue
            area = float(cv2.contourArea(corners.astype(np.float32)))
            total_area += area
            _log.info("[apriltag] tag %d: margin=%.1f area=%.0fpx USED", tag_id, margin, area)
            observations.append(TagObservation(tag_id=tag_id, cam_T_tag=cam_T_tag, weight=area))

        if not observations:
            return None

        est = estimate_camera_pose_from_tags(observations, self._tag_map, self._fuse_method)
        if est is None:
            return None

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

        n_tags = len(observations)
        confidence = min(1.0, _confidence_from_area(total_area) * (0.7 + 0.15 * min(n_tags, 2)))
        used_ids = sorted(o.tag_id for o in observations)
        _log.info("[apriltag] pose: conf=%.2f n_tags=%d used=%s", confidence, n_tags, used_ids)

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