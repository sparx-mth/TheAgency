"""Fuse AprilTag absolute fixes with RGB-D odometry for offline pose estimation."""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import yaml

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

try:
    from sparx_agency.core.localization.rgbd_odometry import RgbdOdometry
    _OPEN3D_OK = True
except ImportError:
    _OPEN3D_OK = False


def _load_tag_map(path: str) -> Tuple[Dict[int, TagWorldPose], Dict[int, float]]:
    with open(path) as f:
        data = yaml.safe_load(f) or {}
    tags = data.get("tags", data)
    poses: Dict[int, TagWorldPose] = {}
    sizes: Dict[int, float] = {}
    for k, v in tags.items():
        tid = int(k)
        poses[tid] = TagWorldPose(
            xyz=tuple(float(x) for x in v["xyz"]),
            rpy=tuple(float(x) for x in v["rpy"]),
        )
        sizes[tid] = float(v.get("size", 0.2))
    return poses, sizes


class PoseFuser:
    """
    Per-frame pose estimator combining AprilTag fixes and RGB-D odometry.

    Strategy:
      1. Every frame: attempt AprilTag detection on full-res RGB.
      2. On fix: record world_T_cam (4×4) and anchor odometry at that pose.
      3. Between fixes: compute odometry delta from anchor, apply to last fix.
      4. If open3d unavailable: AprilTag-only (None returned on frames without tag).

    Output: world_T_cam (4×4), camera-OpenCV → map/world frame (Z=up).
    World frame is defined by new_map.yaml (or whichever tag_map_path is given).
    """

    def __init__(
        self,
        tag_map_path: str,
        rgb_calib_path: str,
        depth_calib_path: str,
        depth_h: int = 392,
        depth_w: int = 504,
        min_margin: float = 15.0,
        tag_family: str = "tag36h11",
        nthreads: int = 2,
    ) -> None:
        self._tag_poses, self._tag_sizes = _load_tag_map(tag_map_path)
        self._calib = load_camera_calib_yaml(rgb_calib_path)
        self._detector = make_detector(tag_family, nthreads=nthreads)
        self._min_margin = float(min_margin)

        self._odom: Optional[RgbdOdometry] = None
        if _OPEN3D_OK:
            self._odom = RgbdOdometry(depth_calib_path, depth_h, depth_w)
        else:
            import warnings
            warnings.warn("open3d not available — odometry disabled, AprilTag-only mode.")

        self._tag_world_T_cam: Optional[np.ndarray] = None
        self._odom_T_cam_at_fix: Optional[np.ndarray] = None
        self._n_fixes = 0
        self._last_tag_ids: List[int] = []
        self._last_tag_total_area: float = 0.0

    @property
    def n_tag_fixes(self) -> int:
        return self._n_fixes

    @property
    def last_tag_ids(self) -> List[int]:
        return self._last_tag_ids

    @property
    def last_tag_total_area(self) -> float:
        """Sum of pixel areas of all tags detected in the last frame."""
        return self._last_tag_total_area

    @property
    def tag_world_xyz(self) -> Dict[int, np.ndarray]:
        """World XYZ (3,) for each known tag — tags are on walls."""
        return {tid: np.array(p.xyz, dtype=np.float64)
                for tid, p in self._tag_poses.items()}

    @property
    def odometry_available(self) -> bool:
        return self._odom is not None

    def update(self, bgr: np.ndarray, depth_m: np.ndarray) -> Optional[np.ndarray]:
        """
        Process one frame. Returns world_T_cam (4×4) or None if no pose available yet.

        bgr: 720×420 raw RGB (for AprilTag).
        depth_m: 504×392 metric float32 (for odometry).
        """
        if self._odom is not None:
            self._odom.update(bgr, depth_m)
            odom_T_cam = self._odom.get_world_T_cam()
        else:
            odom_T_cam = None

        tag_fix, detected_ids, total_area = self._detect_tag_fix(bgr)
        self._last_tag_ids = detected_ids
        self._last_tag_total_area = total_area
        if tag_fix is not None:
            self._tag_world_T_cam = tag_fix
            self._odom_T_cam_at_fix = odom_T_cam.copy() if odom_T_cam is not None else None
            self._n_fixes += 1

        if self._tag_world_T_cam is None:
            return odom_T_cam   # None if no odometry, or raw odom pose if no tag ever seen

        if odom_T_cam is None or self._odom_T_cam_at_fix is None:
            return self._tag_world_T_cam  # odometry unavailable, use last tag fix as-is

        delta = np.linalg.inv(self._odom_T_cam_at_fix) @ odom_T_cam
        return self._tag_world_T_cam @ delta

    def _detect_tag_fix(self, bgr: np.ndarray):
        """Returns (world_T_cam or None, list_of_detected_tag_ids)."""
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        dets = self._detector.detect(gray)
        observations: List[TagObservation] = []
        for d in dets:
            tid = int(d.tag_id)
            if d.decision_margin < self._min_margin or tid not in self._tag_poses:
                continue
            corners = np.array(d.corners, dtype=np.float64).reshape(4, 2)
            obj_pts = tag_object_points(self._tag_sizes[tid])
            cam_T_tag = solvepnp_ippe_square(corners, obj_pts, self._calib.K, self._calib.D)
            if cam_T_tag is None:
                continue
            area = float(cv2.contourArea(corners.astype(np.float32)))
            observations.append(TagObservation(tag_id=tid, cam_T_tag=cam_T_tag, weight=area))
        if not observations:
            return None, [], 0.0
        detected_ids = [obs.tag_id for obs in observations]
        total_area = sum(obs.weight for obs in observations)
        est = estimate_camera_pose_from_tags(
            observations, self._tag_poses, "avg_translation_keep_first_rotation"
        )
        return (est.world_T_cam if est is not None else None), detected_ids, total_area