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

        self._depth_h = depth_h
        self._depth_w = depth_w

        self._tag_world_T_cam: Optional[np.ndarray] = None
        self._odom_T_cam_at_fix: Optional[np.ndarray] = None
        self._n_fixes = 0
        self._last_tag_ids: List[int] = []
        self._last_tag_total_area: float = 0.0
        self._last_observations: List = []

        self._smoothed_depth_scale: Optional[float] = None
        self._scale_ema_alpha: float = 0.1
        _SCALE_MIN, _SCALE_MAX = 0.8, 1.5
        self._scale_clamp = (_SCALE_MIN, _SCALE_MAX)

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
    def last_depth_scale(self) -> Optional[float]:
        """Smoothed DA3 depth scale factor from tag observations. None until first tag seen."""
        return self._smoothed_depth_scale

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
            raw_scale = self._estimate_depth_scale(depth_m, bgr.shape[:2])
            if raw_scale is not None:
                lo, hi = self._scale_clamp
                scale = float(np.clip(raw_scale, lo, hi))
                if self._smoothed_depth_scale is None:
                    self._smoothed_depth_scale = scale
                else:
                    self._smoothed_depth_scale = float(np.clip(
                        self._scale_ema_alpha * scale
                        + (1.0 - self._scale_ema_alpha) * self._smoothed_depth_scale,
                        lo, hi,
                    ))
                if abs(raw_scale - scale) > 0.01:
                    print(f"  [scale] raw={raw_scale:.3f} → clamped to {scale:.3f} "
                          f"(ema={self._smoothed_depth_scale:.3f})")

        if self._tag_world_T_cam is None:
            return odom_T_cam   # None if no odometry, or raw odom pose if no tag ever seen

        if odom_T_cam is None or self._odom_T_cam_at_fix is None:
            return self._tag_world_T_cam  # odometry unavailable, use last tag fix as-is

        delta = np.linalg.inv(self._odom_T_cam_at_fix) @ odom_T_cam
        return self._tag_world_T_cam @ delta

    def _estimate_depth_scale(
        self, depth_m: np.ndarray, rgb_shape: Tuple[int, int]
    ) -> Optional[float]:
        """
        Estimate DA3 metric scale by comparing observed depth at tag centers
        to the expected Z distance from solvePnP (cam_T_tag[2,3]).

        Returns median(expected_z / observed_z) across all visible tags,
        or None if no reliable samples found.
        """
        rgb_h, rgb_w = rgb_shape
        scales = []
        for obs in self._last_observations:
            p_cam = obs.cam_T_tag[:3, 3]
            expected_z = float(p_cam[2])
            if expected_z < 0.2:
                continue

            # Project tag center to depth image (same sensor, different resolution)
            u_rgb = self._calib.K[0, 0] * p_cam[0] / p_cam[2] + self._calib.K[0, 2]
            v_rgb = self._calib.K[1, 1] * p_cam[1] / p_cam[2] + self._calib.K[1, 2]
            u_d = int(round(u_rgb * self._depth_w / rgb_w))
            v_d = int(round(v_rgb * self._depth_h / rgb_h))

            r = 5
            u0 = max(0, u_d - r); u1 = min(self._depth_w,  u_d + r + 1)
            v0 = max(0, v_d - r); v1 = min(self._depth_h, v_d + r + 1)
            if u1 <= u0 or v1 <= v0:
                continue

            patch = depth_m[v0:v1, u0:u1]
            valid = patch[np.isfinite(patch) & (patch > 0.1)]
            if valid.size == 0:
                continue

            observed_z = float(np.median(valid))
            if observed_z < 0.1:
                continue

            scales.append(expected_z / observed_z)

        if not scales:
            return None
        return float(np.median(scales))

    def _detect_tag_fix(self, bgr: np.ndarray):
        """Returns (world_T_cam or None, list_of_detected_tag_ids, total_area)."""
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
        self._last_observations = observations
        if not observations:
            return None, [], 0.0
        detected_ids = [obs.tag_id for obs in observations]
        total_area = sum(obs.weight for obs in observations)
        est = estimate_camera_pose_from_tags(
            observations, self._tag_poses, "avg_translation_keep_first_rotation"
        )
        return (est.world_T_cam if est is not None else None), detected_ids, total_area