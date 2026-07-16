"""Offline Open3D RGB-D frame-to-frame odometry. No ROS2."""
from __future__ import annotations

from typing import Optional

import cv2
import numpy as np
import yaml
import open3d as o3d
from scipy.spatial.transform import Rotation as Rot


def _pinhole_from_yaml(camera_yaml: str, w: int, h: int) -> o3d.camera.PinholeCameraIntrinsic:
    with open(camera_yaml) as f:
        data = yaml.safe_load(f)
    yaml_w = int(data["image_width"])
    yaml_h = int(data["image_height"])
    # Prefer camera_matrix (raw/distorted-image K) over projection_matrix (P):
    # the RGB/depth frames fed into odometry are never undistorted, so P — valid
    # only post-rectification — is the wrong K and can introduce a real fx/fy
    # anisotropy (see run_room_mapper.py::_load_depth_K for the same fix).
    if "camera_matrix" in data:
        K = np.array(data["camera_matrix"]["data"], dtype=np.float64).reshape(3, 3)
        fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    elif "projection_matrix" in data:
        P = np.array(data["projection_matrix"]["data"], dtype=np.float64).reshape(3, 4)
        fx, fy, cx, cy = P[0, 0], P[1, 1], P[0, 2], P[1, 2]
    else:
        fx, fy = float(data["fx"]), float(data["fy"])
        cx, cy = float(data["cx"]), float(data["cy"])
    sx, sy = w / yaml_w, h / yaml_h
    return o3d.camera.PinholeCameraIntrinsic(w, h, fx * sx, fy * sy, cx * sx, cy * sy)


def _to_rgbd(bgr: np.ndarray, depth_m: np.ndarray, depth_trunc_m: float) -> o3d.geometry.RGBDImage:
    """Build Open3D RGBD, resizing BGR to depth resolution if needed."""
    dh, dw = depth_m.shape[:2]
    if bgr.shape[:2] != (dh, dw):
        bgr = cv2.resize(bgr, (dw, dh), interpolation=cv2.INTER_LINEAR)
    d = depth_m.astype(np.float32)
    d[~np.isfinite(d)] = 0.0
    d[d < 0.0] = 0.0
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return o3d.geometry.RGBDImage.create_from_color_and_depth(
        o3d.geometry.Image(rgb),
        o3d.geometry.Image(d),
        depth_scale=1.0,
        depth_trunc=depth_trunc_m,
        convert_rgb_to_intensity=False,
    )


class RgbdOdometry:
    """
    Frame-to-frame RGB-D odometry using Open3D. No ROS2.

    Accumulates world_T_cam starting from identity at the first frame.
    Camera convention: OpenCV (Z=forward, X=right, Y=down).
    Use anchor_to() after an AprilTag fix to align with the world frame.
    """

    def __init__(
        self,
        camera_yaml: str,
        depth_h: int,
        depth_w: int,
        depth_trunc_m: float = 7.0,
        depth_min_m: float = 0.3,
        depth_max_m: float = 5.0,
        depth_diff_max: float = 0.05,
        min_score: float = 5e4,
        max_rot_deg: float = 5.0,
        max_t_norm: float = 0.6,
    ) -> None:
        self._pinhole = _pinhole_from_yaml(camera_yaml, depth_w, depth_h)
        self._depth_trunc_m = depth_trunc_m
        self._min_score = min_score
        self._max_rot_deg = max_rot_deg
        self._max_t_norm = max_t_norm

        opt = o3d.pipelines.odometry.OdometryOption()
        opt.iteration_number_per_pyramid_level = o3d.utility.IntVector([200, 100, 50, 20])
        opt.depth_diff_max = depth_diff_max
        opt.depth_min = depth_min_m
        opt.depth_max = depth_max_m
        self._opt = opt

        self._prev_rgbd: Optional[o3d.geometry.RGBDImage] = None
        self._last_trans = np.eye(4, dtype=np.float64)
        self._world_T_cam = np.eye(4, dtype=np.float64)

    def reset(self, pose: Optional[np.ndarray] = None) -> None:
        """Reset accumulated state. pose sets the initial world_T_cam if provided."""
        self._prev_rgbd = None
        self._last_trans = np.eye(4, dtype=np.float64)
        self._world_T_cam = np.eye(4, dtype=np.float64) if pose is None else pose.astype(np.float64)

    def update(self, bgr: np.ndarray, depth_m: np.ndarray) -> bool:
        """Process a new frame pair. Returns True when pose was updated."""
        rgbd = _to_rgbd(bgr, depth_m, self._depth_trunc_m)
        if self._prev_rgbd is None:
            self._prev_rgbd = rgbd
            return True
        success, trans, info = o3d.pipelines.odometry.compute_rgbd_odometry(
            self._prev_rgbd,
            rgbd,
            self._pinhole,
            self._last_trans,
            o3d.pipelines.odometry.RGBDOdometryJacobianFromColorTerm(),
            self._opt,
        )
        accepted = self._accept(success, trans, info)
        if accepted:
            self._last_trans = trans
            self._world_T_cam = self._world_T_cam @ np.linalg.inv(trans)
        else:
            self._last_trans = np.eye(4, dtype=np.float64)
        self._prev_rgbd = rgbd
        return accepted

    def get_world_T_cam(self) -> np.ndarray:
        """Return 4×4 accumulated pose (OpenCV camera → odometry world)."""
        return self._world_T_cam.copy()

    def _accept(self, success: bool, trans: np.ndarray, info: np.ndarray) -> bool:
        if not success:
            return False
        score = float(np.mean(np.diag(info)))
        if score < self._min_score:
            return False
        rot_deg = float(np.linalg.norm(Rot.from_matrix(trans[:3, :3]).as_rotvec())) * 180.0 / np.pi
        t_norm = float(np.linalg.norm(trans[:3, 3]))
        if rot_deg < 1e-4 and t_norm < 1e-4:
            return False
        return rot_deg <= self._max_rot_deg and t_norm <= self._max_t_norm
