from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import yaml
import cv2
from pupil_apriltags import Detector

@dataclass(frozen=True)
class CameraCalib:
    K: np.ndarray  # (3,3)
    D: np.ndarray  # (n,)

def load_camera_calib_yaml(path: str) -> CameraCalib:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"camera_calib_path does not exist: {path}")
    with p.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    cm = data.get("camera_matrix", {})
    dc = data.get("distortion_coefficients", {})

    K_data = cm.get("data", None) or data.get("K", None)
    D_data = dc.get("data", None) or data.get("D", None) or [0,0,0,0,0]

    if K_data is None:
        raise ValueError("Missing camera matrix (camera_matrix.data or K).")

    K = np.array(K_data, dtype=np.float64).reshape(3, 3)
    D = np.array(D_data, dtype=np.float64).reshape(-1)
    if not np.isfinite(K).all() or K.shape != (3, 3) or K[0, 0] <= 0 or K[1, 1] <= 0:
        raise ValueError("Invalid camera intrinsics K (fx/fy must be > 0).")

    return CameraCalib(K=K, D=D)

def tag_object_points(tag_size_m: float) -> np.ndarray:
    s = tag_size_m / 2.0
    return np.array([[-s,-s,0],[s,-s,0],[s,s,0],[-s,s,0]], dtype=np.float64)

def make_T(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    T = np.eye(4, dtype=float)
    T[:3, :3] = R.astype(float)
    T[:3, 3] = t.reshape(3).astype(float)
    return T

def invert_T(T: np.ndarray) -> np.ndarray:
    R = T[:3, :3]
    t = T[:3, 3]
    Ti = np.eye(4, dtype=float)
    Ti[:3, :3] = R.T
    Ti[:3, 3] = -(R.T @ t)
    return Ti

def make_detector(tag_family: str, nthreads: int = 2) -> Detector:
    if not tag_family:
        raise ValueError("tag_family is empty")
    return Detector(families=tag_family, nthreads=int(nthreads))

def solvepnp_ippe_square(
    corners_2d: np.ndarray,
    obj_pts_3d: np.ndarray,
    K: np.ndarray,
    D: np.ndarray,
) -> Optional[np.ndarray]:
    """
    Returns tag_T_cam (TAG -> CAM) as 4x4.
    corners_2d: (4,2) float64
    obj_pts_3d: (4,3) float64 (tag corners in tag frame)
    """
    ok, rvec, tvec = cv2.solvePnP(
        obj_pts_3d,
        corners_2d,
        K,
        D,
        flags=cv2.SOLVEPNP_IPPE_SQUARE,
    )
    if not ok:
        return None

    R, _ = cv2.Rodrigues(rvec.reshape(3, 1))
    tag_T_cam = make_T(R, tvec.reshape(3, 1))  # maps TAG -> CAM
    return tag_T_cam
