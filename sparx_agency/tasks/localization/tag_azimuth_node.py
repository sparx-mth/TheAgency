from __future__ import annotations

import time
import math
import yaml
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, Optional, List, Tuple, Union
import json
import cv2
import numpy as np
from pupil_apriltags import Detector

from apriltag_localization.core.localization.tag_azimuth_estimator import (
    TagAzimuthEstimator,
    TagObservation,
)
from datetime import datetime

@dataclass(frozen=True)
class CameraCalib:
    K: np.ndarray          # 3x3
    D: np.ndarray          # (n,) usually 5 or 8


def load_tag_config(path: str) -> Dict[int, float]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"tag_config_path does not exist: {path}")

    with p.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    tags = data.get("tags", data)
    if not isinstance(tags, dict) or not tags:
        raise ValueError(f"No tags found in config: {path}")
    
    out: Dict[int, float] = {}
    for k, v in tags.items():
        out[int(k)] = float(v)

    if not out:
        raise ValueError(f"No tags found in config: {path}")
    return out


def load_camera_calib_yaml(path: str) -> CameraCalib:
    """
    Supports common formats like:
    camera_matrix: {data: [fx, 0, cx, 0, fy, cy, 0, 0, 1]}
    distortion_coefficients: {data: [k1,k2,p1,p2,k3]}
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"camera_calib_path does not exist: {path}")

    with p.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    # Try ROS-like calibration YAML keys
    cm = data.get("camera_matrix", {})
    dc = data.get("distortion_coefficients", {})

    K_data = cm.get("data", None) or data.get("K", None)
    D_data = dc.get("data", None) or data.get("D", None)

    if K_data is None:
        raise ValueError("Missing camera matrix in calib file (camera_matrix.data or K).")
    if D_data is None:
        # Allowed: no distortion
        D_data = [0, 0, 0, 0, 0]

    K = np.array(K_data, dtype=np.float64).reshape(3, 3)
    D = np.array(D_data, dtype=np.float64).reshape(-1)
    
    # Basic validation
    if not np.isfinite(K).all() or K.shape != (3, 3) or K[0, 0] <= 0 or K[1, 1] <= 0:
        raise ValueError("Invalid camera intrinsics K (fx/fy must be > 0).")

    return CameraCalib(K=K, D=D)


def tag_object_points(tag_size_m: float) -> np.ndarray:
    """
    3D model points for the 4 tag corners in tag frame (Z=0 plane).
    Order MUST match the detector corners order (pupil_apriltags returns corners in
    consistent order around the tag).
    """
    s = tag_size_m / 2.0
    return np.array(
        [
            [-s, -s, 0],
            [ s, -s, 0],
            [ s,  s, 0],
            [-s,  s, 0],
        ],
        dtype=np.float64,
    )


class TagAzimuthOpenCVTask:
    """
    TASK adapter (no ROS):
    - MUST have:
        1) tag_config_deg (id -> wall azimuth deg)
        2) camera calibration (K, D)
        3) AprilTag detector (pupil_apriltags.Detector)
    - For EACH input image/frame: returns a single azimuth (float degrees 0..360)
    """

    def __init__(
        self,
        tag_config_path: str,
        camera_calib_path: str,
        tag_size_m: float,
        tag_family: str = "tag36h11",
        nthreads: int = 2,
        max_history: int = 20,
        max_time_diff_sec: float = 1.0,
    ):
        # 1) tags config
        self.tag_config_deg = load_tag_config(tag_config_path)
        if not self.tag_config_deg:
            raise ValueError("Tag config is empty. Cannot continue.")
    
        # 2) camera calib
        self.calib = load_camera_calib_yaml(camera_calib_path)
        if self.calib is None or self.calib.K is None:
            raise ValueError("Camera calibration is missing. Cannot continue.")

        # 3) detector
        if not tag_family:
            raise ValueError("tag_family is missing. Cannot create detector.")
        self.detector = Detector(families=tag_family, nthreads=int(nthreads))
        if self.detector is None:
            raise RuntimeError("Failed to create AprilTag detector.")

        self.obj_pts = tag_object_points(float(tag_size_m))

        # Azimuth estimator
        self.estimator = TagAzimuthEstimator(
            tag_config_deg=self.tag_config_deg,
            max_history=max_history,
            max_time_diff_sec=max_time_diff_sec,
        )

    
    def _solve_tag_pose(self, corners_2d: np.ndarray) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """
        corners_2d: (4,2) float64
        returns (rvec, tvec) where tvec is tag position in camera frame.
        """
        ok, rvec, tvec = cv2.solvePnP(
            self.obj_pts,
            corners_2d,
            self.calib.K,
            self.calib.D,
            flags=cv2.SOLVEPNP_IPPE_SQUARE,  
        )
        if not ok:
            return None
        
        if float(tvec[2]) < 0:
            tvec = -tvec
            rvec = -rvec

        return rvec.reshape(3), tvec.reshape(3)


    def compute_azimuth_from_bgr(self, frame_bgr: np.ndarray, stamp_sec: float) -> float:
        """
        Main API: image in -> azimuth out.
        Raises RuntimeError if cannot compute.
        """
        if frame_bgr is None:
            raise ValueError("frame_bgr is None")

        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        dets = self.detector.detect(gray)

        observations: list[TagObservation] = []

        for d in dets:
            tag_id = int(d.tag_id)
            if tag_id not in self.tag_config_deg:
                continue

            corners = np.array(d.corners, dtype=np.float64).reshape(4, 2)
            tvec = self._solve_tag_pose(corners)
            if tvec is None:
                continue

            obs = TagAzimuthEstimator.obs_from_tvec(tag_id=tag_id, tvec=tvec)
            observations.append(obs)

        if not observations:
            raise RuntimeError("No usable known tags detected in image.")

        result = self.estimator.update(observations, stamp_sec=float(stamp_sec))
        if result is None:
            raise RuntimeError("Estimator failed to produce azimuth.")

        yaw_deg, _best_tag = result
        return float(yaw_deg)



def main():
    """
    Example usage:

    python -m sparx_agency.tasks.localization.nodes.tag_azimuth_node \
        --tag_config_path /path/tags.yaml \
        --camera_calib_path /path/cam.yaml \
        --tag_size_m 0.08 \
        --image /path/frame.png \
        --tag_family tag36h11

    Output:
      Prints a single float azimuth in degrees (0..360) to stdout.
      Exits with non-zero code on failure.
    """
    import argparse
    import sys
    import time


    ap = argparse.ArgumentParser()
    ap.add_argument("--tag_config_path", required=True)
    ap.add_argument("--camera_calib_path", required=True)
    ap.add_argument("--tag_size_m", type=float, required=True)
    ap.add_argument("--tag_family", default="tag36h11")
    ap.add_argument("--image", required=True, help="Path to an input image (BGR).")
    ap.add_argument("--nthreads", type=int, default=2)
    ap.add_argument("--history_len", type=int, default=20)
    ap.add_argument("--max_time_diff_sec", type=float, default=1.0)
    args = ap.parse_args()

    # Build task (will raise and stop if tags/calib/detector are missing)
    task = TagAzimuthOpenCVTask(
        tag_config_path=args.tag_config_path,
        camera_calib_path=args.camera_calib_path,
        tag_size_m=args.tag_size_m,
        tag_family=args.tag_family,
        nthreads=args.nthreads,
        max_history=args.history_len,
        max_time_diff_sec=args.max_time_diff_sec,
    )

    frame = cv2.imread(args.image, cv2.IMREAD_COLOR)
    if frame is None:
        print(f"ERROR: Failed to read image: {args.image}", file=sys.stderr)
        raise SystemExit(2)

    try:
        yaw = task.compute_azimuth_from_bgr(frame, stamp_sec=time.time())
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        raise SystemExit(1)

    # Print azimuth only (simple output)
    print(f"{yaw:.6f}")


if __name__ == "__main__":
    main()
