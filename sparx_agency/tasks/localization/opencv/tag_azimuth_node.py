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

from sparx_agency.core.localization.tag_azimuth_estimator import (
    TagAzimuthEstimator,
    TagObservation,
)
from datetime import datetime
from sparx_agency.tasks.localization.common.apriltag_cv_common import (
    CameraCalib,
    load_camera_calib_yaml,
    tag_object_points,
    make_detector,
    solvepnp_ippe_square,
)


def load_tag_config(path: str) -> Dict[int, float]:
    """Load mapping: tag_id -> wall azimuth (deg) from YAML."""

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


class TagAzimuthOpenCVTask:
    """TASK adapter (no ROS): image -> azimuth (deg).

    Requirements:
      1) tag_config_deg (id -> wall azimuth deg)
      2) camera calibration (K, D)
      3) AprilTag detector (pupil_apriltags)

    For each input frame:
      - detect tags
      - solvePnP(IPPE) to get tag translation in camera frame (tvec)
      - convert to TagObservation and run TagAzimuthEstimator
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
        self.calib: CameraCalib = load_camera_calib_yaml(camera_calib_path)

        # 3) detector
        self.detector = make_detector(tag_family=tag_family, nthreads=int(nthreads))

        # tag model points
        self.obj_pts = tag_object_points(float(tag_size_m))

        # Azimuth estimator
        self.estimator = TagAzimuthEstimator(
            tag_config_deg=self.tag_config_deg,
            max_history=max_history,
            max_time_diff_sec=max_time_diff_sec,
        )

    def _solve_tag_tvec(self, corners_2d: np.ndarray) -> Optional[np.ndarray]:
        """Return tvec (3,) where tvec is TAG origin position in CAMERA frame."""

        tag_T_cam = solvepnp_ippe_square(
            corners_2d=corners_2d,
            obj_pts_3d=self.obj_pts,
            K=self.calib.K,
            D=self.calib.D,
        )
        if tag_T_cam is None:
            return None

        tvec = tag_T_cam[:3, 3].copy()   # (3,)

        # Keep behavior you had: flip if OpenCV returns negative Z
        if float(tvec[2]) < 0:
            tvec = -tvec

        return tvec

    def compute_azimuth_from_bgr(self, frame_bgr: np.ndarray, stamp_sec: float) -> float:
        """Main API: image -> azimuth (0..360 deg)."""

        if frame_bgr is None:
            raise ValueError("frame_bgr is None")

        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        dets = self.detector.detect(gray)

        observations: list[TagObservation] = []
        print(f"Detected {len(dets)} tags in image.")
        for d in dets:
            print(f"Detected tag id={d.tag_id}")
            tag_id = int(d.tag_id)
            if tag_id not in self.tag_config_deg:
                continue

            corners = np.array(d.corners, dtype=np.float64).reshape(4, 2)
            tvec = self._solve_tag_tvec(corners)
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
        yaw_deg = (-yaw_deg) % 360.0
        return float(yaw_deg)


def main():
    """CLI example:

    python tag_azimuth_node.py \
        --tag_config_path /path/tags.yaml \
        --camera_calib_path /path/cam.yaml \
        --tag_size_m 0.08 \
        --image /path/frame.png \
        --tag_family tag36h11

    Output:
      Prints a single float azimuth in degrees (0..360) to stdout.
    """

    import argparse
    import sys

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

    print(f"{yaw:.6f}")


if __name__ == "__main__":
    main()
