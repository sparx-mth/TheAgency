#!/usr/bin/env python3
"""
Tag Triangulation (world_T_cam) using AprilTags + OpenCV (no ROS).

Pipeline:
  1) Known tag poses in WORLD (YAML): world_T_tag
  2) AprilTag detection (pupil_apriltags) -> 2D corners
  3) solvePnP (OpenCV IPPE_SQUARE) -> tag_T_cam (TAG -> CAM)
  4) Invert -> cam_T_tag (CAM -> TAG)
  5) Estimate camera pose in world: world_T_cam from multiple tags

Outputs:
  - Prints / overlays (x,y,z) + quaternion (qx,qy,qz,qw)
  - Optional JSONL logging per frame

Usage example:
  python tag_triangulation_opencv_task.py \
      --tag_map_path tags_world.yaml \
      --camera_calib_path camera.yaml \
      --tag_size_m 0.16 \
      --source 0 \
      --out_json /tmp/triangulation_log.jsonl
"""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
import yaml
from pupil_apriltags import Detector

from sparx_agency.core.localization.tag_triangulation import (
    TagWorldPose,
    TagObservation,
    estimate_camera_pose_from_tags,
    transform_to_pose,
)

from sparx_agency.tasks.localization.common.apriltag_cv_common import (
    load_camera_calib_yaml,
    tag_object_points,
    make_detector,
    invert_T,
    solvepnp_ippe_square,
)


def load_tag_world_map(path: str) -> Dict[int, TagWorldPose]:
    """
    YAML format:
      tags:
        14:
          xyz: [x,y,z]
          rpy: [roll,pitch,yaw]   # radians
    or at root:
      14: { xyz: [...], rpy: [...] }
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"tag_map_path does not exist: {path}")

    with p.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    tags = data.get("tags", data)

    out: Dict[int, TagWorldPose] = {}
    for k, v in tags.items():
        tid = int(k)
        xyz = tuple(float(x) for x in v["xyz"])
        rpy = tuple(float(x) for x in v["rpy"])
        out[tid] = TagWorldPose(xyz=xyz, rpy=rpy)

    if not out:
        raise ValueError(f"No tags loaded from: {path}")
    return out


class TagTriangulationOpenCVTask:
    """
    OpenCV adapter (no ROS):
      - Detect AprilTags
      - solvePnP -> tag_T_cam (TAG -> CAM)
      - Convert to cam_T_tag (CAM -> TAG) for core estimator input
      - Compute world_T_cam from known world tag poses
    """

    def __init__(
        self,
        tag_map_path: str,
        camera_calib_path: str,
        tag_size_m: float,
        video_source: Union[int, str] = 0,
        tag_family: str = "tag36h11",
        visualize: bool = True,
        out_json_path: str = "",
        fuse_method: str = "avg_translation_keep_first_rotation",
        nthreads: int = 2,
    ):
        self.tag_map = load_tag_world_map(tag_map_path)
        self.calib = load_camera_calib_yaml(camera_calib_path)
        self.obj_pts = tag_object_points(float(tag_size_m))

        self.detector = make_detector(tag_family, nthreads=nthreads)

        self.visualize = bool(visualize)
        self.fuse_method = str(fuse_method)

        self.out_json_path = out_json_path.strip()
        self._json_f = None
        if self.out_json_path:
            out_p = Path(self.out_json_path).expanduser().resolve()
            out_p.parent.mkdir(parents=True, exist_ok=True)
            self._json_f = out_p.open("a", buffering=1)  # line-buffered

        self.image_dir: Optional[Path] = None
        self.cap: Optional[cv2.VideoCapture] = None

        if isinstance(video_source, str) and video_source.startswith("dir:"):
            self.image_dir = Path(video_source[4:]).expanduser().resolve()
            if not self.image_dir.exists():
                raise FileNotFoundError(f"Image dir does not exist: {self.image_dir}")
        else:
            self.cap = cv2.VideoCapture(video_source)
            if not self.cap.isOpened():
                raise RuntimeError(f"Could not open video source: {video_source}")

    def _log_json(self, record: dict):
        if self._json_f is None:
            return
        self._json_f.write(json.dumps(record, ensure_ascii=False) + "\n")

    def _iter_images_in_dir(self, exts=(".png", ".jpg", ".jpeg", ".bmp")) -> List[Path]:
        assert self.image_dir is not None
        files = [p for p in self.image_dir.iterdir() if p.is_file() and p.suffix.lower() in exts]
        files.sort()
        return files

    def run(self):
        processed: set[str] = set()

        while True:
            # --- get frame ---
            if self.image_dir is not None:
                files = self._iter_images_in_dir()
                next_file = None
                for f in files:
                    if str(f) not in processed:
                        next_file = f
                        break
                if next_file is None:
                    time.sleep(0.2)
                    continue

                frame = cv2.imread(str(next_file), cv2.IMREAD_COLOR)
                processed.add(str(next_file))
                if frame is None:
                    print(f"Failed to read image: {next_file}")
                    continue

                stamp_sec = float(next_file.stat().st_mtime)
                src_name = next_file.name
                src_type = "dir"
            else:
                assert self.cap is not None
                ok, frame = self.cap.read()
                if not ok:
                    break
                stamp_sec = float(time.time())
                src_name = "camera/video"
                src_type = "video"

            record_base = {
                "timestamp_sec": stamp_sec,
                "source_name": src_name,
                "source_type": src_type,
            }

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            dets = self.detector.detect(gray)

            observations: List[TagObservation] = []
            detections_info = []

            for d in dets:
                tag_id = int(d.tag_id)
                if tag_id not in self.tag_map:
                    continue

                corners = np.array(d.corners, dtype=np.float64).reshape(4, 2)

                # tag_T_cam: TAG -> CAM
                tag_T_cam = solvepnp_ippe_square(
                    corners_2d=corners,
                    obj_pts_3d=self.obj_pts,
                    K=self.calib.K,
                    D=self.calib.D,
                )
                if tag_T_cam is None:
                    continue

                # Core expects cam_T_tag (CAM -> TAG)
                cam_T_tag = invert_T(tag_T_cam)

                # IMPORTANT: feed the inverted matrix (cam_T_tag)
                observations.append(TagObservation(tag_id=tag_id, cam_T_tag=cam_T_tag))

                area = float(cv2.contourArea(corners.astype(np.float32)))
                detections_info.append(
                    {
                        "tag_id": tag_id,
                        "area_px2": area,
                        "corners_px": [[float(x), float(y)] for (x, y) in corners],
                    }
                )

            if not observations:
                if self.visualize:
                    cv2.putText(
                        frame,
                        f"{src_name} | No known tags visible",
                        (20, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (255, 255, 255),
                        2,
                    )
                    cv2.imshow("tag_triangulation", frame)
                    k = cv2.waitKey(0) & 0xFF
                    if k in (27, ord("q")):
                        break

                self._log_json(
                    {
                        **record_base,
                        "found": False,
                        "pose": None,
                        "used_tag_ids": [],
                        "detections": detections_info,
                    }
                )
                continue

            est = estimate_camera_pose_from_tags(
                observations=observations,
                tag_map=self.tag_map,
                fuse_method=self.fuse_method,
            )

            if est is None:
                self._log_json(
                    {
                        **record_base,
                        "found": False,
                        "pose": None,
                        "used_tag_ids": [],
                        "detections": detections_info,
                    }
                )
                continue

            (x, y, z), (qx, qy, qz, qw) = transform_to_pose(est.world_T_cam)

            out_pose = {
                "position_xyz_m": [float(x), float(y), float(z)],
                "quat_xyzw": [float(qx), float(qy), float(qz), float(qw)],
            }

            used_ids = [int(i) for i in sorted(set(est.used_tag_ids))]

            self._log_json(
                {
                    **record_base,
                    "found": True,
                    "pose": out_pose,
                    "used_tag_ids": used_ids,
                    "detections": detections_info,
                }
            )

            if self.visualize:
                used_set = set(used_ids)
                for d in dets:
                    tid = int(d.tag_id)
                    if tid not in used_set:
                        continue
                    corners = np.array(d.corners, dtype=np.float64).reshape(4, 2)
                    cv2.polylines(frame, [corners.astype(np.int32)], True, (0, 255, 0), 2)
                    cx = int(np.mean(corners[:, 0]))
                    cy = int(np.mean(corners[:, 1]))
                    cv2.putText(
                        frame,
                        f"id={tid}",
                        (cx, cy),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 255, 0),
                        2,
                    )

                line1 = f"{src_name} | used={sorted(list(used_set))}"
                line2 = f"world pose: x={x:.2f} y={y:.2f} z={z:.2f}"
                line3 = f"quat: [{qx:.3f}, {qy:.3f}, {qz:.3f}, {qw:.3f}]"

                cv2.putText(frame, line1, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.putText(frame, line2, (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.putText(frame, line3, (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

                cv2.imshow("tag_triangulation", frame)
                k = cv2.waitKey(0) & 0xFF
                if k in (27, ord("q")):
                    break
            else:
                print(f"[{stamp_sec:.3f}] world(x,y,z)=({x:.2f},{y:.2f},{z:.2f}) used={used_ids}")

        if self.cap is not None:
            self.cap.release()
        if self._json_f is not None:
            self._json_f.close()
            self._json_f = None
        cv2.destroyAllWindows()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag_map_path", required=True, help="YAML: tag_id -> {xyz:[...], rpy:[...]} (radians)")
    ap.add_argument("--camera_calib_path", required=True, help="YAML camera intrinsics/distortion")
    ap.add_argument("--tag_size_m", type=float, required=True)
    ap.add_argument("--source", default="0", help="camera index (0) or video path")
    ap.add_argument("--image_dir", default="", help="Directory to read images from (instead of camera/video).")
    ap.add_argument("--tag_family", default="tag36h11")
    ap.add_argument("--nthreads", type=int, default=2)
    ap.add_argument("--no_vis", action="store_true")
    ap.add_argument("--out_json", default="", help="Path to output JSONL log (one JSON per frame).")
    ap.add_argument("--fuse_method", default="avg_translation_keep_first_rotation")
    args = ap.parse_args()

    src: Union[int, str]
    if args.image_dir:
        src = f"dir:{args.image_dir}"
    else:
        src = int(args.source) if isinstance(args.source, str) and args.source.isdigit() else args.source

    task = TagTriangulationOpenCVTask(
        tag_map_path=args.tag_map_path,
        camera_calib_path=args.camera_calib_path,
        tag_size_m=args.tag_size_m,
        video_source=src,
        tag_family=args.tag_family,
        visualize=(not args.no_vis),
        out_json_path=args.out_json,
        fuse_method=args.fuse_method,
        nthreads=args.nthreads,
    )
    task.run()


if __name__ == "__main__":
    main()