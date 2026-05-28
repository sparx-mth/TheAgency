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
from operator import inv
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
import yaml
from pupil_apriltags import Detector
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import Image
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Int32MultiArray
from cv_bridge import CvBridge


from sparx_agency.core.localization.tag_triangulation import (
    TagWorldPose,
    TagObservation,
    estimate_camera_pose_from_tags,
    matrix_to_pose,
    print_transform_debug,
    world_T_tag_from_pose,
)

from sparx_agency.tasks.localization.common.apriltag_cv_common import (
    load_camera_calib_yaml,
    tag_object_points,
    make_detector,
    invert_T,
    solvepnp_ippe_square,
)


def load_tag_world_map(path: str, default_size: float) -> Tuple[Dict[int, TagWorldPose], Dict[int, float]]:
    """
    YAML format:
      tags:
        14:
          xyz: [x,y,z]
          rpy: [roll,pitch,yaw]   # radians
          size: 0.16              # optional, meters
    """
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
        
        # Pull size from YAML, fallback to default_size if not specified
        out_sizes[tid] = float(v.get("size", default_size))

    if not out_poses:
        raise ValueError(f"No tags loaded from: {path}")
    return out_poses, out_sizes


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
        ros_topic: str = "",
        tag_family: str = "tag36h11",
        visualize: bool = False,
        out_json_path: str = "",
        fuse_method: str = "avg_translation_keep_first_rotation",
        nthreads: int = 2,
    ):
        self.default_tag_size = float(tag_size_m)
        self.tag_map, self.tag_sizes = load_tag_world_map(tag_map_path, self.default_tag_size)
        for tag_id, pose in self.tag_map.items():
            print(f"\n[DEBUG] Loaded tag {tag_id} with size {self.tag_sizes[tag_id]}m")
            print(f"xyz: {pose.xyz}")
            print(f"rpy input rad: {pose.rpy}")
            print(f"rpy input deg: {[math.degrees(v) for v in pose.rpy]}")

            world_T_tag = world_T_tag_from_pose(pose)
            print_transform_debug(f"world_T_tag for tag {tag_id}", world_T_tag)
            
        self.calib = load_camera_calib_yaml(camera_calib_path)
        #self.obj_pts = tag_object_points(float(tag_size_m))

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

        self.ros_topic = ros_topic
        self.use_ros_image = (str(video_source).lower() == "ros")
        self.latest_ros_image = None
        self.latest_ros_stamp = None

        rclpy.init()
        self.ros_node = rclpy.create_node('opencv_ros_hybrid')
        self.bridge = CvBridge()
        self.pose_pub = self.ros_node.create_publisher(PoseStamped, '/xtend/april_tag_pose', 10)

        self.filtered_x = None
        self.alpha = 0.2
        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=5,
        )
        if self.use_ros_image:
            self.ros_node.create_subscription(Image, self.ros_topic, self.ros_image_cb, qos)
        else:
            if isinstance(video_source, str) and video_source.startswith("dir:"):
                self.image_dir = Path(video_source[4:]).expanduser().resolve()
                if not self.image_dir.exists():
                    raise FileNotFoundError(f"Image dir does not exist: {self.image_dir}")
            else:
                self.cap = cv2.VideoCapture(video_source)
                if not self.cap.isOpened():
                    raise RuntimeError(f"Could not open video source: {video_source}")

    def ros_image_cb(self, msg: Image):
        try:
            self.latest_ros_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            self.latest_ros_stamp = float(msg.header.stamp.sec) + float(msg.header.stamp.nanosec) * 1e-9
        except Exception as e:
            print(f"CV Bridge error: {e}")

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

        try:
            while True:
                # --- get frame ---
                if self.use_ros_image:
                    rclpy.spin_once(self.ros_node, timeout_sec=0.02)
                    if self.latest_ros_image is None:
                        continue
                    frame = self.latest_ros_image.copy()
                    stamp_sec = self.latest_ros_stamp
                    src_name = self.ros_topic
                    src_type = "ros"
                    self.latest_ros_image = None

                elif self.image_dir is not None:
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
                    margin = d.decision_margin
                    
                    if margin < 35:
                        print(f"[DEBUG] Skipping tag {tag_id} - Margin {margin:.2f} too low")
                        continue
                    if tag_id not in self.tag_map:
                        continue

                    corners = np.array(d.corners, dtype=np.float64).reshape(4, 2)

                    specific_tag_size = self.tag_sizes.get(tag_id, self.default_tag_size)
                    current_obj_pts = tag_object_points(specific_tag_size)

                    camera_T_tag = solvepnp_ippe_square(
                        corners_2d=corners,
                        obj_pts_3d=current_obj_pts,
                        K=self.calib.K,
                        D=self.calib.D,
                    )
                    if camera_T_tag is None:
                        continue

                    # Calculate the area of the tag in pixels to use as a confidence weight
                    area = float(cv2.contourArea(corners.astype(np.float32)))

                    # Append observation with the calculated weight
                    observations.append(TagObservation(
                        tag_id=tag_id,
                        cam_T_tag=camera_T_tag,
                        weight=area
                    ))

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

                        exit_requested = False
                        while True:
                            k = cv2.waitKey(50) & 0xFF
                            if k == 32:
                                break
                            elif k in (27, ord("q")):
                                exit_requested = True
                                break
                        if exit_requested:
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

                cv_to_ros = np.array([
                    [0.0, -1.0, 0.0, 0.0],
                    [0.0, 0.0, -1.0, 0.0],
                    [1.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0]
                ], dtype=float)

                world_T_ros = est.world_T_cam @ cv_to_ros

                (x, y, z), (qx, qy, qz, qw) = matrix_to_pose(world_T_ros)

                if self.filtered_x is None:
                    self.filtered_x, self.filtered_y, self.filtered_z = x, y, z
                else:
                    self.filtered_x = self.alpha * x + (1 - self.alpha) * self.filtered_x
                    self.filtered_y = self.alpha * y + (1 - self.alpha) * self.filtered_y
                    self.filtered_z = self.alpha * z + (1 - self.alpha) * self.filtered_z

                x, y, z = self.filtered_x, self.filtered_y, self.filtered_z

                # --- Format and print clean summary ---
                yaw_rad = math.atan2(world_T_ros[1, 0], world_T_ros[0, 0])
                yaw_deg = math.degrees(yaw_rad)

                print(f"\n--- Pose Update [{stamp_sec:.2f}] ---")
                print(f"Position: X={x:.3f}, Y={y:.3f}, Z={z:.3f} | Yaw={yaw_deg:.1f} deg")
                print("Tags Detected:")
                for d in dets:
                    tid = int(d.tag_id)
                    margin = d.decision_margin
                    if tid in est.used_tag_ids:
                        size = self.tag_sizes.get(tid, self.default_tag_size)
                        # Calculate the area again specifically for the printout
                        corners = np.array(d.corners, dtype=np.float64).reshape(4, 2)
                        area = float(cv2.contourArea(corners.astype(np.float32)))
                        print(f"  -> ID: {tid} | Margin: {margin:.1f} | Size: {size}m | Weight: {area:.0f}px")
                print("-----------------------------------")

                out_pose = {
                    "position_xyz_m": [float(self.filtered_x), float(self.filtered_y), float(self.filtered_z)],
                    "quat_xyzw": [float(qx), float(qy), float(qz), float(qw)],
                }

                used_ids = [int(i) for i in sorted(set(est.used_tag_ids))]

                pose_msg = PoseStamped()
                pose_msg.header.frame_id = "world"
                pose_msg.header.stamp.sec = int(stamp_sec)
                pose_msg.header.stamp.nanosec = int((stamp_sec - int(stamp_sec)) * 1e9)

                pose_msg.pose.position.x = float(x)
                pose_msg.pose.position.y = float(y)
                pose_msg.pose.position.z = float(z)
                pose_msg.pose.orientation.x = float(qx)
                pose_msg.pose.orientation.y = float(qy)
                pose_msg.pose.orientation.z = float(qz)
                pose_msg.pose.orientation.w = float(qw)

                self.pose_pub.publish(pose_msg)

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
                        # print(f"[DEBUG] Detected tag {tid} with margin {margin:.2f}")
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

                    yaw_rad = math.atan2(world_T_ros[1, 0], world_T_ros[0, 0])
                    yaw_deg = math.degrees(yaw_rad)
                    line4 = f"yaw: {yaw_deg:.1f} deg"

                    cv2.putText(frame, line1, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                    cv2.putText(frame, line2, (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                    cv2.putText(frame, line3, (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                    cv2.putText(frame, line4, (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

                    cv2.imshow("tag_triangulation", frame)

                    exit_requested = False
                    while True:
                        k = cv2.waitKey(50) & 0xFF
                        if k == 32:
                            break
                        elif k in (27, ord("q")):
                            exit_requested = True
                            break
                    if exit_requested:
                        break
                else:
                    print(f"[{stamp_sec:.3f}] world(x,y,z)=({x:.2f},{y:.2f},{z:.2f}) used={used_ids}")

        except KeyboardInterrupt:
            print("interrupted by user, exiting...")

        finally:
            self.ros_node.destroy_node()
            if rclpy.ok():
                rclpy.shutdown()
            if self.cap is not None:
                self.cap.release()
            if self._json_f is not None:
                self._json_f.close()
                self._json_f = None
            cv2.destroyAllWindows()
            print("Cleaned up resources, exiting.")


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
    ap.add_argument("--image_topic", default="/xtend/rgb", help="ROS image topic to listen to")
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
        ros_topic=args.image_topic,
        tag_family=args.tag_family,
        visualize=(not args.no_vis),
        out_json_path=args.out_json,
        fuse_method=args.fuse_method,
        nthreads=args.nthreads,
    )
    task.run()


if __name__ == "__main__":
    main()