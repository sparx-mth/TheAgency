#!/usr/bin/env python3
"""
depth_to_pointcloud_node.py

Backprojects metric depth into a PointCloud2 in the camera optical frame.
Three input modes, selected by the `mode` parameter:

  ros_image  (default)
      Subscribes to sensor_msgs/Image (32FC1) + CameraInfo.
      Use for bag replay where /xtend/depth_m is recorded.

  file_path
      Subscribes to std_msgs/String: "{/abs/path.npy} {stamp_sec} {stamp_nanosec}"
      published by depth_processor_node.  Intrinsics from config_yaml.

  dir_watch
      Polls `depth_dir` for new .npy depth files on a timer.
      No topic subscription needed — works for offline replay or live pipelines
      where depth_processor_node is saving files to disk.
      Intrinsics from config_yaml.  Files are processed in sorted (temporal) order.

Parameters common to all modes:
  pointcloud_topic   output topic              default: /xtend/pointcloud
  clip_min_m         near clip (m)             default: 0.05
  clip_max_m         far clip (m)              default: 10.0

Parameters for ros_image:
  depth_topic        sensor_msgs/Image topic   default: /xtend/depth_m
  camera_info_topic  CameraInfo topic          default: /xtend/camera_info

Parameters for file_path:
  depth_path_topic   String path topic         default: /xtend/depth_frame_path
  config_yaml        camera calibration yaml

Parameters for dir_watch:
  depth_dir          directory of .npy files   default: /tmp/xtend_depth
  config_yaml        camera calibration yaml
  poll_hz            scan rate (Hz)            default: 2.0
"""
import numpy as np
from pathlib import Path

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, HistoryPolicy, ReliabilityPolicy
from sensor_msgs.msg import Image, CameraInfo, PointCloud2, PointField
from std_msgs.msg import String
from builtin_interfaces.msg import Time
from cv_bridge import CvBridge

from sparx_agency.robots.common.helpers import (
    load_camera_info_from_yaml,
    crop_resize_camera_info,
)

_BE_QOS = QoSProfile(
    history=HistoryPolicy.KEEP_LAST,
    depth=5,
    reliability=ReliabilityPolicy.BEST_EFFORT,
)


class DepthToPointcloudNode(Node):
    def __init__(self):
        super().__init__("depth_to_pointcloud_node")

        self.declare_parameter("mode",               "ros_image")
        self.declare_parameter("pointcloud_topic",   "/xtend/pointcloud")
        self.declare_parameter("clip_min_m",         0.05)
        self.declare_parameter("clip_max_m",         10.0)
        # ros_image params
        self.declare_parameter("depth_topic",        "/xtend/depth_m")
        self.declare_parameter("camera_info_topic",  "/xtend/camera_info")
        # file_path params
        self.declare_parameter("depth_path_topic",   "/xtend/depth_frame_path")
        # dir_watch + file_path shared
        self.declare_parameter("config_yaml", str(
            Path.home() / "GIT/TheAgency/sparx_agency/robots/XTEND/config"
                          "/camera_xtend_ros_calib_720_420.yaml"
        ))
        # dir_watch params
        self.declare_parameter("depth_dir",  "/tmp/xtend_depth")
        self.declare_parameter("poll_hz",    2.0)

        self._mode   = str(self.get_parameter("mode").value)
        self._z_min  = float(self.get_parameter("clip_min_m").value)
        self._z_max  = float(self.get_parameter("clip_max_m").value)
        cloud_topic  = str(self.get_parameter("pointcloud_topic").value)

        self._bridge     = CvBridge()
        self._cam_info   = None   # cache for ros_image mode
        self._fx = self._fy = self._cx = self._cy = None
        self._cam_w = self._cam_h = None

        self._pub = self.create_publisher(PointCloud2, cloud_topic, _BE_QOS)

        if self._mode == "ros_image":
            self._setup_ros_image(cloud_topic)

        elif self._mode == "file_path":
            self._setup_file_path()

        elif self._mode == "dir_watch":
            self._setup_dir_watch()

        else:
            raise ValueError(f"Unknown mode '{self._mode}'. Choose: ros_image | file_path | dir_watch")

    # ── ros_image ─────────────────────────────────────────────────────────────

    def _setup_ros_image(self, cloud_topic):
        depth_topic = str(self.get_parameter("depth_topic").value)
        info_topic  = str(self.get_parameter("camera_info_topic").value)
        self.create_subscription(CameraInfo, info_topic,  self._info_cb,  _BE_QOS)
        self.create_subscription(Image,      depth_topic, self._image_cb, _BE_QOS)
        self.get_logger().info(f"[ros_image] {depth_topic} + {info_topic} → {cloud_topic}")

    def _info_cb(self, msg: CameraInfo) -> None:
        self._cam_info = msg

    def _image_cb(self, msg: Image) -> None:
        if self._cam_info is None:
            return
        depth = self._bridge.imgmsg_to_cv2(msg, desired_encoding="32FC1")
        h, w  = depth.shape
        sx = w / self._cam_info.width  if self._cam_info.width  > 0 else 1.0
        sy = h / self._cam_info.height if self._cam_info.height > 0 else 1.0
        fx = float(self._cam_info.k[0]) * sx
        fy = float(self._cam_info.k[4]) * sy
        cx = float(self._cam_info.k[2]) * sx
        cy = float(self._cam_info.k[5]) * sy
        self._backproject_and_publish(depth, fx, fy, cx, cy, msg.header.stamp)

    # ── file_path ─────────────────────────────────────────────────────────────

    def _setup_file_path(self):
        self._load_intrinsics_from_yaml()
        topic = str(self.get_parameter("depth_path_topic").value)
        self.create_subscription(String, topic, self._path_cb, 10)
        self.get_logger().info(f"[file_path] {topic} → /xtend/pointcloud")

    def _path_cb(self, msg: String) -> None:
        parts = msg.data.strip().split()
        if len(parts) < 3:
            self.get_logger().warn(f"Bad depth path message: '{msg.data}'")
            return
        path, sec, nsec = parts[0], int(parts[1]), int(parts[2])
        depth = self._load_npy(path)
        if depth is None:
            return
        self._publish_with_intrinsics(depth, Time(sec=sec, nanosec=nsec))

    # ── dir_watch ─────────────────────────────────────────────────────────────

    def _setup_dir_watch(self):
        self._load_intrinsics_from_yaml()
        self._depth_dir   = Path(str(self.get_parameter("depth_dir").value)).expanduser()
        self._seen_files: set = set()
        poll_hz = float(self.get_parameter("poll_hz").value)
        self.create_timer(1.0 / poll_hz, self._dir_poll_cb)
        self.get_logger().info(f"[dir_watch] watching {self._depth_dir} at {poll_hz:.1f} Hz")

    def _dir_poll_cb(self) -> None:
        if not self._depth_dir.exists():
            return
        new_files = sorted(
            f for f in self._depth_dir.glob("*.npy")
            if f.name not in self._seen_files
        )
        for fpath in new_files:
            self._seen_files.add(fpath.name)
            depth = self._load_npy(str(fpath))
            if depth is None:
                continue
            stamp = self.get_clock().now().to_msg()
            self._publish_with_intrinsics(depth, stamp)

    # ── shared helpers ────────────────────────────────────────────────────────

    def _load_intrinsics_from_yaml(self) -> None:
        yaml_path = str(self.get_parameter("config_yaml").value)
        info = load_camera_info_from_yaml(yaml_path, frame_id="xtend_camera")
        info = crop_resize_camera_info(
            base=info, crop_left=90, crop_top=0,
            crop_width=540, crop_height=420,
            new_width=504,  new_height=392,
        )
        self._fx, self._fy = float(info.k[0]), float(info.k[4])
        self._cx, self._cy = float(info.k[2]), float(info.k[5])
        self._cam_w, self._cam_h = info.width, info.height

    def _load_npy(self, path: str) -> "np.ndarray | None":
        try:
            return np.load(path).astype(np.float32)
        except Exception as e:
            self.get_logger().error(f"Could not load {path}: {e}", throttle_duration_sec=2.0)
            return None

    def _publish_with_intrinsics(self, depth: np.ndarray, stamp) -> None:
        h, w = depth.shape
        sx = w / self._cam_w if self._cam_w > 0 else 1.0
        sy = h / self._cam_h if self._cam_h > 0 else 1.0
        self._backproject_and_publish(
            depth,
            self._fx * sx, self._fy * sy,
            self._cx * sx, self._cy * sy,
            stamp,
        )

    def _backproject_and_publish(
        self, depth: np.ndarray,
        fx: float, fy: float, cx: float, cy: float,
        stamp,
    ) -> None:
        h, w = depth.shape
        u, v = np.meshgrid(np.arange(w, dtype=np.float32),
                           np.arange(h, dtype=np.float32))
        z    = depth.flatten().astype(np.float32)
        mask = np.isfinite(z) & (z > self._z_min) & (z < self._z_max)
        pts  = np.stack([
            (u.flatten()[mask] - cx) * z[mask] / fx,
            (v.flatten()[mask] - cy) * z[mask] / fy,
            z[mask],
        ], axis=1).astype(np.float32)

        out                  = PointCloud2()
        out.header.stamp     = stamp
        out.header.frame_id  = "xtend_camera"
        out.height           = 1
        out.width            = len(pts)
        out.fields           = [
            PointField(name="x", offset=0,  datatype=PointField.FLOAT32, count=1),
            PointField(name="y", offset=4,  datatype=PointField.FLOAT32, count=1),
            PointField(name="z", offset=8,  datatype=PointField.FLOAT32, count=1),
        ]
        out.is_bigendian     = False
        out.point_step       = 12
        out.row_step         = 12 * len(pts)
        out.is_dense         = True
        out.data             = pts.tobytes()
        self._pub.publish(out)


def main():
    rclpy.init()
    rclpy.spin(DepthToPointcloudNode())
    rclpy.shutdown()


if __name__ == "__main__":
    main()
