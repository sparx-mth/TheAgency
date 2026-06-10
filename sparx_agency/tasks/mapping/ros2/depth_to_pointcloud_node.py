#!/usr/bin/env python3
"""
depth_to_pointcloud_node.py

Backprojects metric depth into a PointCloud2 in the camera optical frame.
Supports two input modes selected by the `mode` parameter:

  ros_image  (default) — subscribes to a sensor_msgs/Image (32FC1) + CameraInfo.
                         Use for bag replay where /xtend/depth_m is recorded.

  file_path            — subscribes to a std_msgs/String with format
                           "{/abs/path/to/depth.npy} {stamp_sec} {stamp_nanosec}"
                         published by depth_processor_node (save_frames pipeline).
                         Intrinsics are loaded from `config_yaml`.

Publications (both modes):
  pointcloud_topic   sensor_msgs/PointCloud2  (camera optical frame)
"""
import numpy as np
from pathlib import Path

import rclpy
from rclpy.node import Node
from rclpy.node import Parameter
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

        self.declare_parameter("mode",               "ros_image")   # ros_image | file_path
        self.declare_parameter("depth_topic",        "/xtend/depth_m")
        self.declare_parameter("camera_info_topic",  "/xtend/camera_info")
        self.declare_parameter("depth_path_topic",   "/xtend/depth_frame_path")
        self.declare_parameter("config_yaml",        str(
            Path.home() / "GIT/TheAgency/sparx_agency/robots/XTEND/config"
                          "/camera_xtend_ros_calib_720_420.yaml"
        ))
        self.declare_parameter("pointcloud_topic",   "/xtend/pointcloud")
        self.declare_parameter("clip_min_m",         0.05)
        self.declare_parameter("clip_max_m",         10.0)

        self._mode       = str(self.get_parameter("mode").value)
        cloud_topic      = str(self.get_parameter("pointcloud_topic").value)
        self._z_min      = float(self.get_parameter("clip_min_m").value)
        self._z_max      = float(self.get_parameter("clip_max_m").value)

        self._bridge     = CvBridge()
        self._cam_info   = None   # populated from CameraInfo msg or yaml
        self._fx = self._fy = self._cx = self._cy = None
        self._cam_w = self._cam_h = None

        self._pub = self.create_publisher(PointCloud2, cloud_topic, _BE_QOS)

        if self._mode == "ros_image":
            depth_topic = str(self.get_parameter("depth_topic").value)
            info_topic  = str(self.get_parameter("camera_info_topic").value)
            self.create_subscription(CameraInfo, info_topic,  self._info_cb,  _BE_QOS)
            self.create_subscription(Image,      depth_topic, self._image_cb, _BE_QOS)
            self.get_logger().info(
                f"[ros_image] {depth_topic} + {info_topic} → {cloud_topic}"
            )

        elif self._mode == "file_path":
            config_yaml    = str(self.get_parameter("config_yaml").value)
            depth_path_topic = str(self.get_parameter("depth_path_topic").value)
            self._load_intrinsics_from_yaml(config_yaml)
            self.create_subscription(String, depth_path_topic, self._path_cb, 10)
            self.get_logger().info(
                f"[file_path] {depth_path_topic} + {config_yaml} → {cloud_topic}"
            )

        else:
            raise ValueError(f"Unknown mode '{self._mode}'. Use 'ros_image' or 'file_path'.")

    # ── ros_image mode ────────────────────────────────────────────────────────

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
        stamp = msg.header.stamp
        self._backproject_and_publish(depth, fx, fy, cx, cy, stamp)

    # ── file_path mode ────────────────────────────────────────────────────────

    def _load_intrinsics_from_yaml(self, yaml_path: str) -> None:
        info = load_camera_info_from_yaml(yaml_path, frame_id="xtend_camera")
        # Apply the same crop_resize that depth_processor_node uses
        info = crop_resize_camera_info(
            base=info, crop_left=90, crop_top=0,
            crop_width=540, crop_height=420,
            new_width=504,  new_height=392,
        )
        self._fx = float(info.k[0])
        self._fy = float(info.k[4])
        self._cx = float(info.k[2])
        self._cy = float(info.k[5])
        self._cam_w = info.width
        self._cam_h = info.height

    def _path_cb(self, msg: String) -> None:
        parts = msg.data.strip().split()
        if len(parts) < 3:
            self.get_logger().warn(f"Bad depth path message: '{msg.data}'")
            return
        path, sec, nsec = parts[0], int(parts[1]), int(parts[2])

        try:
            depth = np.load(path).astype(np.float32)
        except Exception as e:
            self.get_logger().error(f"Could not load {path}: {e}", throttle_duration_sec=2.0)
            return

        h, w = depth.shape
        sx = w / self._cam_w if self._cam_w > 0 else 1.0
        sy = h / self._cam_h if self._cam_h > 0 else 1.0
        stamp = Time(sec=sec, nanosec=nsec)
        self._backproject_and_publish(
            depth,
            self._fx * sx, self._fy * sy,
            self._cx * sx, self._cy * sy,
            stamp,
        )

    # ── shared backprojection ─────────────────────────────────────────────────

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

        pts = np.stack([
            (u.flatten()[mask] - cx) * z[mask] / fx,
            (v.flatten()[mask] - cy) * z[mask] / fy,
            z[mask],
        ], axis=1).astype(np.float32)

        out              = PointCloud2()
        out.header.stamp = stamp
        out.header.frame_id = "xtend_camera"
        out.height       = 1
        out.width        = len(pts)
        out.fields       = [
            PointField(name="x", offset=0,  datatype=PointField.FLOAT32, count=1),
            PointField(name="y", offset=4,  datatype=PointField.FLOAT32, count=1),
            PointField(name="z", offset=8,  datatype=PointField.FLOAT32, count=1),
        ]
        out.is_bigendian = False
        out.point_step   = 12
        out.row_step     = 12 * len(pts)
        out.is_dense     = True
        out.data         = pts.tobytes()
        self._pub.publish(out)


def main():
    rclpy.init()
    rclpy.spin(DepthToPointcloudNode())
    rclpy.shutdown()


if __name__ == "__main__":
    main()
