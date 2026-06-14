#!/usr/bin/env python3
from pathlib import Path

import cv2
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge

from sparx_agency.robots.common.helpers import load_camera_info_from_yaml

_IMG_EXTS = ("jpg", "jpeg", "png", "PNG", "JPG", "JPEG")

_DEFAULT_YAML = str(
    Path.home() / "GIT/TheAgency/sparx_agency/robots/XTEND/config/camera_xtend_ros_calib_720_420.yaml"
)


class FrameSourceNode(Node):
    """
    Publishes RGB frames and CameraInfo on /rgbd/rgb and /rgbd/camera_info.

    mode=mock  — reads sorted images from rgb_dir at publish_hz, loops when done
    mode=live  — subscribes to source_topic and republishes (topic bridge)
    """

    def __init__(self):
        super().__init__("frame_source_node")
        self.bridge = CvBridge()

        self.declare_parameter("mode", "mock")
        self.declare_parameter("rgb_dir", "")
        self.declare_parameter("camera_config_yaml", _DEFAULT_YAML)
        self.declare_parameter("rgb_topic", "/rgbd/rgb")
        self.declare_parameter("camera_info_topic", "/rgbd/camera_info")
        self.declare_parameter("source_topic", "/xtend/rgb")
        self.declare_parameter("frame_id", "xtend_camera")
        self.declare_parameter("publish_hz", 10.0)
        self.declare_parameter("loop", True)
        self.declare_parameter("step", 1)
        self.declare_parameter("start_idx", 0)

        self.mode = self.get_parameter("mode").value
        self.frame_id = self.get_parameter("frame_id").value
        self.rgb_topic = self.get_parameter("rgb_topic").value
        self.camera_info_topic = self.get_parameter("camera_info_topic").value
        self.loop = bool(self.get_parameter("loop").value)

        qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=5,
            reliability=ReliabilityPolicy.BEST_EFFORT,
        )
        self.pub_rgb = self.create_publisher(Image, self.rgb_topic, qos)
        self.pub_info = self.create_publisher(CameraInfo, self.camera_info_topic, qos)

        config_yaml = self.get_parameter("camera_config_yaml").value
        self.cam_info_msg = load_camera_info_from_yaml(config_yaml, self.frame_id)

        if self.mode == "mock":
            self._setup_mock()
        elif self.mode == "live":
            self._setup_live(qos)
        else:
            raise ValueError(f"Unknown mode {self.mode!r}. Use 'mock' or 'live'.")

        self.get_logger().info(f"FrameSourceNode started in {self.mode!r} mode")

    def _setup_mock(self):
        rgb_dir = Path(self.get_parameter("rgb_dir").value).expanduser()
        if not rgb_dir.exists():
            raise RuntimeError(f"rgb_dir does not exist: {rgb_dir}")

        paths = []
        for ext in _IMG_EXTS:
            paths.extend(rgb_dir.glob(f"*.{ext}"))
        self.rgb_paths = sorted(set(paths))

        if not self.rgb_paths:
            raise RuntimeError(f"No images found in: {rgb_dir}")

        step = int(self.get_parameter("step").value)
        start = int(self.get_parameter("start_idx").value)
        self.rgb_paths = self.rgb_paths[start::step]
        self.idx = 0

        hz = float(self.get_parameter("publish_hz").value)
        self.timer = self.create_timer(1.0 / max(hz, 1e-6), self._mock_cb)
        self.get_logger().info(f"Mock: {len(self.rgb_paths)} frames @ {hz} Hz from {rgb_dir}")

    def _setup_live(self, qos):
        source = self.get_parameter("source_topic").value
        self.sub = self.create_subscription(Image, source, self._live_cb, qos)
        self.get_logger().info(f"Live: relaying {source} → {self.rgb_topic}")

    def _mock_cb(self):
        if self.idx >= len(self.rgb_paths):
            if self.loop:
                self.idx = 0
            else:
                self.get_logger().info("All frames published.")
                self.timer.cancel()
                rclpy.shutdown()
                return

        bgr = cv2.imread(str(self.rgb_paths[self.idx]), cv2.IMREAD_COLOR)
        if bgr is None:
            self.get_logger().warn(f"Could not read: {self.rgb_paths[self.idx]}")
            self.idx += 1
            return

        self._publish(bgr)
        self.idx += 1

    def _live_cb(self, msg: Image):
        bgr = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        self._publish(bgr)

    def _publish(self, bgr):
        stamp = self.get_clock().now().to_msg()
        self.cam_info_msg.header.stamp = stamp
        self.pub_info.publish(self.cam_info_msg)

        msg = self.bridge.cv2_to_imgmsg(bgr, encoding="bgr8")
        msg.header.stamp = stamp
        msg.header.frame_id = self.frame_id
        self.pub_rgb.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = FrameSourceNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
