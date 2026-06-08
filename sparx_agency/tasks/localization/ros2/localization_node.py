"""
Generic localization ROS2 node.

Publishes /xtend/localization (PoseStamped) and /xtend/localization_source (String)
regardless of which localization method is active.

Usage:
  python3 -m sparx_agency.tasks.localization.ros2.localization_node \
    --ros-args \
    -p provider_type:=apriltag \
    -p frame_path_topic:=/xtend/rgb_frame_path \
    -p tag_map_path:=/path/to/new_map.yaml \
    -p camera_calib_path:=/path/to/calib.yaml \
    -p tag_size_m:=0.13
"""
from __future__ import annotations

import math
import sys
from typing import Optional

import cv2
import rclpy
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import String

from sparx_agency.core.common.types.perception import Observation, RGBFrame
from sparx_agency.core.localization.base import BaseLocalizationProvider, LocalizationEstimate
from sparx_agency.core.localization.providers import AprilTagLocalizationProvider

_PROVIDERS = {
    "apriltag": AprilTagLocalizationProvider,
    # register new providers here: "optical_flow": OpticalFlowLocalizationProvider,
}

_OUTPUT_TOPIC = "/xtend/localization"
_SOURCE_TOPIC = "/xtend/localization_source"


class LocalizationNode(Node):
    """
    Wraps any BaseLocalizationProvider and publishes its output
    to a fixed topic pair regardless of which method is running.
    """

    def __init__(self) -> None:
        super().__init__("localization_node")

        self.declare_parameter("provider_type", "apriltag")
        self.declare_parameter("frame_path_topic", "/xtend/rgb_frame_path")
        self.declare_parameter("image_topic", "")
        # apriltag params
        self.declare_parameter("tag_map_path", "")
        self.declare_parameter("camera_calib_path", "")
        self.declare_parameter("tag_size_m", 0.13)
        self.declare_parameter("tag_family", "tag36h11")
        self.declare_parameter("min_margin", 10.0)
        self.declare_parameter("alpha", 0.1)

        provider_type = self.get_parameter("provider_type").value
        if provider_type not in _PROVIDERS:
            raise ValueError(
                f"Unknown provider_type '{provider_type}'. Available: {list(_PROVIDERS)}"
            )

        self._provider: BaseLocalizationProvider = self._build_provider(provider_type)
        self.get_logger().info(
            f"Localization provider: {provider_type} → {_OUTPUT_TOPIC}"
        )

        qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.BEST_EFFORT,
        )

        self._pub_pose = self.create_publisher(PoseStamped, _OUTPUT_TOPIC, 10)
        self._pub_source = self.create_publisher(String, _SOURCE_TOPIC, 10)

        frame_path_topic = self.get_parameter("frame_path_topic").value.strip()
        image_topic = self.get_parameter("image_topic").value.strip()

        if frame_path_topic and not image_topic:
            self.create_subscription(String, frame_path_topic, self._on_frame_path, qos)
            self.get_logger().info(f"Subscribed to frame_path_topic: {frame_path_topic}")
        elif image_topic:
            from sensor_msgs.msg import Image
            from cv_bridge import CvBridge
            self._bridge = CvBridge()
            self.create_subscription(Image, image_topic, self._on_image, qos)
            self.get_logger().info(f"Subscribed to image_topic: {image_topic}")
        else:
            raise ValueError("Set frame_path_topic or image_topic.")

    def _build_provider(self, provider_type: str) -> BaseLocalizationProvider:
        if provider_type == "apriltag":
            tag_map = self.get_parameter("tag_map_path").value
            calib = self.get_parameter("camera_calib_path").value
            if not tag_map or not calib:
                raise ValueError("apriltag provider requires tag_map_path and camera_calib_path.")
            return AprilTagLocalizationProvider(
                tag_map_path=tag_map,
                camera_calib_path=calib,
                tag_size_m=float(self.get_parameter("tag_size_m").value),
                tag_family=str(self.get_parameter("tag_family").value),
                min_margin=float(self.get_parameter("min_margin").value),
                alpha=float(self.get_parameter("alpha").value),
            )
        raise ValueError(f"No builder for provider_type: {provider_type}")

    def _on_frame_path(self, msg: String) -> None:
        parts = msg.data.rsplit(" ", 2)
        path = parts[0]
        if len(parts) == 3:
            stamp_sec = int(parts[1]) + int(parts[2]) * 1e-9
        else:
            stamp_sec = float(self.get_clock().now().nanoseconds) * 1e-9

        frame = cv2.imread(path, cv2.IMREAD_COLOR)
        if frame is None:
            self.get_logger().warn(f"Could not read frame: {path}")
            return

        obs = Observation(rgb=RGBFrame(image=frame, stamp_sec=stamp_sec))
        self._process(obs)

    def _on_image(self, msg) -> None:
        stamp_sec = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        frame = self._bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        obs = Observation(rgb=RGBFrame(image=frame, stamp_sec=stamp_sec))
        self._process(obs)

    def _process(self, obs: Observation) -> None:
        if not self._provider.is_healthy():
            self.get_logger().warn(f"Provider {self._provider.source_name} is unhealthy.")
            return

        estimate: Optional[LocalizationEstimate] = self._provider.update(obs)
        if estimate is None:
            return
        pose_msg = _to_pose_stamped(estimate)
        self._pub_pose.publish(pose_msg)

        src_msg = String()
        src_msg.data = estimate.source
        self._pub_source.publish(src_msg)


def _to_pose_stamped(est: LocalizationEstimate) -> PoseStamped:
    """Convert LocalizationEstimate → geometry_msgs/PoseStamped (yaw-only quaternion)."""
    msg = PoseStamped()
    msg.header.frame_id = "world"
    msg.header.stamp.sec = int(est.stamp_sec)
    msg.header.stamp.nanosec = int((est.stamp_sec - int(est.stamp_sec)) * 1e9)

    msg.pose.position.x = est.pose.x
    msg.pose.position.y = est.pose.y
    msg.pose.position.z = est.pose.z

    half = est.pose.yaw / 2.0
    msg.pose.orientation.x = 0.0
    msg.pose.orientation.y = 0.0
    msg.pose.orientation.z = math.sin(half)
    msg.pose.orientation.w = math.cos(half)

    return msg


def main() -> None:
    rclpy.init(args=sys.argv)
    node = LocalizationNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()