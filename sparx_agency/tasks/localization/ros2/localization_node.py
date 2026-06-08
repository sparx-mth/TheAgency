"""
Generic localization ROS2 node.

Publishes /xtend/localization (PoseStamped) and /xtend/localization_source (String)
regardless of which localization method is active.

Usage — AprilTag:
  python3 -m sparx_agency.tasks.localization.ros2.localization_node \
    --ros-args -p provider_type:=apriltag \
    -p frame_path_topic:=/xtend/rgb_frame_path \
    -p tag_map_path:=/path/to/new_map.yaml \
    -p camera_calib_path:=/path/to/calib.yaml \
    -p tag_size_m:=0.13

Usage — Optical Flow:
  python3 -m sparx_agency.tasks.localization.ros2.localization_node \
    --ros-args -p provider_type:=optical_flow \
    -p frame_path_topic:=/xtend/rgb_frame_path \
    -p depth_frame_path_topic:=/xtend/depth_frame_path \
    -p camera_calib_path:=/path/to/calib.yaml \
    -p bearing_topic:=/xtend/bearing \
    -p demo_mode_topic:=/xtend/demo_mode
"""
from __future__ import annotations

import math
import sys
from typing import Optional

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import String

from sparx_agency.core.common.types.perception import Observation, RGBFrame
from sparx_agency.core.localization.base import BaseLocalizationProvider, LocalizationEstimate
from sparx_agency.core.localization.providers import (
    AprilTagLocalizationProvider,
    OpticalFlowLocalizationProvider,
    AmclLocalizationProvider,
)

_PROVIDERS = {
    "apriltag": AprilTagLocalizationProvider,
    "optical_flow": OpticalFlowLocalizationProvider,
    "amcl": AmclLocalizationProvider,
}

_OUTPUT_TOPIC = "/xtend/localization"
_SOURCE_TOPIC = "/xtend/localization_source"


def _parse_path_msg(data: str):
    """Parse '{path} {sec} {nanosec}' string. Returns (path, stamp_sec)."""
    parts = data.rsplit(" ", 2)
    path = parts[0]
    if len(parts) == 3:
        stamp_sec = int(parts[1]) + int(parts[2]) * 1e-9
    else:
        stamp_sec = 0.0
    return path, stamp_sec


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
        # optical_flow params
        self.declare_parameter("depth_frame_path_topic", "")
        self.declare_parameter("bearing_topic", "")
        self.declare_parameter("demo_mode_topic", "/xtend/demo_mode")
        self.declare_parameter("min_depth", 0.05)
        self.declare_parameter("max_depth", 10.0)
        self.declare_parameter("depth_ema_alpha", 0.05)
        self.declare_parameter("vel_alpha", 0.2)
        self.declare_parameter("max_wait_for_depth_sec", 0.1)
        # Velocity deadbands: zero out velocities below these thresholds (m/s).
        # Prevents noise from integrating into position drift when near-stationary.
        # Lower toward 0.0 for slow/calm flights; raise for vibrating live platforms.
        # vx=forward, vy=lateral, vz=vertical — vy/vz are typically noisier.
        self.declare_parameter("deadband_vx", 0.02)
        self.declare_parameter("deadband_vy", 0.20)
        self.declare_parameter("deadband_vz", 0.20)
        # amcl params
        self.declare_parameter("map_dir", "")
        self.declare_parameter("amcl_orientations_n", 32)
        self.declare_parameter("amcl_beams_n", 64)
        self.declare_parameter("amcl_max_range_m", 8.0)
        self.declare_parameter("amcl_window_m", 5.0)
        self.declare_parameter("amcl_initial_loc_m_json", "")
        self.declare_parameter("amcl_initial_orientation_rad", 0.0)
        self.declare_parameter("amcl_uncertainty_cells", 5)

        provider_type = self.get_parameter("provider_type").value
        if provider_type not in _PROVIDERS:
            raise ValueError(
                f"Unknown provider_type '{provider_type}'. Available: {list(_PROVIDERS)}"
            )

        self._provider: BaseLocalizationProvider = self._build_provider(provider_type)
        self._provider_type = provider_type
        self.get_logger().info(f"Localization provider: {provider_type} → {_OUTPUT_TOPIC}")

        qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.BEST_EFFORT,
        )

        existing = self.count_publishers(_OUTPUT_TOPIC)
        if existing > 0:
            self.get_logger().warn(
                f"CONFLICT: {existing} publisher(s) already active on {_OUTPUT_TOPIC}. "
                f"Only one localization provider should run at a time. "
                f"Stop the other node or poses will be overwritten."
            )

        self._pub_pose = self.create_publisher(PoseStamped, _OUTPUT_TOPIC, 10)
        self._pub_source = self.create_publisher(String, _SOURCE_TOPIC, 10)

        if provider_type in ("optical_flow", "amcl"):
            self._setup_optical_flow_subs(qos)
        else:
            self._setup_single_stream_subs(qos)

    def _setup_single_stream_subs(self, qos: QoSProfile) -> None:
        """AprilTag and similar single-stream providers."""
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

    def _setup_optical_flow_subs(self, qos: QoSProfile) -> None:
        """Optical flow uses separate RGB path, depth path, bearing, and demo_mode."""
        rgb_topic = self.get_parameter("frame_path_topic").value.strip()
        depth_topic = self.get_parameter("depth_frame_path_topic").value.strip()

        if not rgb_topic:
            raise ValueError("optical_flow provider requires frame_path_topic.")
        if not depth_topic:
            raise ValueError("optical_flow provider requires depth_frame_path_topic.")

        self.create_subscription(String, rgb_topic, self._on_rgb_path_flow, qos)
        self.create_subscription(String, depth_topic, self._on_depth_path_flow, qos)
        self.get_logger().info(f"[optical_flow] RGB:   {rgb_topic}")
        self.get_logger().info(f"[optical_flow] Depth: {depth_topic}")

        bearing_topic = self.get_parameter("bearing_topic").value.strip()
        if bearing_topic:
            from std_msgs.msg import Float32
            self.create_subscription(Float32, bearing_topic, self._on_bearing, 10)
            self.get_logger().info(f"[optical_flow] Bearing: {bearing_topic}")

        demo_mode_topic = self.get_parameter("demo_mode_topic").value.strip()
        if demo_mode_topic:
            self.create_subscription(String, demo_mode_topic, self._on_demo_mode, 10)
            self.get_logger().info(f"[optical_flow] Demo mode: {demo_mode_topic}")

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
        if provider_type == "optical_flow":
            calib = self.get_parameter("camera_calib_path").value
            if not calib:
                raise ValueError("optical_flow provider requires camera_calib_path.")
            return OpticalFlowLocalizationProvider(
                camera_calib_path=calib,
                min_depth=float(self.get_parameter("min_depth").value),
                max_depth=float(self.get_parameter("max_depth").value),
                depth_ema_alpha=float(self.get_parameter("depth_ema_alpha").value),
                vel_alpha=float(self.get_parameter("vel_alpha").value),
                max_wait_for_depth_sec=float(self.get_parameter("max_wait_for_depth_sec").value),
                deadband_vx=float(self.get_parameter("deadband_vx").value),
                deadband_vy=float(self.get_parameter("deadband_vy").value),
                deadband_vz=float(self.get_parameter("deadband_vz").value),
            )
        if provider_type == "amcl":
            map_dir = self.get_parameter("map_dir").value
            calib = self.get_parameter("camera_calib_path").value
            if not map_dir:
                raise ValueError("amcl provider requires map_dir.")
            if not calib:
                raise ValueError("amcl provider requires camera_calib_path.")
            return AmclLocalizationProvider(
                map_dir=map_dir,
                camera_calib_path=calib,
                num_orientations=int(self.get_parameter("amcl_orientations_n").value),
                num_beams=int(self.get_parameter("amcl_beams_n").value),
                max_range_m=float(self.get_parameter("amcl_max_range_m").value),
                window_m=float(self.get_parameter("amcl_window_m").value),
                initial_loc_m_json=self.get_parameter("amcl_initial_loc_m_json").value,
                initial_orientation_rad=float(self.get_parameter("amcl_initial_orientation_rad").value),
                prediction_uncertainty_cells=int(self.get_parameter("amcl_uncertainty_cells").value),
                max_wait_for_depth_sec=float(self.get_parameter("max_wait_for_depth_sec").value),
            )
        raise ValueError(f"No builder for provider_type: {provider_type}")

    # ------------------------------------------------------------------
    # Single-stream callbacks (apriltag)
    # ------------------------------------------------------------------

    def _on_frame_path(self, msg: String) -> None:
        path, stamp_sec = _parse_path_msg(msg.data)
        frame = cv2.imread(path, cv2.IMREAD_COLOR)
        if frame is None:
            self.get_logger().warn(f"Could not read frame: {path}")
            return
        obs = Observation(rgb=RGBFrame(image=frame, stamp_sec=stamp_sec))
        self._publish_estimate(self._provider.update(obs))

    def _on_image(self, msg) -> None:
        stamp_sec = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        frame = self._bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        obs = Observation(rgb=RGBFrame(image=frame, stamp_sec=stamp_sec))
        self._publish_estimate(self._provider.update(obs))

    # ------------------------------------------------------------------
    # Optical-flow callbacks
    # ------------------------------------------------------------------

    def _on_rgb_path_flow(self, msg: String) -> None:
        path, stamp_sec = _parse_path_msg(msg.data)
        frame = cv2.imread(path, cv2.IMREAD_COLOR)
        if frame is None:
            self.get_logger().warn(f"Could not read RGB frame: {path}")
            return
        self._provider.process_rgb(frame, stamp_sec)

    def _on_depth_path_flow(self, msg: String) -> None:
        path, stamp_sec = _parse_path_msg(msg.data)
        try:
            depth = np.load(path).astype(np.float32)
        except Exception as e:
            self.get_logger().warn(f"Could not load depth: {path} — {e}")
            return
        estimate = self._provider.process_depth(depth, stamp_sec)
        self._publish_estimate(estimate)

    def _on_bearing(self, msg) -> None:
        self._provider.set_yaw(float(msg.data))

    def _on_demo_mode(self, msg: String) -> None:
        mode = msg.data.lower().strip()
        self._provider.set_turning(mode == "turning")

    # ------------------------------------------------------------------
    # Shared publish helper
    # ------------------------------------------------------------------

    def _publish_estimate(self, estimate: Optional[LocalizationEstimate]) -> None:
        if estimate is None:
            return
        self._pub_pose.publish(_to_pose_stamped(estimate))
        src_msg = String()
        src_msg.data = estimate.source
        self._pub_source.publish(src_msg)


def _to_pose_stamped(est: LocalizationEstimate) -> PoseStamped:
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