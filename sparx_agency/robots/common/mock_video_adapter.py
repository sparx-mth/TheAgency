#!/usr/bin/env python3
"""
MockVideoStreamManager: A drop-in replacement for VideoStreamManager
that reads from webcam or video file instead of GStreamer UDP stream.

Usage:
    # In your test main.py, replace:
    # from sparx_agency.robots.ROBOTICAN.adapters.rooster_video_adapter import VideoStreamManager
    # with:
    from sparx_agency.robots.common.mock_video_adapter import MockVideoStreamManager as VideoStreamManager
"""
import argparse
import datetime
import threading
import time
import copy
import math
import os
import sys
from typing import Optional

# Add project root to path so we can import sparx_agency when run standalone
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))

import numpy as np
import cv2

import rclpy
from rclpy.node import Node
from rclpy.duration import Duration
from rclpy.qos import qos_profile_sensor_data

from sensor_msgs.msg import Image, CameraInfo
from builtin_interfaces.msg import Time
from std_srvs.srv import Trigger

from cv_bridge import CvBridge

# AprilTag azimuth calculation (optional - may not be installed)
try:
    from sparx_agency.tasks.localization.opencv.tag_azimuth_node import TagAzimuthOpenCVTask
    APRILTAG_AVAILABLE = True
except ImportError as e:
    APRILTAG_AVAILABLE = False
    TagAzimuthOpenCVTask = None
    print(f"[MockVideoAdapter] AprilTag not available: {e}")


class MockVideoStreamManager(Node):
    """
    Mock version of VideoStreamManager that reads from webcam or video file.
    Provides the same ROS2 interface (topics, services) as the real adapter.
    Includes AprilTag-based azimuth calculation.
    """

    def __init__(
        self,
        drone_id: str = "R1",
        high_resolution: int = 640,
        source=0,  # 0=webcam, or path to video file
        loop_video: bool = True,
        fps: float = 10.0,
        tag_config_path: str = "sparx_agency/tasks/localization/config/tags_azimuth.yaml",
        camera_calib_path: str = "sparx_agency/tasks/localization/config/front_camera_calib.yaml",
        tag_size_m: float = 0.16,
        **kwargs  # Ignore extra args like host_ip, port
    ):
        super().__init__("mock_video_stream")
        self.id = drone_id
        self.width = high_resolution
        self.height = int(self.width * 9 / 16)
        self.source = source
        self.loop_video = loop_video
        self.fps = fps

        self.i = 0
        self.capturing_enabled = False
        self.dir_name = datetime.datetime.now().strftime("%Y_%m_%d___%H_%M_%S")
        self.last_frame_np = None
        self.camera_info_template = None

        self.bridge = CvBridge()

        # AprilTag localization (azimuth calculation)
        self.april_localization = None
        if APRILTAG_AVAILABLE:
            try:
                self.april_localization = TagAzimuthOpenCVTask(
                    tag_config_path=tag_config_path,
                    camera_calib_path=camera_calib_path,
                    tag_size_m=tag_size_m,
                )
                self.get_logger().info("AprilTag localization initialized.")
            except Exception as e:
                self.get_logger().warn(f"AprilTag init failed (azimuth will be 0): {e}")
        else:
            self.get_logger().warn("AprilTag not available (pupil_apriltags not installed)")

        # ROS2 topic names (same as real VideoStreamManager)
        image_raw_topic = f"/{self.id}/camera/image_raw"
        camera_info_topic = f"/{self.id}/camera/camera_info"
        image_used_topic = f"/{self.id}/selected_frame"
        self.camera_frame = f"{self.id}_camera"

        # Publishers
        self.image_pub = self.create_publisher(Image, image_raw_topic, qos_profile_sensor_data)
        self.camera_info_pub = self.create_publisher(CameraInfo, camera_info_topic, qos_profile_sensor_data)
        self.image_used_pub = self.create_publisher(Image, image_used_topic, qos_profile_sensor_data)

        # Services (same as real VideoStreamManager)
        self.create_service(Trigger, f"/{self.id}/start_capture", self.handle_start_capture)
        self.create_service(Trigger, f"/{self.id}/stop_capture", self.handle_stop_capture)

        # Video capture
        self.cap = None
        self.cap_lock = threading.Lock()
        self._open_video_source()

        # Timer for reading frames
        period = 1.0 / self.fps
        self.frame_timer = self.create_timer(period, self.on_frame_timer)

        self.get_logger().info(
            f"MockVideoStreamManager initialized: source={source}, "
            f"resolution={self.width}x{self.height}, fps={fps}"
        )

    def _open_video_source(self):
        """Open the video source (webcam or file)."""
        with self.cap_lock:
            if self.cap is not None:
                self.cap.release()

            self.cap = cv2.VideoCapture(self.source)
            if not self.cap.isOpened():
                self.get_logger().error(f"Failed to open video source: {self.source}")
                self.cap = None
            else:
                self.get_logger().info(f"Opened video source: {self.source}")

    def handle_start_capture(self, request, response):
        self.capturing_enabled = True
        self.dir_name = datetime.datetime.now().strftime("%Y_%m_%d___%H_%M_%S")
        self.get_logger().info("Capture ENABLED by service call")
        response.success = True
        response.message = "Started capturing"
        return response

    def handle_stop_capture(self, request, response):
        self.capturing_enabled = False
        self.get_logger().info("Capture DISABLED by service call")
        response.success = True
        response.message = "Stopped capturing"
        return response

    def on_frame_timer(self):
        """Read a frame and process/publish it."""
        if not self.capturing_enabled:
            return

        frame = self._read_frame()
        if frame is None:
            return

        now = self.get_clock().now()
        self.process_frame(frame, now)

    def _read_frame(self) -> Optional[np.ndarray]:
        """Read a frame from the video source."""
        with self.cap_lock:
            if self.cap is None:
                return None

            ret, frame = self.cap.read()
            if not ret:
                if self.loop_video and isinstance(self.source, str):
                    # Loop video file
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    ret, frame = self.cap.read()
                    if not ret:
                        self.get_logger().warn("Failed to loop video")
                        return None
                else:
                    self.get_logger().info("End of video stream")
                    return None

            # Resize to target resolution
            frame = cv2.resize(frame, (self.width, self.height))
            return frame

    def process_frame(self, frame: np.ndarray, now):
        """Process and publish a frame (same interface as real VideoStreamManager)."""
        h, w = frame.shape[:2]
        stamp_msg = now.to_msg()
        self.i += 1

        # Build CameraInfo template once
        if self.camera_info_template is None:
            self.camera_info_template = self.make_camera_info(frame_id=self.camera_frame)

        if self.i % 10 == 1:
            self.get_logger().info(f"Frame #{self.i}: {w}x{h}")

        # Publish Image
        msg = self.bridge.cv2_to_imgmsg(frame, encoding="bgr8")
        msg.header.stamp = stamp_msg
        msg.header.frame_id = self.camera_frame
        self.image_pub.publish(msg)

        # Store for external access
        self.last_frame_np = frame.copy()

        # Publish CameraInfo
        ci = copy.deepcopy(self.camera_info_template)
        ci.header.stamp = stamp_msg
        ci.header.frame_id = self.camera_frame
        ci.width = w
        ci.height = h
        self.camera_info_pub.publish(ci)

        # Compute azimuth using AprilTag (like real VideoStreamManager)
        timestamp_sec = now.nanoseconds * 1e-9
        last_yaw_deg = None

        if self.april_localization is not None:
            try:
                last_yaw_deg = self.april_localization.compute_azimuth_from_bgr(
                    frame,
                    timestamp_sec,
                )
                self.get_logger().debug(f"AprilTag yaw: {last_yaw_deg:.1f} deg")
            except Exception as e:
                # No tag detected or computation failed
                self.get_logger().debug(f"Azimuth calc failed: {e}")

        if last_yaw_deg is None:
            last_yaw_rad = math.pi  # Default when no tag detected (same as real adapter)
        else:
            last_yaw_rad = math.radians(last_yaw_deg)

        # Publish selected_frame with yaw in frame_id (same format as real adapter)
        msg_used = copy.deepcopy(msg)
        msg_used.header.frame_id = f"{self.dir_name}_____{last_yaw_rad:.5f}"
        self.image_used_pub.publish(msg_used)

    def intrinsic_from_fov(self, hfov_deg=135, vfov_deg=90, half_pixel=True):
        # Kept in sync with rooster_video_adapter.py:VideoStreamManager -
        # hfov=135deg is confirmed against Sphera's scenario config
        # (roosters.R1.main_camera.hfov); vfov=90deg is still unverified.
        theta_x = np.deg2rad(hfov_deg)
        theta_y = np.deg2rad(vfov_deg)

        fx = self.width / (2.0 * np.tan(theta_x / 2.0))
        fy = self.height / (2.0 * np.tan(theta_y / 2.0))

        if half_pixel:
            cx = (self.width - 1) / 2.0
            cy = (self.height - 1) / 2.0
        else:
            cx = self.width / 2.0
            cy = self.height / 2.0

        K = [fx, 0.0, cx, 0.0, fy, cy, 0.0, 0.0, 1.0]
        return K

    def make_camera_info(
        self,
        frame_id: str = "camera",
        stamp: Optional[Time] = None,
        distortion_model: str = "plumb_bob",
    ) -> CameraInfo:
        K = self.intrinsic_from_fov()
        K_list = list(K)

        fx, fy = K_list[0], K_list[4]
        cx, cy = K_list[2], K_list[5]

        msg = CameraInfo()
        if stamp is not None:
            msg.header.stamp = stamp
        msg.header.frame_id = frame_id
        msg.width = self.width
        msg.height = self.height
        msg.k = K_list
        msg.distortion_model = distortion_model
        msg.d = [0.0, 0.0, 0.0, 0.0, 0.0]
        msg.r = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
        msg.p = [fx, 0.0, cx, 0.0, 0.0, fy, cy, 0.0, 0.0, 0.0, 1.0, 0.0]
        return msg

    def destroy_node(self):
        with self.cap_lock:
            if self.cap is not None:
                self.cap.release()
                self.cap = None
        super().destroy_node()


def main(args=None):
    parser = argparse.ArgumentParser(description="Mock Video Stream from webcam/file")
    parser.add_argument("--drone-id", default="R1", help="Drone ID")
    parser.add_argument("--source", default="0", help="0=webcam or path to video file")
    parser.add_argument("--width", type=int, default=640, help="Image width")
    parser.add_argument("--fps", type=float, default=10.0, help="Frames per second")
    parser.add_argument("--loop", action="store_true", help="Loop video file")
    parser.add_argument("--tag-config", default="sparx_agency/tasks/localization/config/tags_azimuth.yaml",
                        help="Path to AprilTag config YAML")
    parser.add_argument("--camera-calib", default="sparx_agency/tasks/localization/config/front_camera_calib.yaml",
                        help="Path to camera calibration YAML")
    parser.add_argument("--tag-size", type=float, default=0.16, help="AprilTag size in meters")
    parsed = parser.parse_args()

    # Parse source
    source = int(parsed.source) if parsed.source.isdigit() else parsed.source

    rclpy.init(args=args)
    node = MockVideoStreamManager(
        drone_id=parsed.drone_id,
        high_resolution=parsed.width,
        source=source,
        loop_video=parsed.loop,
        fps=parsed.fps,
        tag_config_path=parsed.tag_config,
        camera_calib_path=parsed.camera_calib,
        tag_size_m=parsed.tag_size,
    )

    # Auto-enable capturing for standalone test
    node.capturing_enabled = True

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()

