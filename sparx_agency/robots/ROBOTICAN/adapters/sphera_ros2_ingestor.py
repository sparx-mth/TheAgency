#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
import cv2
import numpy as np
from cv_bridge import CvBridge

# ROS 2 & Robotican Interfaces
from fcu_driver_interfaces.msg import UAVState
from rooster_handler_interfaces.msg import KeepAlive
from video_handler_interfaces.srv import SetVideoMode

import gi

gi.require_version("Gst", "1.0")
from gi.repository import Gst


class SpheraRos2Ingestor(Node):
    def __init__(self, pipeline, costmap, drone_id="R2"):
        super().__init__('sphera_ros2_ingestor')
        self.pipeline = pipeline
        self.costmap = costmap
        self.drone_id = drone_id
        self.bridge = CvBridge()

        # 1. Hardware Heartbeat (Keep the Rooster alive)
        self.keep_alive_pub = self.create_publisher(KeepAlive, f"/{self.drone_id}/keep_alive", 10)
        self.create_timer(1.0, self._publish_heartbeat)

        # 2. Drone State (Odometry)
        self.state_sub = self.create_subscription(
            UAVState, f"/{self.drone_id}/fcu/state", self.uav_state_callback, 10)
        self.last_pose = None

        # 3. Video Setup
        self.video_client = self.create_client(SetVideoMode, f"/{self.drone_id}/video/set_video_mode")
        Gst.init(None)

        # GStreamer pipeline for UDP stream from Rooster
        gst_str = (f"udpsrc port=5001 ! rtph264depay ! decodebin ! "
                   "videoconvert ! video/x-raw,format=BGR ! appsink name=sink emit-signals=true sync=false")
        self.gst_pipeline = Gst.parse_launch(gst_str)
        sink = self.gst_pipeline.get_by_name("sink")
        sink.connect("new-sample", self._on_new_frame)

    def _publish_heartbeat(self):
        msg = KeepAlive(is_active=True, requested_flight_mode=1)
        self.keep_alive_pub.publish(msg)

    def start_hardware_stream(self):
        """Call the service to make the drone send H.264 packets."""
        if not self.video_client.wait_for_service(timeout_sec=5.0):
            self.get_logger().error("Video Service NOT ready!")
            return

        req = SetVideoMode.Request()
        req.playing = True
        req.host = "192.168.131.24"  # Set this to your local processing machine IP
        req.port = 5001
        req.resolution_width = 640
        req.resolution_height = 360

        self.get_logger().info("Sending Video Request to Drone...")
        self.video_client.call_async(req)
        self.gst_pipeline.set_state(Gst.State.PLAYING)

    def uav_state_callback(self, msg):
        # Store the latest pose for the next video frame
        self.last_pose = msg.pose

    def _on_new_frame(self, sink):
        sample = sink.emit("pull-sample")
        buf = sample.get_buffer()
        res, map_info = buf.map(Gst.MapFlags.READ)
        if res and self.last_pose:
            # Create OpenCV frame (BGR)
            frame = np.ndarray((360, 640, 3), buffer=map_info.data, dtype=np.uint8)

            # --- EXECUTE MAPPING PIPELINE ---
            # This triggers DepthAnythingV2 + Cloud Generation + Costmap Update
            self.pipeline.process_frame(frame, self.last_pose)

            buf.unmap(map_info)
        return Gst.FlowReturn.OK

    def activate_video_hardware(self):
        """Sends the service request to the drone to start streaming."""
        # Check if the service is ready before calling
        if not self.video_client.wait_for_service(timeout_sec=5.0):
            self.get_logger().error("SetVideoMode service not available!")
            return False

        req = SetVideoMode.Request()
        req.camera_id = 0
        req.playing = True
        # Ensure this is the IP of your processing machine (host)
        req.host = "192.168.131.20"
        req.port = 5001
        req.resolution_width = 640
        req.resolution_height = 360
        req.bitrate = SetVideoMode.Request.BITRATE_1500000

        self.get_logger().info(f"Requesting hardware video stream for {self.drone_id}...")

        # Call the service
        self.video_client.call_async(req)

        # Start the local GStreamer pipeline to receive the packets
        self.gst_pipeline.set_state(Gst.State.PLAYING)
        return True