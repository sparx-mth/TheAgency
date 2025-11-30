#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.client import Client

from video_handler_interfaces.srv import SetVideoMode
from std_msgs.msg import Bool
from sensor_msgs.msg import Image

import numpy as np
import cv2
from cv_bridge import CvBridge

import gi
gi.require_version("Gst", "1.0")
gi.require_version("GstVideo", "1.0")
from gi.repository import Gst, GstVideo

Gst.init(None)


class VideoExample(Node):
    def __init__(self):
        super().__init__("video_example")
        self.id = "R2"
        self.i = 0

        # cv_bridge for publishing
        self.bridge = CvBridge()

        # ---- ROS2: service + keep-alive ----
        self.set_video_mode_srv: Client = self.create_client(
            srv_type=SetVideoMode,
            srv_name=f"/{self.id}/video_handler/set_video_mode",
        )

        self.gcs_keep_alive_publisher = self.create_publisher(
            Bool, f"/{self.id}/gcs_keep_alive", 10
        )
        self.gcs_keep_alive_timer = self.create_timer(
            1.0, self.gcs_keep_alive_timer_callback
        )

        self.video_on_timer = self.create_timer(
            3.0, self.video_on_timer_callback
        )

        self.image_pub = self.create_publisher(
            Image, f"/{self.id}/camera/image_raw", 10
        )

        # ---- GStreamer pipeline with appsink ----
        gst_pipeline = (
            "udpsrc port=5001 "
            "caps=application/x-rtp,media=(string)video,clock-rate=(int)90000,"
            "encoding-name=(string)H264,payload=(int)96 ! "
            "rtph264depay ! "
            "decodebin ! "
            "videoconvert ! "
            "video/x-raw,format=BGR ! "
            "appsink name=mysink emit-signals=true sync=false max-buffers=1 drop=true"
        )

        self.get_logger().info(f"Creating GStreamer pipeline:\n{gst_pipeline}")

        self.pipeline = Gst.parse_launch(gst_pipeline)
        self.appsink = self.pipeline.get_by_name("mysink")
        if self.appsink is None:
            self.get_logger().error("Failed to get appsink from pipeline")
            raise RuntimeError("appsink not found")

        # be explicit about properties
        self.appsink.set_property("emit-signals", True)
        self.appsink.set_property("sync", False)
        self.appsink.set_property("max-buffers", 1)
        self.appsink.set_property("drop", True)

        # connect callback that will be called on every new frame
        self.appsink.connect("new-sample", self.on_new_sample)

        # optional: listen to bus errors
        bus = self.pipeline.get_bus()
        bus.add_signal_watch()
        bus.connect("message::error", self.on_gst_error)

        # start the pipeline
        ret = self.pipeline.set_state(Gst.State.PLAYING)
        if ret == Gst.StateChangeReturn.FAILURE:
            self.get_logger().error("Failed to set GStreamer pipeline to PLAYING")
        else:
            self.get_logger().info("GStreamer pipeline set to PLAYING")

    # ---------- ROS2 timers ----------

    def gcs_keep_alive_timer_callback(self):
        msg = Bool()
        msg.data = True
        self.gcs_keep_alive_publisher.publish(msg)

    def video_on_timer_callback(self):
        # run only once
        self.video_on_timer.cancel()

        if not self.set_video_mode_srv.service_is_ready():
            self.get_logger().warn("set_video_mode service not ready yet, retrying...")
            self.video_on_timer.reset()
            return

        req = SetVideoMode.Request()
        req.camera_id = 0
        req.playing = True
        req.port = 5001
        req.host = "192.168.131.5"   # host IP
        HIGH_RESO = 640
        req.resolution_width = HIGH_RESO
        req.resolution_height = int(HIGH_RESO * 9 / 16)
        req.recording = False
        req.bitrate = SetVideoMode.Request.BITRATE_1500000
        req.fps = 0  # default

        self.get_logger().info(
            f"send {self.set_video_mode_srv.srv_name} [{req}]"
        )

        future = self.set_video_mode_srv.call_async(req)

        def set_video_mode_cb(fut: rclpy.client.Future):
            try:
                result = fut.result()
                self.get_logger().info(
                    f"{self.id}: set_video_mode_cb: success={result.success}, msg='{result.message}'"
                )
            except Exception as e:
                self.get_logger().error(f"{self.id}: set_video_mode failed: {e}")

        future.add_done_callback(set_video_mode_cb)

    # ---------- GStreamer callbacks ----------

    def on_gst_error(self, bus, msg):
        err, debug = msg.parse_error()
        self.get_logger().error(f"GStreamer error: {err} (debug: {debug})")

    def on_new_sample(self, sink):
        """
        Called by GStreamer streaming thread whenever a new frame arrives.
        We convert GstSample → numpy array (BGR) and hand it to process_frame().
        """
        sample = sink.emit("pull-sample")
        if sample is None:
            return Gst.FlowReturn.ERROR

        buf = sample.get_buffer()
        caps = sample.get_caps()

        # Get video info (width/height)
        info = GstVideo.VideoInfo()
        try:
            info.from_caps(caps)
        except Exception as e:
            self.get_logger().error(f"VideoInfo.from_caps failed: {e}")
            return Gst.FlowReturn.ERROR

        width, height = info.width, info.height

        success, map_info = buf.map(Gst.MapFlags.READ)
        if not success:
            self.get_logger().warn("Failed to map buffer")
            return Gst.FlowReturn.ERROR

        try:
            data = map_info.data
            expected = width * height * 3
            if len(data) < expected:
                self.get_logger().warn(
                    f"Mapped buffer too small: len={len(data)}, expected={expected}"
                )
                return Gst.FlowReturn.ERROR

            frame = np.frombuffer(data, dtype=np.uint8)
            try:
                frame = frame.reshape((height, width, 3))
            except ValueError as e:
                self.get_logger().error(
                    f"Reshape failed for frame {height}x{width}: {e}"
                )
                return Gst.FlowReturn.ERROR

            frame = frame.copy()  # detach from Gst buffer

        finally:
            buf.unmap(map_info)

        # now we can use the frame in our own code
        self.process_frame(frame)

        return Gst.FlowReturn.OK

    # ---------- your logic on each frame ----------

    def process_frame(self, frame: np.ndarray):
        """
        frame is a numpy array, shape (H, W, 3), BGR.
        """
        h, w, _ = frame.shape
        self.i += 1

        # occasional debug
        if self.i % 30 == 1:
            self.get_logger().info(f"Frame #{self.i}: {w}x{h}")

        # Optional: save a frame every 100 frames
        # if self.i % 100 == 0:
        #     cv2.imwrite(f"{self.id}_{self.i}.png", frame)

        # ROS2 publish
        msg = self.bridge.cv2_to_imgmsg(frame, encoding="bgr8")
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = f"{self.id}_camera"
        self.image_pub.publish(msg)

    # ---------- cleanup ----------

    def destroy_node(self):
        # stop pipeline
        if hasattr(self, "pipeline") and self.pipeline is not None:
            self.pipeline.set_state(Gst.State.NULL)
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = VideoExample()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
