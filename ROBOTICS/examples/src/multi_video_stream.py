#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.client import Client

from std_msgs.msg import Bool
from sensor_msgs.msg import Image
from video_handler_interfaces.srv import SetVideoMode

import numpy as np
import cv2
from cv_bridge import CvBridge

from gi.repository import Gst

Gst.init(None)

# Configure which drones to try and which ports they use
DRONE_IDS = ["R1", "R2", "R3"]
BASE_PORT = 5001  # R1 -> 5001, R2 -> 5002...


class MultiDroneVideo(Node):
    def __init__(self):
        super().__init__("multi_drone_video")
        self.bridge = CvBridge()
        self.drones = {}

        # Create contexts per drone
        for idx, drone_id in enumerate(DRONE_IDS):
            udp_port = BASE_PORT + idx
            ctx = self._create_drone_context(drone_id, udp_port)
            if ctx is not None:
                self.drones[drone_id] = ctx

        if not self.drones:
            self.get_logger().warn("No drones initialized (no services found?)")

    # -------------------------------------------------------
    # Drone context
    # -------------------------------------------------------
    class DroneContext:
        def __init__(self, drone_id, port, client, keep_alive_pub,
                     keep_alive_timer, video_on_timer, image_pub,
                     pipeline, appsink):
            self.drone_id = drone_id
            self.port = port
            self.client = client
            self.keep_alive_pub = keep_alive_pub
            self.keep_alive_timer = keep_alive_timer
            self.video_on_timer = video_on_timer
            self.image_pub = image_pub
            self.pipeline = pipeline
            self.appsink = appsink
            self.frame_counter = 0

    def _create_drone_context(self, drone_id: str, udp_port: int):
        self.get_logger().info(f"Setting up drone {drone_id} on UDP port {udp_port}")

        # Service client
        client: Client = self.create_client(
            SetVideoMode, f"/{drone_id}/video_handler/set_video_mode"
        )

        # Optional: wait a bit to see if the service exists
        if not client.wait_for_service(timeout_sec=2.0):
            self.get_logger().warn(
                f"{drone_id}: SetVideoMode service not available, skipping this drone"
            )
            return None

        # Keep-alive publisher and timer
        keep_alive_pub = self.create_publisher(
            Bool, f"/{drone_id}/gcs_keep_alive", 10
        )
        keep_alive_timer = self.create_timer(
            1.0, lambda d=drone_id, p=keep_alive_pub: self._keep_alive_cb(d, p)
        )

        # Image publisher
        image_pub = self.create_publisher(
            Image, f"/{drone_id}/camera/image_raw", 10
        )

        # GStreamer pipeline
        gst_pipeline = (
            f"udpsrc port={udp_port} "
            "caps=application/x-rtp,media=(string)video,clock-rate=(int)90000,"
            "encoding-name=(string)H264,payload=(int)96 ! "
            "rtph264depay ! "
            "decodebin ! "
            "videoconvert ! "
            "video/x-raw,format=BGR ! "
            "appsink name=mysink emit-signals=true sync=false max-buffers=1 drop=true"
        )
        self.get_logger().info(f"{drone_id}: creating GStreamer pipeline:\n{gst_pipeline}")

        pipeline = Gst.parse_launch(gst_pipeline)
        appsink = pipeline.get_by_name("mysink")
        if appsink is None:
            self.get_logger().error(f"{drone_id}: failed to get appsink from pipeline")
            return None

        # Connect appsink callback with drone_id as user_data
        appsink.connect("new-sample", self._on_new_sample, drone_id)

        bus = pipeline.get_bus()
        bus.add_signal_watch()
        bus.connect("message::error", self._on_gst_error, drone_id)

        ret = pipeline.set_state(Gst.State.PLAYING)
        if ret == Gst.StateChangeReturn.FAILURE:
            self.get_logger().error(f"{drone_id}: failed to set pipeline to PLAYING")
            return None

        # Timer to send SetVideoMode (one-shot)
        video_on_timer = self.create_timer(
            3.0, lambda d=drone_id, c=client, port=udp_port: self._video_on_cb(d, c, port)
        )

        return self.DroneContext(
            drone_id=drone_id,
            port=udp_port,
            client=client,
            keep_alive_pub=keep_alive_pub,
            keep_alive_timer=keep_alive_timer,
            video_on_timer=video_on_timer,
            image_pub=image_pub,
            pipeline=pipeline,
            appsink=appsink,
        )

    # -------------------------------------------------------
    # ROS callbacks
    # -------------------------------------------------------
    def _keep_alive_cb(self, drone_id: str, pub):
        msg = Bool()
        msg.data = True
        pub.publish(msg)
        # self.get_logger().info(f"{drone_id}: keep-alive")

    def _video_on_cb(self, drone_id: str, client: Client, udp_port: int):
        # Cancel the timer (one-shot behavior)
        timers_to_cancel = []
        for d_id, ctx in self.drones.items():
            if d_id == drone_id:
                timers_to_cancel.append(ctx.video_on_timer)
        for t in timers_to_cancel:
            t.cancel()

        req = SetVideoMode.Request()
        req.camera_id = 0
        req.playing = True
        req.port = udp_port
        req.host = "127.0.0.1"
        HIGH_RESO = 640
        req.resolution_width = HIGH_RESO
        req.resolution_height = int(HIGH_RESO * 9 / 16)
        req.recording = False
        req.bitrate = SetVideoMode.Request.BITRATE_1500000
        req.fps = 0

        self.get_logger().info(f"{drone_id}: sending SetVideoMode [{req}]")

        future = client.call_async(req)

        def done_cb(fut: rclpy.client.Future):
            result = fut.result()
            self.get_logger().info(
                f"{drone_id}: set_video_mode_cb: {getattr(result, 'success', False)}"
            )

        future.add_done_callback(done_cb)

    # -------------------------------------------------------
    # GStreamer callbacks
    # -------------------------------------------------------
    def _on_gst_error(self, bus, msg, drone_id: str):
        err, debug = msg.parse_error()
        self.get_logger().error(
            f"{drone_id}: GStreamer error: {err} (debug: {debug})"
        )

    def _on_new_sample(self, sink, drone_id: str):
        sample = sink.emit("pull-sample")
        if sample is None:
            return Gst.FlowReturn.ERROR

        buf = sample.get_buffer()
        caps = sample.get_caps()
        struct = caps.get_structure(0)

        width = struct.get_value("width")
        height = struct.get_value("height")

        ok, map_info = buf.map(Gst.MapFlags.READ)
        if not ok:
            self.get_logger().warn(f"{drone_id}: failed to map buffer")
            return Gst.FlowReturn.ERROR

        try:
            frame = np.frombuffer(map_info.data, dtype=np.uint8)
            frame = frame.reshape((height, width, 3))
            frame = frame.copy()  # ensure it is safe after unmap
        finally:
            buf.unmap(map_info)

        self._process_frame(drone_id, frame)
        return Gst.FlowReturn.OK

    # -------------------------------------------------------
    # Per-frame processing
    # -------------------------------------------------------
    def _process_frame(self, drone_id: str, frame: np.ndarray):
        ctx = self.drones.get(drone_id)
        if ctx is None:
            return

        ctx.frame_counter += 1

        # Example: save every 100 frames
        # if ctx.frame_counter % 100 == 0:
        #     filename = f"{drone_id}_frame_{ctx.frame_counter:06d}.png"
        #     cv2.imwrite(filename, frame)
        #     self.get_logger().info(f"{drone_id}: saved {filename}")

        # Publish as ROS2 Image
        msg = self.bridge.cv2_to_imgmsg(frame, encoding="bgr8")
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = f"{drone_id}_camera"
        ctx.image_pub.publish(msg)

        # Here you can also send frame to ORB-SLAM / VLM, etc.
        # For example, to create a Torch tensor:
        # import torch
        # tensor = torch.from_numpy(frame).permute(2, 0, 1).unsqueeze(0).to("cuda", non_blocking=True).float() / 255.0
        # result = your_vlm_model(tensor)

    # -------------------------------------------------------
    # Cleanup
    # -------------------------------------------------------
    def destroy_node(self):
        for ctx in self.drones.values():
            ctx.pipeline.set_state(Gst.State.NULL)
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = MultiDroneVideo()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
