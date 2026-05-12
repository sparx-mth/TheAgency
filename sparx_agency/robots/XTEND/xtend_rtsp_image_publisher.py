#!/usr/bin/env python3

import argparse
import threading
import time
from typing import Optional, Tuple

import cv2
import numpy as np
import rclpy
import yaml
from cv_bridge import CvBridge
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from sensor_msgs.srv import SetCameraInfo

from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

from sparx_agency.robots.common.image_utils import pad_width_center


class LatestFrameGrabber:
    def __init__(self, uri: str, backend: str):
        self.uri = uri
        self.backend = backend

        self.lock = threading.Lock()
        self.latest_frame = None
        self.latest_stamp = 0.0

        self.running = False
        self.thread = None

        self.cap = None

        self.gst_lock = threading.Lock()
        self.gst_pipeline = None
        self.gst_appsink = None
        self.gst_available = False
        self.Gst = None
        #
        # if backend == "gstreamer":
        #     self.open_gstreamer_native(uri)
        # else:
        #     self.cap = self.open_capture(uri, backend)

    def open_capture(self, uri: str, backend: str):
        if backend == "ffmpeg":
            cap = cv2.VideoCapture(uri, cv2.CAP_FFMPEG)
        elif backend == "default":
            cap = cv2.VideoCapture(uri)
        else:
            raise RuntimeError(f"Unsupported OpenCV backend: {backend}")

        if not cap.isOpened():
            raise RuntimeError(f"Could not open RTSP stream with backend={backend}: {uri}")

        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        return cap

    def open_gstreamer_native(self, uri: str):
        try:
            import gi
            gi.require_version("Gst", "1.0")
            from gi.repository import Gst
        except Exception as exc:
            raise RuntimeError("Native GStreamer backend requires python gi bindings.") from exc

        Gst.init(None)

        pipeline_str = (
            f"rtspsrc location={uri} latency=100 protocols=tcp drop-on-latency=true ! "
            "rtph264depay ! h264parse ! avdec_h264 ! videoconvert ! "
            "video/x-raw,format=BGR ! "
            "appsink name=appsink emit-signals=false sync=false max-buffers=1 drop=true"
        )

        while True:
            pipeline = Gst.parse_launch(pipeline_str)
            appsink = pipeline.get_by_name("appsink")

            if appsink is None:
                pipeline.set_state(Gst.State.NULL)
                raise RuntimeError("Failed to create appsink element")

            ret = pipeline.set_state(Gst.State.PLAYING)

            if ret != Gst.StateChangeReturn.FAILURE:
                with self.gst_lock:
                    self.Gst = Gst
                    self.gst_pipeline = pipeline
                    self.gst_appsink = appsink
                    self.gst_available = True

                print(f"✓ GStreamer connected to {uri}")
                return

            pipeline.set_state(Gst.State.NULL)
            print(f"[RTSP] Waiting for drone RTSP stream at {uri}...")
            time.sleep(2.0)

    def start(self):
        if self.running:
            print("[RTSP] grabber already running")
            return

        self.running = True

        if self.backend == "gstreamer":
            self.open_gstreamer_native(self.uri)
            self.thread = threading.Thread(
                target=self.gstreamer_loop,
                daemon=True,
            )
            self.thread.start()
        else:
            self.cap = self.open_capture(self.uri, self.backend)
            self.thread = threading.Thread(
                target=self.opencv_loop,
                daemon=True,
            )
            self.thread.start()

    def opencv_loop(self):
        while self.running:
            ok, frame = self.cap.read()

            if not ok or frame is None:
                time.sleep(0.005)
                continue

            with self.lock:
                self.latest_frame = frame
                self.latest_stamp = time.time()

    def gstreamer_loop(self):
        last_frame_time = time.time()
        timeout_ns = int(0.5 * 1e9)
        print("Starting GStreamer frame consumer loop...")

        while self.running:
            with self.gst_lock:
                appsink = self.gst_appsink

            if appsink is None:
                time.sleep(0.1)
                continue

            sample = appsink.emit("try-pull-sample", timeout_ns)

            if sample is None:
                if time.time() - last_frame_time > 3.0:
                    print("[Watchdog] No RTSP data for 3s. Restarting pipeline...")
                    self.reconnect_gstreamer()
                    last_frame_time = time.time()
                else:
                    time.sleep(0.01)
                continue

            buf = sample.get_buffer()
            caps = sample.get_caps()
            structure = caps.get_structure(0)
            width = int(structure.get_value("width"))
            height = int(structure.get_value("height"))

            ok, mapinfo = buf.map(self.Gst.MapFlags.READ)
            if not ok:
                continue

            try:
                data = np.frombuffer(mapinfo.data, dtype=np.uint8)
                frame = data.reshape((height, width, 3)).copy()

                with self.lock:
                    self.latest_frame = frame
                    self.latest_stamp = time.time()

                last_frame_time = time.time()

            finally:
                buf.unmap(mapinfo)

    def reconnect_gstreamer(self):
        """Cleanly stops and restarts the pipeline."""
        if self.gst_pipeline:
            self.gst_pipeline.set_state(self.Gst.State.NULL)

        print("[RTSP] Restarting GStreamer pipeline...")

        self.close_gstreamer_native()
        time.sleep(0.5)
        self.open_gstreamer_native(self.uri)

    def close_gstreamer_native(self):
        with self.gst_lock:
            pipeline = self.gst_pipeline
            self.gst_pipeline = None
            self.gst_appsink = None
            self.gst_available = False

        if pipeline is not None:
            try:
                pipeline.set_state(self.Gst.State.NULL)
            except Exception as exc:
                print(f"[RTSP] Failed to close GStreamer pipeline: {exc}")

    def get_latest(self) -> Tuple[Optional[np.ndarray], float]:
        with self.lock:
            if self.latest_frame is None:
                return None, 0.0
            return self.latest_frame.copy(), self.latest_stamp

    def stop(self):
        self.running = False

        if self.thread is not None:
            self.thread.join(timeout=2.0)

        if self.cap is not None:
            self.cap.release()

        if self.gst_pipeline is not None:
            self.gst_pipeline.set_state(self.Gst.State.NULL)

        self.thread = None
        self.cap = None
        self.gst_pipeline = None
        self.gst_appsink = None





class RtspImagePublisher(Node):
    def __init__(self, args):
        super().__init__("xtend_rtsp_image_publisher_low_latency")

        self.args = args
        self.bridge = CvBridge()

        qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.BEST_EFFORT,
        )
        self.pub = self.create_publisher(
            Image,
            args.image_topic,
            qos,
        )


        self.grabber = LatestFrameGrabber(args.rtsp_uri, args.backend)
        self.grabber.start()

        self.frame_count = 0
        self.timer = self.create_timer(
            1.0 / max(args.publish_hz, 1e-6),
            self.publish_latest,
        )

        self.get_logger().info(f"RTSP: {args.rtsp_uri}")
        self.get_logger().info(f"Backend: {args.backend}")
        self.get_logger().info(f"Publishing latest frame to: {args.image_topic}")
        self.get_logger().info(f"Frame ID: {args.frame_id}")
        self.get_logger().info(f"Publish Hz: {args.publish_hz}")

    def publish_latest(self):
        frame, capture_stamp = self.grabber.get_latest()

        if frame is None:
            self.get_logger().warn("No frame yet", throttle_duration_sec=2.0)
            return

        age_ms = (time.time() - capture_stamp) * 1000.0

        try:
            frame = pad_width_center(frame, self.args.pad_to_width)
        except ValueError as exc:
            self.get_logger().error(str(exc), throttle_duration_sec=2.0)
            return

        msg = self.bridge.cv2_to_imgmsg(frame, encoding="bgr8")
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self.args.frame_id

        self.pub.publish(msg)
        self.frame_count += 1

        log_every = max(int(round(self.args.publish_hz)), 1)
        if self.frame_count % log_every == 0:
            h, w = frame.shape[:2]
            self.get_logger().info(
                f"published={self.frame_count}, size={w}x{h}, frame_age={age_ms:.1f} ms",
                throttle_duration_sec=2.0,
            )

    def destroy_node(self):
        self.get_logger().info("Stopping frame grabber...")
        self.grabber.stop()
        super().destroy_node()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--rtsp-uri",
        default="rtsp://192.0.0.15:8510/active_drone_fpv",
    )
    parser.add_argument(
        "--image-topic",
        default="/xtend/image_raw",
    )
    parser.add_argument(
        "--frame-id",
        default="xtend_camera",
    )
    parser.add_argument(
        "--publish-hz",
        type=float,
        default=15.0,
    )
    parser.add_argument(
        "--backend",
        choices=["gstreamer", "ffmpeg", "default"],
        default="gstreamer",
        help="gstreamer uses native gi.repository.Gst, not OpenCV CAP_GSTREAMER",
    )

    parser.add_argument(
        "--camera-yaml",
        default="/home/user/GIT/TheAgency/sparx_agency/robots/XTEND/config/camera_xtend_ros_calib_504_280_center_crop.yaml",
    )

    parser.add_argument("--pad-to-width", type=int, default=728)

    return parser.parse_args()


def main():
    args = parse_args()

    rclpy.init()
    node = None

    try:
        node = RtspImagePublisher(args)
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node is not None:
            node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()