#!/usr/bin/env python3

import argparse
import threading
import time

import cv2
import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from sensor_msgs.msg import Image
from sensor_msgs.srv import SetCameraInfo


class LatestFrameGrabber:
    def __init__(self, uri: str, backend: str):
        self.uri = uri
        self.backend = backend
        self.cap = self.open_capture(uri, backend)

        self.lock = threading.Lock()
        self.latest_frame = None
        self.latest_stamp = 0.0
        self.running = False
        self.thread = None

    def open_capture(self, uri: str, backend: str):
        if backend == "ffmpeg":
            cap = cv2.VideoCapture(uri, cv2.CAP_FFMPEG)
        elif backend == "gstreamer":
            pipeline = (
                f"rtspsrc location={uri} latency=0 drop-on-latency=true ! "
                "rtph264depay ! h264parse ! avdec_h264 ! "
                "videoconvert ! video/x-raw,format=BGR ! "
                "appsink sync=false max-buffers=1 drop=true"
            )
            cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
        else:
            cap = cv2.VideoCapture(uri)

        if not cap.isOpened():
            raise RuntimeError(f"Could not open RTSP stream with backend={backend}: {uri}")

        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        return cap

    def start(self):
        self.running = True
        self.thread = threading.Thread(target=self.loop, daemon=True)
        self.thread.start()

    def loop(self):
        while self.running:
            ok, frame = self.cap.read()

            if not ok or frame is None:
                time.sleep(0.005)
                continue

            with self.lock:
                self.latest_frame = frame
                self.latest_stamp = time.time()

    def get_latest(self):
        with self.lock:
            if self.latest_frame is None:
                return None, 0.0
            return self.latest_frame.copy(), self.latest_stamp

    def stop(self):
        self.running = False

        if self.thread is not None:
            self.thread.join(timeout=1.0)

        if self.cap is not None:
            self.cap.release()


class RtspImagePublisher(Node):
    def __init__(self, args):
        super().__init__("xtend_rtsp_image_publisher_low_latency")

        self.args = args
        self.bridge = CvBridge()
        self.pub = self.create_publisher(Image, args.image_topic, 10)

        self.set_camera_info_srv = self.create_service(
            SetCameraInfo,
            "/camera/set_camera_info",
            self.handle_set_camera_info,
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
        self.get_logger().info(f"Publish Hz: {args.publish_hz}")
        self.get_logger().info("Serving /camera/set_camera_info")

    def handle_set_camera_info(self, request, response):
        response.success = True
        response.status_message = "Accepted by dummy calibration service"
        return response

    def publish_latest(self):
        frame, capture_stamp = self.grabber.get_latest()

        if frame is None:
            self.get_logger().warn("No frame yet", throttle_duration_sec=2.0)
            return

        age_ms = (time.time() - capture_stamp) * 1000.0

        msg = self.bridge.cv2_to_imgmsg(frame, encoding="bgr8")
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self.args.frame_id

        self.pub.publish(msg)

        self.frame_count += 1
        if self.frame_count % max(int(self.args.publish_hz), 1) == 0:
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
        default=3.0,
    )
    parser.add_argument(
        "--backend",
        choices=["ffmpeg", "gstreamer", "default"],
        default="ffmpeg",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    rclpy.init()
    node = RtspImagePublisher(args)

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()