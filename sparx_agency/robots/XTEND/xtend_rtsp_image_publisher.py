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
from rclpy.qos import qos_profile_sensor_data


def load_camera_info_from_yaml(yaml_path: str, frame_id: str) -> CameraInfo:
    with open(yaml_path, "r") as f:
        data = yaml.safe_load(f)

    msg = CameraInfo()
    msg.header.frame_id = frame_id

    msg.width = int(data["image_width"])
    msg.height = int(data["image_height"])

    msg.distortion_model = data.get("distortion_model", "plumb_bob")
    msg.d = list(data["distortion_coefficients"]["data"])
    msg.k = list(data["camera_matrix"]["data"])
    msg.r = list(data["rectification_matrix"]["data"])
    msg.p = list(data["projection_matrix"]["data"])

    return msg

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

        self.gst_pipeline = None
        self.gst_appsink = None
        self.gst_available = False

        if backend == "gstreamer":
            self.open_gstreamer_native(uri)
        else:
            self.cap = self.open_capture(uri, backend)

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
            f"rtspsrc location={uri} latency=0 protocols=tcp ! "
            "rtph264depay ! h264parse ! avdec_h264 ! videoconvert ! "
            "video/x-raw,format=BGR ! "
            "appsink name=appsink emit-signals=false sync=false max-buffers=1 drop=true"
        )

        # Retry loop to wait for the drone
        connected = False
        while not connected:
            pipeline = Gst.parse_launch(pipeline_str)
            appsink = pipeline.get_by_name("appsink")

            ret = pipeline.set_state(Gst.State.PLAYING)

            # Check if the pipeline actually started
            if ret != Gst.StateChangeReturn.FAILURE:
                print(f"✓ GStreamer connected to {uri}")
                connected = True
                self.Gst = Gst
                self.gst_pipeline = pipeline
                self.gst_appsink = appsink
                self.gst_available = True
            else:
                pipeline.set_state(Gst.State.NULL)
                print(f"Waiting for drone RTSP stream at {uri}...")
                time.sleep(2.0)  # Wait before retrying

    def start(self):
        self.running = True

        if self.backend == "gstreamer":
            self.thread = threading.Thread(target=self.gstreamer_loop, daemon=True)
        else:
            self.thread = threading.Thread(target=self.opencv_loop, daemon=True)

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
        """Modified loop with a non-blocking check and auto-reconnect."""
        last_frame_time = time.time()
        print("Starting GStreamer frame consumer loop...")

        while self.running:
            # We use 'try-pull-sample' with a timeout in nanoseconds (e.g., 0.1s)
            # This prevents the loop from hanging if the drone is off
            timeout_ns = 100 * 1000 * 1000  # 100ms
            sample = self.gst_appsink.emit("try-pull-sample", timeout_ns)

            if sample is None:
                # If no frame for 3 seconds, the stream is likely dead/not started
                if time.time() - last_frame_time > 3.0:
                    print(f"[Watchdog] No RTSP data for 3s. Re-triggering pipeline...")
                    self.gst_pipeline.set_state(self.Gst.State.NULL)
                    time.sleep(0.5)
                    self.gst_pipeline.set_state(self.Gst.State.PLAYING)
                    last_frame_time = time.time()  # Reset watchdog to wait for next attempt
                continue

            # Frame found! Process it.
            buf = sample.get_buffer()
            caps = sample.get_caps()
            structure = caps.get_structure(0)
            width = int(structure.get_value("width"))
            height = int(structure.get_value("height"))

            ok, mapinfo = buf.map(self.Gst.MapFlags.READ)
            if ok:
                try:
                    data = np.frombuffer(mapinfo.data, dtype=np.uint8)
                    frame = data.reshape((height, width, 3)).copy()

                    with self.lock:
                        self.latest_frame = frame
                        self.latest_stamp = time.time()

                    # Update watchdog
                    last_frame_time = time.time()
                finally:
                    buf.unmap(mapinfo)

    def reconnect_gstreamer(self):
        """Cleanly stops and restarts the pipeline."""
        if self.gst_pipeline:
            self.gst_pipeline.set_state(self.Gst.State.NULL)

        time.sleep(1.0)  # Small breather

        try:
            self.gst_pipeline.set_state(self.Gst.State.PLAYING)
            print("GStreamer pipeline signaled to restart.")
        except Exception as e:
            print(f"Reconnect failed: {e}")

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
        self.pub = self.create_publisher(
            Image,
            args.image_topic,
            qos_profile_sensor_data,
        )

        self.camera_info_pub = self.create_publisher(
            CameraInfo,
            args.camera_info_topic,
            qos_profile_sensor_data,
        )

        self.camera_info_msg = load_camera_info_from_yaml(
            args.camera_yaml,
            args.frame_id,
        )
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
        self.get_logger().info(f"Frame ID: {args.frame_id}")
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

        # Crop frame if crop parameters are enabled.
        if self.args.crop_width > 0 and self.args.crop_height > 0:
            x0 = self.args.crop_left
            y0 = self.args.crop_top
            x1 = x0 + self.args.crop_width
            y1 = y0 + self.args.crop_height

            h, w = frame.shape[:2]
            if x0 < 0 or y0 < 0 or x1 > w or y1 > h:
                self.get_logger().error(
                    f"Invalid crop: x={x0}:{x1}, y={y0}:{y1}, frame={w}x{h}",
                    throttle_duration_sec=2.0,
                )
                return

            frame = frame[y0:y1, x0:x1]

        msg = self.bridge.cv2_to_imgmsg(frame, encoding="bgr8")
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self.args.frame_id

        self.pub.publish(msg)

        # Publish matching CameraInfo with the same timestamp.
        cam_info = self.camera_info_msg
        cam_info.header.stamp = msg.header.stamp
        cam_info.header.frame_id = msg.header.frame_id
        self.camera_info_pub.publish(cam_info)

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
        "--camera-info-topic",
        default="/xtend/camera_info",
    )

    parser.add_argument(
        "--camera-yaml",
        default="/home/user/GIT/TheAgency/sparx_agency/robots/XTEND/config/camera_xtend_ros_calib_504_280_center_crop.yaml",
    )

    parser.add_argument("--crop-left", type=int, default=108)
    parser.add_argument("--crop-top", type=int, default=70)
    parser.add_argument("--crop-width", type=int, default=504)
    parser.add_argument("--crop-height", type=int, default=280)

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