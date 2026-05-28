#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
from pathlib import Path

import cv2
import numpy as np

import rclpy
from cv_bridge import CvBridge
from sensor_msgs.msg import Image, CameraInfo
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

from sparx_agency.robots.XTEND.xtend_online_bridge_base import OnlineXtendBridgeBase
from sparx_agency.robots.XTEND.xtend_rtsp_image_publisher import LatestFrameGrabber
from sparx_agency.robots.common.helpers import load_camera_info_from_yaml
from sparx_agency.robots.common.image_utils import pad_width_center, center_crop_resize


class OnlineNavBridgePublisher(OnlineXtendBridgeBase):
    """Online XTEND bridge that publishes preprocessed RTSP frames and matching CameraInfo.

    preprocess_mode="pad"         — pad width to pad_to_width (DA3 METRIC LARGE, default)
    preprocess_mode="crop_resize" — center-crop then resize to output_width x output_height
    """

    def __init__(
        self,
        host: str,
        port: int,
        frequency: float,
        robot_uid: str,
        rtsp_uri: str,
        *,
        image_topic: str = "/xtend/rgb",
        camera_info_topic: str = "/xtend/camera_info",
        camera_info_yaml: str | Path = "",
        frame_id: str = "xtend_camera",
        backend: str = "gstreamer",
        preprocess_mode: str = "pad",
        pad_to_width: int = 728,
        crop_width: int = 540,
        crop_height: int = 420,
        output_width: int = 504,
        output_height: int = 392,
        drop_bad_frames: bool = True,
        bad_frame_mean_min: float = 2.0,
        bad_frame_std_min: float = 1.0,
        bad_frame_sample_step: int = 16,
        bad_frame_log_every: int = 30,
        telemetry_topic: str = "/xtend/local_telemetry",
        bearing_topic: str = "/xtend/bearing",
        telemetry_frame_id: str = "odom",
        telemetry_child_frame_id: str = "xtend_camera",
        log_dir: str | Path = "./xtend_online_publisher_logs",
    ):
        super().__init__(
            host=host,
            port=port,
            frequency=frequency,
            robot_uid=robot_uid,
            telemetry_topic=telemetry_topic,
            bearing_topic=bearing_topic,
            telemetry_frame_id=telemetry_frame_id,
            telemetry_child_frame_id=telemetry_child_frame_id,
            log_dir=log_dir,
        )

        if preprocess_mode not in ("pad", "crop_resize", "resize"):
            raise ValueError(
                f"preprocess_mode must be 'pad', 'crop_resize', or 'resize', got: {preprocess_mode!r}"
            )

        self.rtsp_uri = rtsp_uri
        self.image_topic = image_topic
        self.camera_info_topic = camera_info_topic
        self.camera_info_yaml = Path(camera_info_yaml).expanduser() if camera_info_yaml else None
        self.frame_id = frame_id
        self.preprocess_mode = preprocess_mode

        self.pad_to_width = int(pad_to_width)
        self.crop_width = int(crop_width)
        self.crop_height = int(crop_height)
        self.output_width = int(output_width)
        self.output_height = int(output_height)

        self.drop_bad_frames = bool(drop_bad_frames)
        self.bad_frame_mean_min = float(bad_frame_mean_min)
        self.bad_frame_std_min = float(bad_frame_std_min)
        self.bad_frame_sample_step = max(1, int(bad_frame_sample_step))
        self.bad_frame_log_every = max(1, int(bad_frame_log_every))
        self.bad_frame_count = 0
        self.good_frame_count = 0

        self.bridge = CvBridge()

        image_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.BEST_EFFORT,
        )
        camera_info_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=5,
            reliability=ReliabilityPolicy.BEST_EFFORT,
        )

        self.image_pub = self.ros_node.create_publisher(
            Image,
            self.image_topic,
            image_qos,
        )

        self.camera_info_pub = None
        self.camera_info_msg: CameraInfo | None = None
        if self.camera_info_yaml is not None:
            if not self.camera_info_yaml.exists():
                raise FileNotFoundError(f"CameraInfo YAML not found: {self.camera_info_yaml}")
            self.camera_info_msg = load_camera_info_from_yaml(
                yaml_path=str(self.camera_info_yaml),
                frame_id=self.frame_id,
            )
            self.camera_info_pub = self.ros_node.create_publisher(
                CameraInfo,
                self.camera_info_topic,
                camera_info_qos,
            )

        self.grabber = LatestFrameGrabber(rtsp_uri, backend=backend)
        self.grabber.start()

        print(f"[image] RTSP: {self.rtsp_uri}")
        print(f"[image] topic: {self.image_topic}")
        print(f"[image] preprocess_mode: {self.preprocess_mode}")
        if self.preprocess_mode == "pad":
            print(f"[image] pad_to_width={self.pad_to_width}")
        elif self.preprocess_mode == "resize":
            print(f"[image] resize -> {self.output_width}x{self.output_height}")
        else:
            print(
                f"[image] crop={self.crop_width}x{self.crop_height} "
                f"-> {self.output_width}x{self.output_height}"
            )
        print(f"[image] camera_info: {'enabled' if self.camera_info_pub else 'disabled (no YAML)'}")
        print(
            "[image] bad-frame guard: "
            f"enabled={self.drop_bad_frames}, "
            f"mean_min={self.bad_frame_mean_min}, "
            f"std_min={self.bad_frame_std_min}, "
            f"sample_step={self.bad_frame_sample_step}"
        )

    def _preprocess(self, frame: np.ndarray) -> np.ndarray:
        if self.preprocess_mode == "pad":
            return pad_width_center(frame, self.pad_to_width)
        elif self.preprocess_mode == "resize":
            return cv2.resize(
                frame,
                (self.output_width, self.output_height),
                interpolation=cv2.INTER_AREA,
            )
        return center_crop_resize(
            frame,
            crop_width=self.crop_width,
            crop_height=self.crop_height,
            output_width=self.output_width,
            output_height=self.output_height,
        )

    def is_bad_frame(self, frame) -> tuple[bool, str]:
        """Return True for empty/flat frames that should not enter the pipeline."""
        if frame is None:
            return True, "frame is None"

        if not isinstance(frame, np.ndarray):
            return True, f"not ndarray: {type(frame).__name__}"

        if frame.size == 0:
            return True, "zero-size ndarray"

        if frame.ndim != 3 or frame.shape[2] != 3:
            return True, f"unexpected shape: {frame.shape}"

        h, w = frame.shape[:2]
        if h <= 0 or w <= 0:
            return True, f"invalid shape: {frame.shape}"

        small = frame[:: self.bad_frame_sample_step, :: self.bad_frame_sample_step]

        if not np.isfinite(small).all():
            return True, "contains non-finite values"

        mean_val = float(small.mean())
        std_val = float(small.std())

        if mean_val < self.bad_frame_mean_min:
            return True, f"too dark/empty: mean={mean_val:.3f}"

        if std_val < self.bad_frame_std_min:
            return True, f"too flat/empty: std={std_val:.3f}"

        return False, f"mean={mean_val:.3f}, std={std_val:.3f}"

    def should_publish_frame(self, frame) -> bool:
        if not self.drop_bad_frames:
            return True

        is_bad, reason = self.is_bad_frame(frame)
        if is_bad:
            self.bad_frame_count += 1
            self.good_frame_count = 0

            if self.bad_frame_count == 1 or self.bad_frame_count % self.bad_frame_log_every == 0:
                print(f"[image][drop] bad frame #{self.bad_frame_count}: {reason}")

            return False

        self.good_frame_count += 1
        if self.bad_frame_count > 0:
            print(
                f"[image] recovered after {self.bad_frame_count} dropped bad frame(s); "
                f"first good frame: {reason}"
            )
            self.bad_frame_count = 0

        return True

    async def image_publish_loop(self):
        sleep_time = 1.0 / max(self.frequency, 1e-6)
        print("✓ Image publisher active")

        while True:
            frame, _stamp = self.grabber.get_latest()

            if not self.should_publish_frame(frame):
                await asyncio.sleep(sleep_time)
                continue

            try:
                frame = self._preprocess(frame)
            except ValueError as exc:
                print(f"[image][preprocess] {exc}")
                await asyncio.sleep(sleep_time)
                continue

            now_msg = self.ros_node.get_clock().now().to_msg()

            msg = self.bridge.cv2_to_imgmsg(frame, encoding="bgr8")
            msg.header.stamp = now_msg
            msg.header.frame_id = self.frame_id
            self.image_pub.publish(msg)

            if self.camera_info_pub is not None and self.camera_info_msg is not None:
                self.camera_info_msg.header.stamp = now_msg
                self.camera_info_msg.header.frame_id = self.frame_id
                self.camera_info_pub.publish(self.camera_info_msg)

            await asyncio.sleep(sleep_time)

    def create_extra_tasks(self):
        return [asyncio.create_task(self.image_publish_loop())]

    def on_shutdown(self):
        self.grabber.stop()


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--host", default="192.0.0.15")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--frequency", type=float, default=10.0)
    p.add_argument("--robot-uid", default="drnb177ede2")
    p.add_argument("--rtsp-uri", default="rtsp://192.0.0.15:8510/active_drone_fpv")

    p.add_argument("--image-topic", default="/xtend/rgb")
    p.add_argument("--camera-info-topic", default="/xtend/camera_info")
    p.add_argument("--camera-info-yaml", default="")
    p.add_argument("--frame-id", default="xtend_camera")
    p.add_argument("--backend", choices=["ffmpeg", "gstreamer", "default"], default="gstreamer")

    p.add_argument(
        "--preprocess-mode",
        choices=["pad", "crop_resize", "resize"],
        default="resize",
        help="'pad': pad width to pad-to-width (DA3 LARGE, default). 'crop_resize': center-crop then resize.",
    )
    p.add_argument("--pad-to-width", type=int, default=728)
    p.add_argument("--crop-width", type=int, default=540)
    p.add_argument("--crop-height", type=int, default=420)
    p.add_argument("--output-width", type=int, default=504)
    p.add_argument("--output-height", type=int, default=294)

    p.add_argument("--no-drop-bad-frames", action="store_true")
    p.add_argument("--bad-frame-mean-min", type=float, default=2.0)
    p.add_argument("--bad-frame-std-min", type=float, default=1.0)
    p.add_argument("--bad-frame-sample-step", type=int, default=16)
    p.add_argument("--bad-frame-log-every", type=int, default=30)

    p.add_argument("--telemetry-topic", default="/xtend/local_telemetry")
    p.add_argument("--bearing-topic", default="/xtend/bearing")
    p.add_argument("--telemetry-frame-id", default="odom")
    p.add_argument("--telemetry-child-frame-id", default="xtend_camera")

    p.add_argument(
        "--log-dir",
        default=str(Path.home() / "Documents" / "online_publisher_node" / "logs"),
    )
    return p.parse_args()


async def async_main():
    args = parse_args()
    rclpy.init()

    bridge = OnlineNavBridgePublisher(
        host=args.host,
        port=args.port,
        frequency=args.frequency,
        robot_uid=args.robot_uid,
        rtsp_uri=args.rtsp_uri,
        image_topic=args.image_topic,
        camera_info_topic=args.camera_info_topic,
        camera_info_yaml=args.camera_info_yaml,
        frame_id=args.frame_id,
        backend=args.backend,
        preprocess_mode=args.preprocess_mode,
        pad_to_width=args.pad_to_width,
        crop_width=args.crop_width,
        crop_height=args.crop_height,
        output_width=args.output_width,
        output_height=args.output_height,
        drop_bad_frames=not args.no_drop_bad_frames,
        bad_frame_mean_min=args.bad_frame_mean_min,
        bad_frame_std_min=args.bad_frame_std_min,
        bad_frame_sample_step=args.bad_frame_sample_step,
        bad_frame_log_every=args.bad_frame_log_every,
        log_dir=args.log_dir,
    )

    await bridge.run_bridge()


def main():
    try:
        asyncio.run(async_main())
    except KeyboardInterrupt:
        print("\n[main] stopped by user")


if __name__ == "__main__":
    main()
