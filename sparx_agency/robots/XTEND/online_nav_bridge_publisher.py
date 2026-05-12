#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
from pathlib import Path

import rclpy
from cv_bridge import CvBridge
from sensor_msgs.msg import Image, CameraInfo
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

from sparx_agency.robots.XTEND.xtend_online_bridge_base import OnlineXtendBridgeBase
from sparx_agency.robots.XTEND.xtend_rtsp_image_publisher import LatestFrameGrabber
from sparx_agency.robots.common.helpers import load_camera_info_from_yaml
from sparx_agency.robots.common.image_utils import pad_width_center, center_crop_resize


class OnlineNavBridgePublisher(OnlineXtendBridgeBase):
    """Online XTEND bridge that publishes cropped RTSP frames and matching CameraInfo."""

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
        pad_to_width: int = 728,
        crop_width: int = 540,
        crop_height: int = 420,
        output_width: int = 504,
        output_height: int = 392,
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

        self.rtsp_uri = rtsp_uri
        self.image_topic = image_topic
        self.camera_info_topic = camera_info_topic
        self.camera_info_yaml = Path(camera_info_yaml).expanduser()
        self.frame_id = frame_id

        self.pad_to_width = int(pad_to_width)
        self.crop_width = int(crop_width)
        self.crop_height = int(crop_height)
        self.output_width = int(output_width)
        self.output_height = int(output_height)

        self.bridge = CvBridge()
        image_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.BEST_EFFORT,
        )

        # camera_info_qos = QoSProfile(
        #     history=HistoryPolicy.KEEP_LAST,
        #     depth=5,
        #     reliability=ReliabilityPolicy.BEST_EFFORT,
        # )

        self.image_pub = self.ros_node.create_publisher(
            Image,
            self.image_topic,
            image_qos,
        )

        # self.camera_info_pub = self.ros_node.create_publisher(
        #     CameraInfo,
        #     self.camera_info_topic,
        #     camera_info_qos,
        # )
        # if not self.camera_info_yaml.exists():
        #     raise FileNotFoundError(f"CameraInfo YAML not found: {self.camera_info_yaml}")
        #
        # self.camera_info_msg = load_camera_info_from_yaml(
        #     yaml_path=str(self.camera_info_yaml),
        #     frame_id=self.frame_id,
        # )

        self.grabber = LatestFrameGrabber(rtsp_uri, backend=backend)
        self.grabber.start()

        print(f"[image] RTSP: {self.rtsp_uri}")
        print(f"[image] topic: {self.image_topic}")
        # print(f"[image] no crop, pad_to_width={self.pad_to_width}")


    async def image_publish_loop(self):
        sleep_time = 1.0 / max(self.frequency, 1e-6)
        print("✓ Image publisher active")

        while True:
            frame, _stamp = self.grabber.get_latest()

            if frame is not None:
                h, w = frame.shape[:2]
                # DA3 LARGEMETRIC: 728*420
                # frame = pad_width_center(frame, self.pad_to_width)
                # DA3 SMALL 504*392
                frame = center_crop_resize(
                    frame,
                    crop_width=self.crop_width,
                    crop_height=self.crop_height,
                    output_width=self.output_width,
                    output_height=self.output_height,
                )

                now_msg = self.ros_node.get_clock().now().to_msg()

                msg = self.bridge.cv2_to_imgmsg(frame, encoding="bgr8")
                msg.header.stamp = now_msg
                msg.header.frame_id = self.frame_id

                # self.camera_info_msg.header.stamp = now_msg
                # self.camera_info_msg.header.frame_id = self.frame_id

                # self.camera_info_pub.publish(self.camera_info_msg)
                self.image_pub.publish(msg)

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

    p.add_argument("--pad-to-width", type=int, default=728)


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
