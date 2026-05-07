#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
from pathlib import Path

import rclpy
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from rclpy.qos import qos_profile_sensor_data

from sparx_agency.robots.XTEND.xtend_online_bridge_base import OnlineXtendBridgeBase
from sparx_agency.robots.XTEND.xtend_rtsp_image_publisher import LatestFrameGrabber


class OnlineNavBridgePublisher(OnlineXtendBridgeBase):
    """Online XTEND bridge that publishes cropped RTSP frames to /xtend/image_raw."""

    def __init__(
        self,
        host: str,
        port: int,
        frequency: float,
        robot_uid: str,
        rtsp_uri: str,
        *,
        image_topic: str = "/xtend/image_raw",
        frame_id: str = "xtend_camera",
        backend: str = "gstreamer",
        crop_left: int = 108,
        crop_top: int = 70,
        crop_width: int = 504,
        crop_height: int = 280,
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
        self.frame_id = frame_id

        self.crop_left = int(crop_left)
        self.crop_top = int(crop_top)
        self.crop_width = int(crop_width)
        self.crop_height = int(crop_height)

        self.bridge = CvBridge()
        self.image_pub = self.ros_node.create_publisher(
            Image,
            self.image_topic,
            qos_profile_sensor_data,
        )

        self.grabber = LatestFrameGrabber(rtsp_uri, backend=backend)
        self.grabber.start()

        print(f"[image] RTSP: {self.rtsp_uri}")
        print(f"[image] topic: {self.image_topic}")
        print(
            f"[image] crop: x={self.crop_left}:{self.crop_left + self.crop_width}, "
            f"y={self.crop_top}:{self.crop_top + self.crop_height}"
        )

    async def image_publish_loop(self):
        sleep_time = 1.0 / max(self.frequency, 1e-6)
        print("✓ Image publisher active")

        while True:
            frame, _stamp = self.grabber.get_latest()

            if frame is not None:
                h, w = frame.shape[:2]
                x0 = self.crop_left
                y0 = self.crop_top
                x1 = min(x0 + self.crop_width, w)
                y1 = min(y0 + self.crop_height, h)

                if x0 < w and y0 < h and x1 > x0 and y1 > y0:
                    frame = frame[y0:y1, x0:x1].copy()

                msg = self.bridge.cv2_to_imgmsg(frame, encoding="bgr8")
                msg.header.stamp = self.ros_node.get_clock().now().to_msg()
                msg.header.frame_id = self.frame_id
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
    p.add_argument("--frequency", type=float, default=15.0)
    p.add_argument("--robot-uid", default="drnb177ede2")
    p.add_argument("--rtsp-uri", default="rtsp://192.0.0.15:8510/active_drone_fpv")

    p.add_argument("--image-topic", default="/xtend/image_raw")
    p.add_argument("--frame-id", default="xtend_camera")
    p.add_argument("--backend", choices=["ffmpeg", "gstreamer", "default"], default="gstreamer")

    p.add_argument("--crop-left", type=int, default=108)
    p.add_argument("--crop-top", type=int, default=70)
    p.add_argument("--crop-width", type=int, default=504)
    p.add_argument("--crop-height", type=int, default=280)

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
        frame_id=args.frame_id,
        backend=args.backend,
        crop_left=args.crop_left,
        crop_top=args.crop_top,
        crop_width=args.crop_width,
        crop_height=args.crop_height,
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
