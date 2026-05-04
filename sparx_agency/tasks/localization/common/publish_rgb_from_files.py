#!/usr/bin/env python3

import argparse
from pathlib import Path

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo


def find_rgb_files(rgb_dir: Path, extensions: list[str]) -> list[Path]:
    paths = []
    for ext in extensions:
        paths.extend(rgb_dir.glob(f"*.{ext}"))
        paths.extend(rgb_dir.glob(f"*.{ext.upper()}"))
    return sorted(set(paths))


class RgbFilePublisher(Node):
    def __init__(self, args):
        super().__init__("rgb_file_publisher")

        self.args = args
        self.bridge = CvBridge()

        self.rgb_dir = Path(args.rgb_dir).expanduser()

        if not self.rgb_dir.exists():
            raise RuntimeError(f"RGB directory does not exist: {self.rgb_dir}")

        self.rgb_paths = find_rgb_files(
            self.rgb_dir,
            extensions=args.rgb_extensions,
        )

        if not self.rgb_paths:
            raise RuntimeError(f"No RGB files found in: {self.rgb_dir}")

        if args.end_idx >= 0:
            self.rgb_paths = self.rgb_paths[args.start_idx : args.end_idx : args.step]
        else:
            self.rgb_paths = self.rgb_paths[args.start_idx :: args.step]

        if not self.rgb_paths:
            raise RuntimeError("No frames left after start/end/step filtering.")

        self.rgb_pub = self.create_publisher(Image, args.rgb_topic, 10)

        # CameraInfo Publisher and Message Setup
        self.cam_info_pub = self.create_publisher(CameraInfo, args.camera_info_topic, 10)
        self.cam_info_msg = CameraInfo()
        self.cam_info_msg.header.frame_id = self.args.frame_id
        
        # Image resolution
        self.cam_info_msg.width = 720
        self.cam_info_msg.height = 420
        
        # Distortion model
        self.cam_info_msg.distortion_model = "plumb_bob"
        self.cam_info_msg.d = [-0.2971784717997778, 0.08010222870361268, -0.0037003783730540046, -0.000627696838234576, 0.0]
        
        # Camera matrix (K)
        self.cam_info_msg.k = [
            460.9072976392783, 0.0, 345.80685226685307,
            0.0, 461.9847581630249, 128.61455823829436,
            0.0, 0.0, 1.0
        ]
        
        # Rectification matrix (R)
        self.cam_info_msg.r = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
        
        # Projection matrix (P)
        self.cam_info_msg.p = [
            361.52381185798737, 0.0, 337.3434895805878, 0.0,
            0.0, 410.764442594862, 116.76308616209292, 0.0,
            0.0, 0.0, 1.0, 0.0
        ]

        self.idx = 0
        self.finished_once = False

        period = 1.0 / max(args.publish_hz, 1e-6)
        self.timer = self.create_timer(period, self.timer_cb)

        self.get_logger().info(f"RGB dir: {self.rgb_dir}")
        self.get_logger().info(f"Publishing {len(self.rgb_paths)} RGB frames")
        self.get_logger().info(f"RGB topic: {args.rgb_topic}")
        self.get_logger().info(f"CamInfo topic: {args.camera_info_topic}")
        self.get_logger().info(f"Publish Hz: {args.publish_hz}")
        self.get_logger().info(f"Loop: {args.loop}")

    def timer_cb(self):
        if self.idx >= len(self.rgb_paths):
            if self.args.loop:
                self.idx = 0
                self.finished_once = True
                self.get_logger().info("Looping back to first frame")
            else:
                self.get_logger().info("Finished publishing all frames")
                self.timer.cancel()
                rclpy.shutdown()
                return

        rgb_path = self.rgb_paths[self.idx]

        bgr = cv2.imread(str(rgb_path), cv2.IMREAD_COLOR)
        if bgr is None:
            self.get_logger().warn(f"Could not read RGB: {rgb_path}")
            self.idx += 1
            return

        stamp = self.get_clock().now().to_msg()

        # Publish CameraInfo with the exact same timestamp
        self.cam_info_msg.header.stamp = stamp
        self.cam_info_pub.publish(self.cam_info_msg)

        rgb_msg = self.bridge.cv2_to_imgmsg(bgr, encoding="bgr8")
        rgb_msg.header.stamp = stamp
        rgb_msg.header.frame_id = self.args.frame_id

        self.rgb_pub.publish(rgb_msg)

        if self.idx % max(int(self.args.publish_hz), 1) == 0:
            h, w = bgr.shape[:2]
            self.get_logger().info(
                f"published idx={self.idx} "
                f"rgb={w}x{h} "
                f"name={rgb_path.name}"
            )

        self.idx += 1


def parse_args():
    parser = argparse.ArgumentParser(
        description="Publish RGB images as synchronized ROS 2 Image topics."
    )

    parser.add_argument("--rgb-dir", required=True)

    parser.add_argument(
        "--rgb-extensions",
        nargs="+",
        default=["jpg", "jpeg", "png"],
    )

    parser.add_argument(
        "--rgb-topic",
        default="/xtend/rgb",
    )
    
    parser.add_argument(
        "--camera-info-topic",
        default="/xtend/camera_info",
    )
    
    parser.add_argument(
        "--frame-id",
        default="xtend_camera",
    )
    parser.add_argument(
        "--publish-hz",
        type=float,
        default=10.0,
    )

    parser.add_argument("--start-idx", type=int, default=0)
    parser.add_argument("--end-idx", type=int, default=-1)
    parser.add_argument("--step", type=int, default=1)
    parser.add_argument("--loop", action="store_true")

    return parser.parse_args()


def main():
    args = parse_args()

    rclpy.init()
    node = RgbFilePublisher(args)

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()