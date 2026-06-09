#!/usr/bin/env python3
import argparse
from pathlib import Path

import cv2
import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from sensor_msgs.msg import Image



class CalibFramePublisher(Node):
    def __init__(self, images_dir: str, ext: str, rate_hz: float):
        super().__init__("calib_frame_publisher")

        self.image_paths = sorted(Path(images_dir).expanduser().glob(f"*.{ext}"))
        if not self.image_paths:
            raise RuntimeError(f"No images found in {images_dir} with extension {ext}")

        self.pub = self.create_publisher(Image, "/xtend/image_raw", 10)
        self.bridge = CvBridge()
        self.idx = 0

        period = 1.0 / rate_hz
        self.timer = self.create_timer(period, self.tick)

        self.get_logger().info(f"Loaded {len(self.image_paths)} images")

    def tick(self):
        path = self.image_paths[self.idx]
        bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)

        if bgr is None:
            self.get_logger().warn(f"Could not read {path}")
            self.idx = (self.idx + 1) % len(self.image_paths)
            return

        msg = self.bridge.cv2_to_imgmsg(bgr, encoding="bgr8")
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "xtend_camera"

        self.pub.publish(msg)

        self.get_logger().info(f"Published {self.idx}: {path.name}")
        self.idx = (self.idx + 1) % len(self.image_paths)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--images-dir", required=True)
    parser.add_argument("--ext", default="jpg")
    parser.add_argument("--rate-hz", type=float, default=2.0)
    args = parser.parse_args()

    rclpy.init()
    node = CalibFramePublisher(args.images_dir, args.ext, args.rate_hz)
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()