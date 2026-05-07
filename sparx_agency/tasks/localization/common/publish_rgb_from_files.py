#!/usr/bin/env python3

import argparse
from pathlib import Path
import cv2
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

        self.rgb_paths = find_rgb_files(self.rgb_dir, extensions=args.rgb_extensions)

        # Apply filtering
        if args.end_idx >= 0:
            self.rgb_paths = self.rgb_paths[args.start_idx: args.end_idx: args.step]
        else:
            self.rgb_paths = self.rgb_paths[args.start_idx:: args.step]

        if not self.rgb_paths:
            raise RuntimeError("No frames found after filtering.")

        # --- ULTRA-CACHE: PRE-CONVERT TO ROS MESSAGES ---
        self.get_logger().info(f"Caching {len(self.rgb_paths)} frames to RAM...")
        self.msg_cache = []

        for path in self.rgb_paths:
            bgr = cv2.imread(str(path))
            if bgr is not None:
                # Convert once during init to save CPU during the timer callback
                msg = self.bridge.cv2_to_imgmsg(bgr, encoding="bgr8")
                msg.header.frame_id = self.args.frame_id
                self.msg_cache.append(msg)
            else:
                self.get_logger().warn(f"Could not load: {path}")

        self.cam_info_msg = self.prepare_camera_info()
        self.rgb_pub = self.create_publisher(Image, args.rgb_topic, 10)
        self.cam_info_pub = self.create_publisher(CameraInfo, args.camera_info_topic, 10)

        self.idx = 0
        period = 1.0 / max(args.publish_hz, 1e-6)
        self.timer = self.create_timer(period, self.timer_cb)
        self.get_logger().info(f"RAM Cache Ready. Publishing at {args.publish_hz} Hz.")

    def prepare_camera_info(self):
        """Pre-configure the static CameraInfo message."""
        msg = CameraInfo()
        msg.header.frame_id = self.args.frame_id
        msg.width, msg.height = 720, 420
        msg.distortion_model = "plumb_bob"
        msg.d = [-0.2971784717997778, 0.08010222870361268, -0.0037003783730540046, -0.000627696838234576, 0.0]
        msg.k = [460.9072976392783, 0.0, 345.80685226685307, 0.0, 461.9847581630249, 128.61455823829436, 0.0, 0.0, 1.0]
        msg.r = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
        msg.p = [361.52381185798737, 0.0, 337.3434895805878, 0.0, 0.0, 410.764442594862, 116.76308616209292, 0.0, 0.0,
                 0.0, 1.0, 0.0]
        return msg

    def timer_cb(self):
        """Ultra-fast callback: Only updates timestamps and publishes."""
        if self.idx >= len(self.msg_cache):
            if self.args.loop:
                self.idx = 0
            else:
                self.timer.cancel()
                return

        # Fetch pre-converted message
        rgb_msg = self.msg_cache[self.idx]

        # Sync timestamps
        now = self.get_clock().now().to_msg()
        rgb_msg.header.stamp = now
        self.cam_info_msg.header.stamp = now

        # Publish
        self.rgb_pub.publish(rgb_msg)
        self.cam_info_pub.publish(self.cam_info_msg)

        self.idx += 1


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rgb-dir", required=True)
    parser.add_argument("--rgb-extensions", nargs="+", default=["jpg", "jpeg", "png"])
    parser.add_argument("--rgb-topic", default="/xtend/rgb")
    parser.add_argument("--camera-info-topic", default="/xtend/camera_info")
    parser.add_argument("--frame-id", default="xtend_camera")
    parser.add_argument("--publish-hz", type=float, default=10.0)
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
        rclpy.shutdown()


if __name__ == "__main__":
    main()