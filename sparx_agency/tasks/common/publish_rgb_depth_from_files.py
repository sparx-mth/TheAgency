#!/usr/bin/env python3

import argparse
from pathlib import Path

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from sensor_msgs.msg import Image


def find_rgb_files(rgb_dir: Path, extensions: list[str]) -> list[Path]:
    paths = []
    for ext in extensions:
        paths.extend(rgb_dir.glob(f"*.{ext}"))
        paths.extend(rgb_dir.glob(f"*.{ext.upper()}"))
    return sorted(set(paths))


class RgbDepthFilePublisher(Node):
    def __init__(self, args):
        super().__init__("rgb_depth_file_publisher")

        self.args = args
        self.bridge = CvBridge()

        self.rgb_dir = Path(args.rgb_dir).expanduser()
        self.depth_dir = Path(args.depth_dir).expanduser()

        if not self.rgb_dir.exists():
            raise RuntimeError(f"RGB directory does not exist: {self.rgb_dir}")
        if not self.depth_dir.exists():
            raise RuntimeError(f"Depth directory does not exist: {self.depth_dir}")

        self.rgb_paths = find_rgb_files(
            self.rgb_dir,
            extensions=args.rgb_extensions,
        )

        if not self.rgb_paths:
            raise RuntimeError(f"No RGB files found in: {self.rgb_dir}")

        self.depth_paths = []
        self.valid_rgb_paths = []

        for rgb_path in self.rgb_paths:
            depth_path = self.depth_dir / f"{rgb_path.stem}.npy"
            if depth_path.exists():
                self.valid_rgb_paths.append(rgb_path)
                self.depth_paths.append(depth_path)
            else:
                self.get_logger().warn(f"No matching depth for {rgb_path.name}: {depth_path}")

        self.rgb_paths = self.valid_rgb_paths

        if not self.rgb_paths:
            raise RuntimeError("No matching RGB/depth pairs found.")

        if args.end_idx >= 0:
            self.rgb_paths = self.rgb_paths[args.start_idx : args.end_idx : args.step]
            self.depth_paths = self.depth_paths[args.start_idx : args.end_idx : args.step]
        else:
            self.rgb_paths = self.rgb_paths[args.start_idx :: args.step]
            self.depth_paths = self.depth_paths[args.start_idx :: args.step]

        if not self.rgb_paths:
            raise RuntimeError("No frames left after start/end/step filtering.")

        self.rgb_pub = self.create_publisher(Image, args.rgb_topic, 10)
        self.depth_pub = self.create_publisher(Image, args.depth_topic, 10)

        self.idx = 0
        self.finished_once = False

        period = 1.0 / max(args.publish_hz, 1e-6)
        self.timer = self.create_timer(period, self.timer_cb)

        self.get_logger().info(f"RGB dir: {self.rgb_dir}")
        self.get_logger().info(f"Depth dir: {self.depth_dir}")
        self.get_logger().info(f"Publishing {len(self.rgb_paths)} RGB-D pairs")
        self.get_logger().info(f"RGB topic: {args.rgb_topic}")
        self.get_logger().info(f"Depth topic: {args.depth_topic}")
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
                return

        rgb_path = self.rgb_paths[self.idx]
        depth_path = self.depth_paths[self.idx]

        bgr = cv2.imread(str(rgb_path), cv2.IMREAD_COLOR)
        if bgr is None:
            self.get_logger().warn(f"Could not read RGB: {rgb_path}")
            self.idx += 1
            return

        try:
            depth_m = np.load(depth_path).astype(np.float32)
        except Exception as exc:
            self.get_logger().warn(f"Could not read depth {depth_path}: {exc}")
            self.idx += 1
            return

        if depth_m.ndim == 3:
            if depth_m.shape[-1] == 1:
                depth_m = depth_m[..., 0]
            else:
                self.get_logger().warn(f"Depth has unexpected shape: {depth_m.shape}")
                self.idx += 1
                return

        depth_m[~np.isfinite(depth_m)] = 0.0
        depth_m[depth_m < 0.0] = 0.0

        stamp = self.get_clock().now().to_msg()

        rgb_msg = self.bridge.cv2_to_imgmsg(bgr, encoding="bgr8")
        rgb_msg.header.stamp = stamp
        rgb_msg.header.frame_id = self.args.frame_id

        depth_msg = self.bridge.cv2_to_imgmsg(depth_m, encoding="32FC1")
        depth_msg.header.stamp = stamp
        depth_msg.header.frame_id = self.args.frame_id

        self.rgb_pub.publish(rgb_msg)
        self.depth_pub.publish(depth_msg)

        if self.idx % max(int(self.args.publish_hz), 1) == 0:
            h, w = bgr.shape[:2]
            dh, dw = depth_m.shape[:2]
            valid = depth_m[np.isfinite(depth_m) & (depth_m > 0.0)]
            if valid.size > 0:
                median_depth = float(np.median(valid))
            else:
                median_depth = float("nan")

            self.get_logger().info(
                f"published idx={self.idx} "
                f"rgb={w}x{h} depth={dw}x{dh} "
                f"median_depth={median_depth:.3f}m "
                f"name={rgb_path.name}"
            )

        self.idx += 1


def parse_args():
    parser = argparse.ArgumentParser(
        description="Publish RGB images and .npy metric depth files as synchronized ROS 2 Image topics."
    )

    parser.add_argument("--rgb-dir", required=True)
    parser.add_argument("--depth-dir", required=True)

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
        "--depth-topic",
        default="/xtend/depth_m",
    )
    parser.add_argument(
        "--frame-id",
        default="xtend_camera",
    )
    parser.add_argument(
        "--publish-hz",
        type=float,
        default=1.0,
    )

    parser.add_argument("--start-idx", type=int, default=0)
    parser.add_argument("--end-idx", type=int, default=-1)
    parser.add_argument("--step", type=int, default=2)
    parser.add_argument("--loop", action="store_true")

    return parser.parse_args()


def main():
    args = parse_args()

    rclpy.init()
    node = RgbDepthFilePublisher(args)

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()