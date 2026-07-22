#!/usr/bin/env python3
"""dir_watch_path_publisher.py

Watches a local directory for new files (by mtime) and publishes each one's
absolute path on a ROS2 String topic, in the same "{path} {sec} {nanosec}"
format used across the XTEND/ROBOTICAN pipelines
(online_nav_bridge_dir_publisher.py, rooster_frame_dir_publisher.py).

Generic — used wherever a producer's files land on this machine via some
other means (e.g. dir_push_relay.py transferring them from a different
machine) and a downstream consumer (depth_processor_node.py,
localization_node.py) just needs the path once the file is actually
present, exactly as it already does for a locally-written frame.

Single responsibility: detect new files, publish their path. Does not
write, transfer, or otherwise produce the files itself.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import rclpy
from rclpy.node import Node
from std_msgs.msg import String


class DirWatchPathPublisher(Node):
    def __init__(self, args):
        super().__init__("dir_watch_path_publisher")

        self.watch_dir = Path(args.watch_dir).expanduser().resolve()
        self.watch_dir.mkdir(parents=True, exist_ok=True)
        self.pattern = args.pattern
        self.max_files_kept = max(0, int(args.max_files_kept))
        self.path_pub = self.create_publisher(String, args.path_topic, 10)

        self._last_mtime = 0.0 if args.include_existing else time.time()

        self.create_timer(args.poll_interval, self._on_timer)

        self.get_logger().info(
            f"dir_watch_path_publisher ready\n"
            f"  watch_dir:   {self.watch_dir}\n"
            f"  pattern:     {self.pattern}\n"
            f"  path topic:  {args.path_topic}"
        )

    def _on_timer(self):
        files = [
            p for p in self.watch_dir.glob(self.pattern)
            if p.is_file() and p.stat().st_mtime > self._last_mtime
        ]
        files.sort(key=lambda p: p.stat().st_mtime)

        for path in files:
            self._last_mtime = path.stat().st_mtime
            ros_stamp = self.get_clock().now().to_msg()
            msg = String()
            msg.data = f"{path} {ros_stamp.sec} {ros_stamp.nanosec}"
            self.path_pub.publish(msg)

        if self.max_files_kept > 0:
            existing = sorted(self.watch_dir.glob(self.pattern))
            for old in existing[: max(0, len(existing) - self.max_files_kept)]:
                old.unlink(missing_ok=True)

def parse_args():
    p = argparse.ArgumentParser(
        description="Watch a directory for new files and publish each path on a ROS2 topic.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--watch-dir", required=True)
    p.add_argument("--pattern", default="*.jpg")
    p.add_argument("--path-topic", required=True)
    p.add_argument("--poll-interval", type=float, default=0.05)
    p.add_argument("--max-files-kept", type=int, default=30,
                    help="Rolling window: delete oldest files beyond this count (0=keep all)")
    p.add_argument("--include-existing", action="store_true",
                    help="Publish files already present in --watch-dir at startup (default: only new arrivals)")
    return p.parse_args()


def main():
    args = parse_args()
    rclpy.init()
    node = DirWatchPathPublisher(args)
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, rclpy.executors.ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
