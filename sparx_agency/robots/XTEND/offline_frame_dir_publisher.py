#!/usr/bin/env python3
"""
Offline frame replay publisher.

Reads saved JPEG/PNG frames from an input directory and replays them through
the same /xtend/frame_path topic used by OnlineNavBridgeDirPublisher, so the
full downstream pipeline (depth node, AprilTag, etc.) runs identically without
a live drone or RTSP feed.

Per-frame flow (mirrors the online bridge exactly):
  copy  -> out_dir/frame_XXXXXXXX.tmp   (source dir is never modified)
  rename-> out_dir/frame_XXXXXXXX.jpg   (atomic: visible only when complete)
  publish String("{abs_path} {sec} {nanosec}")
  cleanup out_dir rolling window (max_frames_kept)
"""
from __future__ import annotations

import argparse
import shutil
import time
from pathlib import Path

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from std_msgs.msg import String

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def _collect_frames(input_dir: Path) -> list[Path]:
    frames = sorted(
        p for p in input_dir.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS
    )
    if not frames:
        raise FileNotFoundError(f"No image frames found in: {input_dir}")
    return frames


class OfflineFrameDirPublisher(Node):
    """
    Replays frames from a saved directory through /xtend/frame_path at a
    configurable rate. Copies each frame to out_dir before publishing so the
    same rolling-cleanup mechanism as the online bridge applies.
    """

    def __init__(self, args):
        super().__init__("offline_frame_dir_publisher")

        self.input_dir = Path(args.input_dir).expanduser().resolve()
        if not self.input_dir.exists():
            raise FileNotFoundError(f"input_dir does not exist: {self.input_dir}")

        self.out_dir = Path(args.out_dir).expanduser().resolve()
        self.out_dir.mkdir(parents=True, exist_ok=True)

        self.path_topic = args.path_topic
        self.frequency = max(0.01, float(args.frequency))
        self.loop = bool(args.loop)
        self.max_frames_kept = max(0, int(args.max_frames_kept))

        if not args.no_clear_on_start:
            removed = 0
            for f in self.out_dir.glob("frame_*"):
                if f.suffix in (".jpg", ".tmp"):
                    f.unlink(missing_ok=True)
                    removed += 1
            if removed:
                self.get_logger().info(f"Cleared {removed} file(s) from {self.out_dir}")

        self._frames = _collect_frames(self.input_dir)
        self._index = 0
        self._seq = 0
        self._done = False

        qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
            reliability=ReliabilityPolicy.RELIABLE,
        )
        self._pub = self.create_publisher(String, self.path_topic, qos)

        period = 1.0 / self.frequency
        self._timer = self.create_timer(period, self._tick)

        self.get_logger().info(f"Offline replay: {len(self._frames)} frames from {self.input_dir}")
        self.get_logger().info(f"out_dir: {self.out_dir}  topic: {self.path_topic}  {self.frequency:.1f} Hz  loop={self.loop}")

    def _tick(self):
        if self._done:
            return

        if self._index >= len(self._frames):
            if self.loop:
                self._index = 0
                self.get_logger().info("Looping back to first frame")
            else:
                self.get_logger().info("All frames published. Stopping.")
                self._done = True
                return

        src = self._frames[self._index]
        self._index += 1
        self._seq += 1

        final_path = self.out_dir / f"frame_{self._seq:08d}.jpg"
        tmp_path = final_path.with_suffix(".tmp")

        try:
            shutil.copy2(str(src), str(tmp_path))
            tmp_path.rename(final_path)
        except Exception as e:
            self.get_logger().error(f"Copy/rename failed for {src.name}: {e}")
            return

        # Rolling cleanup of out_dir copies (source dir is untouched)
        if self.max_frames_kept > 0:
            existing = sorted(self.out_dir.glob("frame_*.jpg"))
            for old in existing[: max(0, len(existing) - self.max_frames_kept)]:
                old.unlink(missing_ok=True)

        ros_stamp = self.get_clock().now().to_msg()
        msg = String()
        msg.data = f"{final_path} {ros_stamp.sec} {ros_stamp.nanosec}"
        self._pub.publish(msg)

        self.get_logger().info(
            f"[{self._index}/{len(self._frames)}] {src.name}",
            throttle_duration_sec=2.0,
        )


def parse_args():
    p = argparse.ArgumentParser(
        description="Replay saved frames through /xtend/frame_path for offline pipeline testing."
    )
    p.add_argument(
        "--input-dir", required=True,
        help="Directory of saved JPEG/PNG frames to replay (never modified).",
    )
    p.add_argument(
        "--out-dir", default="/tmp/xtend_frames",
        help="Working directory where frame copies are written (same as online bridge).",
    )
    p.add_argument("--path-topic", default="/xtend/frame_path")
    p.add_argument("--frequency", type=float, default=10.0, help="Replay rate in Hz")
    p.add_argument("--loop", action="store_true", help="Loop the frame sequence indefinitely")
    p.add_argument(
        "--max-frames-kept", type=int, default=30,
        help="Rolling window: delete oldest copies beyond this count (0=keep all)",
    )
    p.add_argument(
        "--no-clear-on-start", action="store_true",
        help="Keep existing files in out_dir from a previous run",
    )
    return p.parse_args()


def main():
    args = parse_args()
    rclpy.init()
    node = None
    try:
        node = OfflineFrameDirPublisher(args)
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
