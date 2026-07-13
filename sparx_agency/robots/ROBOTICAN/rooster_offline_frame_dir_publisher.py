#!/usr/bin/env python3
"""rooster_offline_frame_dir_publisher.py

Offline replay of a rooster_dome_main.py capture session — a faithful mock
of the live ROBOTICAN pipeline for testing downstream consumers (FALCON,
room_mapper, etc.) without Sphera/the drone. Reads the session's
<drone_id>_<timestamp>[_<seq>].{jpg,json,npy} triples and republishes them
on the exact same topics the live pipeline uses:
  /<rooster-id>/rgb_frame_path    (std_msgs/String, "{path} {sec} {nanosec}")
  /<rooster-id>/depth_frame_path  (std_msgs/String, same format)
  /<rooster-id>/localization      (geometry_msgs/PoseStamped)

RGB/depth files are copied into --rgb-out-dir/--depth-out-dir (atomic
tmp-then-rename, matching rooster_frame_dir_publisher.py's convention) so a
consumer mounting the same fixed directories (e.g. FALCON's container, per
run_falcon.sh) doesn't need to know whether it's talking to a live session
or a replay.

This does not re-run DA3 or localization - it replays their already-computed
outputs from the session directly, the same way XTEND's
offline_frame_dir_publisher.py works for its own capture format.
"""
from __future__ import annotations

import argparse
import json
import math
import shutil
from pathlib import Path
from typing import Optional

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import String


def _collect_triples(session_dir: Path) -> list[Path]:
    """Returns sorted .json sidecar paths that have a matching .jpg and .npy."""
    triples = []
    for json_path in sorted(session_dir.glob("*.json")):
        stem = json_path.stem
        jpg_path = session_dir / f"{stem}.jpg"
        npy_path = session_dir / f"{stem}.npy"
        if not jpg_path.is_file():
            print(f"[offline_replay] skipping {stem}: no matching .jpg")
            continue
        triples.append(json_path)
    return triples


class RoosterOfflineFrameDirPublisher(Node):
    def __init__(self, args):
        super().__init__("rooster_offline_frame_dir_publisher")

        self.session_dir = Path(args.session_dir).expanduser().resolve()
        if not self.session_dir.is_dir():
            raise FileNotFoundError(f"session-dir not found: {self.session_dir}")

        self.triples = _collect_triples(self.session_dir)
        if not self.triples:
            raise FileNotFoundError(f"no .jpg/.json triples found in: {self.session_dir}")

        self.rgb_out_dir = Path(args.rgb_out_dir).expanduser().resolve()
        self.depth_out_dir = Path(args.depth_out_dir).expanduser().resolve()
        self.rgb_out_dir.mkdir(parents=True, exist_ok=True)
        self.depth_out_dir.mkdir(parents=True, exist_ok=True)

        self.loop = bool(args.loop)
        self._index = 0
        self._seq = 0
        self._done_logged = False

        self.rgb_pub = self.create_publisher(String, args.rgb_path_topic, 10)
        self.depth_pub = self.create_publisher(String, args.depth_path_topic, 10)
        self.pose_pub = self.create_publisher(PoseStamped, args.pose_topic, 10)

        period = 1.0 / max(args.rate, 1e-6)
        self.create_timer(period, self._on_timer)

        self.get_logger().info(
            f"rooster_offline_frame_dir_publisher ready\n"
            f"  session:   {self.session_dir} ({len(self.triples)} frames)\n"
            f"  rgb out:   {self.rgb_pub.topic_name} <- {self.rgb_out_dir}\n"
            f"  depth out: {self.depth_pub.topic_name} <- {self.depth_out_dir}\n"
            f"  pose out:  {self.pose_pub.topic_name}\n"
            f"  loop:      {self.loop}"
        )

    def _publish_path(self, pub, src: Path, out_dir: Path, ext: str) -> None:
        self._seq += 1
        final_path = out_dir / f"frame_{self._seq:08d}{ext}"
        tmp_path = final_path.with_suffix(".tmp")
        shutil.copy2(src, tmp_path)
        tmp_path.rename(final_path)

        stamp = self.get_clock().now().to_msg()
        msg = String()
        msg.data = f"{final_path} {stamp.sec} {stamp.nanosec}"
        pub.publish(msg)

    def _publish_pose(self, pose: dict) -> None:
        yaw_rad = math.radians(float(pose.get("yaw", 0.0)))
        msg = PoseStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.pose.position.x = float(pose.get("x", 0.0))
        msg.pose.position.y = float(pose.get("y", 0.0))
        msg.pose.position.z = float(pose.get("z", 0.0))
        msg.pose.orientation.z = math.sin(yaw_rad / 2.0)
        msg.pose.orientation.w = math.cos(yaw_rad / 2.0)
        self.pose_pub.publish(msg)

    def _on_timer(self):
        if self._index >= len(self.triples):
            if self.loop:
                self._index = 0
                self.get_logger().info("replay: looping back to start")
            else:
                if not self._done_logged:
                    self.get_logger().info("replay complete - holding on last frame")
                    self._done_logged = True
                return

        json_path = self.triples[self._index]
        self._index += 1

        stem = json_path.stem
        jpg_path = self.session_dir / f"{stem}.jpg"
        npy_path = self.session_dir / f"{stem}.npy"

        self._publish_path(self.rgb_pub, jpg_path, self.rgb_out_dir, ".jpg")
        if npy_path.is_file():
            self._publish_path(self.depth_pub, npy_path, self.depth_out_dir, ".npy")

        try:
            pose = json.loads(json_path.read_text()).get("pose", {})
        except (OSError, json.JSONDecodeError) as exc:
            self.get_logger().warn(f"[{stem}] failed to read pose sidecar: {exc}")
            pose = {}
        self._publish_pose(pose)


def parse_args():
    p = argparse.ArgumentParser(
        description="Replay a rooster_dome_main.py capture session onto the live pipeline's topics.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--session-dir", required=True,
                   help="A capture session dir, e.g. ~/rooster_dome_capture/latest")
    p.add_argument("--rooster-id", default="R1")
    p.add_argument("--rgb-path-topic", default="")
    p.add_argument("--depth-path-topic", default="")
    p.add_argument("--pose-topic", default="")
    p.add_argument("--rgb-out-dir", default="/tmp/rooster_frames")
    p.add_argument("--depth-out-dir", default="/tmp/rooster_depth")
    p.add_argument("--rate", type=float, default=1.0, help="Playback rate in Hz.")
    p.add_argument("--loop", action="store_true", help="Replay continuously instead of stopping at the end.")
    return p.parse_args()


def main():
    args = parse_args()
    if not args.rgb_path_topic:
        args.rgb_path_topic = f"/{args.rooster_id}/rgb_frame_path"
    if not args.depth_path_topic:
        args.depth_path_topic = f"/{args.rooster_id}/depth_frame_path"
    if not args.pose_topic:
        args.pose_topic = f"/{args.rooster_id}/localization"

    rclpy.init()
    node = RoosterOfflineFrameDirPublisher(args)
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, rclpy.executors.ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
