#!/usr/bin/env python3
"""rooster_dome_main.py

360-degree dome sweep mission for ROBOTICAN/Rooster, mirroring
sparx_agency/demos/Demo_No4_XTEND_MapRoom/xtend_dome_main.py's capture
output so room_mapper/run_room_mapper.py works unchanged on either robot's
captures.

Output layout per session (identical to the XTEND version):
  <out_dir>/<YYYYmmdd_HHMMSS>/
    R1_20260127_122951.jpg   .json   .npy
    ...
  <out_dir>/latest  ->  symlink to session dir

Unlike XTEND, ROBOTICAN's command channel is already plain ROS2 topics (no
websocket), so this is a plain rclpy.Node rather than an asyncio task. It
only ever publishes cmd_nav JSON and polls rooster_status/localization —
takeoff/land/arm/disarm are entirely owned by RoosterCommandUnitNode's
RoosterUnit (sparx_agency/robots/ROBOTICAN/helpers/rooster_unit.py), which is
"our side" implementation (ROBOTICAN's FCU has no platform-level takeoff/land
primitive the way XTEND does) — this script never re-implements climb/descent
itself. It also never branches on simulator-vs-real-hardware: everything goes
through the same cmd_nav/rooster_status/localization/frame-path topics either
way.
"""
from __future__ import annotations

import argparse
import json
import math
import shutil
import signal
import threading
import time
from pathlib import Path
from typing import Optional

import rclpy
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import String

from sparx_agency.core.common.types.geometry import normalize_angle
from sparx_agency.robots.common.txt_utils import update_sidecar_json

DEFAULT_AXIS_VALUE = 400.0


def angle_step(cur: float, prev: float) -> float:
    return abs(normalize_angle(cur - prev))


def _parse_path_msg(data: str) -> str:
    """Parse "{path} {sec} {nanosec}" — see rooster_frame_dir_publisher.py."""
    return data.rsplit(" ", 2)[0]


class RoosterDomeMain(Node):
    def __init__(self, args):
        super().__init__("rooster_dome_main")

        self.rooster_id = args.rooster_id
        self.sleep_time = float(args.sleep_time)
        self.axis_value = float(args.axis_value)
        self.capture_interval_sec = float(args.capture_interval_sec)
        self.yaw_bucket_rad = (
            math.radians(args.yaw_bucket_deg) if args.yaw_bucket_deg > 0 else None
        )
        self.rotate_total_deg = float(args.rotate_total_deg)
        self.rotate_step_deg = float(args.rotate_step_deg)
        self.blind_turn_deg_per_sec = float(args.blind_turn_deg_per_sec)
        self.arm_confirm_timeout_sec = float(args.arm_confirm_timeout_sec)
        self.takeoff_confirm_timeout_sec = float(args.takeoff_confirm_timeout_sec)
        self.land_confirm_timeout_sec = float(args.land_confirm_timeout_sec)
        self.video_on_confirm_timeout_sec = float(args.video_on_confirm_timeout_sec)

        self.base_dir = Path(args.out_dir).expanduser().resolve()
        self.base_dir.mkdir(parents=True, exist_ok=True)
        session_ts = time.strftime("%Y%m%d_%H%M%S", time.localtime())
        self.session_dir = self.base_dir / session_ts
        self.session_dir.mkdir(parents=True, exist_ok=True)
        self._seq = 0

        self._shutdown = threading.Event()

        self.armed = False
        self.airborne = False
        self.video_on = False

        self._pose_lock = threading.Lock()
        self._latest_pose: Optional[dict] = None
        self._latest_yaw_rad: Optional[float] = None

        self._rgb_lock = threading.Lock()
        self._latest_rgb_path: Optional[str] = None
        self._depth_lock = threading.Lock()
        self._latest_depth_path: Optional[str] = None

        self._capture_stop = threading.Event()
        self._capture_thread: Optional[threading.Thread] = None

        self.cmd_pub = self.create_publisher(String, f"/{self.rooster_id}/cmd_nav", 10)
        self.create_subscription(
            String, f"/{self.rooster_id}/rooster_status", self._on_status, 10)
        self.create_subscription(PoseStamped, args.pose_topic, self._on_pose, 10)
        self.create_subscription(String, args.rgb_path_topic, self._on_rgb_path, 10)
        self.create_subscription(String, args.depth_path_topic, self._on_depth_path, 10)

        self.get_logger().info(
            f"rooster_dome_main ready for {self.rooster_id}\n"
            f"  command out: /{self.rooster_id}/cmd_nav\n"
            f"  status in:   /{self.rooster_id}/rooster_status\n"
            f"  pose in:     {args.pose_topic}\n"
            f"  rgb in:      {args.rgb_path_topic}\n"
            f"  depth in:    {args.depth_path_topic}\n"
            f"  session dir: {self.session_dir}"
        )

    # ---- subscriptions ----

    def _on_status(self, msg: String):
        try:
            data = json.loads(msg.data)
        except json.JSONDecodeError:
            return
        self.armed = bool(data.get("armed", self.armed))
        self.airborne = bool(data.get("airborne", self.airborne))
        self.video_on = bool(data.get("video_on", self.video_on))

    def _on_pose(self, msg: PoseStamped):
        q = msg.pose.orientation
        yaw_rad = 2.0 * math.atan2(q.z, q.w)
        with self._pose_lock:
            self._latest_pose = {
                "x": round(float(msg.pose.position.x), 5),
                "y": round(float(msg.pose.position.y), 5),
                "z": round(float(msg.pose.position.z), 5),
                "yaw": round(math.degrees(yaw_rad), 5),
            }
            self._latest_yaw_rad = yaw_rad

    def _on_rgb_path(self, msg: String):
        with self._rgb_lock:
            self._latest_rgb_path = _parse_path_msg(msg.data)

    def _on_depth_path(self, msg: String):
        with self._depth_lock:
            self._latest_depth_path = _parse_path_msg(msg.data)

    def _get_latest_pose_dict(self) -> dict:
        with self._pose_lock:
            if self._latest_pose is not None:
                return dict(self._latest_pose)
        return {"x": 0.0, "y": 0.0, "z": 0.0, "yaw": 0.0}

    def _get_latest_yaw_rad(self) -> Optional[float]:
        with self._pose_lock:
            return self._latest_yaw_rad

    def _get_latest_rgb_path(self) -> Optional[str]:
        with self._rgb_lock:
            return self._latest_rgb_path

    def _get_latest_depth_path(self) -> Optional[str]:
        with self._depth_lock:
            return self._latest_depth_path

    # ---- commands ----

    def _pub_cmd(self, action: str, value: Optional[float] = None):
        payload = {"action": action}
        if value is not None:
            payload["value"] = value
        msg = String()
        msg.data = json.dumps(payload)
        self.cmd_pub.publish(msg)

    def _sleep(self, seconds: float):
        self._shutdown.wait(timeout=seconds)

    def _wait_for(self, predicate, timeout_sec: float, what: str) -> bool:
        deadline = time.time() + timeout_sec
        while time.time() < deadline and not self._shutdown.is_set():
            if predicate():
                self.get_logger().info(f"[{what}] confirmed.")
                return True
            time.sleep(0.1)
        self.get_logger().warn(f"[{what}] not confirmed within {timeout_sec}s — proceeding anyway.")
        return False

    # ---- capture ----

    def _update_latest_symlink(self):
        latest_link = self.base_dir / "latest"
        try:
            if latest_link.is_symlink() or latest_link.exists():
                latest_link.unlink()
            latest_link.symlink_to(self.session_dir)
        except OSError as exc:
            self.get_logger().warn(f"[capture] failed to create symlink {latest_link}: {exc}")

    def _save_capture(self):
        rgb_path = self._get_latest_rgb_path()
        if not rgb_path or not Path(rgb_path).is_file():
            self.get_logger().warn("[capture] no rgb frame available yet")
            return

        pose = self._get_latest_pose_dict()
        ts = time.strftime("%Y%m%d_%H%M%S", time.localtime())
        base_name = f"{self.rooster_id}_{ts}"
        jpg_path = self.session_dir / f"{base_name}.jpg"
        if jpg_path.exists():
            base_name = f"{base_name}_{self._seq}"
            jpg_path = self.session_dir / f"{base_name}.jpg"
        json_path = self.session_dir / f"{base_name}.json"

        self._update_latest_symlink()

        shutil.copy2(rgb_path, jpg_path)
        update_sidecar_json(str(json_path), pose, jpg_path.name, vlm_text=None)

        depth_path = self._get_latest_depth_path()
        if depth_path and Path(depth_path).is_file():
            try:
                shutil.copy2(depth_path, self.session_dir / f"{base_name}.npy")
            except OSError as exc:
                self.get_logger().warn(f"[capture] depth copy failed: {exc}")
        else:
            self.get_logger().warn(f"[capture] no depth file for {base_name}")

        self.get_logger().info(f"[capture] saved {jpg_path.name}")
        self._seq += 1

    def _capture_loop(self):
        next_time = time.time()
        next_bucket = self.yaw_bucket_rad
        yaw_travel = 0.0
        yaw_last: Optional[float] = None

        while not self._capture_stop.is_set():
            now = time.time()
            yaw = self._get_latest_yaw_rad()

            if self.yaw_bucket_rad is not None and yaw is not None:
                if yaw_last is None:
                    yaw_last = yaw
                step = angle_step(yaw, yaw_last)
                if step < 1.0:
                    yaw_travel += step
                yaw_last = yaw

            time_due = now >= next_time
            bucket_due = (
                self.yaw_bucket_rad is not None
                and next_bucket is not None
                and yaw_travel >= next_bucket
            )

            if time_due or bucket_due:
                self._save_capture()
                if time_due:
                    next_time = now + self.capture_interval_sec
                if bucket_due and next_bucket is not None:
                    next_bucket += self.yaw_bucket_rad

            time.sleep(0.05)

    def _start_capture(self):
        self._capture_stop.clear()
        self._capture_thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._capture_thread.start()
        self.get_logger().info("[capture] started")

    def _stop_capture(self):
        self._capture_stop.set()
        if self._capture_thread is not None:
            self._capture_thread.join(timeout=2.0)
        self.get_logger().info("[capture] stopped")

    # ---- rotation ----

    def _blind_turn(self, chunk_deg: float):
        """No localization pose available — turn for an estimated duration.

        blind_turn_deg_per_sec is a rough, uncalibrated estimate: tune it for
        the drone/room before relying on this fallback for accurate coverage.
        """
        duration = chunk_deg / max(self.blind_turn_deg_per_sec, 1e-6)
        self._pub_cmd("turn_left", self.axis_value)
        self._sleep(duration)
        self._pub_cmd("stop")

    def rotate_degrees(self, degrees: float, step_deg: float = 90.0):
        target = abs(degrees)
        total_done = 0.0

        while total_done < target and not self._shutdown.is_set():
            chunk_deg = min(step_deg, target - total_done)
            chunk_rad = math.radians(chunk_deg)

            deadline = time.time() + 5.0
            while self._get_latest_yaw_rad() is None and time.time() < deadline and not self._shutdown.is_set():
                time.sleep(0.05)

            start_yaw = self._get_latest_yaw_rad()
            if start_yaw is not None:
                last_rad = start_yaw
                acc = 0.0
                chunk_deadline = time.time() + 30.0
                self._pub_cmd("turn_left", self.axis_value)
                try:
                    while acc < chunk_rad and time.time() < chunk_deadline and not self._shutdown.is_set():
                        time.sleep(0.02)
                        cur = self._get_latest_yaw_rad()
                        if cur is None:
                            continue
                        step = angle_step(cur, last_rad)
                        if step < 0.4:
                            acc += step
                        last_rad = cur
                finally:
                    self._pub_cmd("stop")
            else:
                self.get_logger().warn(
                    f"[rotate] no localization pose — blind turn fallback for {chunk_deg:.0f} deg")
                self._blind_turn(chunk_deg)

            total_done += chunk_deg
            self.get_logger().info(f"[rotate] {total_done:.0f}/{target:.0f} deg done")
            if total_done < target:
                self._sleep(1.0)

    # ---- mission ----

    def run_mission(self):
        try:
            # RoosterCommandUnitNode/RoosterPayload owns SetVideoMode — this
            # is the only place we ask it to start streaming, over the same
            # cmd_nav channel used for every other command.
            self._pub_cmd("video_on")
            self._wait_for(lambda: self.video_on, self.video_on_confirm_timeout_sec, "video_on")
            self._sleep(self.sleep_time)

            self._pub_cmd("disarm")
            self._sleep(self.sleep_time)

            self._pub_cmd("arm")
            self._wait_for(lambda: self.armed, self.arm_confirm_timeout_sec, "arm")
            self._sleep(self.sleep_time)

            self._pub_cmd("takeoff")
            self._wait_for(lambda: self.airborne, self.takeoff_confirm_timeout_sec, "takeoff")
            self._sleep(self.sleep_time)

            self._start_capture()
            self.rotate_degrees(self.rotate_total_deg, step_deg=self.rotate_step_deg)
            self._stop_capture()

            self._pub_cmd("land")
            self._wait_for(lambda: not self.airborne, self.land_confirm_timeout_sec, "land")
            self._sleep(self.sleep_time)

            self._pub_cmd("disarm")
            self._sleep(self.sleep_time)
        except Exception as exc:
            self.get_logger().error(f"[mission] error: {exc}")
        finally:
            self._stop_capture()
            # RoosterUnit.land() already disarms internally, but this mirrors
            # xtend_dome_main.py's defensive finally-guaranteed land+disarm.
            self._pub_cmd("land")
            time.sleep(self.sleep_time)
            self._pub_cmd("disarm")
            time.sleep(self.sleep_time)
            self._pub_cmd("video_off")

    def request_shutdown(self):
        self.get_logger().warn(
            "[safety] signal received — cancelling mission (land+disarm will run in finally)")
        self._shutdown.set()
        self._capture_stop.set()


def parse_args():
    p = argparse.ArgumentParser(
        description="Run a 360-degree ROBOTICAN dome sweep with JPG+JSON+NPY capture output.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--rooster-id", default="R1")
    p.add_argument("--out-dir", default=str(Path.home() / "rooster_dome_capture"))

    p.add_argument("--pose-topic", default="", help="Defaults to /<rooster-id>/localization.")
    p.add_argument("--rgb-path-topic", default="", help="Defaults to /<rooster-id>/rgb_frame_path.")
    p.add_argument("--depth-path-topic", default="", help="Defaults to /<rooster-id>/depth_frame_path.")

    p.add_argument("--capture-interval-sec", type=float, default=1.0)
    p.add_argument("--yaw-bucket-deg", type=float, default=30.0)

    p.add_argument("--rotate-total-deg", type=float, default=360.0)
    p.add_argument("--rotate-step-deg", type=float, default=90.0)
    p.add_argument("--axis-value", type=float, default=DEFAULT_AXIS_VALUE)
    p.add_argument("--blind-turn-deg-per-sec", type=float, default=30.0,
                   help="Rough estimate used only when no localization pose is available.")

    p.add_argument("--sleep-time", type=float, default=2.0)
    p.add_argument("--arm-confirm-timeout-sec", type=float, default=5.0)
    p.add_argument("--takeoff-confirm-timeout-sec", type=float, default=8.0)
    p.add_argument("--land-confirm-timeout-sec", type=float, default=35.0)
    p.add_argument("--video-on-confirm-timeout-sec", type=float, default=5.0)
    return p.parse_args()


def main():
    args = parse_args()
    if not args.pose_topic:
        args.pose_topic = f"/{args.rooster_id}/localization"
    if not args.rgb_path_topic:
        args.rgb_path_topic = f"/{args.rooster_id}/rgb_frame_path"
    if not args.depth_path_topic:
        args.depth_path_topic = f"/{args.rooster_id}/depth_frame_path"

    rclpy.init()
    node = RoosterDomeMain(args)

    executor = SingleThreadedExecutor()
    executor.add_node(node)
    spin_thread = threading.Thread(target=executor.spin, daemon=True)
    spin_thread.start()

    def _on_signal(signum, frame):
        node.request_shutdown()

    signal.signal(signal.SIGINT, _on_signal)
    signal.signal(signal.SIGTERM, _on_signal)

    try:
        node.run_mission()
    finally:
        executor.shutdown()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
