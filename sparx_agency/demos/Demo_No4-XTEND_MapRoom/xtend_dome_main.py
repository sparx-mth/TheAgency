#!/usr/bin/env python3
"""
XTEND dome demo launcher with capture output.

Output per captured frame:
  <base_dir>/<session_id>/<drone_id>_YYYYmmdd_HHMMSS_D.jpg
  <base_dir>/<session_id>/<drone_id>_YYYYmmdd_HHMMSS_D.json

The JSON sidecar contains pose={x,y,z,yaw}, where yaw is saved in degrees.
Internally, rotation still uses yaw in radians.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
import time
from pathlib import Path
from typing import Any, Optional

import cv2

from sparx_agency.robots.XTEND.map_a_room_xtend import XtendMapRoomTaskWithCapture, angle_step

try:
    from sparx_agency.robots.common.txt_utils import update_sidecar_json
except Exception:
    update_sidecar_json = None


class XtendDomeTaskWithCapture(XtendMapRoomTaskWithCapture):
    """XTEND task that keeps the XTEND movement API but saves captures like ImageStateBuffer."""

    def __init__(
        self,
        *args,
        drone_id: str,
        bearing_unit: str,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        # In the parent class, out_dir was used as the final capture directory.
        # Here it is the base directory, matching ImageStateBuffer.base_dir.
        self._last_bearing_print_time = 0.0
        self.current_bearing_raw = None
        self.current_yaw_deg = None
        self.base_dir = Path(self.out_dir).absolute()
        self.base_dir.mkdir(parents=True, exist_ok=True)

        self.drone_id = drone_id
        self.bearing_unit = bearing_unit
        self.unique_out_dir = time.strftime("%Y_%m_%d___%H_%M_%S", time.localtime())
        self.out_dir = self.base_dir / self.unique_out_dir
        self.out_dir.mkdir(parents=True, exist_ok=True)

        self.last_xtend_state: Optional[dict[str, Any]] = None
        self._last_saved_name: Optional[str] = None
        self._same_name_hits = 0

    @staticmethod
    def _safe_float(value: Any, default: float = 0.0) -> float:
        try:
            if value is None:
                return default
            return float(value)
        except (TypeError, ValueError):
            return default

    def _bearing_to_rad(self, bearing: Optional[float]) -> Optional[float]:
        if bearing is None:
            return None

        b = float(bearing)
        if self.bearing_unit == "rad":
            return b
        if self.bearing_unit == "deg":
            return math.radians(b)

        # Auto mode: radians are usually within [-pi, pi] or [0, 2*pi].
        # Degrees are usually much larger, e.g. 90, 180, 336.
        if abs(b) > (2.0 * math.pi + 0.5):
            return math.radians(b)
        return b

    def _bearing_to_deg(self, bearing: Optional[float]) -> float:
        if bearing is None:
            return 0.0

        b = float(bearing)
        if self.bearing_unit == "deg":
            return b
        if self.bearing_unit == "rad":
            return math.degrees(b)

        if abs(b) > (2.0 * math.pi + 0.5):
            return b
        return math.degrees(b)

    def update_robot_telemetry(self, bearing: float):
        self.current_bearing_raw = float(bearing)
        yaw_rad = self._bearing_to_rad(bearing)
        yaw_deg = self._bearing_to_deg(bearing)

        self.current_yaw = yaw_rad
        self.current_yaw_deg = yaw_deg

        now = time.time()
        last_print_time = getattr(self, "_last_bearing_print_time", 0.0)

        if last_print_time is None:
            last_print_time = 0.0

        if now - last_print_time >= 1.0:
            raw_txt = f"{self.current_bearing_raw:.5f}"
            rad_txt = "na" if yaw_rad is None else f"{yaw_rad:.5f}"
            deg_txt = "na" if yaw_deg is None else f"{yaw_deg:.5f}"
            print(f"[bearing] raw={raw_txt}  yaw_rad={rad_txt}  yaw_deg={deg_txt}")
            self._last_bearing_print_time = now

    def extract_pose_from_xtend_state(self, state: Optional[dict[str, Any]]) -> dict[str, float]:
        """
        Best-effort extraction of {x,y,z,yaw} from XTEND ROBOT_STATUS.

        XTEND mapping:
          - position: robot.local_telemetry.x/y/z
          - heading:  robot.telemetry.details.bearing

        Saved yaw is degrees, because your offline RGB-D/yaw code expects yaw-column values
        as degrees.
        """
        pose = {"x": 0.0, "y": 0.0, "z": 0.0, "yaw": 0.0}
        if not state:
            return pose

        local_telemetry = state.get("local_telemetry", {}) or {}
        telemetry = state.get("telemetry", {}) or {}
        details = telemetry.get("details", {}) or {}

        pose["x"] = self._safe_float(local_telemetry.get("x"), 0.0)
        pose["y"] = self._safe_float(local_telemetry.get("y"), 0.0)
        pose["z"] = self._safe_float(local_telemetry.get("z"), 0.0)
        pose["yaw"] = self._bearing_to_deg(details.get("bearing"))

        for key, value in pose.items():
            pose[key] = round(float(value), 5)

        return pose

    async def receive_message(self, websocket):
        """Receive XTEND status and keep latest state for capture sidecars."""
        try:
            async for message in websocket:
                try:
                    data = json.loads(message)
                    header = data.get("header", {}) or {}
                    content = data.get("content", {}) or {}

                    if header.get("command") != "ROBOT_STATUS":
                        continue

                    for robot in content.get("robots", []) or []:
                        if robot.get("robot_uid") != self.robot_uid:
                            continue

                        self.last_xtend_state = robot

                        telemetry = robot.get("telemetry", {}) or {}
                        details = telemetry.get("details", {}) or {}
                        bearing = details.get("bearing")
                        if bearing is not None:
                            self.update_robot_telemetry(float(bearing))

                        local_telemetry = robot.get("local_telemetry", {}) or {}
                        self.x = local_telemetry.get("x")
                        self.y = local_telemetry.get("y")
                        self.z = local_telemetry.get("z")
                        break

                except json.JSONDecodeError:
                    print("[RECV] Received non-JSON message")
                except Exception as exc:
                    print(f"[RECV] Error: {exc}")
        except asyncio.CancelledError:
            print("Receiver stopped.")
            raise

    def _write_sidecar_json(self, json_path: Path, pose: dict[str, float], img_basename: str):
        if update_sidecar_json is not None:
            update_sidecar_json(str(json_path), pose, img_basename, vlm_text=None)
            return

        # Fallback for running this file outside the full sparx_agency tree.
        payload = {
            "image": img_basename,
            "pose": pose,
            "vlm_text": None,
        }
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

    def _update_latest_symlink(self):
        latest_link = self.base_dir / "latest"
        latest_ann_link = self.base_dir / "latest_ann"
        try:
            if os.path.lexists(latest_ann_link):
                os.remove(latest_ann_link)
            if os.path.lexists(latest_link):
                os.remove(latest_link)
            os.symlink(self.out_dir, latest_link)
        except Exception as exc:
            print(f"[capture] Failed to create symlink {latest_link}: {exc}")

    def save_capture(self, bgr, t_sec: float):
        pose = self.extract_pose_from_xtend_state(self.last_xtend_state)

        decisec = int(round((t_sec % 1), 1) * 10) % 10
        ts_str = time.strftime("%Y%m%d_%H%M%S", time.localtime(t_sec))
        base_name = f"{self.drone_id}_{ts_str}_{decisec}"

        # Extra guard in case both time and yaw bucket trigger in the same decisecond.
        if base_name == self._last_saved_name:
            self._same_name_hits += 1
            base_name = f"{base_name}_{self._same_name_hits}"
        else:
            self._same_name_hits = 0
        self._last_saved_name = base_name

        jpg_path = self.out_dir / f"{base_name}.jpg"
        json_path = self.out_dir / f"{base_name}.json"
        img_basename = jpg_path.name

        self.out_dir.mkdir(parents=True, exist_ok=True)
        self._update_latest_symlink()

        ok = cv2.imwrite(
            str(jpg_path),
            bgr,
            [int(cv2.IMWRITE_JPEG_QUALITY), int(self.jpeg_quality)],
        )
        if not ok:
            print(f"[capture] Failed to save image: {jpg_path}")
            return

        self._write_sidecar_json(json_path, pose, img_basename)
        print(f"[capture] Saved: {jpg_path}")

    async def _capture_loop(self):
        assert self._rtsp is not None

        while True:
            out = self._rtsp.get_latest()
            if out is None:
                await asyncio.sleep(0.01)
                continue

            bgr, stamp = out
            now = time.time()
            t_sec = float(stamp) if isinstance(stamp, (int, float)) and stamp > 0 else now

            yaw = None
            if self.current_yaw is not None:
                yaw = float(self.current_yaw)

            if self.yaw_bucket_rad is not None and yaw is not None:
                if self._yaw_last is None:
                    self._yaw_last = yaw
                step = angle_step(yaw, self._yaw_last)
                if step < 1.0:
                    self._yaw_travel += step
                self._yaw_last = yaw

            time_due = now >= self._next_time
            bucket_due = (
                self.yaw_bucket_rad is not None
                and self._next_bucket is not None
                and self._yaw_travel >= self._next_bucket
            )

            if time_due or bucket_due:
                self.save_capture(bgr, t_sec)
                self._seq += 1

                if time_due:
                    self._next_time = now + self.capture_interval_sec
                if bucket_due and self._next_bucket is not None and self.yaw_bucket_rad is not None:
                    self._next_bucket += self.yaw_bucket_rad

            if self.show_video:
                cv2.imshow("XTEND capture", bgr)
                if (cv2.waitKey(1) & 0xFF) == ord("q"):
                    return

            await asyncio.sleep(0.005)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run XTEND dome demo with JPG+JSON capture output.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument("--host", default="192.0.0.15")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--frequency", type=float, default=30.0)
    parser.add_argument("--robot-uid", default="drnb177ede2")
    parser.add_argument("--drone-id", default="R2")

    parser.add_argument("--rtsp-uri", default="rtsp://192.0.0.15:8510/active_drone_fpv")
    parser.add_argument("--rtsp-latency-ms", type=int, default=0)

    parser.add_argument("--out-dir", default="./xtend_dome_capture")
    parser.add_argument("--capture-interval-sec", type=float, default=0.5)
    parser.add_argument("--yaw-bucket-deg", type=float, default=30.0)
    parser.add_argument("--jpeg-quality", type=int, default=90)
    parser.add_argument("--show-video", action="store_true")

    parser.add_argument("--sleep-time", type=float, default=2.0)
    parser.add_argument(
        "--bearing-unit",
        choices=["auto", "rad", "deg"],
        default="auto",
        help="Unit of telemetry.details.bearing. Use auto unless you already verified it.",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    if args.frequency <= 0:
        raise ValueError("--frequency must be greater than 0")

    task = XtendDomeTaskWithCapture(
        host=args.host,
        port=args.port,
        frequency=args.frequency,
        robot_uid=args.robot_uid,
        rtsp_uri=args.rtsp_uri,
        rtsp_latency_ms=args.rtsp_latency_ms,
        out_dir=args.out_dir,
        capture_interval_sec=args.capture_interval_sec,
        yaw_bucket_deg=args.yaw_bucket_deg,
        jpeg_quality=args.jpeg_quality,
        show_video=args.show_video,
        sleep_time=args.sleep_time,
        drone_id=args.drone_id,
        bearing_unit=args.bearing_unit,
    )

    asyncio.run(task.run())


if __name__ == "__main__":
    main()
