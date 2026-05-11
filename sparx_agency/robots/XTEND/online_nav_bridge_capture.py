#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
import json
import time
from pathlib import Path

import cv2
import rclpy

from sparx_agency.robots.XTEND.xtend_online_bridge_base import OnlineXtendBridgeBase


class OnlineNavBridgeCapture(OnlineXtendBridgeBase):
    """Online XTEND bridge that saves RTSP frames as JPG + JSON sidecars."""

    def __init__(
        self,
        host: str,
        port: int,
        frequency: float,
        robot_uid: str,
        rtsp_uri: str,
        *,
        out_dir: str | Path = "./captures",
        drone_id: str = "R1",
        capture_interval_sec: float = 0.5,
        jpeg_quality: int = 90,
        log_dir: str | Path | None = None,
    ):
        out_dir = Path(out_dir).expanduser().resolve()
        self.capture_base_dir = out_dir
        self.capture_base_dir.mkdir(parents=True, exist_ok=True)

        self.drone_id = drone_id
        self.session_dir = self.capture_base_dir / time.strftime("%Y_%m_%d___%H_%M_%S")
        self.session_dir.mkdir(parents=True, exist_ok=True)

        if log_dir is None:
            log_dir = self.session_dir / "logs"

        super().__init__(
            host=host,
            port=port,
            frequency=frequency,
            robot_uid=robot_uid,
            log_dir=log_dir,
        )

        self.rtsp_uri = rtsp_uri
        self.capture_interval_sec = float(capture_interval_sec)
        self.jpeg_quality = int(jpeg_quality)

        self.cap = None
        self._last_save_time = 0.0

        print(f"[capture] session: {self.session_dir}")
        print(f"[capture] RTSP: {self.rtsp_uri}")

    def extract_pose(self) -> dict:
        pose = {"x": 0.0, "y": 0.0, "z": 0.0, "yaw": 0.0}
        if not self.last_xtend_state:
            return pose

        local = self.last_xtend_state.get("local_telemetry", {}) or {}
        details = (self.last_xtend_state.get("telemetry", {}) or {}).get("details", {}) or {}

        pose["x"] = local.get("x", 0.0)
        pose["y"] = local.get("y", 0.0)
        pose["z"] = local.get("z", 0.0)
        pose["yaw"] = details.get("bearing", 0.0)
        return pose

    async def capture_loop(self):
        print("[capture] opening RTSP stream")
        self.cap = cv2.VideoCapture(self.rtsp_uri)

        while True:
            ret, frame = self.cap.read()
            if not ret or frame is None:
                await asyncio.sleep(0.1)
                continue

            now = time.time()
            if now - self._last_save_time >= self.capture_interval_sec:
                ts_str = time.strftime("%Y%m%d_%H%M%S", time.localtime(now))
                base_name = f"{self.drone_id}_{ts_str}"

                jpg_path = self.session_dir / f"{base_name}.jpg"
                json_path = self.session_dir / f"{base_name}.json"

                suffix = 1
                while jpg_path.exists() or json_path.exists():
                    jpg_path = self.session_dir / f"{base_name}_{suffix}.jpg"
                    json_path = self.session_dir / f"{base_name}_{suffix}.json"
                    suffix += 1

                cv2.imwrite(
                    str(jpg_path),
                    frame,
                    [int(cv2.IMWRITE_JPEG_QUALITY), self.jpeg_quality],
                )

                with open(json_path, "w", encoding="utf-8") as f:
                    json.dump(
                        {
                            "image": jpg_path.name,
                            "pose": self.extract_pose(),
                        },
                        f,
                        indent=2,
                    )

                print(f"[capture] saved: {jpg_path.name}")
                self._last_save_time = now

            await asyncio.sleep(0.01)

    def create_extra_tasks(self):
        return [asyncio.create_task(self.capture_loop())]

    def on_shutdown(self):
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        cv2.destroyAllWindows()


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--host", default="192.0.0.15")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--frequency", type=float, default=10.0)
    p.add_argument("--robot-uid", default="drnb177ede2")
    p.add_argument("--rtsp-uri", default="rtsp://192.0.0.15:8510/active_drone_fpv")

    p.add_argument("--out-dir", default="./captures")
    p.add_argument("--drone-id", default="R1")
    p.add_argument("--capture-interval-sec", type=float, default=0.5)
    p.add_argument("--jpeg-quality", type=int, default=90)
    p.add_argument("--log-dir", default=None)
    return p.parse_args()


async def async_main():
    args = parse_args()
    rclpy.init()

    bridge = OnlineNavBridgeCapture(
        host=args.host,
        port=args.port,
        frequency=args.frequency,
        robot_uid=args.robot_uid,
        rtsp_uri=args.rtsp_uri,
        out_dir=args.out_dir,
        drone_id=args.drone_id,
        capture_interval_sec=args.capture_interval_sec,
        jpeg_quality=args.jpeg_quality,
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
