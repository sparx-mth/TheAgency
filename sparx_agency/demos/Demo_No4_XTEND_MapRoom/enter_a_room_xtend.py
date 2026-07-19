#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
import math
import os
import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from sparx_agency.robots.XTEND.automation import ControllerAutomation
from sparx_agency.robots.XTEND.get_xtend_probe import RtspProbe


def normalize_angle(a: float) -> float:
    while a <= -math.pi:
        a += 2.0 * math.pi
    while a > math.pi:
        a -= 2.0 * math.pi
    return a


def angle_step(cur: float, prev: float) -> float:
    return abs(normalize_angle(cur - prev))


def fmt_num(v, width: int = 6, prec: int = 2) -> str:
    if v is None or not np.isfinite(v):
        return "na"
    s = f"{v:+0{width}.{prec}f}"
    return s.replace(".", "p")


def make_filename(seq: int, x, y, z, yaw_rad) -> str:
    yaw_deg = yaw_rad * 180.0 / math.pi if yaw_rad is not None else None
    x = np.random.randint(0, 1000)
    y = np.random.randint(0, 1000)
    z = np.random.randint(0, 1000)
    return (
        f"{seq:04d}"
        f"x{fmt_num(x)}"
        f"y{fmt_num(y)}"
        f"z{fmt_num(z)}"
        f"yaw{fmt_num(yaw_deg, prec=1)}.jpg"
    )
class ScenarioDone(Exception):
    pass


class XtendMapRoomTaskWithCapture(ControllerAutomation):
    def __init__(
        self,
        host: str,
        port: int,
        frequency: float,
        robot_uid: str,
        rtsp_uri: str,
        rtsp_latency_ms: int,
        out_dir: str,
        capture_interval_sec: float,
        yaw_bucket_deg: float,
        jpeg_quality: int,
        show_video: bool,
        sleep_time: float,
    ):
        super().__init__(host, port, frequency, robot_uid)

        self.rtsp_uri = rtsp_uri
        self.rtsp_latency_ms = rtsp_latency_ms

        self.out_dir = Path(out_dir)
        time_now = time.strftime('%Y%m%d_%H%M%S')
        self.out_dir = self.out_dir / time_now

        self.out_dir.mkdir(parents=True, exist_ok=True)

        self.capture_interval_sec = float(capture_interval_sec)
        self.yaw_bucket_deg = float(yaw_bucket_deg)
        self.yaw_bucket_rad = None if yaw_bucket_deg <= 0 else (yaw_bucket_deg * math.pi / 180.0)

        self.jpeg_quality = int(jpeg_quality)
        self.show_video = bool(show_video)

        self.sleep_time = float(sleep_time)

        # RTSP
        self._rtsp: Optional[RtspProbe] = None
        self._capture_task: Optional[asyncio.Task] = None

        # capture state
        self._seq = 0
        self._next_time = 0.0
        self._yaw_last: Optional[float] = None
        self._yaw_travel = 0.0
        self._next_bucket: Optional[float] = None
        self.current_yaw: Optional[float] = None

        # Telemetry placeholders: fill later when you parse x/y/z
        self.x: Optional[float] = None
        self.y: Optional[float] = None
        self.z: Optional[float] = None

    async def rotate_degrees(self, degrees: float, direction: int = +1, yaw_cmd: int = 1000):
        """Rotate by degrees using telemetry yaw integration (robust 360)."""
        target = abs(degrees) * math.pi / 180.0

        # wait until yaw arrives
        while self.current_yaw is None:
            await asyncio.sleep(0.01)

        acc = 0.0
        last = float(self.current_yaw)

        # left: negative, right: positive
        self.send_command["axes"][3] = (-yaw_cmd if direction == +1 else +yaw_cmd)
        try:
            while acc < target:
                await asyncio.sleep(0.02)
                cur = float(self.current_yaw)
                step = angle_step(cur, last)
                if step < 0.4:  # reject glitches
                    acc += step
                last = cur
        finally:
            self.send_command["axes"][3] = 0

    async def _start_capture(self):

        print("[capture] starting RTSP probe")

        self._rtsp = RtspProbe(uri=self.rtsp_uri, latency_ms=self.rtsp_latency_ms)
        self._rtsp.start()
        self._next_time = time.time()

        # init bucket logic
        self._yaw_last = None
        self._yaw_travel = 0.0
        self._next_bucket = self.yaw_bucket_rad if self.yaw_bucket_rad is not None else None

        self._capture_task = asyncio.create_task(self._capture_loop())

    async def _stop_capture(self):
        if self._capture_task is not None:
            self._capture_task.cancel()
            await asyncio.gather(self._capture_task, return_exceptions=True)
            self._capture_task = None

        if self._rtsp is not None:
            self._rtsp.stop()
            self._rtsp = None

        if self.show_video:
            cv2.destroyAllWindows()

    async def _capture_loop(self):
        assert self._rtsp is not None


        while True:
            out = self._rtsp.get_latest()
            if out is None:
                await asyncio.sleep(0.01)
                continue

            bgr, _stamp = out
            now = time.time()

            yaw = None
            if self.current_yaw is not None:
                yaw = float(self.current_yaw)

            # bucket accumulation
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
                fname = make_filename(self._seq, self.x, self.y, self.z, yaw)

                path = self.out_dir / fname
                cv2.imwrite(str(path), bgr, [int(cv2.IMWRITE_JPEG_QUALITY), int(self.jpeg_quality)])
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

    async def create_scenario(self):
        sleep_time = 3
        land_sleep_time = 5
        finally_sleep_time = 2

        await asyncio.sleep(2)  # let telemetry stabilize
        await self._start_capture()   # your capture start
        try:
            scenario = [
                self.disarm_robot(),
                self.arm_robot(),
                self.takeoff(),
                self.move_down(500),
                self.move_forward(4000),
                # self.move_backward(400),
                # self.move_left(100),
                # self.move_right(100),
                # self.move_up(100),

                # self.rotate_left(3000),
                self.rotate_right(2000),
                self.move_forward(2500),
                # self.full_rotation(1),
                # self.full_rotation(-1),
                self.move_down(400),
                self.land(),
                self.disarm_robot(),
            ]

            # for step_fn in scenario:
            #     await step_fn()
            #     if step_fn == self.land:
            #         await asyncio.sleep(land_sleep_time)  # let it sink into landing
            #     else:
            #         await asyncio.sleep(sleep_time)

            for step in scenario:
                await step
                print("before sleep")
                await asyncio.sleep(sleep_time)
                print("after sleep")

        except Exception as e:
            print(f"[scenario] error: {e}")
            # fallthrough to finally for landing/disarm
        finally:
            # HARD SAFETY: always try to land+disarm even if something broke
            try:
                await self.land()
                await asyncio.sleep(finally_sleep_time)
            except Exception as e:
                print(f"[scenario] land failed: {e}")
            try:
                await self.disarm_robot()
                await asyncio.sleep(finally_sleep_time)
            except Exception as e:
                print(f"[scenario] disarm failed: {e}")

            await self._stop_capture()
            await asyncio.sleep(finally_sleep_time // 2)

def parse_args():
    p = argparse.ArgumentParser()

    p.add_argument("--host", default="192.0.0.15")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--frequency", type=float, default=30.0)
    p.add_argument("--robot-uid", default="drndfb3eeb1")

    p.add_argument("--rtsp-uri", default="rtsp://192.0.0.15:8510/active_drone_fpv")
    p.add_argument("--rtsp-latency-ms", type=int, default=0)

    p.add_argument("--out-dir", default="./xtend_capture_out")
    p.add_argument("--capture-interval-sec", type=float, default=0.5)
    p.add_argument("--yaw-bucket-deg", type=float, default=30.0)
    p.add_argument("--jpeg-quality", type=int, default=90)
    p.add_argument("--show-video", action="store_true")

    p.add_argument("--sleep-time", type=float, default=2.0)
    return p.parse_args()


def main():
    args = parse_args()
    task = XtendMapRoomTaskWithCapture(
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
    )
    asyncio.run(task.run())


if __name__ == "__main__":
    main()
