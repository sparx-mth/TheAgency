#!/usr/bin/env python3
"""
XTEND dome demo launcher with capture output.

Output layout per session:
  <out_dir>/<drone_id>/<YYYYmmdd_HHMMSS>/
    R1_20260127_122951.jpg   .json   .npy
    R1_20260127_122952.jpg   .json   .npy
    ...
  <out_dir>/latest  →  symlink to session dir

The JSON sidecar contains pose={x,y,z,yaw} (yaw in degrees).
Depth NPY files come from depth_processor_node via --depth-topic.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
import shutil
import threading
import time
from pathlib import Path
from typing import Any, Optional

import signal as _signal

try:
    import rclpy
    from geometry_msgs.msg import PoseStamped as RosPoseStamped
    from std_msgs.msg import String as RosString
    from rclpy.executors import SingleThreadedExecutor as _RclpyExecutor
    _ROS2_AVAILABLE = True
except ImportError:
    _ROS2_AVAILABLE = False

import cv2

from sparx_agency.robots.XTEND.map_a_room_xtend import XtendMapRoomTaskWithCapture, angle_step

try:
    from sparx_agency.robots.common.txt_utils import update_sidecar_json
except Exception:
    update_sidecar_json = None


class _LocalizationListener:
    """Background ROS2 subscriber that pulls /pose from OnlineRgbdLocalizationNode."""

    def __init__(self, pose_topic: str):
        if not _ROS2_AVAILABLE:
            raise RuntimeError("rclpy not available — cannot use --pose-topic")
        if not rclpy.ok():
            rclpy.init()
        self._node = rclpy.create_node("dome_loc_listener")
        self._latest: Optional[dict[str, float]] = None
        self._lock = threading.Lock()
        self._node.create_subscription(RosPoseStamped, pose_topic, self._cb, 10)
        self._cmd_pub = self._node.create_publisher(RosString, "/xtend/cmd_nav", 10)
        self._executor = _RclpyExecutor()
        self._executor.add_node(self._node)
        self._thread = threading.Thread(target=self._executor.spin, daemon=True)
        self._thread.start()
        print(f"[localization] listening on {pose_topic}")

    def _cb(self, msg: "RosPoseStamped"):
        q = msg.pose.orientation
        # Localization node encodes yaw as: z=sin(yaw/2), w=cos(yaw/2), x=y=0
        yaw_rad = 2.0 * math.atan2(q.z, q.w)
        with self._lock:
            self._latest = {
                "x": round(float(msg.pose.position.x), 5),
                "y": round(float(msg.pose.position.y), 5),
                "z": round(float(msg.pose.position.z), 5),
                "yaw": round(math.degrees(yaw_rad), 5),
            }

    def get_pose(self) -> Optional[dict[str, float]]:
        with self._lock:
            return dict(self._latest) if self._latest is not None else None

    def pub_cmd(self, action: str, value: int = 0):
        msg = RosString()
        msg.data = json.dumps({"action": action, "value": value})
        self._cmd_pub.publish(msg)

    def shutdown(self):
        try:
            self._executor.shutdown(wait=False)
        except Exception:
            pass
        try:
            self._node.destroy_node()
        except Exception:
            pass


class _DepthListener:
    """Background ROS2 subscriber for depth NPY paths from depth_processor_node."""

    def __init__(self, depth_topic: str):
        if not _ROS2_AVAILABLE:
            raise RuntimeError("rclpy not available — cannot use --depth-topic")
        if not rclpy.ok():
            rclpy.init()
        self._node = rclpy.create_node("dome_depth_listener")
        self._latest_path: Optional[str] = None
        self._latest_stamp: Optional[float] = None
        self._lock = threading.Lock()
        self._node.create_subscription(RosString, depth_topic, self._cb, 10)
        self._executor = _RclpyExecutor()
        self._executor.add_node(self._node)
        self._thread = threading.Thread(target=self._executor.spin, daemon=True)
        self._thread.start()
        print(f"[depth] listening on {depth_topic}")

    def _cb(self, msg: "RosString"):
        # Format: "{path} {sec} {nanosec}" — sec/nanosec is the ORIGINAL RGB
        # frame's capture stamp, carried through by depth_processor_node from
        # the frame it ran DA3 inference on (not the completion time).
        parts = msg.data.rsplit(" ", 2)
        path = parts[0]
        stamp = None
        if len(parts) == 3:
            try:
                stamp = float(parts[1]) + float(parts[2]) * 1e-9
            except ValueError:
                stamp = None
        with self._lock:
            self._latest_path = path
            self._latest_stamp = stamp

    def get_latest_path(self, current_t_sec: Optional[float] = None,
                         max_age_s: float = 0.5) -> Optional[str]:
        """Return the latest depth path, or None if it's older than max_age_s
        relative to current_t_sec (the RGB frame currently being saved).

        DA3 inference takes ~300-450ms, so without this check a depth file
        computed for a PREVIOUS RGB frame — from before the drone finished
        rotating to its current heading — can get silently paired with the
        current frame ("depth from before" / smeared objects on the map).
        Pass current_t_sec=None to skip the freshness check (old behavior).
        """
        with self._lock:
            path, stamp = self._latest_path, self._latest_stamp
        if path is None:
            return None
        if current_t_sec is not None and stamp is not None:
            age = current_t_sec - stamp
            if age > max_age_s:
                print(f"[depth] stale depth ({age:.2f}s old) — skipping, not pairing with current frame")
                return None
        return path

    def shutdown(self):
        try:
            self._executor.shutdown(wait=False)
        except Exception:
            pass
        try:
            self._node.destroy_node()
        except Exception:
            pass


class XtendDomeTaskWithCapture(XtendMapRoomTaskWithCapture):
    """XTEND task that keeps the XTEND movement API but saves captures like ImageStateBuffer."""

    def __init__(
        self,
        *args,
        drone_id: str,
        bearing_unit: str,
        loc_pose_topic: str = "",
        depth_topic: str = "",
        max_depth_age_sec: float = 0.5,
        min_climb_m: float = 0.15,
        takeoff_verify_timeout_s: float = 3.0,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self._loc_listener: Optional[_LocalizationListener] = None
        if loc_pose_topic:
            self._loc_listener = _LocalizationListener(loc_pose_topic)

        self._depth_listener: Optional[_DepthListener] = None
        if depth_topic:
            self._depth_listener = _DepthListener(depth_topic)
        self.max_depth_age_sec = max_depth_age_sec
        self.min_climb_m = min_climb_m
        self.takeoff_verify_timeout_s = takeoff_verify_timeout_s

        # base_dir is the root; out_dir is the per-session flat capture directory.
        self._last_bearing_print_time = 0.0
        self.current_bearing_raw = None
        self.current_yaw_deg = None
        self.base_dir = Path(self.out_dir).absolute()
        self.base_dir.mkdir(parents=True, exist_ok=True)

        self.drone_id = drone_id
        self.bearing_unit = bearing_unit
        self.unique_out_dir = time.strftime("%Y%m%d_%H%M%S", time.localtime())
        # Session dir directly under base_dir — drone_id is already in each filename.
        self.out_dir = self.base_dir / self.unique_out_dir
        self.out_dir.mkdir(parents=True, exist_ok=True)

        self.last_xtend_state: Optional[dict[str, Any]] = None
        # When True, send_message() forwards frames so we can bypass the bridge.
        self._direct_cmd_active = False

    # ------------------------------------------------------------------
    # Route drone commands through /xtend/cmd_nav (bridge executes them).
    # land() and disarm_robot() also send direct WebSocket frames as a
    # fallback so the drone is safe even if the bridge process is dead.
    # ------------------------------------------------------------------

    async def send_message(self, websocket):
        """Normally silent — only sends when _direct_cmd_active is set."""
        while True:
            if self._direct_cmd_active:
                self.virtual_controller['content'] = self.send_command
                await websocket.send(
                    json.dumps(self.virtual_controller, separators=(',', ':'))
                )
            await asyncio.sleep(self.interval)

    def _pub_cmd(self, action: str, value: int = 0):
        if self._loc_listener is not None:
            self._loc_listener.pub_cmd(action, value)
        else:
            print(f"[cmd] no loc_listener — cannot publish cmd_nav '{action}'")

    async def arm_robot(self):
        print("Arming robot... !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        self._pub_cmd("arm")
        await asyncio.sleep(1.0)
        print("Robot armed... !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")

    def _get_altitude(self) -> Optional[float]:
        """Read local_telemetry.z (a rough proxy for altitude) from the last
        XTEND websocket state, if any has arrived yet."""
        state = self.last_xtend_state or {}
        local_telemetry = state.get("local_telemetry", {}) or {}
        z = local_telemetry.get("z")
        try:
            return float(z) if z is not None else None
        except (TypeError, ValueError):
            return None

    async def takeoff(self, duration=3.3, value=1000,
                       min_climb_m: Optional[float] = None,
                       verify_timeout_s: Optional[float] = None):
        min_climb_m = self.min_climb_m if min_climb_m is None else min_climb_m
        verify_timeout_s = self.takeoff_verify_timeout_s if verify_timeout_s is None else verify_timeout_s
        """Issue takeoff, then verify it actually happened before declaring success.

        arm/takeoff are fire-and-forget over the XTEND websocket link — with no
        check here, a failed takeoff (dead battery, prop fault, dropped command)
        would silently print "Taken off..." and the mission would proceed to
        rotate/capture with the drone still on the floor. Raises RuntimeError on
        a confirmed failure (altitude never moved), which create_scenario()'s
        try/except/finally already catches and turns into a safe land+disarm
        instead of running the rest of the dome sweep.
        """
        print("Taking off... !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        baseline_z = self._get_altitude()
        self._pub_cmd("takeoff")
        await asyncio.sleep(duration + 1.0)

        if baseline_z is None:
            print("[takeoff] WARNING: no telemetry available — cannot verify takeoff, "
                  "proceeding blind.")
        else:
            climbed = False
            t0 = time.time()
            current_z = baseline_z
            while time.time() - t0 < verify_timeout_s:
                current_z = self._get_altitude()
                if current_z is not None and abs(current_z - baseline_z) >= min_climb_m:
                    climbed = True
                    break
                await asyncio.sleep(0.2)
            if not climbed:
                raise RuntimeError(
                    f"Takeoff verification failed — altitude did not change "
                    f"(baseline={baseline_z:.3f}m, current={current_z}, "
                    f"threshold={min_climb_m}m). Drone likely never left the "
                    f"ground (dead battery, prop fault, or dropped command)."
                )

        print("Taken off... !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")

    async def land(self, duration=4.1):
        print("Landing... !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        self._pub_cmd("land")
        # Direct WebSocket backup: works even if bridge is already dead.
        self._direct_cmd_active = True
        try:
            self.send_command['buttons'][3] = 1
            await asyncio.sleep(duration)
            self.send_command['buttons'][3] = 0
            await asyncio.sleep(2.0)
        finally:
            self._direct_cmd_active = False
        print("Landed... !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")

    async def disarm_robot(self):
        print("Disarming robot... !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        self._pub_cmd("disarm")
        # Direct WebSocket backup.
        self._direct_cmd_active = True
        try:
            self.send_command['buttons'][0] = 1
            await asyncio.sleep(0.1)
            self.send_command['buttons'][0] = 0
            await asyncio.sleep(0.1)
            self.send_command['buttons'][0] = 1
            await asyncio.sleep(0.1)
            self.send_command['buttons'][0] = 0
        finally:
            self._direct_cmd_active = False
        print("Robot disarmed... !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")

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
        if self._loc_listener is not None:
            pose = self._loc_listener.get_pose() or self.extract_pose_from_xtend_state(self.last_xtend_state)
        else:
            pose = self.extract_pose_from_xtend_state(self.last_xtend_state)

        ts = time.strftime("%Y%m%d_%H%M%S", time.localtime(t_sec))
        base_name = f"{self.drone_id}_{ts}"
        jpg_path = self.out_dir / f"{base_name}.jpg"
        if jpg_path.exists():
            base_name = f"{base_name}_{self._seq}"
            jpg_path = self.out_dir / f"{base_name}.jpg"
        json_path = self.out_dir / f"{base_name}.json"

        self._update_latest_symlink()

        ok = cv2.imwrite(
            str(jpg_path),
            bgr,
            [int(cv2.IMWRITE_JPEG_QUALITY), int(self.jpeg_quality)],
        )
        if not ok:
            print(f"[capture] Failed to save image: {jpg_path}")
            return

        self._write_sidecar_json(json_path, pose, jpg_path.name)

        if self._depth_listener is not None:
            depth_src = self._depth_listener.get_latest_path(
                current_t_sec=t_sec, max_age_s=self.max_depth_age_sec)
            if depth_src and os.path.isfile(depth_src):
                try:
                    shutil.copy2(depth_src, self.out_dir / f"{base_name}.npy")
                except Exception as exc:
                    print(f"[capture] depth copy failed: {exc}")
            else:
                print(f"[capture] no depth file for {base_name}")

        print(f"[capture] saved {jpg_path.name}")

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


    async def rotate_degrees(
        self,
        degrees: float,
        direction: int = +1,
        yaw_cmd: int = 1000,
        step_deg: float = 90.0,
    ):
        """
        Rotate `degrees` total in `step_deg` chunks, measuring each chunk via
        the localization node's yaw (/xtend/localization PoseStamped).

        If the localization node has no pose when a chunk begins, falls back to
        XTEND bearing integration for that chunk only.
        """
        if self._loc_listener is None:
            await super().rotate_degrees(degrees, direction, yaw_cmd)
            return

        target = abs(degrees)
        total_done = 0.0

        while total_done < target:
            chunk_deg = min(step_deg, target - total_done)
            chunk_rad = math.radians(chunk_deg)

            # Wait up to 5 s for a fresh localization pose before this chunk.
            deadline = asyncio.get_event_loop().time() + 5.0
            while (
                self._loc_listener.get_pose() is None
                and asyncio.get_event_loop().time() < deadline
            ):
                await asyncio.sleep(0.05)

            start_pose = self._loc_listener.get_pose()

            if start_pose is not None:
                last_rad = math.radians(start_pose["yaw"])
                last_pose_yaw = last_rad  # track last NEW pose to detect stale
                last_update_t = asyncio.get_event_loop().time()
                acc = 0.0
                chunk_deadline = asyncio.get_event_loop().time() + 30.0
                rot_action = "rotate_left" if direction == +1 else "rotate_right"
                self._pub_cmd(rot_action, yaw_cmd)
                try:
                    while acc < chunk_rad:
                        now = asyncio.get_event_loop().time()
                        if now > chunk_deadline:
                            print(f"[rotate] chunk timeout — proceeding after 30 s")
                            break
                        await asyncio.sleep(0.02)
                        pose = self._loc_listener.get_pose()
                        if pose is None:
                            continue
                        cur_rad = math.radians(pose["yaw"])
                        if cur_rad != last_pose_yaw:   # new localization update
                            last_pose_yaw = cur_rad
                            last_update_t = asyncio.get_event_loop().time()
                        step = angle_step(cur_rad, last_rad)
                        if step < 0.4:
                            acc += step
                        last_rad = cur_rad
                finally:
                    self._pub_cmd("stop")
            else:
                # Localization not available — fall back to XTEND bearing for this chunk.
                print(f"[rotate] no localization pose — XTEND bearing fallback for {chunk_deg:.0f}°")
                await super().rotate_degrees(chunk_deg, direction, yaw_cmd)

            total_done += chunk_deg
            print(f"[rotate] {total_done:.0f}° / {target:.0f}° done")
            if total_done < target:
                await asyncio.sleep(1.0)

    async def create_scenario(self):
        """360° dome sweep using yaw-integrated rotation."""
        sleep_time = self.sleep_time

        print("Creating scenario... !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")

        await asyncio.sleep(5)
        print("Scenario created... !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        try:
            # Callables — coroutines created only when reached so un-run steps
            # don't emit "coroutine was never awaited" on early exit.
            steps = [
                (self.disarm_robot,   ()),
                (self.arm_robot,      ()),
                (self.takeoff,        ()),
                (self._start_capture, ()),
                (self.rotate_degrees, (360,)),
                (self._stop_capture,  ()),
                (self.move_down,      (500,)),
                (self.move_down,      (400,)),
                (self.land,           ()),
                (self.disarm_robot,   ()),
            ]

            for fn, fn_args in steps:
                await fn(*fn_args)
                print("before sleep")
                await asyncio.sleep(sleep_time)
                print("after sleep")

        except Exception as e:
            print(f"[scenario] error: {e}")
        finally:
            try:
                await self.land()
                await asyncio.sleep(sleep_time)
            except Exception as e:
                print(f"[scenario] land failed: {e}")
            try:
                await self.disarm_robot()
                await asyncio.sleep(sleep_time)
            except Exception as e:
                print(f"[scenario] disarm failed: {e}")

            await self._stop_capture()
            if self._loc_listener is not None:
                self._loc_listener.shutdown()
            if self._depth_listener is not None:
                self._depth_listener.shutdown()
            await asyncio.sleep(sleep_time // 2)


    async def run(self):
        """
        Override automation.run() so that SIGTERM/SIGINT cancel scenario_task first
        (firing the finally → land + disarm) before send_task is stopped.
        Without this, tmux kill-session / systemctl stop terminate the process
        immediately and the drone is left armed.
        """
        import websockets  # noqa: PLC0415 — avoids circular at module level

        loop = asyncio.get_running_loop()
        _stop = asyncio.Event()

        def _on_signal():
            print("[safety] Signal received — cancelling scenario (land+disarm will run in finally)...")
            _stop.set()

        loop.add_signal_handler(_signal.SIGTERM, _on_signal)
        loop.add_signal_handler(_signal.SIGINT,  _on_signal)

        try:
            async with websockets.connect(self.uri) as websocket:
                print(f"✓ Connected to {self.uri}")
                # send_message is normally silent (_direct_cmd_active=False);
                # it only transmits during land/disarm as a bridge-down fallback.
                send_task     = asyncio.create_task(self.send_message(websocket))
                receive_task  = asyncio.create_task(self.receive_message(websocket))
                scenario_task = asyncio.create_task(self.create_scenario())
                stop_task     = asyncio.create_task(_stop.wait())

                done, _ = await asyncio.wait(
                    [scenario_task, stop_task],
                    return_when=asyncio.FIRST_COMPLETED,
                )

                if stop_task in done:
                    # Cancel scenario; its finally block publishes land+disarm
                    # (cmd_nav + direct WebSocket) before we close.
                    scenario_task.cancel()

                stop_task.cancel()

                # Wait for the scenario finally block to finish BEFORE closing comms.
                await asyncio.gather(scenario_task, return_exceptions=True)

                for t in (send_task, receive_task):
                    t.cancel()
                await asyncio.gather(send_task, receive_task, return_exceptions=True)

        except websockets.exceptions.WebSocketException as e:
            print(f"✗ WebSocket error: {e}")
        except ConnectionRefusedError:
            print(f"✗ Connection refused at {self.uri}")
        except Exception as e:
            print(f"✗ Unexpected error: {e}")
        finally:
            try:
                loop.remove_signal_handler(_signal.SIGTERM)
                loop.remove_signal_handler(_signal.SIGINT)
            except Exception:
                pass


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run XTEND dome demo with JPG+JSON capture output.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument("--host", default="192.0.0.15")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--frequency", type=float, default=30.0)
    parser.add_argument("--robot-uid", default="drnb177ede2")
    parser.add_argument("--drone-id", default="xtend")

    parser.add_argument("--rtsp-uri", default="rtsp://192.0.0.15:8510/active_drone_fpv")
    parser.add_argument("--rtsp-latency-ms", type=int, default=0)

    parser.add_argument("--out-dir", default="/home/user/jetson-containers/data/captures")
    parser.add_argument("--capture-interval-sec", type=float, default=1.0,
                        help="Minimum seconds between frame captures.")
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
    parser.add_argument(
        "--pose-topic",
        default="",
        help="PoseStamped topic from the localization node "
             "(e.g. /xtend/localization). "
             "Leave empty to use raw XTEND telemetry for JSON sidecars and XTEND bearing for rotation.",
    )
    parser.add_argument(
        "--depth-topic",
        default="/xtend/depth_frame_path",
        help="String topic publishing depth NPY paths from depth_processor_node. "
             "Leave empty to skip depth capture.",
    )
    parser.add_argument(
        "--max-depth-age-sec", type=float, default=0.5,
        help="Reject a depth file whose embedded RGB-frame timestamp is older than "
             "this many seconds relative to the frame currently being saved — DA3 "
             "inference lags ~300-450ms behind the RGB feed, so without this check "
             "a fast rotation can pair the current frame with a depth map computed "
             "for an earlier heading (stale/mismatched 'depth from before').",
    )
    parser.add_argument(
        "--min-climb-m", type=float, default=0.15,
        help="Minimum altitude change (m) from pre-takeoff baseline required to "
             "consider takeoff successful. Below this after --takeoff-verify-timeout-sec, "
             "takeoff() raises instead of silently proceeding into the dome sweep "
             "with the drone still on the ground.",
    )
    parser.add_argument(
        "--takeoff-verify-timeout-sec", type=float, default=3.0,
        help="How long to poll telemetry for the altitude climb before declaring "
             "takeoff failed.",
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
        loc_pose_topic=args.pose_topic,
        depth_topic=args.depth_topic,
        max_depth_age_sec=args.max_depth_age_sec,
        min_climb_m=args.min_climb_m,
        takeoff_verify_timeout_s=args.takeoff_verify_timeout_sec,
    )

    asyncio.run(task.run())


if __name__ == "__main__":
    main()
