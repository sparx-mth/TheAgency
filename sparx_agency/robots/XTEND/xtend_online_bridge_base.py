#!/usr/bin/env python3
from __future__ import annotations

import asyncio
import csv
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import rclpy
from std_msgs.msg import String, Float32
from nav_msgs.msg import Odometry
import math
import websockets

from sparx_agency.robots.XTEND.automation import ControllerAutomation
from sparx_agency.robots.common.spatial_math import yaw_to_quaternion


def clamp_axis(value: int, limit: int = 1000) -> int:
    return max(-limit, min(limit, int(value)))


class OnlineXtendBridgeBase(ControllerAutomation):
    """
    Shared online XTEND bridge.

    Owns:
      - one XTEND WebSocket connection
      - /drone/cmd_nav subscriber
      - hold-style movement commands
      - telemetry logging
      - action logging
      - clean shutdown

    Subclasses should only add video behavior:
      - save JPG/JSON frames
      - publish /xtend/image_raw
    """

    def __init__(
        self,
        host: str,
        port: int,
        frequency: float,
        robot_uid: str,
        *,
        cmd_topic: str = "/xtend/cmd_nav",
        telemetry_topic: str = "/xtend/local_telemetry",
        bearing_topic: str = "/xtend/bearing",
        telemetry_frame_id: str = "odom",
        telemetry_child_frame_id: str = "xtend_camera",
        log_dir: str | Path = "./xtend_online_logs",
    ):
        super().__init__(host, port, frequency, robot_uid)

        self.cmd_topic = cmd_topic
        self.cmd_queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
        self.loop: asyncio.AbstractEventLoop | None = None

        self.ros_node = rclpy.create_node("xtend_online_bridge")
        self.cmd_sub = self.ros_node.create_subscription(
            String,
            self.cmd_topic,
            self.ros_callback,
            10,
        )
        self.telemetry_topic = telemetry_topic
        self.bearing_topic = bearing_topic
        self.telemetry_frame_id = telemetry_frame_id
        self.telemetry_child_frame_id = telemetry_child_frame_id

        self.telemetry_pub = self.ros_node.create_publisher(
            Odometry,
            self.telemetry_topic,
            10,
        )

        self.bearing_pub = self.ros_node.create_publisher(
            Float32,
            self.bearing_topic,
            10,
        )

        print(f"[bridge] telemetry topic: {self.telemetry_topic}")
        print(f"[bridge] bearing topic:   {self.bearing_topic}")

        self.last_xtend_state: dict[str, Any] | None = None
        self.x = None
        self.y = None
        self.z = None

        self.active_action: str | None = None
        self.active_action_start_t: float | None = None
        self.action_log: list[dict[str, Any]] = []

        self.log_dir = Path(log_dir).expanduser().resolve()
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self.run_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.telemetry_log_path = self.log_dir / f"xtend_telemetry_{self.run_stamp}.csv"
        self.action_log_path = self.log_dir / f"xtend_actions_{self.run_stamp}.csv"

        self.telemetry_fp = open(self.telemetry_log_path, "w", newline="", encoding="utf-8")
        self.telemetry_writer = csv.writer(self.telemetry_fp)
        self.telemetry_writer.writerow([
            "time_sec",
            "iso_time",
            "robot_uid",
            "x",
            "y",
            "z",
            "bearing_raw",
            "active_action",
            "axis_lateral",
            "axis_vertical",
            "axis_forward",
            "axis_yaw",
            "axis_marker_vertical",
        ])

        print(f"[bridge] command topic: {self.cmd_topic}")
        print(f"[log] telemetry: {self.telemetry_log_path}")
        print(f"[log] actions:   {self.action_log_path}")

    def ros_callback(self, msg: String) -> None:
        try:
            data = json.loads(msg.data)
            if not isinstance(data, dict):
                raise ValueError("JSON command must be an object")
            if self.loop is None:
                self.ros_node.get_logger().warn("Async loop is not ready yet; dropping command")
                return
            self.loop.call_soon_threadsafe(self.cmd_queue.put_nowait, data)
        except Exception as exc:
            self.ros_node.get_logger().error(f"Failed to parse command: {exc}")

    def set_axes(
        self,
        lateral: int = 0,
        vertical: int = 0,
        forward: int = 0,
        yaw: int = 0,
        marker_vertical: int = 0,
    ) -> None:
        self.send_command["axes"][0] = clamp_axis(lateral)
        self.send_command["axes"][1] = clamp_axis(vertical)
        self.send_command["axes"][2] = clamp_axis(forward)
        self.send_command["axes"][3] = clamp_axis(yaw)
        self.send_command["axes"][4] = clamp_axis(marker_vertical)

    def stop_motion(self, reason: str = "stop") -> None:
        self.end_action_timer(reason=reason)
        self.set_axes(0, 0, 0, 0, 0)

    def hold_forward(self, thrust: int = 400) -> None:
        thrust = clamp_axis(thrust)
        self.start_action_timer(f"forward_{thrust}")
        print(f"[hold] forward thrust={thrust}")
        self.set_axes(forward=thrust, yaw=0)

    def hold_backward(self, thrust: int = 400) -> None:
        thrust = clamp_axis(thrust)
        self.start_action_timer(f"backward_{thrust}")
        print(f"[hold] backward thrust={thrust}")
        self.set_axes(forward=-thrust, yaw=0)

    def hold_turn_left(self, thrust: int = 1000) -> None:
        thrust = clamp_axis(thrust)
        self.start_action_timer(f"turn_left_{thrust}")
        print(f"[hold] turn_left thrust={thrust}")
        self.set_axes(forward=0, yaw=-thrust)

    def hold_turn_right(self, thrust: int = 1000) -> None:
        thrust = clamp_axis(thrust)
        self.start_action_timer(f"turn_right_{thrust}")
        print(f"[hold] turn_right thrust={thrust}")
        self.set_axes(forward=0, yaw=thrust)

    def hold_lateral_left(self, thrust: int = 400) -> None:
        thrust = clamp_axis(thrust)
        self.start_action_timer(f"left_{thrust}")
        print(f"[hold] lateral left thrust={thrust}")
        self.set_axes(lateral=-thrust)

    def hold_lateral_right(self, thrust: int = 400) -> None:
        thrust = clamp_axis(thrust)
        self.start_action_timer(f"right_{thrust}")
        print(f"[hold] lateral right thrust={thrust}")
        self.set_axes(lateral=thrust)

    def hold_up(self, thrust: int = 400) -> None:
        thrust = clamp_axis(thrust)
        self.start_action_timer(f"up_{thrust}")
        print(f"[hold] up thrust={thrust}")
        self.set_axes(vertical=thrust)

    def hold_down(self, thrust: int = 400) -> None:
        thrust = clamp_axis(thrust)
        self.start_action_timer(f"down_{thrust}")
        print(f"[hold] down thrust={thrust}")
        self.set_axes(vertical=-thrust)

    def start_action_timer(self, action_name: str) -> None:
        self.end_action_timer(reason=f"interrupted_by_{action_name}")
        self.active_action = action_name
        self.active_action_start_t = time.time()
        print(f"[action] START {action_name}")

    def end_action_timer(self, reason: str = "stop") -> None:
        if self.active_action is None or self.active_action_start_t is None:
            return

        now = time.time()
        duration = now - self.active_action_start_t

        entry = {
            "action": self.active_action,
            "start_t": self.active_action_start_t,
            "end_t": now,
            "duration_sec": duration,
            "reason": reason,
        }
        self.action_log.append(entry)

        print(
            f"[action] END {self.active_action} "
            f"duration={duration:.3f}s reason={reason}"
        )

        self.active_action = None
        self.active_action_start_t = None

    def _safe_float(self, value, default=None):
        try:
            if value is None or value == "":
                return default
            return float(value)
        except (TypeError, ValueError):
            return default

    def bearing_to_yaw_rad(self, bearing) -> float | None:
        """XTEND bearing is in radians."""
        return self._safe_float(bearing, None)

    def publish_telemetry(self, robot: dict[str, Any]) -> None:
        local = robot.get("local_telemetry", {}) or {}
        telemetry = robot.get("telemetry", {}) or {}
        details = telemetry.get("details", {}) or {}

        x = self._safe_float(local.get("x"), None)
        y = self._safe_float(local.get("y"), None)
        z = self._safe_float(local.get("z"), None)
        bearing_raw = self._safe_float(details.get("bearing"), None)

        # Publish bearing even if local position is missing.
        if bearing_raw is not None:
            bearing_msg = Float32()
            bearing_msg.data = float(bearing_raw)
            self.bearing_pub.publish(bearing_msg)

        # If local telemetry is empty, do not publish fake zero odometry.
        if x is None or y is None or z is None:
            return

        msg = Odometry()
        msg.header.stamp = self.ros_node.get_clock().now().to_msg()
        msg.header.frame_id = self.telemetry_frame_id
        msg.child_frame_id = self.telemetry_child_frame_id

        msg.pose.pose.position.x = x
        msg.pose.pose.position.y = y
        msg.pose.pose.position.z = z

        yaw_rad = self.bearing_to_yaw_rad(bearing_raw)
        if yaw_rad is not None:
            qx, qy, qz, qw = yaw_to_quaternion(yaw_rad)
            msg.pose.pose.orientation.x = qx
            msg.pose.pose.orientation.y = qy
            msg.pose.pose.orientation.z = qz
            msg.pose.pose.orientation.w = qw
        else:
            msg.pose.pose.orientation.w = 1.0

        self.telemetry_pub.publish(msg)

    def save_action_log_csv(self) -> None:
        with open(self.action_log_path, "w", newline="", encoding="utf-8") as fp:
            writer = csv.writer(fp)
            writer.writerow([
                "index",
                "action",
                "start_t",
                "end_t",
                "duration_sec",
                "reason",
            ])

            for i, entry in enumerate(self.action_log):
                writer.writerow([
                    i,
                    entry["action"],
                    entry["start_t"],
                    entry["end_t"],
                    entry["duration_sec"],
                    entry["reason"],
                ])

        print(f"[log] saved actions: {self.action_log_path}")

    def log_telemetry(self, robot: dict[str, Any]) -> None:
        now = time.time()
        iso_time = datetime.fromtimestamp(now).isoformat(timespec="milliseconds")

        local = robot.get("local_telemetry", {}) or {}
        telemetry = robot.get("telemetry", {}) or {}
        details = telemetry.get("details", {}) or {}
        axes = self.send_command.get("axes", [0, 0, 0, 0, 0])

        self.telemetry_writer.writerow([
            now,
            iso_time,
            robot.get("robot_uid", self.robot_uid),
            local.get("x", ""),
            local.get("y", ""),
            local.get("z", ""),
            details.get("bearing", ""),
            self.active_action or "",
            axes[0],
            axes[1],
            axes[2],
            axes[3],
            axes[4],
        ])


    async def timed_async_action(self, action_name: str, coro):
        self.start_action_timer(action_name)
        try:
            return await coro
        finally:
            self.end_action_timer(reason="completed")

    async def receive_message(self, websocket):
        """Receive XTEND telemetry, update state, and log telemetry."""
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

                        details = (robot.get("telemetry", {}) or {}).get("details", {}) or {}
                        bearing = details.get("bearing")
                        if bearing is not None:
                            self.update_robot_telemetry(float(bearing))

                        local = robot.get("local_telemetry", {}) or {}
                        self.x = local.get("x", self.x)
                        self.y = local.get("y", self.y)
                        self.z = local.get("z", self.z)

                        self.log_telemetry(robot)
                        self.publish_telemetry(robot)
                        break

                except json.JSONDecodeError:
                    print("[RECV] Received non-JSON message")
                except Exception as exc:
                    print(f"[RECV] Error: {exc}")

        except asyncio.CancelledError:
            print("Receiver stopped.")
            raise

    async def handle_custom_command(self, command: dict[str, Any]) -> bool:
        return False

    async def dynamic_executor(self):
        print(f"ONLINE MODE: hold-style commands from {self.cmd_topic}")

        while True:
            command = await self.cmd_queue.get()
            action = command.get("action")
            thrust = int(command.get("thrust", command.get("value", 400)))

            print(f"[cmd] action={action}, thrust={thrust}")

            try:
                if action == "arm":
                    await self.timed_async_action("arm", self.arm_robot())

                elif action == "disarm":
                    self.stop_motion(reason="disarm")
                    await self.timed_async_action("disarm", self.disarm_robot())

                elif action == "takeoff":
                    await self.timed_async_action("takeoff", self.takeoff())

                elif action == "land":
                    self.stop_motion(reason="land")
                    await self.timed_async_action("land", self.land())

                elif action == "stop":
                    self.stop_motion(reason="stop")

                elif action == "forward":
                    self.hold_forward(thrust)

                elif action == "backward":
                    self.hold_backward(thrust)

                elif action in ("turn_left", "rotate_left"):
                    self.hold_turn_left(thrust)

                elif action in ("turn_right", "rotate_right"):
                    self.hold_turn_right(thrust)

                elif action == "left":
                    self.hold_lateral_left(thrust)

                elif action == "right":
                    self.hold_lateral_right(thrust)

                elif action == "up":
                    self.hold_up(thrust)

                elif action in ("down", "move_down"):
                    self.hold_down(thrust)

                elif await self.handle_custom_command(command):
                    pass

                else:
                    print(f"[cmd] Unknown action: {action}")

            finally:
                self.cmd_queue.task_done()

    def create_extra_tasks(self) -> list[asyncio.Task]:
        return []

    def on_shutdown(self) -> None:
        pass

    async def _ros_spin_loop(self):
        """Spin ROS callbacks in the event loop without blocking it."""
        executor = rclpy.executors.SingleThreadedExecutor()
        executor.add_node(self.ros_node)
        try:
            while True:
                executor.spin_once(timeout_sec=0)
                await asyncio.sleep(0.01)
        except asyncio.CancelledError:
            pass
        finally:
            executor.shutdown()

    async def _telemetry_flush_loop(self):
        """Flush the telemetry CSV at 1 Hz instead of on every message."""
        try:
            while True:
                await asyncio.sleep(1.0)
                if hasattr(self, "telemetry_fp") and not self.telemetry_fp.closed:
                    self.telemetry_fp.flush()
        except asyncio.CancelledError:
            pass

    async def run_bridge(self):
        self.loop = asyncio.get_running_loop()
        tasks: list[asyncio.Task] = []

        try:
            while True:
                try:
                    async with websockets.connect(self.uri) as websocket:
                        print(f"✓ Connected to XTEND at {self.uri}")

                        tasks = [
                            asyncio.create_task(self.send_message(websocket)),
                            asyncio.create_task(self.receive_message(websocket)),
                            asyncio.create_task(self.dynamic_executor()),
                            asyncio.create_task(self._ros_spin_loop()),
                            asyncio.create_task(self._telemetry_flush_loop()),
                        ]
                        tasks.extend(self.create_extra_tasks())

                        await asyncio.gather(*tasks)

                except (websockets.exceptions.WebSocketException, OSError) as exc:
                    print(f"[bridge] WebSocket disconnected: {exc}. Reconnecting in 3s...")
                    for task in tasks:
                        task.cancel()
                    if tasks:
                        await asyncio.gather(*tasks, return_exceptions=True)
                    tasks = []
                    await asyncio.sleep(3.0)

        except (KeyboardInterrupt, asyncio.CancelledError):
            pass

        finally:
            print("[shutdown] stopping motion and closing resources")

            try:
                self.stop_motion(reason="shutdown")
            except Exception as exc:
                print(f"[shutdown] stop_motion failed: {exc}")

            for task in tasks:
                task.cancel()

            if tasks:
                await asyncio.gather(*tasks, return_exceptions=True)

            try:
                self.save_action_log_csv()
            except Exception as exc:
                print(f"[shutdown] save_action_log_csv failed: {exc}")

            try:
                if hasattr(self, "telemetry_fp") and not self.telemetry_fp.closed:
                    self.telemetry_fp.flush()
                    self.telemetry_fp.close()
                    print(f"[log] closed telemetry: {self.telemetry_log_path}")
            except Exception as exc:
                print(f"[shutdown] telemetry close failed: {exc}")

            try:
                self.on_shutdown()
            except Exception as exc:
                print(f"[shutdown] subclass cleanup failed: {exc}")

            try:
                self.ros_node.destroy_node()
            except Exception:
                pass

            if rclpy.ok():
                rclpy.shutdown()
