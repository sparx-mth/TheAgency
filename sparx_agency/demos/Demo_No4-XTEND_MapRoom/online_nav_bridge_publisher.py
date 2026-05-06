import asyncio
import csv
from datetime import datetime

import rclpy
import json
import websockets
import cv2
import time
import threading
import numpy as np
from pathlib import Path
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from std_msgs.msg import String
from rclpy.qos import qos_profile_sensor_data

# Base automation class
from sparx_agency.robots.XTEND.automation import ControllerAutomation
from sparx_agency.robots.XTEND.xtend_rtsp_image_publisher import LatestFrameGrabber


class OnlineNavBridgePublisher(ControllerAutomation):
    def __init__(self, host, port, frequency, robot_uid, rtsp_uri, out_dir):
        super().__init__(host, port, frequency, robot_uid)

        # Event loop and Queue for Nav Commands
        self.loop = asyncio.get_event_loop()
        self.cmd_queue = asyncio.Queue()
        self.bridge = CvBridge()

        self.active_action = None
        self.active_action_start_t = None
        self.action_log = []
        self.log_dir =Path.home() / "Documents" / out_dir  / "logs"
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
            "axes_0_lateral",
            "axes_1_vertical",
            "axes_2_forward",
            "axes_3_yaw",
            "axes_4_marker_vertical",
        ])

        print(f"[log] telemetry: {self.telemetry_log_path}")
        print(f"[log] actions:   {self.action_log_path}")

        # ROS 2 Setup
        self.ros_node = rclpy.create_node('drone_bridge_publisher_node')

        # Publisher: FPV Camera
        self.image_pub = self.ros_node.create_publisher(
            Image,
            '/xtend/image_raw',
            qos_profile_sensor_data
        )

        # Subscriber: Navigation Commands
        self.subscription = self.ros_node.create_subscription(
            String,
            '/drone/cmd_nav',
            self.ros_callback,
            10
        )

        # Performance Frame Grabber
        self.grabber = LatestFrameGrabber(rtsp_uri, backend='gstreamer')
        self.grabber.start()

    def ros_callback(self, msg):
        """Moves ROS messages into the Asyncio loop safely."""
        try:
            data = json.loads(msg.data)
            self.loop.call_soon_threadsafe(self.cmd_queue.put_nowait, data)
        except Exception as e:
            self.ros_node.get_logger().error(f"Failed to parse command: {e}")

    def set_axes(self, lateral=0, vertical=0, forward=0, yaw=0, marker_vertical=0):
        self.send_command["axes"][0] = int(lateral)
        self.send_command["axes"][1] = int(vertical)
        self.send_command["axes"][2] = int(forward)
        self.send_command["axes"][3] = int(yaw)
        self.send_command["axes"][4] = int(marker_vertical)

    def hold_forward(self, thrust=500):
        self.start_action_timer(f"forward_{thrust}")
        self.set_axes(forward=thrust, yaw=0)

    def hold_backward(self, thrust=500):
        self.start_action_timer(f"backward_{thrust}")
        self.set_axes(forward=-thrust, yaw=0)

    def hold_turn_left(self, thrust=700):
        self.start_action_timer(f"turn_left_{thrust}")
        self.set_axes(forward=0, yaw=-thrust)

    def hold_turn_right(self, thrust=700):
        self.start_action_timer(f"turn_right_{thrust}")
        self.set_axes(forward=0, yaw=thrust)

    def stop_motion(self, reason: str = "stop"):
        self.end_action_timer(reason=reason)
        self.set_axes(0, 0, 0, 0, 0)

    def start_action_timer(self, action_name: str):
        self.end_action_timer(reason=f"interrupted_by_{action_name}")

        self.active_action = action_name
        self.active_action_start_t = time.time()
        print(f"[action] START {action_name}")

    def end_action_timer(self, reason: str = "stop"):
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

    def log_telemetry(self, robot: dict):
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
        self.telemetry_fp.flush()

    def save_action_log_csv(self):
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

    async def image_publish_loop(self):
        """Independent loop to publish frames at 30Hz with cropping."""
        # Define crop parameters from your config
        crop_left, crop_top = 108, 70
        crop_width, crop_height = 504, 280
        sleep_time = 1.0 / max(self.frequency, 1e-6)
        print("✓ Image Publisher Active with Cropping")
        while True:
            frame, _ = self.grabber.get_latest()
            if frame is not None:
                # Apply the crop slicing: frame[y:y+h, x:x+w]
                cropped_frame = frame[crop_top:crop_top + crop_height,
                crop_left:crop_left + crop_width]

                msg = self.bridge.cv2_to_imgmsg(cropped_frame, encoding="bgr8")
                msg.header.stamp = self.ros_node.get_clock().now().to_msg()
                msg.header.frame_id = "xtend_camera"

                self.image_pub.publish(msg)

            await asyncio.sleep(sleep_time)

    async def dynamic_executor(self):
        """Consumes UI commands from /drone/cmd_nav and applies hold-style control."""
        print("ONLINE MODE: hold-style commands from /drone/cmd_nav.")

        while True:
            command = await self.cmd_queue.get()
            action = command.get("action")
            thrust = int(command.get("thrust", command.get("value", 500)))

            print(f"[cmd] action={action}, thrust={thrust}")

            try:
                if action == "arm":
                    await self.arm_robot()

                elif action == "disarm":
                    self.stop_motion()
                    await self.disarm_robot()

                elif action == "takeoff":
                    await self.takeoff()

                elif action == "land":
                    self.stop_motion()
                    await self.land()

                elif action == "stop":
                    self.stop_motion()

                elif action == "forward":
                    self.hold_forward(thrust)

                elif action == "backward":
                    self.hold_backward(thrust)

                elif action in ("turn_left", "rotate_left"):
                    self.hold_turn_left(thrust)

                elif action in ("turn_right", "rotate_right"):
                    self.hold_turn_right(thrust)

                elif action == "left":
                    self.set_axes(lateral=-thrust)

                elif action == "right":
                    self.set_axes(lateral=thrust)

                elif action in ("down", "move_down"):
                    self.set_axes(vertical=-thrust)

                elif action == "up":
                    self.set_axes(vertical=thrust)

                elif action == "disarm":
                    self.stop_motion(reason="disarm")
                    await self.disarm_robot()

                elif action == "land":
                    self.stop_motion(reason="land")
                    await self.land()

                elif action == "stop":
                    self.stop_motion(reason="stop")

                else:
                    print(f"[cmd] Unknown action: {action}")

            finally:
                self.cmd_queue.task_done()

    async def receive_message(self, websocket):
        """Receive XTEND telemetry, update local state, and log telemetry."""
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

                        telemetry = robot.get("telemetry", {}) or {}
                        details = telemetry.get("details", {}) or {}
                        bearing = details.get("bearing")

                        if bearing is not None:
                            self.update_robot_telemetry(float(bearing))

                        local = robot.get("local_telemetry", {}) or {}
                        self.x = local.get("x", getattr(self, "x", None))
                        self.y = local.get("y", getattr(self, "y", None))
                        self.z = local.get("z", getattr(self, "z", None))

                        self.log_telemetry(robot)
                        break

                except json.JSONDecodeError:
                    print("[RECV] Received non-JSON message")
                except Exception as exc:
                    print(f"[RECV] Error: {exc}")

        except asyncio.CancelledError:
            print("Receiver stopped.")
            raise

    async def run_bridge(self):
        """Main entry point to run all concurrent tasks."""
        ros_thread = asyncio.to_thread(rclpy.spin, self.ros_node)
        tasks = []

        try:
            async with websockets.connect(self.uri) as websocket:
                print(f"✓ Connected to Drone at {self.uri}")

                tasks = [
                    asyncio.create_task(self.send_message(websocket)),
                    asyncio.create_task(self.receive_message(websocket)),
                    asyncio.create_task(self.dynamic_executor()),
                    asyncio.create_task(self.image_publish_loop()),
                    asyncio.create_task(ros_thread),
                ]

                await asyncio.gather(*tasks)

        finally:
            print("[shutdown] stopping motion and closing resources")

            try:
                self.stop_motion()
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
                    self.telemetry_fp.close()
                    print(f"[log] closed telemetry: {self.telemetry_log_path}")
            except Exception as exc:
                print(f"[shutdown] telemetry close failed: {exc}")

            try:
                self.grabber.stop()
            except Exception as exc:
                print(f"[shutdown] grabber stop failed: {exc}")

            try:
                self.ros_node.destroy_node()
            except Exception:
                pass

            if rclpy.ok():
                rclpy.shutdown()


async def main():
    rclpy.init()
    bridge = OnlineNavBridgePublisher(
        host="192.0.0.15",
        port=8000,
        frequency=15.0,
        robot_uid="drnb177ede2",
        rtsp_uri="rtsp://192.0.0.15:8510/active_drone_fpv",
        out_dir="online_publisher_node"
    )
    await bridge.run_bridge()


if __name__ == "__main__":
    asyncio.run(main())