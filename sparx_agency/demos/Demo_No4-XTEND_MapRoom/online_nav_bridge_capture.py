import asyncio
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
import json
import websockets
import cv2
import time
import os
from pathlib import Path

# Base automation class and capture utilities
from sparx_agency.robots.XTEND.automation import ControllerAutomation
# Note: Ensure XtendMapRoomTaskWithCapture is available for its RTSP/Buffer utilities
from sparx_agency.robots.XTEND.map_a_room_xtend import XtendMapRoomTaskWithCapture


class OnlineNavBridgeCapture(ControllerAutomation):
    def __init__(self, host, port, frequency, robot_uid, rtsp_uri, out_dir="./captures"):
        super().__init__(host, port, frequency, robot_uid)

        self.loop = asyncio.get_event_loop()
        self.cmd_queue = asyncio.Queue()

        # ROS 2 Setup
        self.ros_node = rclpy.create_node('drone_bridge_node')
        self.subscription = self.ros_node.create_subscription(
            String, '/drone/cmd_nav', self.ros_callback, 10
        )

        # Capture Setup
        self.rtsp_uri = rtsp_uri
        self.drone_id = "42B"
        self.base_dir = Path(out_dir).absolute()
        self.unique_session_dir = time.strftime("%Y_%m_%d___%H_%M_%S", time.localtime())
        self.out_dir = self.base_dir / self.unique_session_dir
        self.out_dir.mkdir(parents=True, exist_ok=True)

        self.last_xtend_state = None
        self.jpeg_quality = 90
        self.capture_interval_sec = 10000  # Save a frame every 0.5s

    def ros_callback(self, msg):
        try:
            data = json.loads(msg.data)
            self.loop.call_soon_threadsafe(self.cmd_queue.put_nowait, data)
        except Exception as e:
            self.ros_node.get_logger().error(f"Callback Error: {e}")

    def extract_pose(self):
        """Extracts {x, y, z, yaw} for the JSON sidecar."""
        pose = {"x": 0.0, "y": 0.0, "z": 0.0, "yaw": 0.0}
        if not self.last_xtend_state:
            return pose

        local = self.last_xtend_state.get("local_telemetry", {}) or {}
        telemetry = self.last_xtend_state.get("telemetry", {}) or {}
        details = telemetry.get("details", {}) or {}

        pose["x"] = local.get("x", 0.0)
        pose["y"] = local.get("y", 0.0)
        pose["z"] = local.get("z", 0.0)
        pose["yaw"] = details.get("bearing", 0.0)  # Assume degrees as per source
        return pose

    async def receive_message(self, websocket):
        """Standard receiver plus state tracking for sidecars[cite: 5]."""
        async for message in websocket:
            try:
                data = json.loads(message)
                content = data.get("content", {}) or {}
                for robot in content.get("robots", []):
                    if robot.get("robot_uid") == self.robot_uid:
                        self.last_xtend_state = robot  # Save for capture pose[cite: 5]
            except Exception:
                pass

    async def capture_loop(self):
        """Captures FPV frames and saves them with JSON pose sidecars[cite: 5]."""
        print(f"Opening RTSP Stream: {self.rtsp_uri}")
        cap = cv2.VideoCapture(self.rtsp_uri)

        last_save_time = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                await asyncio.sleep(0.1)
                continue

            now = time.time()
            if now - last_save_time >= self.capture_interval_sec:
                pose = self.extract_pose()
                ts_str = time.strftime("%Y%m%d_%H%M%S", time.localtime(now))
                base_name = f"{self.drone_id}_{ts_str}"

                # Save Image[cite: 5]
                jpg_path = self.out_dir / f"{base_name}.jpg"
                cv2.imwrite(str(jpg_path), frame, [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality])

                # Save JSON Sidecar[cite: 5]
                json_path = self.out_dir / f"{base_name}.json"
                with open(json_path, "w") as f:
                    json.dump({"image": jpg_path.name, "pose": pose}, f, indent=2)

                print(f"[capture] Saved frame and pose: {base_name}")
                last_save_time = now

            await asyncio.sleep(0.01)

    async def dynamic_executor(self):
        """Consumes UI commands from /drone/cmd_nav and calls XTEND API methods."""
        print("ONLINE MODE: Executing UI commands from /drone/cmd_nav.")

        while True:
            command = await self.cmd_queue.get()
            action = command.get("action")

            # Backward compatibility with old UI:
            # UI sends {"action": "...", "value": 1500}
            # For movement, treat value as duration unless duration is explicitly provided.
            value = command.get("value", 0)
            duration = command.get("duration", None)
            thrust = command.get("thrust", None)

            if duration is None:
                duration = value

            if thrust is None:
                thrust = 500

            print(f"[cmd] action={action}, value={value}, duration={duration}, thrust={thrust}")

            try:
                if action == "arm":
                    await self.arm_robot()

                elif action == "disarm":
                    await self.disarm_robot()

                elif action == "takeoff":
                    await self.takeoff()

                elif action == "land":
                    await self.land()

                elif action == "stop":
                    self.hover()

                elif action == "forward":
                    await self.move_forward(duration=duration, value=thrust)

                elif action == "backward":
                    await self.move_backward(duration=duration, value=thrust)

                elif action == "left":
                    await self.move_left(duration=duration, value=thrust)

                elif action == "right":
                    await self.move_right(duration=duration, value=thrust)

                elif action == "up":
                    await self.move_up(duration=duration, value=thrust)

                elif action in ("down", "move_down"):
                    await self.move_down(duration=duration, value=thrust)

                elif action == "rotate_left":
                    await self.rotate_left(duration=duration, value=1000)

                elif action == "rotate_right":
                    await self.rotate_right(duration=duration, value=1000)

                else:
                    print(f"[cmd] Unknown action: {action}")

            finally:
                self.cmd_queue.task_done()

    async def run_bridge(self):
        ros_thread = asyncio.to_thread(rclpy.spin, self.ros_node)

        async with websockets.connect(self.uri) as websocket:
            print(f"✓ Connected to {self.uri}")
            await asyncio.gather(
                self.send_message(websocket),
                self.receive_message(websocket),
                self.dynamic_executor(),
                self.capture_loop(),  # New capture task[cite: 5]
                ros_thread
            )


async def main():
    rclpy.init()
    bridge = OnlineNavBridgeCapture(
        host="192.0.0.15", port=8000, frequency=30.0,
        robot_uid="drnb177ede2",
        rtsp_uri="rtsp://192.0.0.15:8510/active_drone_fpv"
    )
    await bridge.run_bridge()


if __name__ == "__main__":
    asyncio.run(main())