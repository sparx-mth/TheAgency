# !/usr/bin/env python3
"""
WebSocket Virtual Controller Client
Sends and/or receives virtual controller data via WebSocket at a configurable rate.
Supports send-only, listen-only, and bidirectional modes.
"""
import copy
import json
import math
import asyncio
import argparse
import websockets
from datetime import datetime

class ControllerAutomation:
    """Controller Automation class"""

    def __init__(self, host: str, port: int, frequency: float, robot_uid: str):
        self.host = host
        self.port = port
        self.frequency = frequency
        self.interval = 1.0 / frequency if frequency > 0 else 1.0
        self.uri = f"ws://{host}:{port}"

        self.robot_uid = robot_uid
        self.pilot_station_uid = "gcu12345678"
        self.user_uid = "user12345"
        self.controller_type = 1  # mnf
        self.current_yaw = 0.0

        self.virtual_controller = { 
                "header": {
                    "timestamp": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
                    "command": "VIRTUAL_CONTROLLER"
                },
                "content": {}
            }

        self.base_command = {
            "robot_uid": self.robot_uid,
            "pilot_station_uid": self.pilot_station_uid,
            "user_uid": self.user_uid,
            "type": self.controller_type,
            "buttons": [
                0,  # Switch - off
                0,  # Side - on
                0,  # A - on
                0,  # B - off
                0,  # C - off
                0,  # Joystick - off
            ],
            "axes": [
                0,   # [0] Lateral (left/right)
                0,   # [1] Vertical (up/down)
                0,   # [2] Forward/Backward
                0,   # [3] Yaw
                0,   # [4] Marker Vertical
            ]
        }

        self.send_command = copy.deepcopy(self.base_command)

    async def run(self):
        
        try:
            async with websockets.connect(self.uri) as websocket:
                print(f"✓ Connected to {self.uri}")
                
                # Create tasks for sending and receiving
                send_task = asyncio.create_task(self.send_message(websocket))
                receive_task = asyncio.create_task(self.receive_message(websocket))
                scenario_task = asyncio.create_task(self.create_scenario())
                
                # Wait for both tasks
                try:
                    # Wait only for scenario to finish (success or error)
                    await scenario_task
                finally:
                    # Stop background loops cleanly
                    for t in (send_task, receive_task):
                        t.cancel()
                    await asyncio.gather(send_task, receive_task, return_exceptions=True)
                    
        except websockets.exceptions.WebSocketException as e:
            print(f"✗ WebSocket error: {e}")
        except ConnectionRefusedError as e:
            print(f"✗ Connection refused. Is the server running at {self.uri}?")
            print(f"ConnectionRefusedError {e}")
        except KeyboardInterrupt:
            print(f"\n\n✓ Stopped.")
        except Exception as e:
            print(f"✗ Unexpected error: {e}")

    async def send_message(self, websocket):
        try:
            while True:
                self.virtual_controller['content'] = self.send_command
                message_json = json.dumps(self.virtual_controller, separators=(',', ':'))
                await websocket.send(message_json)
                
                # Wait for next interval
                await asyncio.sleep(self.interval)
        except asyncio.CancelledError:
            print(f"Sender stopped.")
            raise

    async def receive_message(self, websocket):
        """Receive XTEND telemetry, store latest state, and log telemetry."""
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

                        local = robot.get("local_telemetry", {}) or {}
                        self.x = local.get("x", getattr(self, "x", None))
                        self.y = local.get("y", getattr(self, "y", None))
                        self.z = local.get("z", getattr(self, "z", None))

                        break

                except json.JSONDecodeError:
                    print("[RECV] Received non-JSON message")
                except Exception as exc:
                    print(f"[RECV] Error: {exc}")

        except asyncio.CancelledError:
            print("Receiver stopped.")
            raise

    async def arm_robot(self):
        """Arm the robot."""
        print("Arming robot...")
        self.send_command['buttons'][0] = 1
        await asyncio.sleep(0.1)
        self.send_command['buttons'][0] = 0
        await asyncio.sleep(0.1)
        self.send_command['buttons'][0] = 1
        await asyncio.sleep(0.3)
        self.send_command['buttons'][0] = 0
        print("Robot armed.")

    async def disarm_robot(self):
        """Disarm the robot."""
        print("Disarming robot...")
        self.send_command['buttons'][0] = 1
        await asyncio.sleep(0.1)
        self.send_command['buttons'][0] = 0
        await asyncio.sleep(0.1)
        self.send_command['buttons'][0] = 1
        await asyncio.sleep(0.1)
        self.send_command['buttons'][0] = 0
        print("Robot disarmed.")

    async def takeoff(self, duration: float = 3.3, value: int = 1000):
        """Takeoff. duration in seconds."""
        print("Taking off...")
        self.send_command['axes'][1] = value
        await asyncio.sleep(duration)
        print("Taken off.")
        self.send_command['axes'][1] = 0

    async def land(self, duration: float = 4.1):
        """Land the robot. duration in seconds."""
        print("Landing...")
        self.send_command['buttons'][3] = 1
        await asyncio.sleep(duration)
        print("Landed.")
        self.send_command['buttons'][3] = 0
        await asyncio.sleep(2.0)

    async def move_forward(self, duration: float, value=500):
        """Move the robot forward. duration in milliseconds."""
        print(f"Moving forward...")
        self.send_command['axes'][2] = value
        await asyncio.sleep(duration * 0.001)
        print(f"Moved forward.")
        self.send_command['axes'][2] = 0

        await asyncio.sleep(2)
    
    async def move_backward(self, duration: float, value=700):
        """Move the robot backward. duration in milliseconds."""
        print(f"Moving backward...")
        self.send_command['axes'][2] = -value
        await asyncio.sleep(duration * 0.001)
        print(f"Moved backward.")
        self.send_command['axes'][2] = 0

        await asyncio.sleep(2)
    
    async def move_left(self, duration: float, value=700):
        """Move the robot left. duration in milliseconds."""
        print(f"Moving left...")
        self.send_command['axes'][0] = -value
        await asyncio.sleep(duration * 0.001)
        self.send_command['axes'][0] = 0

    async def move_right(self, duration: float, value=700):
        """Move the robot right. duration in milliseconds."""
        print(f"Moving right...")
        self.send_command['axes'][0] = value
        await asyncio.sleep(duration * 0.001)
        print(f"Moved right.")
        self.send_command['axes'][0] = 0

    async def move_up(self, duration: float, value=700):
        """Move the robot up. duration in milliseconds."""
        print(f"Moving up...")
        self.send_command['axes'][1] = value
        await asyncio.sleep(duration * 0.001)
        print(f"Moved up.")
        self.send_command['axes'][1] = 0

    async def move_down(self, duration: float, value=500):
        """Move the robot down. duration in milliseconds."""
        print(f"Moving down...")
        self.send_command['axes'][1] = -value
        await asyncio.sleep(duration * 0.001)
        print(f"Moved down.")
        self.send_command['axes'][1] = 0
    
    async def rotate_left(self, duration_ms: float, value=1000):
        """Rotate the robot left. duration_ms in milliseconds."""
        print(f"Rotating left...")
        self.send_command['axes'][3] = -value
        await asyncio.sleep(duration_ms * 0.001)
        print(f"Rotated left.")
        self.send_command['axes'][3] = 0

    async def rotate_right(self, duration_ms: float, value=1000):
        """Rotate the robot right. duration_ms in milliseconds."""
        print(f"Rotating right...")
        self.send_command['axes'][3] = value
        await asyncio.sleep(duration_ms * 0.001)
        print(f"Rotated right.")
        self.send_command['axes'][3] = 0

    async def full_rotation(self, direction: int, duration_ms: float = 3600.0):
        """Rotate the robot full circle in the given direction. duration_ms in milliseconds."""
        if direction == 1:
            await self.rotate_left(duration_ms)
        elif direction == -1:
            await self.rotate_right(duration_ms)
        else:
            raise ValueError("Invalid direction. Must be 1 or -1.")

    def hover(self):
        """Hover the robot"""
        self.send_command = copy.deepcopy(self.base_command)

    def update_robot_telemetry(self, yaw: float):
        """Update the robot telemetry with the given yaw"""
        self.current_yaw = yaw
        # print(f"Current yaw: {self.current_yaw} radians -> {self.current_yaw * 180 / math.pi} degrees")

    async def create_scenario(self):
        """Test scenario — override in subclasses for custom sequences."""
        sleep_time = 3
        await asyncio.sleep(5)

        scenario = [
            self.disarm_robot(),
            self.arm_robot(),
            self.takeoff(),
            self.move_forward(400),
            self.move_backward(400),
            self.rotate_left(3000),
            self.land(),
            self.disarm_robot(),
        ]

        for step in scenario:
            await step
            await asyncio.sleep(sleep_time)


def main():
    """Main entry point with command-line argument parsing"""
    parser = argparse.ArgumentParser(
        description="WebSocket Virtual Controller Client",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        "--host",
        type=str,
        default="192.0.0.15",
        help="WebSocket server IP address or hostname"
    )
    
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="WebSocket server port"
    )
    
    parser.add_argument(
        "--frequency",
        type=float,
        default=30.0,
        help="Message transmission frequency in Hz"
    )

    args = parser.parse_args()
    
    # Validate frequency
    if args.frequency <= 0:
        print("Error: Frequency must be greater than 0")
        return

    controller = ControllerAutomation(args.host, args.port, args.frequency, "drn120ea1b0")
    asyncio.run(controller.run())

if __name__ == "__main__":
    main()


"""
show video stream

URI = "rtsp://192.0.0.15:8556/osd_snapshot"
gst-launch-1.0 rtspsrc location=rtsp://$URI latency=0 ! rtph264depay ! h264parse ! avdec_h264 ! videoconvert ! osxvideosink sync=false

"""
