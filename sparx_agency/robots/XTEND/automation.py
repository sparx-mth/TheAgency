# !/usr/bin/env python3
"""
WebSocket Virtual Controller Client
Sends and/or receives virtual controller data via WebSocket at a configurable rate.
Supports send-only, listen-only, and bidirectional modes.
"""

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
                0,   # Joystick Horizontal 
                0,   # Joystick Vertical
                0,   # Trigger
                0,   # Marker Horizontal 
                0    # Marker Vertical
            ]
        }

        self.send_command = self.base_command

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
                # Convert to JSON
                message_json = json.dumps(self.virtual_controller, indent=2)
                
                # Send message
                await websocket.send(message_json)
                
                # Wait for next interval
                await asyncio.sleep(self.interval)
        except asyncio.CancelledError:
            print(f"Sender stopped.")
            raise

    async def receive_message(self, websocket):
        try:
            async for message in websocket:
                
                try:
                    # Parse the JSON message
                    data = json.loads(message)
                    
                    # Extract key information
                    header = data.get('header', {})
                    content = data.get('content', {})
                    command = header.get('command', 'N/A')

                    # Display the received message
                    if command == "ROBOT_STATUS":
                        for robot in content['robots']:
                            if robot['robot_uid'] == self.robot_uid:
                                self.update_robot_telemetry(robot['telemetry']['details']['bearing'])
                                
                            lt = robot.get("local_telemetry", {})
                            self.x = lt.get("x", None)
                            self.y = lt.get("y", None)
                            self.z = lt.get("z", None)
                            break

                except json.JSONDecodeError:
                    print(f"[RECV] Received non-JSON message")
                except Exception as e:
                    print(f"[RECV] Error: {e}")
        except asyncio.CancelledError:
            print(f"Receiver stopped.")
            raise

    async def arm_robot(self):
        """Arm the robot"""
        print(f"Arming robot... !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")

        self.send_command['buttons'][0] = 1
        await asyncio.sleep(0.1)
        self.send_command['buttons'][0] = 0
        await asyncio.sleep(0.1)
        self.send_command['buttons'][0] = 1
        await asyncio.sleep(0.3)
        self.send_command['buttons'][0] = 0

        print(f"Robot armed... !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")

    async def disarm_robot(self):
        """Disarm the robot"""
        print(f"Disarming robot... !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        
        self.send_command['buttons'][0] = 1
        await asyncio.sleep(0.1)
        self.send_command['buttons'][0] = 0
        await asyncio.sleep(0.1)
        self.send_command['buttons'][0] = 1
        await asyncio.sleep(0.1)
        self.send_command['buttons'][0] = 0

        print(f"Robot disarmed... !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")

    async def takeoff(self):
        """Get the robot status"""
        print(f"Taking off... !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        self.send_command['axes'][1] = 1000
        # await asyncio.sleep(3.1)  original from Tamir xtend
        await asyncio.sleep(3.3) # + 0.2 sec
        print(f"Taken off... !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")

        self.send_command['axes'][1] = 0

    async def land(self):
        """Land the robot"""
        print(f"Landing... !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        self.send_command['buttons'][3] = 1
        # await asyncio.sleep(3.1) original from Tamir xtend
        await asyncio.sleep(4.1) # +1 sec
        print(f"Landed... !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        self.send_command['buttons'][3] = 0

        await asyncio.sleep(2.0)

    async def move_forward(self, duration: float):
        """Move the robot forward"""

        print(f"Moving forward... !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        self.send_command['axes'][2] = 500 # original from Tamir Xtend 700
        await asyncio.sleep(duration * 0.001)
        print(f"Moved forward... !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        self.send_command['axes'][2] = 0

        await asyncio.sleep(2)
    
    async def move_backward(self, duration: float):
        """Move the robot backward"""
        
        print(f"Moving forward... !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        self.send_command['axes'][2] = -700
        await asyncio.sleep(duration * 0.001)
        print(f"Moved forward... !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        self.send_command['axes'][2] = 0

        await asyncio.sleep(2)
    
    async def move_left(self, duration: float):
        """Move the robot left"""

        print(f"Moving left... !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        self.send_command['axes'][0] = -700
        await asyncio.sleep(duration * 0.001)
        self.send_command['axes'][0] = 0
    
    async def move_right(self, duration: float):
        """Move the robot right"""
        print(f"Moving right... !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        self.send_command['axes'][0] = 700
        await asyncio.sleep(duration * 0.001)
        print(f"Moved right... !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        self.send_command['axes'][0] = 0
    
    async def move_up(self, duration: float):
        """Move the robot up"""
        print(f"Moving up... !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        self.send_command['axes'][1] = 700
        await asyncio.sleep(duration * 0.001)
        print(f"Moved up... !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        self.send_command['axes'][1] = 0
    
    async def move_down(self, duration: float):
        """Move the robot down"""
        print(f"Moving down... !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        self.send_command['axes'][1] = -700
        await asyncio.sleep(duration * 0.001)
        print(f"Moved down... !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        self.send_command['axes'][1] = 0
    
    async def rotate_left(self, duration: float = 0.0):
        """Rotate the robot left"""
        print(f"Rotating left... !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        self.send_command['axes'][3] = -1000
        if duration == 0:
            starting_yaw = self.current_yaw
            while starting_yaw != self.current_yaw:
                await asyncio.sleep(0.01)
        else:
            await asyncio.sleep(duration * 0.001)
        print(f"Rotated left... !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        self.send_command['axes'][3] = 0
    
    async def rotate_right(self, duration: float = 0.0):
        """Rotate the robot right"""
        
        print(f"Rotating right... !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        self.send_command['axes'][3] = 1000
        if duration == 0:
            starting_yaw = self.current_yaw
            while starting_yaw != self.current_yaw:
                await asyncio.sleep(0.01)
        else:
            await asyncio.sleep(duration * 0.001)
        print(f"Rotated right... !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        self.send_command['axes'][3] = 0

    async def full_rotation(self, direction: int):
        """Rotate the robot full circle in the given direction"""
        if direction == 1:
            await self.rotate_left()
        elif direction == -1:
            await self.rotate_right()
        else:
            raise ValueError("Invalid direction. Must be 1 or -1.")
        pass

    def hover(self):
        """Hover the robot"""
        self.send_command = self.base_command

    def update_robot_telemetry(self, yaw: float):
        """Update the robot telemetry with the given yaw"""
        self.current_yaw = yaw
        # print(f"Current yaw: {self.current_yaw} radians -> {self.current_yaw * 180 / math.pi} degrees")

    async def create_scenario(self):
        """Create a scenario"""
        sleep_time = 3

        print(f"Creating scenario... !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")

        await asyncio.sleep(5)
        print(f"Scenario created... !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")

        scenario = [
            self.disarm_robot(),
            self.arm_robot(),
            self.takeoff(),
            self.move_forward(400),
            self.move_backward(400),
            # self.move_left(100),
            # self.move_right(100),
            # self.move_up(100),
            # self.move_down(100),
            self.rotate_left(3000),
            # self.rotate_right(100),
            # self.full_rotation(1),
            # self.full_rotation(-1),
            self.land(),
            self.disarm_robot(),
        ]

        for step in scenario:
            await step
            print("before sleep")
            await asyncio.sleep(sleep_time)
            print("after sleep")


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
