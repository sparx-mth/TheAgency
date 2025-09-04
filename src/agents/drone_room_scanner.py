import asyncio
import websockets
import json
import math


class WebSocketClient:
    def __init__(self, uri="ws://127.0.0.1:8000"):
        self.uri = uri
        self.websocket = None
        self.callbacks = {}
        self.robot_positions = {}  # Store current positions

    async def connect(self):
        self.websocket = await websockets.connect(self.uri)
        print("Connected to WebSocket server.")
        asyncio.create_task(self.listen())

    async def listen(self):
        try:
            async for message in self.websocket:
                try:
                    data = json.loads(message)
                    command = data.get("command")
                    content = data.get("content", {})

                    # Capture robot status messages
                    if command == "ROBOT_STATUS":
                        self.update_robot_positions(content)

                    if command in self.callbacks:
                        self.callbacks[command](content)
                except json.JSONDecodeError:
                    print("Invalid JSON received:", message)
        except websockets.exceptions.ConnectionClosed:
            print("WebSocket connection closed.")

    def update_robot_positions(self, status_content):
        """Update stored robot positions from status messages"""
        robots = status_content.get("robots", [])
        for robot in robots:
            robot_id = robot.get("uid")
            telemetry = robot.get("telemetry", {})
            if robot_id and telemetry:
                self.robot_positions[robot_id] = {
                    "latitude": telemetry.get("latitude"),
                    "longitude": telemetry.get("longitude"),
                    "altitude": telemetry.get("altitude")
                }

    async def send_command(self, command, content={}):
        if self.websocket:
            message = json.dumps({"command": command, "content": content})
            await self.websocket.send(message)
            print(f"Sent: {message}")
        else:
            print("WebSocket not connected, cannot send command.")

    async def request_robot_status(self):
        """Request current robot status"""
        await self.send_command("GET_ROBOT_STATUS", {})

    def add_callback(self, command_id, callback):
        self.callbacks[command_id] = callback

    def get_robot_position(self, robot_id):
        """Get current position of a robot"""
        return self.robot_positions.get(robot_id)

    @staticmethod
    def calculate_scan_points(center_lat, center_lon, radius, num_points, altitude, altitude_type, speed):
        """Calculate circular points for room scanning"""
        points = []
        for i in range(num_points):
            angle = 2 * math.pi * i / num_points
            # Calculate circular waypoints
            lat = center_lat + (radius / 111320) * math.cos(angle)
            lon = center_lon + (radius / (111320 * math.cos(math.radians(center_lat)))) * math.sin(angle)
            points.append({
                "latitude": lat,
                "longitude": lon,
                "altitude": altitude,
                "altitude_type": altitude_type,
                "details": {"speed": speed}
            })
        # Add first point again to complete the circle
        points.append(points[0])
        return points


class DroneController:
    def __init__(self, drone_id, target_uid, client: WebSocketClient):
        self.drone_id = drone_id
        self.target_uid = target_uid
        self.client = client
        self.points = None

    def set_scan_pattern_from_current_position(self):
        """Set scan pattern based on drone's current position"""
        position = self.client.get_robot_position(self.drone_id)

        if position and position["latitude"] and position["longitude"]:
            print(f"Drone {self.drone_id} current position: lat={position['latitude']}, lon={position['longitude']}")

            # Create circular scan pattern around current position
            self.points = WebSocketClient.calculate_scan_points(
                center_lat=position["latitude"],
                center_lon=position["longitude"],
                radius=5,  # 5 meter radius for room coverage
                num_points=12,  # 12 points for smooth circular path
                altitude=2.5,  # 2.5 meters height as requested
                altitude_type=0,  # Absolute altitude
                speed=1.0  # Moderate speed for scanning
            )
            return True
        else:
            print(f"Could not get position for drone {self.drone_id}")
            return False

    async def upload_mission(self):
        if not self.points:
            print(f"No points set for drone {self.drone_id}")
            return

        await self.client.send_command("UPLOAD_MISSION", {
            "uid": f"upload_{self.drone_id}",
            "pilot_station_uid": "OMEN-ADSL",
            "robot_uid": self.drone_id,
            "point_type": 1,
            "target_uid": self.target_uid,
            "mission_type": 2,
            "points": self.points
        })

    async def engage(self):
        await self.client.send_command("AUTONOMOUS_CONTROL", {
            "pilot_station_uid": "OMEN-ADSL",
            "robot_uid": self.drone_id,
            "target_uid": self.target_uid,
            "uid": f"engage_{self.drone_id}",
            "user_uid": "daphna",
            "action": 3  # Engage
        })

    async def land(self):
        await self.client.send_command("AUTONOMOUS_CONTROL", {
            "pilot_station_uid": "OMEN-ADSL",
            "robot_uid": self.drone_id,
            "target_uid": self.target_uid,
            "uid": f"land_{self.drone_id}",
            "user_uid": "daphna",
            "action": 7  # Land
        })


async def main():
    client = WebSocketClient()

    await client.connect()
    await asyncio.sleep(2)

    # Request robot status to get current positions
    print("Getting current drone positions...")
    await client.request_robot_status()
    await asyncio.sleep(3)  # Wait for status response

    # Create drone controllers
    drone1 = DroneController("drn12345678", target_uid=1, client=client)

    # Set scan patterns based on current positions
    if not drone1.set_scan_pattern_from_current_position():
        # Fallback to hardcoded position if can't get current position
        print("Using fallback position for drone1")
        drone1.points = WebSocketClient.calculate_scan_points(
            center_lat=32.481201111,
            center_lon=34.174553111,
            radius=5,
            num_points=12,
            altitude=2.5,  # 2.5 meters height
            altitude_type=0,
            speed=1.0
        )


    # Upload missions
    await asyncio.gather(
        drone1.upload_mission(),
    )

    await asyncio.sleep(3)

    # Engage autonomous flight
    await asyncio.gather(
        drone1.engage(),
    )

    # Scan for a while...
    await asyncio.sleep(60)

    # Land both drones
    await asyncio.gather(
        drone1.land(),
    )

    # Keep alive
    while True:
        await asyncio.sleep(1)


if __name__ == "__main__":
    asyncio.run(main())