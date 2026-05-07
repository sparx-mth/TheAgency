import asyncio
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
import json
import websockets

from sparx_agency.robots.XTEND.automation import ControllerAutomation


class OnlineNavBridge(ControllerAutomation):
    def __init__(self, host, port, frequency, robot_uid):
        super().__init__(host, port, frequency, robot_uid)

        # 1. Capture the main asyncio loop while we are in the main thread
        self.loop = asyncio.get_event_loop()

        self.cmd_queue = asyncio.Queue()
        self.ros_node = rclpy.create_node('drone_bridge_node')

        self.subscription = self.ros_node.create_subscription(
            String,
            '/drone/cmd_nav',
            self.ros_callback,
            10
        )

    def ros_callback(self, msg):
        """Thread-safe callback to move data from ROS thread to Asyncio thread."""
        try:
            data = json.loads(msg.data)
            # 2. Use call_soon_threadsafe to interact with the queue from the ROS thread
            self.loop.call_soon_threadsafe(self.cmd_queue.put_nowait, data)
            self.ros_node.get_logger().info(f"Queued: {data.get('action')}")
        except Exception as e:
            self.ros_node.get_logger().error(f"Callback Error: {e}")

    async def run_connection(self):
        """Maintains the drone heartbeat/telemetry without a fixed scenario."""
        try:
            async with websockets.connect(self.uri) as websocket:
                print(f"✓ Connected to {self.uri}")

                # Use the existing logic from automation.py for sending/receiving
                send_task = asyncio.create_task(self.send_message(websocket))
                receive_task = asyncio.create_task(self.receive_message(websocket))

                await asyncio.gather(send_task, receive_task)
        except Exception as e:
            print(f"Connection Error: {e}")

    async def dynamic_executor(self):
        """Listens to the queue and executes drone movements as they arrive."""
        print("ONLINE MODE: Ready for commands...")
        while True:
            command = await self.cmd_queue.get()
            action = command.get("action")
            value = command.get("value")

            if action == "takeoff":
                await self.arm_robot()
                await self.takeoff()
            elif action == "forward":
                await self.move_forward(value)
            elif action == "backward":
                await self.move_backward(value)
            elif action == "left":
                await self.move_left(value)
            elif action == "right":
                await self.move_right(value)
            elif action == "rotate_left":
                await self.rotate_left(value)
            elif action == "rotate_right":
                await self.rotate_right(value)
            elif action == "land":
                await self.land()
                await self.disarm_robot()
                break

            self.cmd_queue.task_done()

    async def run_bridge(self):
        """Runs the ROS 2 node and Drone logic in parallel."""
        # Spin ROS 2 in a background thread
        ros_thread = asyncio.to_thread(rclpy.spin, self.ros_node)

        try:
            await asyncio.gather(
                self.run_connection(),
                self.dynamic_executor(),
                ros_thread
            )
        finally:
            self.ros_node.destroy_node()
            rclpy.shutdown()


async def main():
    rclpy.init()
    bridge = OnlineNavBridge(
        host="192.0.0.15",
        port=8000,
        frequency=30.0,
        robot_uid="drn120ea1b0"
    )
    try:
        await bridge.run_bridge()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    asyncio.run(main())