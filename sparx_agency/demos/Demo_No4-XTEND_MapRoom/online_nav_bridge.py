import asyncio
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
import json

# Importing the base class from your automation.py
from automation import ControllerAutomation


class OnlineNavBridge(ControllerAutomation):
    def __init__(self, host, port, frequency, robot_uid):
        super().__init__(host, port, frequency, robot_uid)

        # Initialize an asynchronous queue for incoming commands
        self.cmd_queue = asyncio.Queue()

        # ROS 2 Node setup
        self.ros_node = rclpy.create_node('drone_bridge_node')

        # Subscriber for direction and time (assumed JSON string for this example)
        self.subscription = self.ros_node.create_subscription(
            String,
            '/drone/cmd_nav',
            self.ros_callback,
            10
        )

    def ros_callback(self, msg):
        """
        Receives direction and time from the point-cloud controller.
        Expected format: {"action": "forward", "value": 1500} (time in ms)
        """
        try:
            data = json.loads(msg.data)
            # Use thread-safe method to push to the asyncio queue
            asyncio.get_event_loop().call_soon_threadsafe(self.cmd_queue.put_nowait, data)
        except Exception as e:
            self.ros_node.get_logger().error(f"Failed to parse nav command: {e}")

    async def dynamic_executor(self):
        """
        Consumes commands from the queue and executes them in real-time.
        Replaces the static create_scenario method.
        """
        print("ONLINE MODE: Waiting for navigation commands...")
        while True:
            # Wait for the next command from the ROS controller
            command = await self.cmd_queue.get()

            action = command.get("action")
            value = command.get("value")  # duration in ms or degrees

            print(f"Executing: {action} with value {value}")

            if action == "forward":
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
                break

            self.cmd_queue.task_done()

    async def run_bridge(self):
        """
        Overriding the execution loop to include ROS 2 spinning.
        """
        # Start the ROS 2 executor in a separate thread to avoid blocking asyncio
        ros_thread = asyncio.to_thread(rclpy.spin, self.ros_node)

        # Run the standard communication loops from ControllerAutomation
        await asyncio.gather(
            self.run(),  # Includes send_message and receive_message
            self.dynamic_executor(),  # Our new dynamic command loop
            ros_thread
        )


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
    finally:
        bridge.ros_node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    asyncio.run(main())