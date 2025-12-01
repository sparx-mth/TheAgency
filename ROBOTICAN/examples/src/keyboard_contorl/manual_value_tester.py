#!/usr/bin/env python3

from typing import Optional

import rclpy
from rclpy.node import Node

from fcu_driver_interfaces.msg import ManualControl
from rooster_handler_interfaces.msg import KeepAlive
from rooster_manager_interfaces.msg import RoosterState

from manual_core import ManualCommandModel, CsvLogger


class ManualValueTesterNode(Node):
    """
    Interactive tester:

    - Lets you enter exact x/y/z/r and duration.
    - Publishes constant command for that duration.
    - Logs everything to CSV.
    """

    def __init__(self):
        super().__init__("manual_value_tester")

        self.declare_parameter("rooster_id", "R1")
        self.declare_parameter("log_path", "manual_value_log.csv")
        self.declare_parameter("flight_mode", 1)

        self.rooster_id = self.get_parameter("rooster_id").get_parameter_value().string_value
        self.flight_mode = self.get_parameter("flight_mode").get_parameter_value().integer_value
        log_path = self.get_parameter("log_path").get_parameter_value().string_value

        self.command_model = ManualCommandModel(step=10.0)
        self.logger = CsvLogger(self, log_path)

        self.last_state: Optional[RoosterState] = None

        manual_topic = f"/{self.rooster_id}/manual_control"
        keep_alive_topic = f"/{self.rooster_id}/keep_alive"
        state_topic = f"/{self.rooster_id}/state"

        self.manual_pub = self.create_publisher(ManualControl, manual_topic, 10)
        self.keep_alive_pub = self.create_publisher(KeepAlive, keep_alive_topic, 10)
        self.create_subscription(RoosterState, state_topic, self.state_callback, 10)

        self.get_logger().info(
            f"ManualValueTester for {self.rooster_id}. "
            "Use: enter x y z r duration_sec (e.g. '200 0 0 0 1.5'). "
            "Empty line to quit."
        )

    def state_callback(self, msg: RoosterState):
        self.last_state = msg

    def send_keep_alive(self):
        msg = KeepAlive()
        msg.is_active = True
        msg.requested_flight_mode = int(self.flight_mode)
        msg.command_reboot = False
        self.keep_alive_pub.publish(msg)

    def send_constant_command(self, x: float, y: float, z: float, r: float, duration: float):
        import time

        self.command_model.set_axes(x, y, z, r)
        end_time = time.time() + duration
        self.get_logger().info(
            f"Sending x={x}, y={y}, z={z}, r={r} for {duration:.2f} seconds"
        )
        while time.time() < end_time and rclpy.ok():
            axes = self.command_model.get_scaled_axes()
            msg = ManualControl()
            msg.x = axes.x
            msg.y = axes.y
            msg.z = axes.z
            msg.r = axes.r
            msg.buttons = 0

            self.manual_pub.publish(msg)
            now = self.get_clock().now().nanoseconds / 1e9
            self.logger.log_command("manual", msg, self.last_state, now)

            self.send_keep_alive()
            time.sleep(1.0 / 40.0)

        # Back to zero at the end
        self.command_model.reset_axes()
        axes_zero = self.command_model.get_scaled_axes()
        msg = ManualControl()
        msg.x = axes_zero.x
        msg.y = axes_zero.y
        msg.z = axes_zero.z
        msg.r = axes_zero.r
        msg.buttons = 0
        self.manual_pub.publish(msg)

    def interactive_loop(self):
        import sys

        while rclpy.ok():
            sys.stdout.write(
                "\nEnter 'x y z r duration_sec' "
                "(or empty line to quit): "
            )
            sys.stdout.flush()
            line = sys.stdin.readline()
            if not line:
                break
            line = line.strip()
            if not line:
                self.get_logger().info("Empty line, exiting interactive loop.")
                break

            parts = line.split()
            if len(parts) != 5:
                self.get_logger().warn("Expected 5 values: x y z r duration_sec")
                continue

            try:
                x = float(parts[0])
                y = float(parts[1])
                z = float(parts[2])
                r = float(parts[3])
                duration = float(parts[4])
                if duration <= 0:
                    self.get_logger().warn("Duration must be positive.")
                    continue
            except ValueError:
                self.get_logger().warn("Failed to parse numbers.")
                continue

            self.send_constant_command(x, y, z, r, duration)


def main(args=None):
    rclpy.init(args=args)
    node = ManualValueTesterNode()

    try:
        # Spin in a side thread for ROS callbacks (state)
        import threading

        def spin_thread():
            while rclpy.ok():
                rclpy.spin_once(node, timeout_sec=0.1)

        t = threading.Thread(target=spin_thread, daemon=True)
        t.start()

        node.interactive_loop()
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
