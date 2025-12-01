#!/usr/bin/env python3

import time
from typing import Optional, List, Tuple

import rclpy
from rclpy.node import Node

from fcu_driver_interfaces.msg import ManualControl
from rooster_handler_interfaces.msg import KeepAlive
from rooster_manager_interfaces.msg import RoosterState

from manual_core import ManualCommandModel, CsvLogger


class AxisSweepTesterNode(Node):
    """
    Automatic test runner for each axis and simple combinations.

    Runs sequences like:
    - x: 0 -> +A -> 0 -> -A -> 0
    for x, y, z, r, then some (x,y) and (x,z) combinations.
    """

    def __init__(self):
        super().__init__("axis_sweep_tester")

        self.declare_parameter("rooster_id", "R1")
        self.declare_parameter("amplitude", 400.0)
        self.declare_parameter("step_amplitude", 100.0)
        self.declare_parameter("hold_sec", 1.0)
        self.declare_parameter("log_path", "axis_sweep_log.csv")
        self.declare_parameter("flight_mode", 1)

        self.rooster_id = self.get_parameter("rooster_id").get_parameter_value().string_value
        self.amplitude = self.get_parameter("amplitude").get_parameter_value().double_value
        self.step_amp = self.get_parameter("step_amplitude").get_parameter_value().double_value
        self.hold_sec = self.get_parameter("hold_sec").get_parameter_value().double_value
        self.flight_mode = self.get_parameter("flight_mode").get_parameter_value().integer_value
        log_path = self.get_parameter("log_path").get_parameter_value().string_value

        self.command_model = ManualCommandModel(step=self.step_amp)
        self.logger = CsvLogger(self, log_path)

        self.last_state: Optional[RoosterState] = None

        manual_topic = f"/{self.rooster_id}/manual_control"
        keep_alive_topic = f"/{self.rooster_id}/keep_alive"
        state_topic = f"/{self.rooster_id}/state"

        self.manual_pub = self.create_publisher(ManualControl, manual_topic, 10)
        self.keep_alive_pub = self.create_publisher(KeepAlive, keep_alive_topic, 10)
        self.create_subscription(RoosterState, state_topic, self.state_callback, 10)

        self.get_logger().info(
            f"AxisSweepTester for {self.rooster_id}: amplitude={self.amplitude}, "
            f"step={self.step_amp}, hold_sec={self.hold_sec}"
        )

    def state_callback(self, msg: RoosterState):
        self.last_state = msg

    def publish_and_log(self, src: str):
        axes = self.command_model.get_scaled_axes()
        msg = ManualControl()
        msg.x = axes.x
        msg.y = axes.y
        msg.z = axes.z
        msg.r = axes.r
        msg.buttons = 0

        self.manual_pub.publish(msg)
        now = self.get_clock().now().nanoseconds / 1e9
        self.logger.log_command(src, msg, self.last_state, now)

    def send_keep_alive(self):
        msg = KeepAlive()
        msg.is_active = True
        msg.requested_flight_mode = int(self.flight_mode)
        msg.command_reboot = False
        self.keep_alive_pub.publish(msg)

    def _hold_current(self, src: str, seconds: float):
        end_time = time.time() + seconds
        while time.time() < end_time and rclpy.ok():
            self.publish_and_log(src)
            self.send_keep_alive()
            time.sleep(1.0 / 40.0)

    def _test_single_axis(self, axis: str):
        self.get_logger().info(f"Testing axis {axis}: 0 -> +A -> 0 -> -A -> 0")

        self.command_model.reset_axes()
        self._hold_current(f"{axis}_zero_start", self.hold_sec)

        # +amplitude
        self.command_model.set_axes(
            x=self.amplitude if axis == "x" else 0.0,
            y=self.amplitude if axis == "y" else 0.0,
            z=self.amplitude if axis == "z" else 0.0,
            r=self.amplitude if axis == "r" else 0.0,
        )
        self._hold_current(f"{axis}_plus", self.hold_sec)

        # back to zero
        self.command_model.reset_axes()
        self._hold_current(f"{axis}_back_zero", self.hold_sec)

        # -amplitude
        self.command_model.set_axes(
            x=-self.amplitude if axis == "x" else 0.0,
            y=-self.amplitude if axis == "y" else 0.0,
            z=-self.amplitude if axis == "z" else 0.0,
            r=-self.amplitude if axis == "r" else 0.0,
        )
        self._hold_current(f"{axis}_minus", self.hold_sec)

        # back to zero
        self.command_model.reset_axes()
        self._hold_current(f"{axis}_end_zero", self.hold_sec)

    def _test_combo(self, name: str, axes_values: Tuple[float, float, float, float]):
        x, y, z, r = axes_values
        self.get_logger().info(f"Testing combo {name}: x={x}, y={y}, z={z}, r={r}")

        self.command_model.reset_axes()
        self._hold_current(f"{name}_zero_start", self.hold_sec)

        self.command_model.set_axes(x, y, z, r)
        self._hold_current(f"{name}_active", self.hold_sec)

        self.command_model.reset_axes()
        self._hold_current(f"{name}_end_zero", self.hold_sec)

    def run_tests(self):
        self.get_logger().info("Starting axis sweep tests in 3 seconds...")
        for _ in range(3 * 40):
            self._hold_current("idle", 1.0 / 40.0)

        # Single axes
        for axis in ["x", "y", "z", "r"]:
            self._test_single_axis(axis)

        # Simple combinations
        half = self.amplitude / 2.0
        combos: List[Tuple[str, Tuple[float, float, float, float]]] = [
            ("xy_forward_right", (half, half, 0.0, 0.0)),
            ("xz_forward_up", (half, 0.0, half, 0.0)),
        ]
        for name, vals in combos:
            self._test_combo(name, vals)

        self.get_logger().info("Axis sweep tests finished.")


def main(args=None):
    rclpy.init(args=args)
    node = AxisSweepTesterNode()

    try:
        node.run_tests()
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
