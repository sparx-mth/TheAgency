#!/usr/bin/env python3

import sys
import threading
from typing import Optional

import rclpy
from rclpy.node import Node

from fcu_driver_interfaces.msg import ManualControl
from rooster_handler_interfaces.msg import KeepAlive
from rooster_manager_interfaces.msg import RoosterState

from manual_core import ManualCommandModel, CsvLogger


class KeyboardManualControlNode(Node):
    """
    Keyboard teleop for Robotican / Rooster:

    - Publishes /<ROOSTER_ID>/manual_control at 40 Hz.
    - Publishes /<ROOSTER_ID>/keep_alive at 1 Hz.
    - Logs commands + RoosterState to CSV.

    ROLL mode semantics by default:
    - x: forward/backward rolling
    """

    def __init__(self):
        super().__init__("keyboard_manual_control")

        # Parameters
        self.declare_parameter("rooster_id", "R1")
        self.declare_parameter("step", 10.0)
        self.declare_parameter("log_path", "keyboard_manual_control.csv")
        self.declare_parameter("flight_mode", 1)  # your ROLL/GROUND_ROLL enum

        self.rooster_id: str = self.get_parameter("rooster_id").get_parameter_value().string_value
        step: float = self.get_parameter("step").get_parameter_value().double_value
        self.flight_mode: int = self.get_parameter("flight_mode").get_parameter_value().integer_value
        log_path: str = self.get_parameter("log_path").get_parameter_value().string_value

        # Core models
        self.command_model = ManualCommandModel(step=step, turtle_scale=0.5)
        self.logger = CsvLogger(self, log_path)

        # State
        self.last_state: Optional[RoosterState] = None
        self.shutdown_flag = False

        # Topics
        manual_topic = f"/{self.rooster_id}/manual_control"
        keep_alive_topic = f"/{self.rooster_id}/keep_alive"
        state_topic = f"/{self.rooster_id}/state"

        self.manual_pub = self.create_publisher(ManualControl, manual_topic, 10)
        self.keep_alive_pub = self.create_publisher(KeepAlive, keep_alive_topic, 10)

        self.create_subscription(RoosterState, state_topic, self.state_callback, 10)

        # Timers
        self.manual_timer = self.create_timer(1.0 / 40.0, self.manual_timer_cb)
        self.keep_alive_timer = self.create_timer(1.0, self.keep_alive_timer_cb)

        self.get_logger().info(
            f"KeyboardManualControlNode for {self.rooster_id} "
            f"(step={step}, flight_mode={self.flight_mode})"
        )
        self.get_logger().info(
            "Keys: e/x (x+/x- fwd/back), l/j (y+/y- right/left), "
            "i/k (z+/z- up/down), d/a (r+/r- yaw), SPACE (zero), "
            "t (turtle), q (quit)"
        )

    # ---------- ROS callbacks ----------

    def state_callback(self, msg: RoosterState):
        self.last_state = msg

    def manual_timer_cb(self):
        axes = self.command_model.get_scaled_axes()

        msg = ManualControl()
        msg.x = axes.x
        msg.y = axes.y
        msg.z = axes.z
        msg.r = axes.r
        msg.buttons = 0

        self.manual_pub.publish(msg)

        now = self.get_clock().now().nanoseconds / 1e9
        self.logger.log_command("cmd", msg, self.last_state, now)

    def keep_alive_timer_cb(self):
        msg = KeepAlive()
        msg.is_active = True
        msg.requested_flight_mode = int(self.flight_mode)
        msg.command_reboot = False
        self.keep_alive_pub.publish(msg)

    # ---------- For keyboard thread ----------

    def handle_key(self, ch: str):
        quit_requested = False

        if ch == "e":
            self.command_model.apply_increment("x", +1)
        elif ch == "x":
            self.command_model.apply_increment("x", -1)
        elif ch == "l":
            self.command_model.apply_increment("y", +1)
        elif ch == "j":
            self.command_model.apply_increment("y", -1)
        elif ch == "i":
            self.command_model.apply_increment("z", +1)
        elif ch == "k":
            self.command_model.apply_increment("z", -1)
        elif ch == "d":
            self.command_model.apply_increment("r", +1)
        elif ch == "a":
            self.command_model.apply_increment("r", -1)
        elif ch == " ":
            self.command_model.reset_axes()
            self.get_logger().info("Axes reset to zero")
        elif ch == "t":
            state = self.command_model.toggle_turtle()
            self.get_logger().info(f"Turtle mode: {state}")
        elif ch == "q":
            self.get_logger().info("Quit requested from keyboard.")
            quit_requested = True

        axes = self.command_model.axes
        self.get_logger().info(
            f"Axes: x={axes.x:.1f}, y={axes.y:.1f}, "
            f"z={axes.z:.1f}, r={axes.r:.1f}"
        )

        if quit_requested:
            self.shutdown_flag = True

    def destroy_node(self):
        self.logger.close()
        super().destroy_node()


def keyboard_thread(node: KeyboardManualControlNode):
    import termios
    import tty
    import select

    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    tty.setcbreak(fd)
    try:
        while rclpy.ok() and not node.shutdown_flag:
            rlist, _, _ = select.select([sys.stdin], [], [], 0.1)
            if rlist:
                ch = sys.stdin.read(1)
                if ch:
                    node.handle_key(ch)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
        node.get_logger().info("Keyboard thread stopped")


def main(args=None):
    rclpy.init(args=args)
    node = KeyboardManualControlNode()

    t = threading.Thread(target=keyboard_thread, args=(node,), daemon=True)
    t.start()

    try:
        while rclpy.ok() and not node.shutdown_flag:
            rclpy.spin_once(node, timeout_sec=0.1)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
