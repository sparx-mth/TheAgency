#!/usr/bin/python3

import time

import rclpy
from rclpy.node import Node
from std_srvs.srv import SetBool

from fcu_driver_interfaces.msg import ManualControl
from rooster_handler_interfaces.msg import KeepAlive
from rooster_manager_interfaces.msg import RoosterState


class OneSecondRoll(Node):
    """
    Sequence:
    1. Set manual control to zero
    2. Request GROUND_ROLL flight mode via keep_alive
    3. Force arm
    4. Command roll for 1 second
    5. Stop (manual control back to zero)
    """

    def __init__(self):
        super().__init__('one_second_roll_demo')

        # Select drone
        self.id = "R2"

        # Track state
        self.arm_state = False
        self.flight_mode = None
        self.mode_num = KeepAlive.FLIGHT_MODE_GROUND_ROLL

        # Service client: force arm
        self.force_arm = self.create_client(SetBool, f'/{self.id}/fcu/command/force_arm')

        # Manual control publisher
        self.manual_control = ManualControl()
        self.manual_control.x = 0.0
        self.manual_control.y = 0.0
        self.manual_control.z = 0.0
        self.manual_control.r = 0.0
        self.manual_control.buttons = 0

        self.manual_control_pub = self.create_publisher(
            ManualControl,
            f'/{self.id}/manual_control',
            10
        )
        # Publish manual control at 40 Hz
        self.timer_manual = self.create_timer(1.0 / 40.0, self.publish_manual_control)

        # State subscription
        self.state_node_sub = self.create_subscription(
            RoosterState,
            f'/{self.id}/state',
            self.drone_state_cb,
            10
        )

        # Keep-alive / flight mode publisher
        self.flight_mode_pub = self.create_publisher(
            KeepAlive,
            f'/{self.id}/keep_alive',
            10
        )
        self.keep_alive_timer = self.create_timer(1.0, self.publish_keep_alive)

        # Step machine
        self.step = 1
        self.roll_start_time = None
        # Check steps at 10 Hz to get nicer timing for the 1-second roll
        self.step_timer = self.create_timer(0.1, self.step_timer_cb)

    # --- Callbacks ---

    def drone_state_cb(self, msg: RoosterState):
        # Simple debug print when armed / flight mode change
        if self.arm_state != msg.armed or self.flight_mode != msg.flight_mode:
            self.get_logger().info(
                f"state armed: {msg.armed} flight_mode: {msg.flight_mode}"
            )
            self.arm_state = msg.armed
            self.flight_mode = msg.flight_mode

    def publish_keep_alive(self):
        keep_alive_msg = KeepAlive()
        keep_alive_msg.is_active = True
        keep_alive_msg.requested_flight_mode = self.mode_num
        keep_alive_msg.command_reboot = False
        self.flight_mode_pub.publish(keep_alive_msg)

    def publish_manual_control(self):
        self.manual_control_pub.publish(self.manual_control)

    def step_timer_cb(self):
        # Step 1: make sure manual control is zero
        if self.step == 1:
            self.get_logger().info("Step 1: Set manual control to zero")
            self.manual_control.x = 0.0
            self.manual_control.y = 0.0
            self.manual_control.z = 0.0
            self.manual_control.r = 0.0
            self.step = 2

        # Step 2: request arm via force_arm
        elif self.step == 2:
            self.get_logger().info("Step 2: Request arm")

            if not self.force_arm.wait_for_service(timeout_sec=1.0):
                self.get_logger().warn("force_arm service not available yet")
                return

            req = SetBool.Request()
            req.data = True
            future = self.force_arm.call_async(req)

            def arm_callback(fut):
                try:
                    result = fut.result()
                except Exception as e:
                    self.get_logger().error(f"Arm service call failed: {e}")
                    return

                if result.success:
                    self.get_logger().info("Armed. Proceed to roll step.")
                    self.step = 3
                else:
                    self.get_logger().error(f"Arm failed: {result.message}")

            future.add_done_callback(arm_callback)
            # Avoid spamming the service; wait for callback to advance
            self.step = 2.5  # temporary state to prevent re-calls

        # Step 3: start the roll command for 1 second
        elif self.step == 3:
            self.get_logger().info("Step 3: Start rolling for 1 second")

            # Example values – tune to your platform
            # x: forward / roll command, z: throttle
            self.manual_control.x = 500.0
            self.manual_control.y = 0.0
            self.manual_control.z = 200.0
            self.manual_control.r = 0.0

            self.roll_start_time = time.time()
            self.step = 4

        # Step 4: check if 1 second has passed, then stop
        elif self.step == 4:
            if self.roll_start_time is None:
                # Should not happen, but guard anyway
                self.roll_start_time = time.time()

            elapsed = time.time() - self.roll_start_time
            if elapsed >= 4.0:
                self.get_logger().info("Step 4: Roll duration reached, stopping")
                self.manual_control.x = 0.0
                self.manual_control.y = 0.0
                self.manual_control.z = 0.0
                self.manual_control.r = 0.0
                self.step = 5

        # Step 5: done (keep keep-alive running, manual control stays zero)
        elif self.step == 5:
            # No-op; you could optionally disarm here or shutdown node
            pass


def main(args=None):
    rclpy.init(args=args)
    node = OneSecondRoll()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
