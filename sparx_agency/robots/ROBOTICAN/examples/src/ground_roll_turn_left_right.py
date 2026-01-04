#!/usr/bin/python3

import time

import rclpy
from rclpy.node import Node
from std_srvs.srv import SetBool

from fcu_driver_interfaces.msg import ManualControl
from rooster_handler_interfaces.msg import KeepAlive
from rooster_manager_interfaces.msg import RoosterState


class GroundRollTurnLR(Node):
    """
    Sequence:
    1. Set manual control to zero
    2. Request GROUND_ROLL mode via keep_alive
    3. Force arm
    4. Turn left (yaw) for 1 second
    5. Stop for 0.5 seconds
    6. Turn right (yaw) for 1 second
    7. Stop and remain idle
    """

    def __init__(self):
        super().__init__("ground_roll_turn_left_right")

        # Drone ID
        self.id = "R2"

        # State tracking
        self.arm_state = False
        self.flight_mode = None
        self.mode_num = KeepAlive.FLIGHT_MODE_GROUND_ROLL

        # Timing
        self.step = 1
        self.phase_start_time = None

        # Service: force arm
        self.force_arm = self.create_client(
            SetBool,
            f"/{self.id}/fcu/command/force_arm",
        )

        # Manual control message and publisher
        self.manual_control = ManualControl()
        self.manual_control.x = 0.0
        self.manual_control.y = 0.0
        self.manual_control.z = 0.0
        self.manual_control.r = 0.0
        self.manual_control.buttons = 0

        self.manual_pub = self.create_publisher(
            ManualControl,
            f"/{self.id}/manual_control",
            10,
        )
        # Publish at 40 Hz
        self.manual_timer = self.create_timer(1.0 / 40.0, self.publish_manual)

        # State subscription
        self.state_sub = self.create_subscription(
            RoosterState,
            f"/{self.id}/state",
            self.drone_state_cb,
            10,
        )

        # Keep-alive / flight mode publisher
        self.keep_alive_pub = self.create_publisher(
            KeepAlive,
            f"/{self.id}/keep_alive",
            10,
        )
        self.keep_alive_timer = self.create_timer(1.0, self.publish_keep_alive)

        # Step machine timer
        self.step_timer = self.create_timer(0.1, self.step_timer_cb)

    # --- Callbacks ---

    def drone_state_cb(self, msg: RoosterState):
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
        self.keep_alive_pub.publish(keep_alive_msg)

    def publish_manual(self):
        self.manual_pub.publish(self.manual_control)

    def step_timer_cb(self):
        # Step 1: ensure manual control is zero
        if self.step == 1:
            self.get_logger().info("Step 1: zero manual control")
            self.manual_control.x = 0.0
            self.manual_control.y = 0.0
            self.manual_control.z = 0.0
            self.manual_control.r = 0.0
            self.step = 2

        # Step 2: arm via force_arm
        elif self.step == 2:
            self.get_logger().info("Step 2: request arm")

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
                    self.get_logger().info("Armed. Proceed to turn left.")
                    self.step = 3
                else:
                    self.get_logger().error(f"Arm failed: {result.message}")

            future.add_done_callback(arm_callback)
            self.step = 2.5  # temporary state to avoid multiple calls

        # Step 3: turn left for 1 second
        elif self.step == 3:
            self.get_logger().info("Step 3: turn LEFT for 1 second")

            # Base drive force on z, yaw left on r.
            self.manual_control.x = 0.0
            self.manual_control.y = 0.0
            self.manual_control.z = 400.0  # base "force"
            self.manual_control.r = 200.0  # positive yaw = left (adjust if needed)

            self.phase_start_time = time.time()
            self.step = 4

        # Step 4: wait for end of left turn
        elif self.step == 4:
            if self.phase_start_time is None:
                self.phase_start_time = time.time()

            elapsed = time.time() - self.phase_start_time
            if elapsed >= 1.5:
                self.get_logger().info("Left turn complete, stopping")
                self.manual_control.x = 0.0
                self.manual_control.y = 0.0
                self.manual_control.z = 0.0
                self.manual_control.r = 0.0

                self.phase_start_time = time.time()
                self.step = 5

        # Step 5: pause 0.5 seconds
        elif self.step == 5:
            elapsed = time.time() - self.phase_start_time
            if elapsed >= 2.5:
                self.get_logger().info("Pause complete, start RIGHT turn")
                self.step = 6

        # Step 6: turn right for 1 second
        elif self.step == 6:
            self.get_logger().info("Step 6: turn RIGHT for 1 second")

            self.manual_control.x = 0.0
            self.manual_control.y = 0.0
            self.manual_control.z = 400.0
            self.manual_control.r = -400.0  # negative yaw = right (adjust if needed)

            self.phase_start_time = time.time()
            self.step = 7

        # Step 7: wait for end of right turn
        elif self.step == 7:
            elapsed = time.time() - self.phase_start_time
            if elapsed >= 2.0:
                self.get_logger().info("Right turn complete, stopping and idling")
                self.manual_control.x = 0.0
                self.manual_control.y = 0.0
                self.manual_control.z = 0.0
                self.manual_control.r = 0.0
                self.step = 8

        # Step 8: done; keep keep_alive and zero commands
        elif self.step == 8:
            # You can optionally disarm here or exit.
            pass


def main(args=None):
    rclpy.init(args=args)
    node = GroundRollTurnLR()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
