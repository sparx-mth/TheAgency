#!/usr/bin/python3

import time

import rclpy
from rclpy.node import Node
from std_srvs.srv import SetBool

from fcu_driver_interfaces.msg import ManualControl
from rooster_handler_interfaces.msg import KeepAlive
from rooster_manager_interfaces.msg import RoosterState


class GroundRollCourse(Node):
    """
    Sequence for R2 in GROUND_ROLL:

    1. Forward 5 s  (x=400, z=200)
    2. Right turn 2.0 s (z=400, r=-200)
    3. Forward 7 s  (x=400, z=200)
    4. Right turn 2.5 s (z=400, r=-200)
    5. Forward 5 s  (x=500, z=200)
    6. Then disarm and exit.
    """

    def __init__(self):
        super().__init__("ground_roll_course")

        self.id = "R3"

        # State
        self.arm_state = False
        self.flight_mode = None
        self.mode_num = KeepAlive.FLIGHT_MODE_GROUND_ROLL

        self.step = 1
        self.phase_start_time = None

        # Services
        self.force_arm = self.create_client(
            SetBool,
            f"/{self.id}/fcu/command/force_arm",
        )

        # Manual control
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
        # 40 Hz to satisfy watchdog
        self.manual_timer = self.create_timer(1.0 / 40.0, self.publish_manual)

        # State subscription
        self.state_sub = self.create_subscription(
            RoosterState,
            f"/{self.id}/state",
            self.drone_state_cb,
            10,
        )

        # Keep alive / flight mode
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

    def now(self):
        # monotonic is nicer than time.time() for durations
        return time.monotonic()

    def step_timer_cb(self):
        # Step 1: zero manual control
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
                    self.get_logger().info("Armed. Start course.")
                    self.step = 3
                else:
                    self.get_logger().error(f"Arm failed: {result.message}")

            future.add_done_callback(arm_callback)
            self.step = 2.5  # avoid repeated calls

        # Step 3: forward 5 s (x=400, z=200)
        elif self.step == 3:
            self.get_logger().info("Step 3: forward 5 s (x=400, z=200)")
            self.manual_control.x = 400.0
            self.manual_control.y = 0.0
            self.manual_control.z = 200.0
            self.manual_control.r = 0.0

            self.phase_start_time = self.now()
            self.step = 4

        # Step 4: wait for end of forward 5 s
        elif self.step == 4:
            elapsed = self.now() - self.phase_start_time
            if elapsed >= 5.0:
                self.get_logger().info("Forward 1 complete, stopping")
                self.manual_control.x = 0.0
                self.manual_control.y = 0.0
                self.manual_control.z = 0.0
                self.manual_control.r = 0.0
                self.step = 5

        # Step 5: right turn 2.0 s
        elif self.step == 5:
            self.get_logger().info("Step 5: right turn 2.0 s")
            self.manual_control.x = 0.0
            self.manual_control.y = 0.0
            self.manual_control.z = 400.0
            self.manual_control.r = -200.0  # right

            self.phase_start_time = self.now()
            self.step = 6

        # Step 6: wait for end of right turn 2.0 s
        elif self.step == 6:
            elapsed = self.now() - self.phase_start_time
            if elapsed >= 2.0:
                self.get_logger().info("Right turn 1 complete, stopping")
                self.manual_control.x = 0.0
                self.manual_control.y = 0.0
                self.manual_control.z = 0.0
                self.manual_control.r = 0.0
                self.step = 7

        # Step 7: forward 7 s (x=400, z=200)
        elif self.step == 7:
            self.get_logger().info("Step 7: forward 7 s (x=400, z=200)")
            self.manual_control.x = 400.0
            self.manual_control.y = 0.0
            self.manual_control.z = 200.0
            self.manual_control.r = 0.0

            self.phase_start_time = self.now()
            self.step = 8

        # Step 8: wait for end of forward 7 s
        elif self.step == 8:
            elapsed = self.now() - self.phase_start_time
            if elapsed >= 7.0:
                self.get_logger().info("Forward 2 complete, stopping")
                self.manual_control.x = 0.0
                self.manual_control.y = 0.0
                self.manual_control.z = 0.0
                self.manual_control.r = 0.0
                self.step = 9

        # Step 9: right turn 2.5 s
        elif self.step == 9:
            self.get_logger().info("Step 9: right turn 2.5 s")
            self.manual_control.x = 0.0
            self.manual_control.y = 0.0
            self.manual_control.z = 400.0
            self.manual_control.r = -200.0  # right

            self.phase_start_time = self.now()
            self.step = 10

        # Step 10: wait for end of right turn 2.5 s
        elif self.step == 10:
            elapsed = self.now() - self.phase_start_time
            if elapsed >= 2.5:
                self.get_logger().info("Right turn 2 complete, stopping")
                self.manual_control.x = 0.0
                self.manual_control.y = 0.0
                self.manual_control.z = 0.0
                self.manual_control.r = 0.0
                self.step = 11

        # Step 11: forward 5 s (x=500, z=200)
        elif self.step == 11:
            self.get_logger().info("Step 11: forward 5 s (x=500, z=200)")
            self.manual_control.x = 500.0
            self.manual_control.y = 0.0
            self.manual_control.z = 200.0
            self.manual_control.r = 0.0

            self.phase_start_time = self.now()
            self.step = 12

        # Step 12: wait for end of forward 5 s
        elif self.step == 12:
            elapsed = self.now() - self.phase_start_time
            if elapsed >= 5.0:
                self.get_logger().info("Forward 3 complete, stopping")
                self.manual_control.x = 0.0
                self.manual_control.y = 0.0
                self.manual_control.z = 0.0
                self.manual_control.r = 0.0
                self.step = 13

        # Step 13: disarm and then exit
        elif self.step == 13:
            self.get_logger().info("Step 13: disarm and quit")

            if not self.force_arm.wait_for_service(timeout_sec=1.0):
                self.get_logger().warn("force_arm service not available for disarm")
                return

            req = SetBool.Request()
            req.data = False  # disarm
            future = self.force_arm.call_async(req)

            def disarm_callback(fut):
                try:
                    result = fut.result()
                except Exception as e:
                    self.get_logger().error(f"Disarm service call failed: {e}")
                    self.destroy_node()
                    return

                if result.success:
                    self.get_logger().info("Disarmed successfully, shutting down node")
                else:
                    self.get_logger().error(f"Disarm failed: {result.message}")

                self.destroy_node()

            future.add_done_callback(disarm_callback)
            self.step = 14  # prevent multiple disarm calls

        # Step 14: waiting for disarm callback to destroy node
        elif self.step == 14:
            pass


def main(args=None):
    rclpy.init(args=args)
    node = GroundRollCourse()
    try:
        rclpy.spin(node)
    finally:
        # If we reached here because destroy_node() was called:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
