#!/usr/bin/env python3
import csv
import os
import time
from typing import List, Tuple

import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor

from std_srvs.srv import SetBool
from std_msgs.msg import Bool
from fcu_driver_interfaces.msg import ManualControl
from rooster_handler_interfaces.msg import KeepAlive
from rooster_manager_interfaces.msg import RoosterState
from fcu_driver_interfaces.msg import Battery as FcuBattery
from sensor_msgs.msg import Image, Imu

from geometry_msgs.msg import PoseStamped
from sphera_common_interfaces.msg import SpheraPawnState
from tf_transformations import quaternion_from_euler


# Altitude / throttle tuning – adjust to your system
TAKEOFF_Z = 590.0               # short gentle boost
HOVER_Z = 550.0              # lower hover so they do not climb too high

GROUND_Z_WORLD = 0.3        # Drone on the ground z < 0.3
DESCENT_Z_WORLD = 1.0       # Drone in the air
LAND_CMD_DESCENT = 450.0    # Stick for landing
LAND_CMD_GROUND  = 400.0    # Stick for ground


# Timing
SPREAD_DURATION = 3.0        # seconds to initially spread drones apart
PATTERN_SEGMENT_DURATION = 2.0  # seconds per segment in the pattern

LOW_BATT_THRESH = 0.3  # 30%


class SingleDroneFlying(Node):
    """
    Force-arm + takeoff + pattern patrol, for a single drone id.

    Sequence:
    1. Step 1: log throttle 0 (initial state)
    2. Step 2: call force_arm
    3. Step 3: short takeoff boost (z=TAKEOFF_Z)
    4. Step 4: switch to hover (z=HOVER_Z) and move away from origin (spread_vector)
    5. Step 5: after SPREAD_DURATION seconds, start pattern
    6. Step 6: cycle through pattern segments every PATTERN_SEGMENT_DURATION seconds
    """

    def __init__(
        self,
        drone_id: str,
        spread_vector: Tuple[float, float],
        pattern: List[Tuple[float, float]],
        pattern_name: str,
    ):
        super().__init__(f"flying_node_{drone_id}")


        self.id = drone_id
        self.mode_num = KeepAlive.FLIGHT_MODE_MANUAL
        self.arm_state = False
        self.flight_mode = None
        self.step = 1
        self.takeoff_start_time = None
        self.last_state_z = None

        # Spread movement (initial separation)
        self.spread_x, self.spread_y = spread_vector
        self.spread_start_time: float | None = None

        # Patrol pattern (circle/square/X)
        self.pattern = pattern
        self.pattern_name = pattern_name
        self.pattern_index = 0
        self.last_pattern_switch_time = 0.0

        # Service client: force arm
        self.force_arm = self.create_client(
            SetBool, f"/{self.id}/fcu/command/force_arm"
        )

        # Manual control publisher
        self.manual_control = ManualControl()
        self.manual_control.x = 0.0
        self.manual_control.y = 0.0
        self.manual_control.z = 0.0
        self.manual_control.r = 0.0
        self.manual_control.buttons = 0

        self.manual_control_pub = self.create_publisher(
            ManualControl, f"/{self.id}/manual_control", 10
        )
        self.timer_arm = self.create_timer(1.0 / 40.0, self.publish_manual_control)

        # Keep-alive publisher
        self.flight_mode_pub = self.create_publisher(
            KeepAlive, f"/{self.id}/keep_alive", 10
        )
        self.keep_alive_timer = self.create_timer(1.0, self.publish_keep_alive)

        # Optional GCS keep-alive (so links/video infra stay alive if needed)
        self.gcs_keep_alive_pub = self.create_publisher(
            Bool, f"/{self.id}/gcs_keep_alive", 10
        )

        # State subscription (for debug / visibility)
        self.state_node_sub = self.create_subscription(
            RoosterState, f"/{self.id}/state", self.drone_state_cb, 10
        )

        # Optional video + sensor subscriptions
        self.image_sub = self.create_subscription(
            Image, f"/{self.id}/camera/image_raw", self.image_cb, 10
        )
        self.imu_sub = self.create_subscription(
            Imu, f"/{self.id}/imu/data", self.imu_cb, 10
        )
        # Battery state subscription
        self.battery_percentage: float | None = None
        self.low_battery_triggered: bool = False
        self.seen_high_battery: bool = False

        self.battery_sub = self.create_subscription(
            FcuBattery, f"/{self.id}/fcu/battery", self.battery_cb, 10
        )
        self.path_samples = []
        # Pose publisher from sphera/state (tracking / sim truth)
        self.pose_pub = self.create_publisher(
            PoseStamped, f"/{self.id}/tracking_pose", 10
        )

        # Sphera/state subscription to get location/rotation
        self.sphera_state_sub = self.create_subscription(
            SpheraPawnState, f"/{self.id}/sphera/state", self.sphera_state_cb, 10
        )

        # Mission step timer (1 Hz)
        self.step_timer = self.create_timer(1.0, self.step_timer_cb)

        self.get_logger().info(
            f"{self.id}: Initialized pattern='{self.pattern_name}', "
            f"spread=({self.spread_x}, {self.spread_y}), "
            f"pattern={self.pattern}"
        )

    # ------------------------------------------------------------------
    # Callbacks
    # ------------------------------------------------------------------

    def drone_state_cb(self, msg: RoosterState):
        if self.arm_state != msg.armed or self.flight_mode != msg.flight_mode:
            self.get_logger().info(
                f"{self.id}: state change: armed={msg.armed}, flight_mode={msg.flight_mode}"
            )
            self.arm_state = msg.armed
            self.flight_mode = msg.flight_mode

    def sphera_state_cb(self, msg: SpheraPawnState):
        # Publish PoseStamped for RViz / other tools
        pose_msg = PoseStamped()
        pose_msg.header.stamp = msg.header.stamp
        pose_msg.header.frame_id = "world"  # or whatever frame you use

        pose_msg.pose.position.x = msg.location.x
        pose_msg.pose.position.y = msg.location.y
        pose_msg.pose.position.z = msg.location.z

        qx, qy, qz, qw = quaternion_from_euler(
            msg.rotation.roll,
            msg.rotation.pitch,
            msg.rotation.yaw,
        )
        pose_msg.pose.orientation.x = qx
        pose_msg.pose.orientation.y = qy
        pose_msg.pose.orientation.z = qz
        pose_msg.pose.orientation.w = qw

        self.pose_pub.publish(pose_msg)
        # self.get_logger().info(f"{self.id}: Sphera state published")
        # Record path sample for offline plotting
        self.last_state_z = msg.location.z
        t_sec = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        self.path_samples.append((t_sec, msg.location.x, msg.location.y, msg.location.z))

    def image_cb(self, msg: Image):
        self.get_logger().debug(f"{self.id}: image frame received")

    def imu_cb(self, msg: Imu):
        self.get_logger().debug(f"{self.id}: IMU sample received")

    def battery_cb(self, msg: FcuBattery):
        # msg.percentage is in [0.0, 1.0] (0%..100%)
        self.battery_percentage = msg.percentage
        if self.battery_percentage > 0.5:
            self.seen_high_battery = True
        # Trigger once when battery drops below threshold
        if (
                self.arm_state
                and not self.low_battery_triggered
                and self.battery_percentage is not None
                and self.battery_percentage <= LOW_BATT_THRESH
        ):
            self.low_battery_triggered = True
            self.low_batt_trigger_value = float(self.battery_percentage)
            self.takeoff_start_time = None
            self.get_logger().warn(
                f"{self.id}: LOW BATTERY TRIGGER at {self.low_batt_trigger_value:.2f}"
            )

    def handle_low_battery(self, now: float):
        # Stop horizontal motion immediately
        self.manual_control.x = 0.0
        self.manual_control.y = 0.0

        if self.step != 999:
            self.step = 999
            self.landing_start_time = now
            self.get_logger().info(
                f"{self.id}: Entering LOW BATTERY landing mode "
                f"(trigger was {self.low_batt_trigger_value:.2f})"
            )

        elapsed = now - getattr(self, "landing_start_time", now)
        z = self.last_state_z
        if z is None:
            elapsed = now - getattr(self, "landing_start_time", now)
            if elapsed < 1.0:
                self.manual_control.z = HOVER_Z
            elif elapsed < 6.0:
                alpha = (elapsed - 1.0) / 5.0
                alpha = min(max(alpha, 0.0), 1.0)
                self.manual_control.z = HOVER_Z + (LAND_CMD_DESCENT - HOVER_Z) * alpha
            else:
                self.manual_control.z = LAND_CMD_GROUND

        else:
            if z > DESCENT_Z_WORLD:
                self.manual_control.z = LAND_CMD_DESCENT
            elif z > GROUND_Z_WORLD:
                self.manual_control.z = (LAND_CMD_DESCENT + LAND_CMD_GROUND) * 0.5
            else:
                self.manual_control.z = LAND_CMD_GROUND

        self.get_logger().info(
            f"{self.id}: Low battery landing: z={self.manual_control.z:.1f}, "
            f"battery_now={self.battery_percentage:.2f}, "
            f"trigger_at={self.low_batt_trigger_value:.2f}"
        )

    def step_timer_cb(self):
        now = time.time()
        # Low-battery override: if triggered, ignore mission logic and land
        if self.low_battery_triggered:
            self.handle_low_battery(now)
            return


        if self.id == "R4":
            if self.step == 1:
                self.get_logger().info(f"{self.id}: Step 1 - Throttle 0, waiting to arm")
                self.step = 2

            if self.step == 2:
                if not self.force_arm.service_is_ready():
                    self.get_logger().warn(f"{self.id}: force_arm service not ready yet")
                    return

                self.get_logger().info(f"{self.id}: Step 2 - Sending force_arm request")
                arm_req = SetBool.Request()
                arm_req.data = True

                future = self.force_arm.call_async(arm_req)

                def arm_callback(fut):
                    try:
                        result = fut.result()
                        if result.success:
                            self.get_logger().info(f"{self.id}: Armed successfully")
                            self.step = 3
                        else:
                            self.get_logger().warn(
                                f"{self.id}: Arm failed: {result.message}"
                            )
                    except Exception as ex:
                        self.get_logger().error(
                            f"{self.id}: force_arm call failed: {ex}"
                        )

                future.add_done_callback(arm_callback)

            if self.step == 3:
                self.manual_control.z = 600.0
                self.takeoff_start_time = now
                self.get_logger().info(
                    f"{self.id}: Step 3 - Takeoff, z={self.manual_control.z}"
                )
                self.step = 4

            if self.step == 4 and self.takeoff_start_time is not None:
                dt = now - self.takeoff_start_time
                if dt > 2.0:
                    self.manual_control.z = 568.0
                    self.manual_control.x = 100.0
                    self.get_logger().info(
                        f"{self.id}: Step 4 - Forward, z={self.manual_control.z}, x={self.manual_control.x}"
                    )
                if dt > 10.0:
                    # stop forward motion even if battery not low yet
                    self.manual_control.x = 0.0
                    self.get_logger().info(f"{self.id}: stopping forward motion")
                return
        else:
            # Step 1: throttle 0 (just log)
            if self.step == 1:
                self.get_logger().info(f"{self.id}: Step 1 - Throttle 0, waiting to arm")
                self.step = 2

            # Step 2: request arm
            if self.step == 2:
                if not self.force_arm.service_is_ready():
                    self.get_logger().warn(f"{self.id}: force_arm service not ready yet")
                    return

                self.get_logger().info(f"{self.id}: Step 2 - Sending force_arm request")
                arm_req = SetBool.Request()
                arm_req.data = True

                future = self.force_arm.call_async(arm_req)

                def arm_callback(fut):
                    try:
                        result = fut.result()
                        if result.success:
                            self.get_logger().info(f"{self.id}: Armed successfully")
                            self.step = 3
                        else:
                            self.get_logger().warn(
                                f"{self.id}: Arm failed: {result.message}"
                            )
                    except Exception as ex:
                        self.get_logger().error(
                            f"{self.id}: force_arm call failed: {ex}"
                        )

                future.add_done_callback(arm_callback)

            # Step 3: short takeoff boost
            if self.step == 3:
                self.manual_control.z = TAKEOFF_Z
                self.takeoff_start_time = now
                self.get_logger().info(
                    f"{self.id}: Step 3 - Takeoff, z={self.manual_control.z}"
                )
                self.step = 4

            # Step 4: after short time, switch to hover + spread apart
            if self.step == 4 and self.takeoff_start_time is not None:
                if (now - self.takeoff_start_time) > 1.0:
                    self.manual_control.z = HOVER_Z
                    self.spread_start_time = now
                    self.get_logger().info(
                        f"{self.id}: Step 4 - Hover at lower height z={self.manual_control.z}"
                    )
                    self.step = 5

            # Step 5: wait SPREAD_DURATION, then start pattern
            if self.step == 5 and self.spread_start_time is not None:
                if (now - self.spread_start_time) > SPREAD_DURATION:
                    self.pattern_index = 0
                    vx, vy = self.pattern[self.pattern_index]
                    self.manual_control.z =
                    self.manual_control.x = vx
                    self.manual_control.y = vy
                    self.last_pattern_switch_time = now
                    self.get_logger().info(
                        f"{self.id}: Step 5 -> Step 6 - Start pattern '{self.pattern_name}', "
                        f"segment {self.pattern_index}, x={vx}, y={vy}, z={self.manual_control.z}"
                    )
                    self.step = 6

            # Step 6: run pattern forever
            if self.step == 6:
                if (now - self.last_pattern_switch_time) > PATTERN_SEGMENT_DURATION:
                    self.pattern_index = (self.pattern_index + 1) % len(self.pattern)
                    vx, vy = self.pattern[self.pattern_index]
                    self.manual_control.x = vx
                    self.manual_control.y = vy
                    self.last_pattern_switch_time = now
                    self.get_logger().info(
                        f"{self.id}: Pattern '{self.pattern_name}' segment {self.pattern_index}, "
                        f"x={vx}, y={vy}, z={self.manual_control.z}"
                    )
                # Step stays 6; you can add landing logic later

    def publish_keep_alive(self):
        keep_alive_msg = KeepAlive()
        keep_alive_msg.is_active = True
        keep_alive_msg.requested_flight_mode = self.mode_num
        keep_alive_msg.command_reboot = False
        self.flight_mode_pub.publish(keep_alive_msg)

        gcs_msg = Bool()
        gcs_msg.data = True
        self.gcs_keep_alive_pub.publish(gcs_msg)

    def publish_manual_control(self):
        self.manual_control_pub.publish(self.manual_control)


    def destroy_node(self):
        # Dump path to CSV if we have samples
        if self.path_samples:
            os.makedirs("paths", exist_ok=True)
            filename = os.path.join("paths", f"{self.id}_path.csv")
            with open(filename, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["t_sec", "x", "y", "z"])
                writer.writerows(self.path_samples)
            self.get_logger().info(
                f"{self.id}: wrote {len(self.path_samples)} path samples to {filename}"
            )

        super().destroy_node()



def main(args=None):
    rclpy.init(args=args)

    # Bigger movements: increase velocity commands a bit
    V = 150.0  # bigger so you clearly see motion

    pattern_circle = [
        (V, 0.0),
        (0.0, V),
        (-V, 0.0),
        (0.0, -V),
    ]

    pattern_square = [
        (V, 0.0),
        (0.0, V),
        (-V, 0.0),
        (0.0, -V),
    ]

    pattern_x = [
        (V, V),
        (-V, V),
        (-V, -V),
        (V, -V),
    ]

    # Spread vectors to push them to different areas first
    spread_R1 = (300.0, 0.0)     # move forward
    spread_R2 = (-300.0, 0.0)    # move backward
    spread_R3 = (0.0, 300.0)     # move to the right

    drones: List[SingleDroneFlying] = [
        SingleDroneFlying("R1", spread_R1, pattern_circle, "circle"),
        SingleDroneFlying("R2", spread_R2, pattern_square, "square"),
        SingleDroneFlying("R3", spread_R3, pattern_x, "X"),
    ]

    executor = MultiThreadedExecutor()
    for node in drones:
        executor.add_node(node)

    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        for node in drones:
            node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()

