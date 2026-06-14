#!/usr/bin/env python3

from __future__ import annotations

import math
import time

import rclpy
from rclpy.node import Node
from rclpy.duration import Duration
from rclpy.qos import qos_profile_sensor_data

from geometry_msgs.msg import Twist
from std_msgs.msg import Bool
from fcu_driver_interfaces.msg import UAVState


def clamp(value: float, limit: float) -> float:
    return max(-limit, min(limit, float(value)))


class RoosterTwistControlNode(Node):
    """
    ROBOTICAN Rooster control adapter.

    Input:
      /<rooster_id>/cmd_vel     geometry_msgs/Twist

    Twist mapping:
      linear.x   forward/backward
      linear.y   lateral, optional
      linear.z   vertical, optional
      angular.z  yaw

    Notes:
      For ground-roll mode, use linear.x and angular.z only.
    """

    def __init__(
        self,
        rooster_id: str = "R1",
        flight_mode: int = 1,
        cmd_vel_topic: str | None = None,
        max_linear_x: float = 0.25,
        max_linear_y: float = 0.0,
        max_linear_z: float = 0.0,
        max_yaw_rate: float = 0.5,
        command_hz: float = 20.0,
        cmd_timeout_sec: float = 0.4,
    ):
        super().__init__(f"{rooster_id.lower()}_twist_control")

        self.rooster_id = rooster_id
        self.flight_mode = int(flight_mode)

        self.max_linear_x = float(max_linear_x)
        self.max_linear_y = float(max_linear_y)
        self.max_linear_z = float(max_linear_z)
        self.max_yaw_rate = float(max_yaw_rate)

        self.cmd_timeout = Duration(seconds=float(cmd_timeout_sec))
        self.last_cmd_time = self.get_clock().now()

        self.current_twist = Twist()
        self.last_state: UAVState | None = None

        self.cmd_vel_topic = cmd_vel_topic or f"/{self.rooster_id}/cmd_vel"

        self.cmd_sub = self.create_subscription(
            Twist,
            self.cmd_vel_topic,
            self.cmd_vel_callback,
            10,
        )

        self.state_sub = self.create_subscription(
            UAVState,
            f"/{self.rooster_id}/fcu/state",
            self.state_callback,
            qos_profile_sensor_data,
        )

        self.gcs_keep_alive_pub = self.create_publisher(
            Bool,
            f"/{self.rooster_id}/gcs_keep_alive",
            10,
        )

        self.keep_alive_timer = self.create_timer(
            1.0,
            self.publish_keep_alive,
        )

        self.command_timer = self.create_timer(
            1.0 / float(command_hz),
            self.command_timer_callback,
        )

        self.get_logger().info(
            f"RoosterTwistControlNode ready\n"
            f"  cmd_vel:       {self.cmd_vel_topic}\n"
            f"  state:         /{self.rooster_id}/fcu/state\n"
            f"  keep_alive:    /{self.rooster_id}/gcs_keep_alive\n"
            f"  flight_mode:   {self.flight_mode}"
        )

    def cmd_vel_callback(self, msg: Twist) -> None:
        self.current_twist.linear.x = clamp(msg.linear.x, self.max_linear_x)
        self.current_twist.linear.y = clamp(msg.linear.y, self.max_linear_y)
        self.current_twist.linear.z = clamp(msg.linear.z, self.max_linear_z)
        self.current_twist.angular.z = clamp(msg.angular.z, self.max_yaw_rate)

        self.last_cmd_time = self.get_clock().now()

    def state_callback(self, msg: UAVState) -> None:
        self.last_state = msg

    def publish_keep_alive(self) -> None:
        msg = Bool()
        msg.data = True
        self.gcs_keep_alive_pub.publish(msg)

    def command_timer_callback(self) -> None:
        now = self.get_clock().now()

        if (now - self.last_cmd_time) > self.cmd_timeout:
            self.stop_motion()
            return

        self.publish_robotican_velocity_command(self.current_twist)

    def stop_motion(self) -> None:
        zero = Twist()
        self.current_twist = zero
        self.publish_robotican_velocity_command(zero)

    def publish_robotican_velocity_command(self, twist: Twist) -> None:
        """
        TODO:
        Replace this body with the same low-level command publisher used inside
        rooster_control_adapter.PathRunnerNode.

        The intended values are:

          forward_mps = twist.linear.x
          lateral_mps = twist.linear.y
          vertical_mps = twist.linear.z
          yaw_rate_radps = twist.angular.z

        For ground-roll mode:
          use forward_mps and yaw_rate_radps only.
        """

        forward_mps = twist.linear.x
        yaw_rate_radps = twist.angular.z

        # Temporary debug until connected to the real ROBOTICAN command message.
        self.get_logger().debug(
            f"cmd forward={forward_mps:.3f} m/s, yaw={yaw_rate_radps:.3f} rad/s"
        )