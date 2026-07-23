#!/usr/bin/env python3
"""rooster_twist_control_adapter.py

Translates a planner's geometry_msgs/Twist (FALCON's waypoint_follower_node.py
publishes on /cmd_vel, bridged ROS1->ROS2) into cmd_nav "move" JSON commands
for rooster_command_unit.py.

This does NOT publish ManualControl directly. rooster_command_unit.py's
RoosterUnit is the single owner of /<rooster_id>/manual_control and
/<rooster_id>/keep_alive (see that module's docstring: "there is exactly one
place per drone that talks to the FCU, regardless of who issued the
command"). A second, independent ManualControl publisher would fight it for
the same topic - in particular it would zero the z axis (throttle/altitude-
hold) on every Twist that doesn't set linear.z, dropping the drone out of the
sky the moment a planner's Twist arrived. Routing through cmd_nav's "move"
action (which only ever touches x/y/r, never z) is what keeps that guarantee.

Twist mapping:
  linear.x   forward/backward -> axis x
  linear.y   lateral          -> axis y
  angular.z  yaw rate         -> axis r
  linear.z is ignored - altitude is rooster_command_unit.py's job alone.
"""

from __future__ import annotations

import json

import rclpy
from rclpy.node import Node
from rclpy.duration import Duration

from geometry_msgs.msg import Twist
from std_msgs.msg import String

from sparx_agency.robots.common.math_utils import clamp_axis


class RoosterTwistControlNode(Node):
    def __init__(
        self,
        rooster_id: str = "R1",
        cmd_vel_topic: str | None = None,
        max_linear_x: float = 0.25,
        max_linear_y: float = 0.25,
        max_yaw_rate: float = 0.5,
        command_hz: float = 20.0,
        cmd_timeout_sec: float = 0.4,
    ):
        super().__init__(f"{rooster_id.lower()}_twist_control")

        self.rooster_id = rooster_id
        self.max_linear_x = float(max_linear_x)
        self.max_linear_y = float(max_linear_y)
        self.max_yaw_rate = float(max_yaw_rate)

        self.cmd_timeout = Duration(seconds=float(cmd_timeout_sec))
        self.last_cmd_time = self.get_clock().now()
        self.current_twist = Twist()

        # FALCON's real_drone.launch/sphera_drone.launch default drone_ns to
        # "" for Rooster, so waypoint_follower publishes plain /cmd_vel, not
        # /R1/cmd_vel - matches that unless explicitly overridden.
        self.cmd_vel_topic = cmd_vel_topic if cmd_vel_topic is not None else "/cmd_vel"
        self.cmd_nav_topic = f"/{self.rooster_id}/cmd_nav"

        self.cmd_sub = self.create_subscription(
            Twist, self.cmd_vel_topic, self.cmd_vel_callback, 10)
        self.cmd_nav_pub = self.create_publisher(String, self.cmd_nav_topic, 10)

        self.command_timer = self.create_timer(
            1.0 / float(command_hz), self.command_timer_callback)

        self.get_logger().info(
            f"RoosterTwistControlNode ready\n"
            f"  cmd_vel: {self.cmd_vel_topic}\n"
            f"  cmd_nav: {self.cmd_nav_topic} (action=move, x/y/r only)"
        )

    def cmd_vel_callback(self, msg: Twist) -> None:
        self.current_twist = msg
        self.last_cmd_time = self.get_clock().now()

    def command_timer_callback(self) -> None:
        now = self.get_clock().now()
        if (now - self.last_cmd_time) > self.cmd_timeout:
            self.stop_motion()
            return
        self.publish_move(self.current_twist)

    def stop_motion(self) -> None:
        self.current_twist = Twist()
        self._publish_cmd_nav("stop")

    def publish_move(self, twist: Twist) -> None:
        axes = {
            "x": clamp_axis(twist.linear.x / self.max_linear_x * 1000.0
                            if self.max_linear_x else 0.0),
            "y": clamp_axis(twist.linear.y / self.max_linear_y * 1000.0
                            if self.max_linear_y else 0.0),
            "r": clamp_axis(twist.angular.z / self.max_yaw_rate * 1000.0
                            if self.max_yaw_rate else 0.0),
        }
        self._publish_cmd_nav("move", axes=axes)

    def _publish_cmd_nav(self, action: str, **payload) -> None:
        msg = String()
        msg.data = json.dumps({"action": action, **payload})
        self.cmd_nav_pub.publish(msg)


def main(args=None):
    import argparse

    parser = argparse.ArgumentParser(description="Rooster Twist -> cmd_nav control adapter")
    parser.add_argument("--rooster-id", default="R1")
    parser.add_argument("--cmd-vel-topic", default=None)
    parser.add_argument("--max-linear-x", type=float, default=0.25)
    parser.add_argument("--max-linear-y", type=float, default=0.25)
    parser.add_argument("--max-yaw-rate", type=float, default=0.5)
    parser.add_argument("--command-hz", type=float, default=20.0)
    parser.add_argument("--cmd-timeout-sec", type=float, default=0.4)
    parsed, _ = parser.parse_known_args()

    rclpy.init(args=args)
    node = RoosterTwistControlNode(
        rooster_id=parsed.rooster_id,
        cmd_vel_topic=parsed.cmd_vel_topic,
        max_linear_x=parsed.max_linear_x,
        max_linear_y=parsed.max_linear_y,
        max_yaw_rate=parsed.max_yaw_rate,
        command_hz=parsed.command_hz,
        cmd_timeout_sec=parsed.cmd_timeout_sec,
    )
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
