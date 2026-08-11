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
  angular.z  yaw rate         -> axis r, NEGATED (see below)
  linear.z is ignored - altitude is rooster_command_unit.py's job alone.

angular.z is negated: REP103 has positive angular.z = left, but this drone's
FCU axis convention has positive r = right (same convention as
rooster_command_unit.py's turn_left/turn_right). See LESSONS.md.

max_linear_x/max_linear_y/max_yaw_rate are "real-world rate produced at full
axis deflection (1000)" - the scale factor a planner's Twist is normalized
against before becoming an axis value. max_yaw_rate was recalibrated
2026-07-30 from 0.5 to 1.8 rad/s: a logged manual flight (command+pose,
see docs/progress/entries/007-rooster-velocity-controller.md) showed
axis r=500 (turn_right) produced ~55 deg/s (~0.96 rad/s) over 8 isolated
turn segments, i.e. axis 1000 -> ~1.9 rad/s - the old 0.5 rad/s default was
never live-validated and was ~4x too low, meaning any planner asking for
even a modest yaw rate was actually commanding a much faster real turn than
intended. See LESSONS.md for the full derivation. max_linear_x/max_linear_y
were left unchanged: the same flight's forward/lateral segments were too
short and interleaved with turns (leftover momentum contaminates each
segment) to extract a trustworthy number - a dedicated calibration flight
(isolated single-axis moves, no interleaving) is needed before touching
those with the same confidence.

Yaw axis (r) is also slew-rate-limited (max_yaw_axis_step_per_sec): PX4's
own yaw-rate loop has zero derivative gain (MC_YAWRATE_D=0.0, P/I only), so
an instantaneous step change in commanded rate excites it into oscillation.
Ramping the output here smooths that without touching the PX4 param.
max_yaw_axis_step_per_sec=2500 is a first, conservative guess -- live-test
and retune. See LESSONS.md.
"""

from __future__ import annotations

import json

import rclpy
from rclpy.node import Node
from rclpy.duration import Duration

from geometry_msgs.msg import Twist
from std_msgs.msg import String

from sparx_agency.robots.common.math_utils import clamp_axis, slew


class RoosterTwistControlNode(Node):
    def __init__(
        self,
        rooster_id: str = "R1",
        cmd_vel_topic: str | None = None,
        max_linear_x: float = 0.25,
        max_linear_y: float = 0.25,
        max_yaw_rate: float = 1.8,
        max_yaw_axis_step_per_sec: float = 2500.0,
        command_hz: float = 20.0,
        cmd_timeout_sec: float = 0.4,
    ):
        super().__init__(f"{rooster_id.lower()}_twist_control")

        self.rooster_id = rooster_id
        self.max_linear_x = float(max_linear_x)
        self.max_linear_y = float(max_linear_y)
        self.max_yaw_rate = float(max_yaw_rate)
        self.max_yaw_axis_step_per_sec = float(max_yaw_axis_step_per_sec)
        self.command_hz = float(command_hz)

        self.cmd_timeout = Duration(seconds=float(cmd_timeout_sec))
        self.last_cmd_time = self.get_clock().now()
        self.current_twist = Twist()
        # Slew-limited state for the r axis only (see module docstring) --
        # x/y aren't implicated in the reported oscillation and stay as a
        # direct pass-through.
        self._r_axis = 0.0

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
        self._r_axis = 0.0  # stop is immediate, never slew-limited
        self._publish_cmd_nav("stop")

    def publish_move(self, twist: Twist) -> None:
        # Negated -- see module docstring.
        target_r = (-twist.angular.z / self.max_yaw_rate * 1000.0
                    if self.max_yaw_rate else 0.0)
        max_step = self.max_yaw_axis_step_per_sec / self.command_hz
        self._r_axis = slew(target_r, self._r_axis, max_step)
        axes = {
            "x": clamp_axis(twist.linear.x / self.max_linear_x * 1000.0
                            if self.max_linear_x else 0.0),
            "y": clamp_axis(twist.linear.y / self.max_linear_y * 1000.0
                            if self.max_linear_y else 0.0),
            "r": clamp_axis(self._r_axis),
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
    parser.add_argument("--max-yaw-rate", type=float, default=1.8)
    parser.add_argument("--max-yaw-axis-step-per-sec", type=float, default=2500.0)
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
        max_yaw_axis_step_per_sec=parsed.max_yaw_axis_step_per_sec,
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
