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


def velocity_to_axis(v_mps: float, deadzone: float, v_full: float,
                     min_command_mps: float) -> float:
    """Convert a requested body velocity (m/s) into a ManualControl axis.

    The horizontal axes are NOT linear from zero: measured against Sphera
    ground truth 2026-08-17, forward is dead until ~620 counts and then ramps
    to ~1.25 m/s at 1000 (lateral: dead until ~700, ~1.02 m/s at 1000). The
    old ``v / max_v * 1000`` scale therefore put every normal FALCON request
    (0.15-0.2 m/s) right on the deadzone edge, where real speed flips between
    0.00 and 0.26 m/s for a few counts -- the actual cause of the long-standing
    jerky tracking. See LESSONS.md and the axis-calibration memory.

    Args:
        v_mps: Requested velocity along this axis, m/s (signed).
        deadzone: Axis counts below which the platform does not move at all.
        v_full: Velocity produced at axis 1000, m/s.
        min_command_mps: Requests smaller than this are emitted as 0 rather
            than as a deadzone-edge command, because speeds below roughly
            ``v_full * (1 - deadzone/1000)``'s first step are not continuously
            achievable on this platform at all.

    Returns:
        Axis value in [-1000, 1000].
    """
    if v_full <= 0.0 or abs(v_mps) < min_command_mps:
        return 0.0
    span = max(1.0, 1000.0 - deadzone)
    magnitude = deadzone + min(1.0, abs(v_mps) / v_full) * span
    return clamp_axis(magnitude if v_mps > 0.0 else -magnitude)


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
        # Ground-truth-measured axis response (2026-08-17). See
        # velocity_to_axis() above and LESSONS.md; these replace the
        # never-validated linear max_linear_x/y scaling, which is now only
        # used when use_measured_axis_curve=False.
        use_measured_axis_curve: bool = True,
        x_deadzone: float = 620.0,
        x_v_full: float = 1.25,
        y_deadzone: float = 700.0,
        y_v_full: float = 1.02,
        min_command_mps: float = 0.15,
        # The lateral axis is the worst-behaved one: measured dead until
        # ~axis 1000, so ANY sideways demand slams the stick and produces
        # 27-36 deg of roll plus a speed overshoot past the measured cap.
        # There is no gentle middle on this axis. Capping it keeps the
        # airframe flat, which also matters for map quality -- depth comes
        # from monocular DA3, and steep roll skews the geometry it infers.
        # 0.0 disables lateral entirely (turn-and-go via yaw instead).
        max_lateral_axis: float = 0.0,
    ):
        super().__init__(f"{rooster_id.lower()}_twist_control")

        self.rooster_id = rooster_id
        self.max_linear_x = float(max_linear_x)
        self.max_linear_y = float(max_linear_y)
        self.max_yaw_rate = float(max_yaw_rate)
        self.use_measured_axis_curve = bool(use_measured_axis_curve)
        self.x_deadzone = float(x_deadzone)
        self.x_v_full = float(x_v_full)
        self.y_deadzone = float(y_deadzone)
        self.y_v_full = float(y_v_full)
        self.min_command_mps = float(min_command_mps)
        self.max_lateral_axis = float(max_lateral_axis)
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
        if self.use_measured_axis_curve:
            ax_x = velocity_to_axis(twist.linear.x, self.x_deadzone,
                                    self.x_v_full, self.min_command_mps)
            ax_y = velocity_to_axis(twist.linear.y, self.y_deadzone,
                                    self.y_v_full, self.min_command_mps)
        else:
            ax_x = clamp_axis(twist.linear.x / self.max_linear_x * 1000.0
                              if self.max_linear_x else 0.0)
            ax_y = clamp_axis(twist.linear.y / self.max_linear_y * 1000.0
                              if self.max_linear_y else 0.0)
        # Clamp lateral to keep the airframe flat -- see max_lateral_axis.
        if self.max_lateral_axis <= 0.0:
            ax_y = 0.0
        else:
            ax_y = max(-self.max_lateral_axis, min(self.max_lateral_axis, ax_y))
        axes = {
            "x": ax_x,
            "y": ax_y,
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
    parser.add_argument("--max-lateral-axis", type=float, default=0.0)
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
        max_lateral_axis=parsed.max_lateral_axis,
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
