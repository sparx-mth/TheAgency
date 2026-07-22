#!/usr/bin/env python3
"""Standalone Twist -> /xtend/cmd_nav relay.

TWO MODES, and picking the wrong one silently breaks a multi-axis controller:

  * **single-axis (default)** -- emits one discrete ``{"action": ..., "value": ...}``
    per Twist, choosing ONE axis by priority (yaw, then forward, then, if
    ``--allow-multi-axes``, vertical/lateral). Every other axis is DISCARDED. This
    is fine for a controller that only ever drives one axis at a time (the legacy
    waypoint follower), and wrong for anything else: a controller that flies
    forward while correcting sideways would have its correction thrown away on
    every tick where forward is non-zero, i.e. all of them.
  * **combined (``--combined``)** -- routes the Twist through the shared
    :class:`TwistToCmdNavConverter` and emits a single ``set_axes`` command with
    all four axes populated. Required by any multi-axis controller, including
    ``drift_pid``, ``multi_axis``, ``pure_pursuit`` and ``roll_assist``.

Note the in-process sibling used by ``online_nav_bridge`` (``OnlineXtendBridgeBase``)
has always been combined; only this standalone relay defaults to single-axis, and
only for backward compatibility.
"""
from __future__ import annotations

import argparse
import json
import time

import rclpy
from geometry_msgs.msg import Twist
from rclpy.node import Node
from std_msgs.msg import String

from sparx_agency.robots.XTEND.adapters.axis_calibration import XTEND_CALIBRATION
from sparx_agency.robots.XTEND.adapters.twist_to_cmd_nav_converter import (
    TwistToCmdNavConverter,
)

# Derived from the single source of truth so a re-calibration cannot leave this
# file disagreeing with the converter. See axis_calibration.py.
FORWARD_REF_VEL = XTEND_CALIBRATION.forward.ref_velocity
FORWARD_REF_VALUE = XTEND_CALIBRATION.forward.ref_counts
FORWARD_MAX_VALUE = XTEND_CALIBRATION.forward.max_counts

TURN_REF_ANGULAR = XTEND_CALIBRATION.yaw.ref_velocity
TURN_REF_VALUE = XTEND_CALIBRATION.yaw.ref_counts
TURN_MAX_VALUE = XTEND_CALIBRATION.yaw.max_counts

class XtendTwistToCmdNav(Node):
    def __init__(
        self,
        cmd_vel_topic: str,
        cmd_nav_topic: str,
        angular_delta: float,
        linear_delta: float,
        timeout_sec: float,
        publish_stop_on_timeout: bool,
        allow_multi_axes: bool = False,
        combined: bool = False,
    ):
        super().__init__("xtend_twist_to_cmd_nav")

        self.forward_value = 0
        self.turn_value = 0
        self.angular_delta = float(angular_delta)
        self.linear_delta = float(linear_delta)
        self.timeout_sec = float(timeout_sec)
        self.publish_stop_on_timeout = bool(publish_stop_on_timeout)

        self.last_twist_time = 0.0
        self.last_action = None
        # Vertical (up/down) and lateral (left/right) axes. OFF by default: the
        # planar nav stack hardwires linear.z = 0 and never crabs, so enabling
        # them would only widen what a stray Twist can do. Turn it ON for a stack
        # that genuinely commands a climb -- the lost-localization recovery does,
        # and without this its climb rungs fall through to "stop" and do nothing.
        self.allow_multi_axes = bool(allow_multi_axes)
        # Combined mode: emit one set_axes with every axis populated, instead of
        # one discrete single-axis action. Mandatory for a multi-axis controller
        # -- see the module docstring.
        self.combined = bool(combined)
        self._converter = TwistToCmdNavConverter(
            angular_delta=angular_delta,
            linear_delta=linear_delta,
            timeout_sec=timeout_sec,
            publish_stop_on_timeout=publish_stop_on_timeout,
        ) if self.combined else None

        # Planner may publish short zero Twist messages between yaw commands.
        # XTEND commands are hold-style, so do not stop on the first zero Twist.
        self.zero_stop_count = 0
        self.zero_stop_required_count = 2

        # ROS2 publishers and subscribers
        self.pub = self.create_publisher(String, cmd_nav_topic, 10)
        self.sub = self.create_subscription(Twist, cmd_vel_topic, self.twist_cb, 10)

        self.timer = self.create_timer(0.05, self.watchdog_cb)

        self.get_logger().info(f"Listening Twist: {cmd_vel_topic}")
        self.get_logger().info(f"Publishing cmd_nav: {cmd_nav_topic}")
        self.get_logger().info(
            f"Defaults: forward={self.forward_value}, turn={self.turn_value}, "
            f"angular_delta={self.angular_delta}, linear_delta={self.linear_delta}"
        )
        if self.combined:
            self.get_logger().info(
                "Mode: COMBINED -- one set_axes per Twist, all four axes kept"
            )
        else:
            self.get_logger().info(
                f"Axes: x/yaw always; up-down + left-right "
                f"{'ENABLED' if self.allow_multi_axes else 'DISABLED (linear.z -> stop)'}"
            )
            self.get_logger().warn(
                "Mode: SINGLE-AXIS -- only the highest-priority axis of each "
                "Twist is sent, the rest are DISCARDED. Pass --combined for any "
                "multi-axis controller (drift_pid / multi_axis / pure_pursuit / "
                "roll_assist), or its corrections will never reach the drone."
            )

    def publish_action(self, action: str, value: int = 0):
        # Avoid spamming same hold command at 30 Hz.
        # Bridge holds the command until STOP or another command.
        key = (action, int(value))
        if key == self.last_action:
            return

        msg = String()
        msg.data = json.dumps({"action": action, "value": int(value)})
        self.pub.publish(msg)

        self.last_action = key
        self.get_logger().info(f"Published: {msg.data}")

    def scale_translation_axis(self, value: float) -> int:
        value_abs = abs(float(value))
        axis = int(round((value_abs / FORWARD_REF_VEL) * FORWARD_REF_VALUE))
        return max(0, min(FORWARD_MAX_VALUE, axis))

    def scale_yaw_axis(self, angular_z: float) -> int:
        value_abs = abs(float(angular_z))
        axis = int(round((value_abs / TURN_REF_ANGULAR) * TURN_REF_VALUE))
        return max(0, min(TURN_MAX_VALUE, axis))

    def choose_cmd_from_twist(self, msg: Twist) -> tuple[str, int]:
        lx = float(msg.linear.x)
        ly = float(msg.linear.y)
        lz = float(msg.linear.z)
        az = float(msg.angular.z)

        if az > self.angular_delta:
            return "turn_left", self.scale_yaw_axis(az)

        if az < -self.angular_delta:
            return "turn_right", self.scale_yaw_axis(az)

        if lx > self.linear_delta:
            return "forward", self.scale_translation_axis(lx)

        if lx < -self.linear_delta:
            return "backward", self.scale_translation_axis(lx)

        if self.allow_multi_axes:
            if lz > self.linear_delta:
                return "up", self.scale_translation_axis(lz)

            if lz < -self.linear_delta:
                return "down", self.scale_translation_axis(lz)

            if ly > self.linear_delta:
                return "left", self.scale_translation_axis(ly)

            if ly < -self.linear_delta:
                return "right", self.scale_translation_axis(ly)

        return "stop", 0

    def publish_axes(self, cmd) -> None:
        """Publish one combined set_axes command (all four axes at once)."""
        msg = String()
        msg.data = json.dumps({
            "action": "set_axes",
            "forward": cmd.forward,
            "lateral": cmd.lateral,
            "vertical": cmd.vertical,
            "yaw": cmd.yaw,
        })
        self.pub.publish(msg)
        self.get_logger().info("Published: %s (%s)" % (msg.data, cmd.describe()))

    def twist_cb(self, msg: Twist):
        self.last_twist_time = time.time()

        if self.combined:
            # The converter owns dedup and the stop debounce in this mode.
            cmd = self._converter.process(
                msg.linear.x, msg.linear.y, msg.linear.z, msg.angular.z)
            if cmd is not None:
                self.publish_axes(cmd)
            return

        action, value = self.choose_cmd_from_twist(msg)

        if action == "stop":
            self.zero_stop_count += 1

            # Ignore the first zero/stop Twist after an active hold command.
            # Stop only after repeated zero Twist messages.
            if (
                self.last_action is not None
                and self.last_action[0] != "stop"
                and self.zero_stop_count < self.zero_stop_required_count
            ):
                self.get_logger().debug(
                    f"Ignoring transient stop Twist "
                    f"({self.zero_stop_count}/{self.zero_stop_required_count})"
                )
                return
        else:
            self.zero_stop_count = 0

        self.publish_action(action, value)

    def watchdog_cb(self):
        if self.combined:
            cmd = self._converter.check_timeout()
            if cmd is not None:
                self.publish_axes(cmd)
            return

        if self.last_twist_time <= 0.0:
            return

        age = time.time() - self.last_twist_time
        if age > self.timeout_sec and self.publish_stop_on_timeout:
            self.publish_action("stop", 0)



def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--cmd-vel-topic", default="/cmd_vel")
    p.add_argument("--cmd-nav-topic", default="/xtend/cmd_nav")

    # Planner thresholds
    p.add_argument("--angular-delta", type=float, default=0.05)
    p.add_argument("--linear-delta", type=float, default=0.05)

    p.add_argument("--timeout-sec", type=float, default=1.5)
    p.add_argument("--no-stop-on-timeout", action="store_true")
    p.add_argument(
        "--allow-multi-axes",
        action="store_true",
        help="Honour linear.z (up/down) and linear.y (left/right) in addition to "
             "linear.x and angular.z. Without it linear.z is silently dropped as "
             "'stop'. Pass it if the stack commands a climb -- the "
             "lost-localization recovery does. NOTE the sibling in-process "
             "converter (TwistToCmdNavConverter, used by online_nav_bridge) has "
             "these axes ON; this standalone script keeps them OFF by default so "
             "its existing behaviour is unchanged.",
    )
    p.add_argument(
        "--combined",
        action="store_true",
        help="Emit ONE set_axes command with all four axes populated instead of "
             "a single discrete action per Twist. REQUIRED by any multi-axis "
             "controller (drift_pid, multi_axis, pure_pursuit, roll_assist): "
             "without it the relay keeps only the highest-priority axis and "
             "silently discards the rest, so e.g. a sideways drift correction "
             "riding along with forward flight never reaches the drone.",
    )
    return p.parse_args()


def main():
    args = parse_args()

    rclpy.init()
    node = XtendTwistToCmdNav(
        cmd_vel_topic=args.cmd_vel_topic,
        cmd_nav_topic=args.cmd_nav_topic,
        angular_delta=args.angular_delta,
        linear_delta=args.linear_delta,
        timeout_sec=args.timeout_sec,
        publish_stop_on_timeout=not args.no_stop_on_timeout,
        allow_multi_axes=args.allow_multi_axes,
        combined=args.combined,
    )

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.publish_action("stop", 0)
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()