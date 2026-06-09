#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass

import rclpy
from rclpy.node import Node
from rclpy.qos import (
    DurabilityPolicy,
    HistoryPolicy,
    QoSProfile,
    ReliabilityPolicy,
)
from std_msgs.msg import Empty, String

try:
    from sparx_agency.demos.Demo_No4_XTEND_MapRoom.demo_modes import DemoMode
except ImportError:
    # Useful when running this file directly from Demo_No4-XTEND_MapRoom
    # before installing/updating PYTHONPATH.
    from demo_modes import DemoMode


@dataclass
class DemoModeEvent:
    mode: DemoMode
    source: str = "unknown"
    reason: str = ""


class XtendDroneDemoManager(Node):
    """Central mode manager for the XTEND room-mapping demo.

    Responsibilities:
    - Accept mode requests from planner/UI/test tools.
    - Publish the current authoritative mode.
    - Send land/disarm sequence when FINISH is requested.

    First-version scope:
    - FLY_STRAIGHT: declared only.
    - TURNING: declared so localization can disable/down-weight optical flow.
    - VISUAL_SERVOING: declared only; implementation will come later.
    - FINISH: stop, land, wait, disarm.
    """

    def __init__(
        self,
        request_topic: str,
        mode_topic: str,
        cmd_nav_topic: str,
        reset_odom_topic: str,
        cmd_nav_state_sub_topic: str,
        disarm_delay_sec: float,
        publish_period_sec: float,
        initial_mode: DemoMode,
    ):
        super().__init__("xtend_drone_demo_manager")

        self.request_topic = request_topic
        self.mode_topic = mode_topic
        self.cmd_nav_topic = cmd_nav_topic
        self.reset_odom_topic = reset_odom_topic
        self.cmd_nav_state_sub_topic = cmd_nav_state_sub_topic
        self.disarm_delay_sec = float(disarm_delay_sec)

        self.current_mode = initial_mode
        self.previous_mode = DemoMode.IDLE
        self.finish_started = False
        self.disarm_timer = None

        mode_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )

        default_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
            reliability=ReliabilityPolicy.RELIABLE,
        )

        self.mode_pub = self.create_publisher(String, self.mode_topic, mode_qos)
        self.cmd_nav_pub = self.create_publisher(String, self.cmd_nav_topic, default_qos)
        self.reset_odom_pub = self.create_publisher(Empty, self.reset_odom_topic, default_qos)

        self.request_sub = self.create_subscription(
            String,
            self.request_topic,
            self.mode_request_cb,
            default_qos,
        )

        self.publish_timer = self.create_timer(
            float(publish_period_sec),
            self.publish_current_mode,
        )

        self.get_logger().info(f"Mode request topic: {self.request_topic}")
        self.get_logger().info(f"Current mode topic: {self.mode_topic}")
        self.get_logger().info(f"Command topic:      {self.cmd_nav_topic}")
        self.get_logger().info(f"Command watch topic:{self.cmd_nav_state_sub_topic}")
        self.get_logger().info(f"Reset odom topic:   {self.reset_odom_topic}")
        self.get_logger().info(f"Initial mode:       {self.current_mode.value}")

        self.publish_current_mode()

    def parse_request(self, raw: str) -> DemoModeEvent | None:
        text = str(raw).strip()

        if not text:
            return None

        if text.startswith("{"):
            try:
                data = json.loads(text)
            except json.JSONDecodeError:
                self.get_logger().warn(f"Invalid JSON mode request: {text}")
                return None

            mode = DemoMode.from_text(str(data.get("mode", "")))
            if mode is None:
                return None

            return DemoModeEvent(
                mode=mode,
                source=str(data.get("source", "unknown")),
                reason=str(data.get("reason", "")),
            )

        mode = DemoMode.from_text(text)
        if mode is None:
            return None

        return DemoModeEvent(mode=mode, source="string_request", reason=text)

    def mode_request_cb(self, msg: String) -> None:
        event = self.parse_request(msg.data)

        if event is None:
            self.get_logger().warn(f"Unknown demo mode request: {msg.data}")
            return

        self.set_mode(event.mode, source=event.source, reason=event.reason)

    def parse_cmd_nav_action(self, raw: str) -> str | None:
        text = str(raw).strip()

        if not text:
            return None

        if text.startswith("{"):
            try:
                data = json.loads(text)
            except json.JSONDecodeError:
                self.get_logger().warn(f"Invalid cmd_nav JSON: {text}")
                return None

            action = str(data.get("action", "")).strip().lower()
            return action or None

        return text.lower()

    def cmd_nav_state_cb(self, msg: String) -> None:
        action = self.parse_cmd_nav_action(msg.data)

        if action not in ("land", "disarm"):
            return

        if self.current_mode == DemoMode.IDLE:
            return

        self.get_logger().warn(
            f"Observed cmd_nav action={action}; returning demo mode to IDLE"
        )
        self.set_mode(
            DemoMode.IDLE,
            source="cmd_nav_observer",
            reason=f"observed {action}",
        )

    def set_mode(self, new_mode: DemoMode, source: str = "unknown", reason: str = "") -> None:
        if new_mode == self.current_mode:
            self.publish_current_mode()
            return

        old_mode = self.current_mode
        self.previous_mode = old_mode
        self.current_mode = new_mode

        self.get_logger().info(
            f"Mode changed: {old_mode.value} -> {new_mode.value} "
            f"source={source} reason={reason}"
        )

        self.on_exit_mode(old_mode)
        self.on_enter_mode(new_mode)
        self.publish_current_mode()

    def on_exit_mode(self, mode: DemoMode) -> None:
        if mode == DemoMode.TURNING:
            self.get_logger().info("Exiting TURNING mode")
        elif mode == DemoMode.VISUAL_SERVOING:
            self.get_logger().info("Exiting VISUAL_SERVOING mode")

    def on_enter_mode(self, mode: DemoMode) -> None:
        if mode == DemoMode.IDLE:
            self.on_enter_idle()
        elif mode == DemoMode.FLY_STRAIGHT:
            self.on_enter_fly_straight()
        elif mode == DemoMode.TURNING:
            self.on_enter_turning()
        elif mode == DemoMode.VISUAL_SERVOING:
            self.on_enter_visual_servoing()
        elif mode == DemoMode.FINISH:
            self.on_enter_finish()

    def on_enter_idle(self) -> None:
        self.get_logger().info("IDLE mode active")

    def on_enter_fly_straight(self) -> None:
        self.get_logger().info("FLY_STRAIGHT mode active")

    def on_enter_turning(self) -> None:
        self.get_logger().info(
            "TURNING mode active. Localization should disable or down-weight optical flow."
        )

    def on_enter_visual_servoing(self) -> None:
        self.get_logger().info("VISUAL_SERVOING mode active")
        # First version only declares the mode.
        # Later this can coordinate object detector / image-center controller.
        pass

    def on_enter_finish(self) -> None:
        self.start_finish_sequence()

    def publish_current_mode(self) -> None:
        msg = String()
        msg.data = self.current_mode.value
        self.mode_pub.publish(msg)

    def publish_cmd_nav(self, action: str, value: int = 0) -> None:
        msg = String()
        msg.data = json.dumps({"action": action, "value": int(value)})
        self.cmd_nav_pub.publish(msg)
        self.get_logger().info(f"Published cmd_nav: {msg.data}")

    def publish_reset_odom(self) -> None:
        """Publish reset odometry request.

        This is intentionally not called automatically yet.
        Later we can call this on a chosen transition, for example
        IDLE -> FLY_STRAIGHT, or from a dedicated reset command.
        """
        self.reset_odom_pub.publish(Empty())
        self.get_logger().warn(f"Published reset odom on {self.reset_odom_topic}")

    def start_finish_sequence(self) -> None:
        if self.finish_started:
            self.get_logger().warn("FINISH sequence already started")
            return

        self.finish_started = True
        self.get_logger().warn("FINISH mode active: sending STOP then LAND")

        self.publish_cmd_nav("stop", 0)
        self.publish_cmd_nav("land", 0)

        self.disarm_timer = self.create_timer(
            self.disarm_delay_sec,
            self.disarm_once_cb,
        )

    def disarm_once_cb(self) -> None:
        self.publish_cmd_nav("disarm", 0)

        if self.disarm_timer is not None:
            self.disarm_timer.cancel()
            self.destroy_timer(self.disarm_timer)
            self.disarm_timer = None

        self.get_logger().warn("FINISH sequence completed: DISARM sent")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--request-topic", default="/xtend/demo_mode_request")
    parser.add_argument("--mode-topic", default="/xtend/demo_mode")
    parser.add_argument("--cmd-nav-topic", default="/xtend/cmd_nav")
    parser.add_argument("--reset-odom-topic", default="/xtend/reset_odom")
    parser.add_argument("--cmd-nav-state-sub-topic", default="/xtend/cmd_nav")
    parser.add_argument("--disarm-delay-sec", type=float, default=8.0)
    parser.add_argument("--publish-period-sec", type=float, default=1.0)
    parser.add_argument("--initial-mode", default=DemoMode.IDLE.value)
    return parser.parse_args()


def main():
    args = parse_args()

    initial_mode = DemoMode.from_text(args.initial_mode)
    if initial_mode is None:
        raise ValueError(f"Invalid initial mode: {args.initial_mode}")

    rclpy.init()
    node = XtendDroneDemoManager(
        request_topic=args.request_topic,
        mode_topic=args.mode_topic,
        cmd_nav_topic=args.cmd_nav_topic,
        reset_odom_topic=args.reset_odom_topic,
        cmd_nav_state_sub_topic=args.cmd_nav_state_sub_topic,
        disarm_delay_sec=args.disarm_delay_sec,
        publish_period_sec=args.publish_period_sec,
        initial_mode=initial_mode,
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
