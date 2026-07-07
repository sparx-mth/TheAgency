#!/usr/bin/env python3
"""rooster_command_unit.py

RoosterCommandUnitNode is the single command gateway for one Rooster drone.

Every commander - the manual Tkinter UI (see ROBOTICAN/ui.py) today, a
planner node in the future - publishes flight commands to the same
/<rooster_id>/cmd_nav topic. This node is the only thing that owns a
RoosterUnit for that drone, so there is exactly one place per drone that
talks to the FCU, regardless of who issued the command.

cmd_nav payload (std_msgs/String, JSON):
    {"action": "arm|disarm|takeoff|land|forward|backward|left|right|
                up|down|turn_left|turn_right|stop|video_on|video_off",
     "value": <float, optional axis magnitude, default 400>}
"""

from __future__ import annotations

import json

import rclpy
from rclpy.node import Node
from std_msgs.msg import String

from sparx_agency.robots.ROBOTICAN.helpers.rooster_unit import RoosterUnit
from sparx_agency.robots.ROBOTICAN.helpers.rooster_payload import RoosterPayload

DEFAULT_AXIS_VALUE = 400.0

_MOVE_ACTIONS = {
    "forward": dict(x=1),
    "backward": dict(x=-1),
    "left": dict(y=1),
    "right": dict(y=-1),
    "up": dict(z=1),
    "down": dict(z=-1),
    "turn_left": dict(r=1),
    "turn_right": dict(r=-1),
}


class RoosterCommandUnitNode(Node):
    def __init__(self):
        super().__init__("rooster_command_unit")

        self.declare_parameter("rooster_id", "R1")
        self.declare_parameter("climb_z", 600.0)
        self.declare_parameter("hover_z", 550.0)
        self.declare_parameter("land_step", 75.0)
        self.declare_parameter("land_step_interval_sec", 1.0)
        self.declare_parameter("land_timeout_sec", 30.0)
        self.declare_parameter("publish_hz", 40.0)
        self.declare_parameter("video_host", "127.0.0.1")
        self.declare_parameter("video_port", 5001)
        self.declare_parameter("video_width", 540)
        self.declare_parameter("video_height", 360)

        rooster_id = self.get_parameter("rooster_id").value
        climb_z = float(self.get_parameter("climb_z").value)
        hover_z = float(self.get_parameter("hover_z").value)
        land_step = float(self.get_parameter("land_step").value)
        land_step_interval_sec = float(self.get_parameter("land_step_interval_sec").value)
        land_timeout_sec = float(self.get_parameter("land_timeout_sec").value)
        publish_hz = float(self.get_parameter("publish_hz").value)
        video_host = self.get_parameter("video_host").value
        video_port = int(self.get_parameter("video_port").value)
        video_width = int(self.get_parameter("video_width").value)
        video_height = int(self.get_parameter("video_height").value)

        self.rooster_id = rooster_id
        self.unit = RoosterUnit(
            self, rooster_id,
            climb_z=climb_z, hover_z=hover_z,
            land_step=land_step,
            land_step_interval_sec=land_step_interval_sec,
            land_timeout_sec=land_timeout_sec,
        )
        self.payload = RoosterPayload(
            self, rooster_id,
            video_host=video_host, video_port=video_port,
            video_width=video_width, video_height=video_height,
        )

        self.cmd_sub = self.create_subscription(
            String, f"/{rooster_id}/cmd_nav", self._on_cmd_nav, 10)
        self.status_pub = self.create_publisher(
            String, f"/{rooster_id}/rooster_status", 10)

        self.create_timer(1.0 / publish_hz, self.unit.publish_manual)
        self.create_timer(1.0, self.unit.publish_keep_alive)
        self.create_timer(1.0, self.payload.publish_gcs_keep_alive)
        self.create_timer(0.2, self._publish_status)

        self.get_logger().info(
            f"RoosterCommandUnitNode ready for {rooster_id}\n"
            f"  command in:  /{rooster_id}/cmd_nav\n"
            f"  status out:  /{rooster_id}/rooster_status"
        )

    def _publish_status(self):
        msg = String()
        msg.data = json.dumps({
            "armed": bool(self.unit.armed),
            "airborne": bool(self.unit.airborne),
            "busy_action": self.unit.busy_action,
            "battery_pct": self.payload.battery_percentage,
            "battery_voltage": self.payload.battery_voltage,
            "video_on": self.payload.video_on,
        })
        self.status_pub.publish(msg)

    def _on_cmd_nav(self, msg: String):
        try:
            cmd = json.loads(msg.data)
        except json.JSONDecodeError:
            self.get_logger().warn(f"Ignoring malformed cmd_nav payload: {msg.data!r}")
            return

        action = cmd.get("action")
        value = float(cmd.get("value", DEFAULT_AXIS_VALUE))
        unit = self.unit

        if action == "arm":
            unit.arm()
        elif action == "disarm":
            unit.disarm()
        elif action == "takeoff":
            unit.takeoff()
        elif action == "land":
            unit.land()
        elif action == "stop":
            unit.stop()
        elif action == "video_on":
            self.payload.set_video(True)
        elif action == "video_off":
            self.payload.set_video(False)
        elif action in _MOVE_ACTIONS:
            # Only override the axis this action controls - z in particular
            # is throttle/altitude-hold, and defaulting it to 0 here would
            # cut hover power and drop the drone (confirmed: turning while
            # flying without this fix caused an immediate fall).
            axes = dict(x=unit.axes.x, y=unit.axes.y, z=unit.axes.z, r=unit.axes.r)
            axes.update({axis: sign * value for axis, sign in _MOVE_ACTIONS[action].items()})
            unit.set_axes(**axes)
        else:
            self.get_logger().warn(f"Unknown cmd_nav action: {action!r}")


def main(args=None):
    rclpy.init(args=args)
    node = RoosterCommandUnitNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
