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

    {"action": "move", "axes": {"x": <float>, "y": <float>, "r": <float>}}
    Continuous-control variant for a planner (e.g. rooster_twist_control_adapter.py
    translating FALCON's /cmd_vel): sets any subset of x/y/r directly instead of a
    single named direction. z is deliberately never accepted here - same reason
    the named directional actions preserve it (it's throttle/altitude-hold;
    a planner has no business touching it, only the takeoff/land climb logic).
"""

from __future__ import annotations

import json

import rclpy
from rclpy.node import Node
from std_msgs.msg import String

from sparx_agency.robots.ROBOTICAN.helpers.rooster_unit import (
    FLIGHT_MODE_ALTITUDE, FLIGHT_MODE_POSITION, RoosterUnit,
)
from sparx_agency.robots.ROBOTICAN.helpers.rooster_payload import RoosterPayload

DEFAULT_AXIS_VALUE = 400.0
# 2026-08-17: metres, not axis units -- see the "up"/"down" special-case in
# _on_cmd_nav below. Separate default from DEFAULT_AXIS_VALUE on purpose,
# different units.
DEFAULT_ALTITUDE_NUDGE_M = 0.5

_MOVE_ACTIONS = {
    "forward": dict(x=1),
    "backward": dict(x=-1),
    "left": dict(y=-1),
    "right": dict(y=1),
    "up": dict(z=1),
    "down": dict(z=-1),
    # Flipped 2026-07-13: user reported turn_left/turn_right swapped live.
    # Same class of bug as the left/right (y-axis) fix above -- this r-axis
    # sign was only ever assumed from doc convention, never live-validated
    # (see the "adjust if needed" comment in
    # examples/src/ground_roll_turn_left_right.py). Re-confirm with a live
    # test after this change, same as the y-axis fix was.
    "turn_left": dict(r=-1),
    "turn_right": dict(r=1),
}


class RoosterCommandUnitNode(Node):
    def __init__(self):
        super().__init__("rooster_command_unit")

        self.declare_parameter("rooster_id", "R1")
        # See rooster_unit.py's RoosterUnit.__init__ for why these three
        # takeoff params have the values they do.
        self.declare_parameter("climb_z", 1000.0)
        self.declare_parameter("hover_z", 550.0)
        self.declare_parameter("land_step", 75.0)
        self.declare_parameter("land_step_interval_sec", 1.0)
        self.declare_parameter("land_timeout_sec", 30.0)
        self.declare_parameter("climb_duration_sec", 1.0)
        self.declare_parameter("climb_settle_sec", 1.0)
        self.declare_parameter("altitude_hold_kp", 500.0)
        # See RoosterUnit._altitude_hold_tick -- the z axis is three-zone,
        # so descent needs a larger gain than climb to reach its zone at all.
        self.declare_parameter("altitude_hold_kp_down", 1500.0)
        self.declare_parameter("altitude_hold_kd", 600.0)
        # See RoosterUnit.__init__ (rooster_unit.py) for why the default was
        # raised from 200 to 380 -- 2026-08-13.
        self.declare_parameter("altitude_hold_max_correction", 380.0)
        # See RoosterUnit.__init__ -- added 2026-08-17, the actual fix for
        # the "rises and falls out of control" symptom: bounds how much z
        # may change in ONE tick, tighter than altitude_hold_max_correction.
        self.declare_parameter("altitude_hold_max_step", 8.0)
        self.declare_parameter("altitude_hold_velocity_filter_tau_s", 0.5)
        # See RoosterUnit.__init__ -- 2026-08-17, "position" | "altitude"
        # (case-insensitive), selects FLIGHT_MODE_POSITION (3, default,
        # unchanged) or FLIGHT_MODE_ALTITUDE (4, drops horizontal position-
        # hold entirely -- self-level attitude + altitude hold only).
        self.declare_parameter("requested_flight_mode", "position")
        # See RoosterUnit.__init__ for why this was raised from 1.0 to
        # 0.1 -- 2026-08-13 (matches /R1/state's real ~10Hz ranger rate).
        self.declare_parameter("altitude_hold_interval_sec", 0.1)
        # <=0.0 disables -- see RoosterUnit.__init__ for why this exists.
        self.declare_parameter("max_ranger_m", 0.0)
        # <=0.0 disables -- see RoosterUnit.__init__ for why this exists.
        self.declare_parameter("target_ranger_m", 0.0)
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
        climb_duration_sec = float(self.get_parameter("climb_duration_sec").value)
        climb_settle_sec = float(self.get_parameter("climb_settle_sec").value)
        altitude_hold_kp = float(self.get_parameter("altitude_hold_kp").value)
        altitude_hold_kp_down = float(self.get_parameter("altitude_hold_kp_down").value)
        altitude_hold_kd = float(self.get_parameter("altitude_hold_kd").value)
        altitude_hold_max_correction = float(self.get_parameter("altitude_hold_max_correction").value)
        altitude_hold_max_step = float(self.get_parameter("altitude_hold_max_step").value)
        altitude_hold_velocity_filter_tau_s = float(
            self.get_parameter("altitude_hold_velocity_filter_tau_s").value)
        flight_mode_name = str(self.get_parameter("requested_flight_mode").value).strip().lower()
        requested_flight_mode = {
            "position": FLIGHT_MODE_POSITION,
            "altitude": FLIGHT_MODE_ALTITUDE,
        }.get(flight_mode_name)
        if requested_flight_mode is None:
            raise ValueError(
                "requested_flight_mode must be 'position' or 'altitude', got %r"
                % (flight_mode_name,))
        altitude_hold_interval_sec = float(self.get_parameter("altitude_hold_interval_sec").value)
        max_ranger_m = float(self.get_parameter("max_ranger_m").value)
        target_ranger_m = float(self.get_parameter("target_ranger_m").value)
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
            climb_duration_sec=climb_duration_sec,
            climb_settle_sec=climb_settle_sec,
            altitude_hold_kp=altitude_hold_kp,
            altitude_hold_kp_down=altitude_hold_kp_down,
            altitude_hold_kd=altitude_hold_kd,
            altitude_hold_max_correction=altitude_hold_max_correction,
            altitude_hold_max_step=altitude_hold_max_step,
            altitude_hold_velocity_filter_tau_s=altitude_hold_velocity_filter_tau_s,
            altitude_hold_interval_sec=altitude_hold_interval_sec,
            max_ranger_m=max_ranger_m,
            target_ranger_m=target_ranger_m,
            requested_flight_mode=requested_flight_mode,
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
        elif action in ("up", "down") and unit.holding_altitude:
            # While altitude hold is engaged it overwrites axes.z every tick
            # (see RoosterUnit._altitude_hold_tick) -- a raw z pulse here
            # would be fought and mostly cancelled within one tick. Move the
            # setpoint the PD loop is chasing instead. "value" is METRES
            # here, not axis units (see DEFAULT_ALTITUDE_NUDGE_M).
            nudge_m = float(cmd.get("value", DEFAULT_ALTITUDE_NUDGE_M))
            unit.nudge_altitude_target(nudge_m if action == "up" else -nudge_m)
        elif action in _MOVE_ACTIONS:
            # Only override the axis this action controls - z in particular
            # is throttle/altitude-hold, and defaulting it to 0 here would
            # cut hover power and drop the drone (confirmed: turning while
            # flying without this fix caused an immediate fall).
            axes = dict(x=unit.axes.x, y=unit.axes.y, z=unit.axes.z, r=unit.axes.r)
            axes.update({axis: sign * value for axis, sign in _MOVE_ACTIONS[action].items()})
            unit.set_axes(**axes)
        elif action == "move":
            # Continuous-control variant for a planner (see module docstring):
            # an explicit {"x":..,"y":..,"r":..} dict instead of one named
            # direction. Same z-preservation rule as _MOVE_ACTIONS above -
            # z is never accepted here, only x/y/r are ever taken from the
            # payload, so a planner can never touch throttle/altitude-hold.
            requested = cmd.get("axes") or {}
            axes = dict(x=unit.axes.x, y=unit.axes.y, z=unit.axes.z, r=unit.axes.r)
            for axis in ("x", "y", "r"):
                if axis in requested:
                    axes[axis] = float(requested[axis])
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
