#!/usr/bin/env python3
"""
NoMaD ROS2 Bridge Node — Visual Navigation

Subscribes to /waypoint from NoMaD (explore.py or navigate.py),
runs PD control, publishes velocity commands (Twist and/or ManualControl).

NoMaD runs as a separate process and handles all vision/planning.
This bridge only converts waypoints → Rooster R1 commands.

ManualControl axes (Ground Roll mode):
  x: forward / backward  (-1000 .. 1000)
  y: not used
  z: thrust / power       (-1000 .. 1000)
  r: yaw rotation         (-1000 .. 1000)
"""

import sys
import time
import math
import threading
import json
from typing import Optional

import yaml
import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.callback_groups import ReentrantCallbackGroup, MutuallyExclusiveCallbackGroup
from rclpy.executors import MultiThreadedExecutor

from std_msgs.msg import String, Float32MultiArray
from geometry_msgs.msg import Twist

# ── Platform actuation ────────────────────────────────────────────────
# Arming, the KeepAlive heartbeat and the hold-then-stop ManualControl latch are
# Rooster R1 concerns, not NoMaD concerns, so they live with the robot. This
# bridge only decides the axis VALUES (see _publish_velocity: `z` is thrust here,
# unlike OmniVLA where it is tilt).
from sparx_agency.robots.ROBOTICAN.adapters.rooster_manual_control import (
    HAS_ROOSTER,
    ManualAxes,
    RoosterManualControl,
)


# ══════════════════════════════════════════════════════════════════════
class NoMADBridge(Node):
    """ROS2 ↔ NoMaD bridge: waypoint → ManualControl."""

    def __init__(self, config_path: Optional[str] = None):
        super().__init__("nomad_bridge")
        self.declare_parameter("config_path", "")
        config_path = config_path or self.get_parameter("config_path").value

        # ── Load YAML config ──────────────────────────────────────────
        cfg = self._load_yaml(config_path)
        self.cfg = cfg

        # ── Shared state (lock-protected) ─────────────────────────────
        self.lock = threading.Lock()

        # ── PD parameters ─────────────────────────────────────────────
        pd = cfg["bridge"]
        self.kp_lin = pd.get("kp_linear", 1.0)
        self.kp_ang = pd.get("kp_angular", 2.0)
        self.max_v = pd.get("max_linear_vel", 0.3)
        self.max_w = pd.get("max_angular_vel", 0.5)
        self.reached_m = pd.get("waypoint_reached_m", 0.08)
        self.turn_in_place_rad = pd.get("turn_in_place_rad", 1.0)

        # ── Callback groups ───────────────────────────────────────────
        self.input_cg = ReentrantCallbackGroup()
        self.ctrl_cg = MutuallyExclusiveCallbackGroup()

        # ── Subscriber: NoMaD waypoint ────────────────────────────────
        inp = cfg["inputs"]
        self.create_subscription(
            Float32MultiArray, inp["waypoint"]["topic"],
            self._on_waypoint, 1, callback_group=self.input_cg,
        )

        # ── Publishers ────────────────────────────────────────────────
        out = cfg["outputs"]

        if out.get("twist", {}).get("enabled"):
            self.twist_pub = self.create_publisher(Twist, out["twist"]["topic"], 1)

        if out.get("feedback", {}).get("enabled"):
            self.fb_pub = self.create_publisher(String, out["feedback"]["topic"], 1)

        # ── Platform actuation (Rooster R1) ───────────────────────────
        # One adapter owns the publisher, the arm gate, the heartbeat and the
        # hold-then-stop latch. It is inert (send() -> False) when the Rooster
        # interface packages are unavailable, which is how this node runs on a
        # dev box for inference-only testing.
        mc = out.get("manual_control", {})
        ka = out.get("keep_alive", {})
        self.rooster = None
        if mc.get("enabled") or ka.get("enabled"):
            self.rooster = RoosterManualControl(
                self,
                rooster_id=inp.get("robot_id", "R1"),
                manual_control_topic=mc.get("topic"),
                keep_alive_topic=ka.get("topic"),
                publish_rate_hz=mc.get("publish_rate_hz", 50.0),
                keep_alive_rate_hz=ka.get("publish_rate_hz", 1.0),
                flight_mode=ka.get("requested_flight_mode", 1),
                callback_group=self.ctrl_cg,
            )
            self.rooster.attach()

        self._log_summary()

    # ══════════════════════════════════════════════════════════════════
    #  WAYPOINT → PD CONTROL
    # ══════════════════════════════════════════════════════════════════
    def _on_waypoint(self, msg: Float32MultiArray):
        if len(msg.data) < 2:
            return
        dx, dy = float(msg.data[0]), float(msg.data[1])

        # PD control: dx=forward, dy=left (robot frame)
        dist = math.sqrt(dx * dx + dy * dy)
        heading = math.atan2(dy, dx)

        if dist < self.reached_m:
            lin, ang = 0.0, 0.0
        elif abs(heading) > self.turn_in_place_rad:
            lin = 0.0
            ang = float(np.clip(self.kp_ang * heading, -self.max_w, self.max_w))
        else:
            lin = float(np.clip(self.kp_lin * dx, 0.0, self.max_v))
            ang = float(np.clip(self.kp_ang * heading, -self.max_w, self.max_w))

        self._publish_velocity(lin, ang)

        self.get_logger().info(
            f"wp=({dx:.2f},{dy:.2f})  v={lin:.3f}  w={ang:.3f}"
        )

        if hasattr(self, "fb_pub"):
            m = String()
            m.data = json.dumps({
                "dx": dx, "dy": dy, "linear": lin, "angular": ang,
                "t": time.time(),
            })
            self.fb_pub.publish(m)

    # ══════════════════════════════════════════════════════════════════
    #  OUTPUT PUBLISHING
    # ══════════════════════════════════════════════════════════════════
    def _publish_velocity(self, lin: float, ang: float):
        # Twist
        if hasattr(self, "twist_pub"):
            tw = Twist()
            tw.linear.x = lin
            tw.angular.z = ang
            self.twist_pub.publish(tw)

        # ManualControl (Rooster ground-roll mode)
        #   x: forward/backward  ·  y: unused in roll mode
        #   z: THRUST (NoMaD drives the ground wheels; OmniVLA uses z as tilt)
        #   r: yaw rotation
        # Only the axis VALUES are decided here. Arming, the heartbeat, the
        # -1000..1000 clamp and the hold-then-stop latch belong to the robot and
        # live in robots/ROBOTICAN/adapters/rooster_manual_control.py.
        if self.rooster is not None:
            self.rooster.send(self._to_axes(lin, ang))

    def _to_axes(self, lin: float, ang: float) -> ManualAxes:
        """NoMaD's ``(v, w)`` -> Rooster axes, with thrust held for traction.

        Three regimes, unchanged from the original bridge: stopped (no thrust at
        all), turn-in-place (thrust only for wheel traction, no forward axis),
        and cruise (forward + steer at a constant cruise thrust).
        """
        mc = self.cfg["outputs"]["manual_control"]
        ang_axis = ang * mc.get("angular_scale", 2000.0)
        hold_s = mc.get("duration_sec", 0.3)

        is_moving = abs(lin) > 0.01 or abs(ang) > 0.05
        if not is_moving:
            return ManualAxes(hold_s=hold_s)
        if abs(lin) < 0.01:
            return ManualAxes(z=mc.get("turn_thrust", 300.0), r=ang_axis,
                              hold_s=hold_s)
        return ManualAxes(x=lin * mc.get("linear_scale", 800.0),
                          z=mc.get("cruise_thrust", 400.0), r=ang_axis,
                          hold_s=hold_s)

    # ── Config ────────────────────────────────────────────────────────
    @staticmethod
    def _load_yaml(path: str) -> dict:
        if path and path.strip():
            with open(path) as f:
                return yaml.safe_load(f)
        raise RuntimeError("Config path required.  --config <path.yaml>")

    def _log_summary(self):
        i, o = self.cfg["inputs"], self.cfg["outputs"]
        self.get_logger().info("═" * 50)
        self.get_logger().info("NoMaD Bridge — Visual Navigation")
        self.get_logger().info(f"  Waypoint:    {i['waypoint']['topic']}")
        if o.get("twist", {}).get("enabled"):
            self.get_logger().info(f"  Twist out:   {o['twist']['topic']}")
        if o.get("manual_control", {}).get("enabled") and HAS_ROOSTER:
            self.get_logger().info(f"  ManualCtrl:  {o['manual_control']['topic']}")
        self.get_logger().info(f"  PD:          kp_lin={self.kp_lin}  kp_ang={self.kp_ang}")
        self.get_logger().info(f"  Rooster:     {HAS_ROOSTER}")
        self.get_logger().info("═" * 50)


# ══════════════════════════════════════════════════════════════════════
def main(args=None):
    rclpy.init(args=args)
    config_path = None
    for i, a in enumerate(sys.argv):
        if a == "--config" and i + 1 < len(sys.argv):
            config_path = sys.argv[i + 1]
    node = NoMADBridge(config_path)
    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()