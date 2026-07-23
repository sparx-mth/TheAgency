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

from std_msgs.msg import String, Header, Float32MultiArray
from geometry_msgs.msg import Twist

# ── Optional Rooster messages ─────────────────────────────────────────
try:
    from fcu_driver_interfaces.msg import ManualControl
    from rooster_handler_interfaces.msg import KeepAlive
    HAS_ROOSTER = True
except ImportError:
    HAS_ROOSTER = False
    ManualControl = KeepAlive = None

try:
    from std_srvs.srv import SetBool
except ImportError:
    SetBool = None


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
        self.arm_state = False

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

        if out.get("manual_control", {}).get("enabled") and HAS_ROOSTER:
            mc = out["manual_control"]
            self.mc_pub = self.create_publisher(ManualControl, mc["topic"], 10)
            self.create_timer(1.0 / mc.get("publish_rate_hz", 50.0),
                              self._mc_tick, callback_group=self.ctrl_cg)
            self._mc_msg: Optional[ManualControl] = None
            self._mc_expire: float = 0.0

        if out.get("keep_alive", {}).get("enabled") and HAS_ROOSTER:
            ka = out["keep_alive"]
            self.ka_pub = self.create_publisher(KeepAlive, ka["topic"], 10)
            self._ka_cfg = ka
            self.create_timer(1.0 / ka.get("publish_rate_hz", 1.0),
                              self._ka_tick, callback_group=self.ctrl_cg)

        if out.get("feedback", {}).get("enabled"):
            self.fb_pub = self.create_publisher(String, out["feedback"]["topic"], 1)

        # ── Arm service (Rooster) ─────────────────────────────────────
        self._arm_client = None
        rid = inp.get("robot_id", "R1")
        if SetBool and HAS_ROOSTER:
            self._arm_client = self.create_client(
                SetBool, f"/{rid}/fcu/command/force_arm")

        if HAS_ROOSTER:
            from fcu_driver_interfaces.msg import UAVState
            self.create_subscription(
                UAVState, f"/{rid}/fcu/state",
                self._on_uav_state, 10, callback_group=self.input_cg)

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
        out = self.cfg["outputs"]

        # Twist
        if hasattr(self, "twist_pub"):
            tw = Twist()
            tw.linear.x = lin
            tw.angular.z = ang
            self.twist_pub.publish(tw)

        # ManualControl (Rooster ground-roll mode)
        #   x: forward/backward  (-1000 .. 1000)
        #   y: not used in roll mode
        #   z: thrust/power      (-1000 .. 1000)
        #   r: yaw rotation      (-1000 .. 1000)
        if hasattr(self, "mc_pub") and HAS_ROOSTER:
            mc_cfg = out["manual_control"]
            if not self.arm_state:
                self._request_arm()
                return

            msg = ManualControl()
            msg.header = Header()
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.y = 0.0
            msg.buttons = 0

            is_moving = abs(lin) > 0.01 or abs(ang) > 0.05

            if not is_moving:
                # Stopped — no thrust, no movement
                msg.x = 0.0
                msg.z = 0.0
                msg.r = 0.0
            elif abs(lin) < 0.01:
                # Turn in place — constant thrust for traction
                msg.x = 0.0
                msg.z = float(mc_cfg.get("turn_thrust", 300.0))
                msg.r = float(np.clip(ang * mc_cfg.get("angular_scale", 2000.0),
                                      -1000.0, 1000.0))
            else:
                # Move forward + steer — constant cruise thrust
                msg.x = float(np.clip(lin * mc_cfg.get("linear_scale", 800.0),
                                      -1000.0, 1000.0))
                msg.z = float(mc_cfg.get("cruise_thrust", 400.0))
                msg.r = float(np.clip(ang * mc_cfg.get("angular_scale", 2000.0),
                                      -1000.0, 1000.0))

            dur = mc_cfg.get("duration_sec", 0.3)
            with self.lock:
                self._mc_msg = msg
                self._mc_expire = time.time() + dur

    def _mc_tick(self):
        if not hasattr(self, "mc_pub"):
            return
        with self.lock:
            if self._mc_msg and time.time() < self._mc_expire:
                msg = self._mc_msg
            else:
                # No active waypoint → full stop
                msg = ManualControl()
                msg.header = Header()
                msg.x = 0.0
                msg.y = 0.0
                msg.z = 0.0
                msg.r = 0.0
                msg.buttons = 0
                self._mc_msg = None
        msg.header.stamp = self.get_clock().now().to_msg()
        self.mc_pub.publish(msg)

    def _ka_tick(self):
        if not hasattr(self, "ka_pub"):
            return
        msg = KeepAlive()
        msg.header = Header()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.is_active = self._ka_cfg.get("is_active", True)
        msg.requested_flight_mode = self._ka_cfg.get("requested_flight_mode", 1)
        msg.command_reboot = False
        self.ka_pub.publish(msg)

    def _on_uav_state(self, msg):
        with self.lock:
            self.arm_state = msg.armed

    def _request_arm(self):
        if self._arm_client and self._arm_client.service_is_ready():
            req = SetBool.Request()
            req.data = True
            self._arm_client.call_async(req)

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