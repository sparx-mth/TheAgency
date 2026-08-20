#!/usr/bin/env python3
"""Follow N1's committed route and drive the SJTU drone's one control input.

The "tracking the trajectory" half of the stack, and the mirror of FALCON's
``waypoint_follower_node``: subscribe the world-frame ``nav_msgs/Path`` the N1
policy node commits, pursue it, and publish a body twist on
``/simple_drone/cmd_vel``. Nothing here is N1-specific -- it would fly a NavDP
path, an A* path or a hand-drawn one identically, which is the point of putting a
plain ``Path`` on the wire between the two.

The controller is the shared holonomic
:class:`~sparx_agency.core.planning.trackers.pure_pursuit.tracker.PurePursuitTracker3D`,
which emits a **world-frame** velocity aimed at a lookahead point. The SJTU plugin
reads ``cmd_vel`` in the **yaw-aligned body frame**, so this node performs the one
rotation world->body and clamps the result with the platform's own
:mod:`~sparx_agency.robots.SJTU.adapters.velocity_command` adapter -- the same
horizontal-pair clamp the airframe needs so a speed limit never becomes a
steering error.

CPU-only, no torch: the GPU is the network's.
"""
from __future__ import annotations

import os
import threading
from math import cos, sin

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import rclpy
import yaml
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry, Path
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy

from sparx_agency.core.common.types import (
    KinematicLimits,
    Pose3D,
    State3D,
    Twist3D,
)
from sparx_agency.core.planning.interfaces.tracker import TrackerRequest
from sparx_agency.core.planning.trackers.pure_pursuit.params import PurePursuitParams3D
from sparx_agency.core.planning.trackers.pure_pursuit.tracker import PurePursuitTracker3D
from sparx_agency.robots.SJTU.adapters.velocity_command import (
    BodyVelocityLimits,
    BodyTwistCommand,
    fill_twist,
    twist_fields,
    zero_twist_fields,
)
from sparx_agency.tasks.planning.sjtu_internvla_n1.path_trajectory import (
    trajectory_from_points,
)


def _yaw_from_quat(q):
    from math import atan2
    return atan2(2.0 * (q.w * q.z + q.x * q.y),
                 1.0 - 2.0 * (q.y * q.y + q.z * q.z))


def _load_config(path):
    if not path or not os.path.isfile(path):
        return {}
    with open(path, "r") as handle:
        return yaml.safe_load(handle) or {}


class TrajectoryFollowerNode(Node):
    """Pursue a world path and command the SJTU drone's body twist."""

    def __init__(self):
        super().__init__("trajectory_follower_node")
        self.declare_parameter("config_file", "")
        cfg = _load_config(self.get_parameter("config_file").value)

        topics = cfg.get("topics", {})
        foll = cfg.get("follower", {})

        self._path_topic = topics.get("trajectory", "/simple_drone/n1/trajectory")
        self._odom_topic = topics.get("odom", "/simple_drone/odom")
        self._cmd_topic = topics.get("cmd_vel", "/simple_drone/cmd_vel")

        self._cruise = float(foll.get("cruise_speed", 0.4))
        self._target_alt = float(foll.get("target_altitude_m", 1.2))
        self._control_rate = float(foll.get("control_rate_hz", 20.0))
        max_speed_xy = float(foll.get("max_speed_xy", 1.0))
        max_speed_z = float(foll.get("max_speed_z", 0.5))
        max_yaw_rate = float(foll.get("max_yaw_rate", 1.2))

        self._tracker = PurePursuitTracker3D(PurePursuitParams3D(
            cruise_speed=self._cruise,
            max_speed=max(self._cruise, float(foll.get("max_track_speed", self._cruise))),
            max_speed_z=max_speed_z,
            max_yaw_rate=max_yaw_rate,
            base_lookahead=float(foll.get("lookahead_m", 0.8)),
            goal_tolerance=float(foll.get("goal_tolerance_m", 0.2)),
            yaw_lookahead=float(foll.get("yaw_lookahead_m", 1.5)),
        ))
        self._limits = KinematicLimits(max_speed_xy=max_speed_xy, max_speed_z=max_speed_z,
                                       max_yaw_rate=max_yaw_rate)
        self._body_limits = BodyVelocityLimits(max_speed_xy=max_speed_xy,
                                               max_speed_z=max_speed_z,
                                               max_yaw_rate=max_yaw_rate)

        self._lock = threading.Lock()
        self._path_xy = []   # list of (x, y) world
        self._state = None   # State3D

        latched = QoSProfile(reliability=ReliabilityPolicy.RELIABLE,
                             durability=DurabilityPolicy.TRANSIENT_LOCAL,
                             history=HistoryPolicy.KEEP_LAST, depth=1)
        sensor_qos = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT,
                                history=HistoryPolicy.KEEP_LAST, depth=1)

        self.create_subscription(Path, self._path_topic, self._on_path, latched)
        self.create_subscription(Odometry, self._odom_topic, self._on_odom, sensor_qos)
        self._cmd_pub = self.create_publisher(Twist, self._cmd_topic, 1)

        self.create_timer(1.0 / max(1e-3, self._control_rate), self._control)
        self.get_logger().info(
            "trajectory_follower_node up: path=%s odom=%s -> cmd_vel=%s "
            "(cruise %.2f m/s, altitude %.2f m)"
            % (self._path_topic, self._odom_topic, self._cmd_topic,
               self._cruise, self._target_alt))

    def _on_path(self, msg):
        xy = [(ps.pose.position.x, ps.pose.position.y) for ps in msg.poses]
        with self._lock:
            self._path_xy = xy

    def _on_odom(self, msg):
        p = msg.pose.pose
        t = msg.twist.twist
        state = State3D(
            pose=Pose3D(p.position.x, p.position.y, p.position.z, _yaw_from_quat(p.orientation)),
            twist=Twist3D(t.linear.x, t.linear.y, t.linear.z, t.angular.z))
        with self._lock:
            self._state = state

    def _control(self):
        with self._lock:
            state = self._state
            path_xy = list(self._path_xy)
        if state is None:
            return
        if len(path_xy) < 2:
            self._publish(zero_twist_fields())
            return

        # Hold the configured cruise altitude: every reference point sits at it,
        # so the 3D pursuit commands vz toward it while it tracks xy.
        points = [(x, y, self._target_alt) for (x, y) in path_xy]
        try:
            trajectory = trajectory_from_points(points, self._cruise)
        except ValueError:
            self._publish(zero_twist_fields())
            return

        result = self._tracker.step(TrackerRequest(
            state=state, trajectory=trajectory, t=0.0, limits=self._limits))
        cmd = result.command  # world-frame vx, vy, vz + yaw_rate

        body = self._world_to_body(cmd.x, cmd.y, cmd.z, state.pose.yaw)
        fields = twist_fields(
            BodyTwistCommand(vx=body[0], vy=body[1], vz=body[2], yaw_rate=cmd.yaw_rate),
            self._body_limits)
        self._publish(fields)

    @staticmethod
    def _world_to_body(vx, vy, vz, yaw):
        """Rotate a world-frame velocity into the yaw-aligned body frame."""
        c, s = cos(yaw), sin(yaw)
        return (c * vx + s * vy, -s * vx + c * vy, vz)

    def _publish(self, fields):
        self._cmd_pub.publish(fill_twist(Twist(), fields))


def main(args=None):
    rclpy.init(args=args)
    node = TrajectoryFollowerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()


