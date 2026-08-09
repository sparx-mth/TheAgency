#!/usr/bin/env python3
"""falcon_exploration_follower_node.py -- fly FALCON's own exploration plan.

FALCON's native ``exploration_node`` (the paper's own algorithm: space
decomposition, HGrid coverage planning, frontier selection) has been running on
every Sphera mission all along, already fed real pose/depth -- see LESSONS.md.
Nothing has ever consumed its output. ``fast_planner``'s ``traj_server`` turns
that into a continuous, dynamically-optimal 50 Hz setpoint stream
(``quadrotor_msgs/PositionCommand`` on ``/planning/pos_cmd``: position, velocity,
acceleration, yaw). This node is the missing last stage: it tracks that setpoint
and drives Rooster's actual velocity-command interface, the same job
waypoint_follower_node.py does for A*/NavDP paths and object_approach_node.py does
for the visual servo -- just for a continuous reference instead of a discrete path.

The loop is closed by core.planning.trackers.reference_tracker_3d.ReferenceTracker3D
-- already used for exactly this (FALCON trajectory -> real velocity command) on a
different platform, see tasks/planning/falcon_pegasus/isaac/mission.py's
``_explore()``. It emits a world-frame velocity + absolute heading; this node
rotates that into Rooster's body frame, turns the heading into a yaw RATE, then
force-shapes the result the same way object_approach_node.py does (Rooster's real
FCU needs fixed-magnitude pulses, not raw continuous low-magnitude commands) --
but WITHOUT ClosureGait's move-a-little/stop-and-look cadence: that gait exists so
the visual servo's camera gets a fresh look between bursts, which has no analogue
here and would just make otherwise-smooth trajectory tracking needlessly jerky.

A stale or invalid reference (traj_server not READY, or too old) makes the
tracker hold station rather than chase a dead setpoint -- built into
ReferenceTracker3D itself, not reimplemented here.

ROS I/O:
  in   ~pos_cmd_topic          quadrotor_msgs/PositionCommand (traj_server)
  in   ~odom_topic             nav_msgs/Odometry (falcon_adapter_node's /odom_world)
  in   ~demo_mode_topic        std_msgs/String   (granted mode, from the arbiter)
  out  ~demo_mode_request_topic std_msgs/String  (this node's mode request)
  out  ~cmd_vel_topic          geometry_msgs/Twist (gated cmd_vel_raw)

Usage: roslaunch falcon_adapter sphera_exploration.launch
"""
from __future__ import annotations

import math

import rospy
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from quadrotor_msgs.msg import PositionCommand
from std_msgs.msg import String

from sparx_agency.core.common.spatial_math import quat_to_yaw
from sparx_agency.core.common.types import ControlCommand, KinematicLimits, TrajectoryPoint
from sparx_agency.core.planning.trackers.multi_axis_follower.allocation import saturate
from sparx_agency.core.planning.trackers.reference_tracker_3d import (
    ReferenceTracker3D, ReferenceTrackerParams,
)
from sparx_agency.core.planning.visual_servo import AxisForceProfile, PulseShaper

#: The mode this node requests/holds while it owns /cmd_vel -- an arbiter-generic
#: string like every other follower's, see rooster_demo_mode_manager.py.
MODE_EXPLORING = "exploring"


def _param_bool(name, default):
    val = rospy.get_param(name, default)
    if isinstance(val, str):
        return val.strip().lower() in ("1", "true", "yes", "on")
    return bool(val)


class FalconExplorationFollowerNode:

    def __init__(self):
        rospy.init_node("falcon_exploration_follower")
        G = rospy.get_param

        self.drone_ns = str(G("~drone_ns", "")).rstrip("/")
        self.cmd_vel_topic = str(G("~cmd_vel_topic", self.drone_ns + "/cmd_vel"))
        self.pos_cmd_topic = str(G("~pos_cmd_topic", "/planning/pos_cmd"))
        self.odom_topic = str(G("~odom_topic", "/odom_world"))
        self.demo_mode_topic = str(G("~demo_mode_topic", "/R1/demo_mode"))
        self.demo_mode_request_topic = str(
            G("~demo_mode_request_topic", "/R1/demo_mode_request"))
        self.request_repeat_sec = float(G("~request_repeat_sec", 0.5))
        self.ctrl_hz = float(G("~ctrl_hz", 20.0))

        limits = KinematicLimits(
            max_speed_xy=float(G("~max_speed_xy", 1.2)),
            max_speed_z=float(G("~max_speed_z", 0.6)),
            max_yaw_rate=math.radians(float(G("~max_yaw_rate_deg", 45.0))),
            max_accel_xy=float(G("~max_accel_xy", 1.5)),
            max_accel_z=float(G("~max_accel_z", 1.0)),
        )
        self.tracker = ReferenceTracker3D(ReferenceTrackerParams(
            limits=limits,
            reference_timeout_s=float(G("~reference_timeout_s", 1.0)),
        ))
        self.yaw_kp = float(G("~yaw_kp", 1.0))

        # ── Rooster force shaping: fixed-magnitude pulses, no docking gait ──
        # See the module docstring for why ClosureGait is deliberately not used
        # here even though object_approach_node.py uses it alongside PulseShaper.
        self.force_mode = str(G("~force_mode", "fixed")).strip().lower()
        release_frac = float(G("~force_release_frac", 0.5))
        min_vxy = float(G("~min_vxy", 0.06))
        min_wz = math.radians(float(G("~min_wz_deg", 8.0)))

        def _axis(min_mag, max_mag, fixed_mag):
            return AxisForceProfile(
                min_magnitude=float(min_mag), max_magnitude=float(max_mag),
                release_frac=release_frac, mode=self.force_mode,
                fixed_magnitude=(None if fixed_mag is None or float(fixed_mag) <= 0.0
                                 else float(fixed_mag)))

        self.shaper = PulseShaper(
            vx=_axis(min_vxy, limits.max_speed_xy, G("~fixed_vx", 0.3)),
            vy=_axis(min_vxy, limits.max_speed_xy, G("~fixed_vy", 0.3)),
            wz=_axis(min_wz, limits.max_yaw_rate,
                     math.radians(float(G("~fixed_wz_deg", math.degrees(0.7))))),
            min_burst_ticks=int(G("~min_burst_ticks", 2)),
            brake_ticks=int(G("~brake_ticks", 0)))

        self._pose = None          # (x, y, z), world frame
        self._yaw = None           # radians
        self._velocity = None      # (vx, vy, vz), world frame

        self._reference = None         # TrajectoryPoint, last received
        self._reference_stamp = None   # rospy.Time of that reference
        self._reference_ready = False  # trajectory_flag == READY

        self.current_demo_mode = None
        self._requested_mode = None
        self._last_request_pub_t = rospy.Time(0)
        self._last_setpoint = None
        self._prev_tick_t = None

        # ── ROS I/O (publishers before subscribers) ──────────────────
        self.cmd_pub = rospy.Publisher(self.cmd_vel_topic, Twist, queue_size=1)
        self.demo_req_pub = rospy.Publisher(self.demo_mode_request_topic, String,
                                            queue_size=1, latch=True)

        rospy.Subscriber(self.odom_topic, Odometry, self._odom_cb, queue_size=10)
        rospy.Subscriber(self.pos_cmd_topic, PositionCommand, self._pos_cmd_cb, queue_size=1)
        rospy.Subscriber(self.demo_mode_topic, String, self._demo_mode_cb, queue_size=10)

        rospy.Timer(rospy.Duration(1.0 / max(self.ctrl_hz, 1.0)), self._tick)
        rospy.Timer(rospy.Duration(2.0), self._hb)

        self._banner()

    def _banner(self):
        limits = self.tracker.params.limits
        rospy.loginfo(
            "falcon_exploration_follower ready\n"
            "  pos_cmd in  = %s\n"
            "  odom in     = %s\n"
            "  cmd_vel out = %s (via '%s' hand-off on %s)\n"
            "  limits: xy=%.2f m/s  z=%.2f m/s  yaw=%.1f deg/s",
            self.pos_cmd_topic, self.odom_topic, self.cmd_vel_topic, MODE_EXPLORING,
            self.demo_mode_topic, limits.max_speed_xy, limits.max_speed_z,
            math.degrees(limits.max_yaw_rate))

    # ── Sensor callbacks ────────────────────────────────────────────
    def _odom_cb(self, msg):
        p = msg.pose.pose.position
        q = msg.pose.pose.orientation
        v = msg.twist.twist.linear
        self._pose = (p.x, p.y, p.z)
        self._yaw = quat_to_yaw(q.x, q.y, q.z, q.w)
        self._velocity = (v.x, v.y, v.z)

    def _pos_cmd_cb(self, msg):
        self._reference = TrajectoryPoint(
            t=msg.header.stamp.to_sec(),
            x=msg.position.x, y=msg.position.y, z=msg.position.z,
            vx=msg.velocity.x, vy=msg.velocity.y, vz=msg.velocity.z,
            ax=msg.acceleration.x, ay=msg.acceleration.y, az=msg.acceleration.z,
            yaw=msg.yaw)
        # traj_server is expected to stamp every command; fall back to "now" rather
        # than let an unset (zero) stamp read as infinitely old and always stale.
        self._reference_stamp = (msg.header.stamp if msg.header.stamp.to_sec() > 0
                                 else rospy.Time.now())
        self._reference_ready = (msg.trajectory_flag
                                 == PositionCommand.TRAJECTORY_STATUS_READY)

    def _demo_mode_cb(self, msg):
        mode = str(msg.data).strip().lower()
        if mode == MODE_EXPLORING and self.current_demo_mode != MODE_EXPLORING:
            # Freshly granted the channel -- start clean, not mid-way through
            # whatever the integrators/heading slew accumulated while idle.
            self.tracker.reset(yaw=self._yaw)
            self.shaper.reset()
        self.current_demo_mode = mode

    # ── Demo-mode hand-off (mirrors object_approach_node.py's pattern) ──
    def _driving(self):
        return self.current_demo_mode == MODE_EXPLORING

    def _request_mode(self, mode):
        if self._requested_mode != mode:
            self._requested_mode = mode
            self._last_request_pub_t = rospy.Time(0)   # force an immediate publish
            rospy.loginfo("falcon_exploration_follower: request demo_mode=%s (current=%s)",
                          mode, self.current_demo_mode)
        if self.current_demo_mode == mode:
            return
        now = rospy.Time.now()
        if (now - self._last_request_pub_t).to_sec() >= self.request_repeat_sec:
            self.demo_req_pub.publish(String(data=mode))
            self._last_request_pub_t = now

    # ── Control loop ─────────────────────────────────────────────────
    def _tick(self, _evt):
        try:
            self._step()
        except Exception as e:   # noqa: BLE001 -- resilience is the point
            rospy.logwarn_throttle(2.0, "falcon_exploration_follower: tick error (%s: %s)",
                                   type(e).__name__, e)

    def _step(self):
        now = rospy.Time.now()
        now_s = now.to_sec()
        dt = 0.0 if self._prev_tick_t is None else max(1e-3, now_s - self._prev_tick_t)
        self._prev_tick_t = now_s

        if self._pose is None or self._yaw is None:
            return   # no odometry yet -- nothing to track from

        self._request_mode(MODE_EXPLORING)
        if not self._driving():
            return   # not granted the channel -- stay silent, don't fight anyone

        reference = self._reference if self._reference_ready else None
        reference_age = ((now - self._reference_stamp).to_sec()
                         if self._reference_stamp is not None else float("inf"))

        setpoint = self.tracker.update(
            reference, self._pose, self._yaw, dt,
            velocity=self._velocity, reference_age=reference_age)
        self._last_setpoint = setpoint

        # World-frame velocity -> body frame (Rooster's cmd_vel convention).
        cos_y, sin_y = math.cos(self._yaw), math.sin(self._yaw)
        body_vx = setpoint.vx * cos_y + setpoint.vy * sin_y
        body_vy = -setpoint.vx * sin_y + setpoint.vy * cos_y

        # The tracker gives an absolute heading; a yaw RATE command is what
        # cmd_vel needs. Proportional on the same signed error the tracker
        # already computed (reference heading vs. measured), capped at the
        # platform's own ceiling.
        yaw_rate = saturate(self.yaw_kp * setpoint.yaw_error_rad,
                            self.tracker.params.limits.max_yaw_rate)

        self._publish_cmd(body_vx, body_vy, yaw_rate)

    def _publish_cmd(self, vx, vy, wz):
        shaped = self.shaper.shape(ControlCommand.velocity(float(vx), float(vy), 0.0, float(wz)))
        m = Twist()
        m.linear.x = float(shaped.x)
        m.linear.y = float(shaped.y)
        m.linear.z = 0.0
        m.angular.z = float(shaped.yaw_rate)
        self.cmd_pub.publish(m)

    def _hb(self, _evt):
        sp = self._last_setpoint
        rospy.loginfo(
            "falcon_exploration_follower hb  demo=%s  ref_ready=%s  "
            "pos_err=%s  holding=%s",
            self.current_demo_mode, self._reference_ready,
            "-" if sp is None else "%.2fm" % sp.position_error_m,
            "-" if sp is None else sp.holding)


def main():
    FalconExplorationFollowerNode()
    rospy.spin()


if __name__ == "__main__":
    main()
