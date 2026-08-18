#!/usr/bin/env python3
"""rooster_bspline_follower_node.py -- fly FALCON's raw B-spline with the
measured-plant velocity servo, instead of the sampled-point + plain-PID path.

Sibling of ``falcon_exploration_follower_node.py`` (FEF), which tracks
``traj_server``'s 50Hz sampled ``/planning/pos_cmd`` via
``core.planning.trackers.reference_tracker_3d`` and force-shapes the result
down to a FIXED bang-bang magnitude by default -- discarding the tracker's
own computed error size entirely. This node instead:

  1. Subscribes FALCON's own raw curve on ``/planning/bspline`` directly (no
     ``traj_server`` in the loop) and rebuilds it with
     ``core.planning.trajectories.bspline.BsplineTrajectory`` -- the exact
     same construction FALCON's C++ uses, proven live on the SJTU deployment
     (see ``falcon_sjtu/adapter/scripts/bspline_follower_node.py`` and
     ``core/control/README.md``).
  2. Drives ``core.control.velocity_servo.VelocityServo``, which inverts
     Rooster's own velocity-loop lag with a feedforward lead term read
     straight off the spline's acceleration, instead of a plain P+feedforward
     law -- measured 2.8-3.8x tighter cross-track on SJTU's airframe.
  3. Publishes the SAME body Twist on the SAME ``cmd_vel_raw`` topic FEF
     does, through the SAME downstream chain (cmd_vel_gate -> bridge ->
     rooster_twist_control_adapter -> rooster_command_unit -> ManualControl).
     Deliberately unchanged: this isolates "does the control law improve
     tracking" as the one variable, per this repo's own repeated lesson about
     measuring one thing at a time. Only one of FEF / this node may run at
     once (nav_stack.launch's ``exploration_follower`` arg selects which) --
     two publishers on the same topic is exactly the "competing publisher"
     class of bug LESSONS.md documents three times over for this stack.

**UNMEASURED for Rooster/Sphera**: the plant constants below are
``core/control``'s own generic defaults, not a measured step response --
see ``~plant_*`` params and ``core/control/README.md``'s 20-second
measurement procedure. Wrong plant numbers "read exactly like a mistuned
position gain" per that doc; do not trust this node's tracking numbers until
they are replaced with a real measurement.

**Known, deliberately out of scope for this first pass**: altitude (z) is
NOT wired anywhere downstream of ``rooster_twist_control_adapter.py``, which
drops ``linear.z`` -- Rooster's altitude is a completely separate, FALCON-
blind PD loop in ``rooster_command_unit.py``. This node still computes and
carries ``vz``/``world_vz`` in its diagnostics (so a future altitude-coupling
fix has real numbers to start from) but the wire command's z is informational
only until that gap is closed. Also out of scope: SJTU's contact/retreat/
unstick/survey safety reflexes (Gazebo-bumper-specific, not a direct port).
**Not out of scope, added 2026-08-17 after a live capsize during first
flight test**: the tilt-cutoff reflex (``~tilt_limit_deg``, mirrors SJTU's
bspline_follower_node.py exactly) that cuts horizontal drive the instant
roll/pitch crosses a threshold. A follower with no such reflex kept
commanding translation while the aircraft was already past recoverable tilt
after an unseen contact, which did not cause the capsize but did not help it
either -- see LESSONS.md's 2026-08-17 entry. This node also keeps the one
Rooster-proven safety measure FEF already has (``max_measured_speed_xy``,
below).

**Second correction, same day**: the first cut of this reflex read roll/
pitch off ``/odom_world``'s orientation quaternion, which is ALWAYS
yaw-only (``rooster_ground_truth_localization.py`` publishes
``x=y=0, z=sin(yaw/2), w=cos(yaw/2)`` by deliberate contract with other
consumers) -- so the check silently never fired through a second real
tilt event either. Attitude now comes from ``~attitude_topic``
(``/R1/attitude_rpy``, a plain ``geometry_msgs/Vector3`` bridged from that
same node's new raw-roll/pitch/yaw publisher), not from odometry.

ROS I/O:
  in   ~bspline_topic          trajectory/Bspline (exploration_node's raw plan)
  in   ~replan_topic           std_msgs/Int32 (FALCON's own trajectory verdict)
  in   ~odom_topic             nav_msgs/Odometry (falcon_adapter_node's /odom_world)
  in   ~demo_mode_topic        std_msgs/String   (granted mode, from the arbiter)
  out  ~demo_mode_request_topic std_msgs/String  (this node's mode request)
  out  ~cmd_vel_topic          geometry_msgs/Twist (gated cmd_vel_raw)

Usage: roslaunch falcon_adapter sphera_drone.launch nav_mode:=exploration \
           exploration_follower:=bspline
"""
from __future__ import annotations

import math

import rospy
from geometry_msgs.msg import Twist, Vector3
from nav_msgs.msg import Odometry
from std_msgs.msg import Int32, String
from trajectory.msg import Bspline

from sparx_agency.core.common.spatial_math import quat_to_yaw
from sparx_agency.core.control.reference.params import ReferenceParams
from sparx_agency.core.control.velocity_servo import (
    AxisPlant, VelocityLimits, VelocityPlant, VelocityServo, VelocityServoParams,
)
from sparx_agency.core.planning.trajectories.bspline import BsplineTrajectory
from sparx_agency.core.planning.trajectories.bspline.projection import ProjectionParams

#: The mode this node requests/holds while it owns /cmd_vel -- same
#: arbiter-generic string FEF and every other follower uses.
MODE_EXPLORING = "exploring"

#: FALCON's own /planning/replan verdicts (see falcon_sjtu's follower for
#: the derivation -- these values come from exploration_fsm.cpp itself).
REPLAN_TRAJECTORY_UNSAFE = 1
REPLAN_EXPLORATION_FINISHED = 2


class RoosterBsplineFollowerNode:
    def __init__(self):
        rospy.init_node("rooster_bspline_follower")
        G = rospy.get_param

        self.drone_ns = str(G("~drone_ns", "")).rstrip("/")
        self.cmd_vel_topic = str(G("~cmd_vel_topic", self.drone_ns + "/cmd_vel"))
        self.bspline_topic = str(G("~bspline_topic", "/planning/bspline"))
        self.replan_topic = str(G("~replan_topic", "/planning/replan"))
        self.odom_topic = str(G("~odom_topic", "/odom_world"))
        self.demo_mode_topic = str(G("~demo_mode_topic", "/R1/demo_mode"))
        self.demo_mode_request_topic = str(
            G("~demo_mode_request_topic", "/R1/demo_mode_request"))
        # See module docstring: NOT derived from odom (always yaw-only).
        self.attitude_topic = str(G("~attitude_topic", "/R1/attitude_rpy"))
        self.request_repeat_sec = float(G("~request_repeat_sec", 0.5))
        self.ctrl_hz = float(G("~ctrl_hz", 20.0))
        self.state_timeout_s = float(G("~state_timeout_s", 0.5))

        # Same backstop and same rationale as falcon_exploration_follower_node.py:
        # checks MEASURED speed (odometry), not the commanded reference --
        # Rooster's own FCU velocity loop is a closed loop we don't own and can
        # overshoot a commanded pulse independently of anything computed here.
        self.max_measured_speed_xy = float(G("~max_measured_speed_xy", 0.0))

        # Same default as falcon_sjtu's bspline_follower_node.py -- cuts
        # horizontal drive the instant roll/pitch crosses this, well before
        # the airframe's own physical recoverability ceiling (~35deg on that
        # platform; unmeasured on Rooster/Sphera). See module docstring.
        self.tilt_limit_deg = float(G("~tilt_limit_deg", 15.0))

        self._servo = VelocityServo(self._build_params(G))

        self._pose = None          # (x, y, z), world frame
        self._yaw = None           # radians
        self._roll_deg = 0.0
        self._pitch_deg = 0.0
        self._attitude_at = None   # rospy.Time of the last real attitude message
        self._velocity = None      # (vx, vy, vz), world frame
        self._odom_at = None       # rospy.Time of the last odom message

        self._stopped = False      # FALCON condemned the live trajectory
        self._finished = False     # FALCON reported exploration finished
        self._last_command = None  # BodyTwistCommand, for the heartbeat
        self._prev_tick_t = None

        self.current_demo_mode = None
        self._requested_mode = None
        self._last_request_pub_t = rospy.Time(0)

        # ── ROS I/O (publishers before subscribers) ──────────────────
        self.cmd_pub = rospy.Publisher(self.cmd_vel_topic, Twist, queue_size=1)
        self.demo_req_pub = rospy.Publisher(self.demo_mode_request_topic, String,
                                            queue_size=1, latch=True)

        rospy.Subscriber(self.odom_topic, Odometry, self._odom_cb, queue_size=10)
        rospy.Subscriber(self.attitude_topic, Vector3, self._attitude_cb, queue_size=5)
        rospy.Subscriber(self.bspline_topic, Bspline, self._bspline_cb, queue_size=5)
        rospy.Subscriber(self.replan_topic, Int32, self._replan_cb, queue_size=5)
        rospy.Subscriber(self.demo_mode_topic, String, self._demo_mode_cb, queue_size=10)

        rospy.Timer(rospy.Duration(1.0 / max(self.ctrl_hz, 1.0)), self._tick)
        rospy.Timer(rospy.Duration(2.0), self._hb)

        self._banner()

    # ── Setup ────────────────────────────────────────────────────────────
    def _build_params(self, G):
        """Assemble VelocityServoParams from rospy params. See the module
        docstring: every ``~plant_*`` default here is UNMEASURED for Rooster.
        """
        plant = VelocityPlant(
            horizontal=AxisPlant(
                dc_gain=float(G("~plant_xy_gain", 1.0)),
                time_constant_s=float(G("~plant_xy_tau", 0.5)),
                delay_s=float(G("~plant_xy_delay", 0.15))),
            vertical=AxisPlant(
                dc_gain=float(G("~plant_z_gain", 1.0)),
                time_constant_s=float(G("~plant_z_tau", 0.4)),
                delay_s=float(G("~plant_z_delay", 0.05))),
            yaw=AxisPlant(
                dc_gain=float(G("~plant_yaw_gain", 1.0)),
                time_constant_s=float(G("~plant_yaw_tau", 0.5)),
                delay_s=float(G("~plant_yaw_delay", 0.06))))
        # Slow-flight-first ceiling: default well under cruise (matches
        # nav_stack.launch's explore_max_speed_xy) until tracking is proven
        # on a real measured plant, then raise.
        limits = VelocityLimits(
            max_speed_xy=float(G("~max_speed_xy", 0.4)),
            max_speed_up=float(G("~max_speed_up", 0.4)),
            max_speed_down=float(G("~max_speed_down", 0.3)),
            max_accel_xy=float(G("~max_accel_xy", 1.0)),
            max_accel_z=float(G("~max_accel_z", 1.0)),
            max_yaw_rate=math.radians(float(G("~max_yaw_rate_deg", 45.0))),
            max_yaw_accel=float(G("~max_yaw_accel", 3.0)))
        reference = ReferenceParams(
            projection=ProjectionParams(
                search_back_s=float(G("~proj_back_s", 0.5)),
                search_ahead_s=float(G("~proj_ahead_s", 0.30))))
        return VelocityServoParams(
            plant=plant, limits=limits, reference=reference,
            yaw_gain=float(G("~yaw_gain", 1.5)),
            use_feedforward_lead=bool(G("~use_feedforward_lead", True)),
            predict_reference=bool(G("~predict_reference", True)),
            # Conservative vs. SJTU's tuned 0.25/0.35 -- Rooster's plant is
            # unmeasured, so start narrow and widen once it is.
            max_overspeed=float(G("~max_overspeed", 0.15)),
            max_catchup_speed=float(G("~max_catchup_speed", 0.1)))

    def _banner(self):
        limits = self._servo.params.limits
        rospy.loginfo(
            "rooster_bspline_follower ready\n"
            "  bspline in  = %s\n"
            "  odom in     = %s\n"
            "  cmd_vel out = %s (via '%s' hand-off on %s)\n"
            "  limits: xy=%.2f m/s  yaw=%.1f deg/s  (UNMEASURED plant -- see module docstring)",
            self.bspline_topic, self.odom_topic, self.cmd_vel_topic, MODE_EXPLORING,
            self.demo_mode_topic, limits.max_speed_xy, math.degrees(limits.max_yaw_rate))

    # ── Sensor / plan callbacks ──────────────────────────────────────────
    def _odom_cb(self, msg):
        p = msg.pose.pose.position
        q = msg.pose.pose.orientation
        v = msg.twist.twist.linear
        self._pose = (p.x, p.y, p.z)
        self._yaw = quat_to_yaw(q.x, q.y, q.z, q.w)
        self._velocity = (v.x, v.y, v.z)
        self._odom_at = rospy.Time.now()

    def _attitude_cb(self, msg):
        # Vector3.x/y = raw roll/pitch, radians, sign UNVERIFIED -- magnitude
        # only. See module docstring for why this is a separate topic from
        # /odom_world (always yaw-only there).
        self._roll_deg = math.degrees(msg.x)
        self._pitch_deg = math.degrees(msg.y)
        self._attitude_at = rospy.Time.now()

    def _bspline_cb(self, msg):
        """Rebuild FALCON's curve and queue it -- FALCON's own construction
        rules, so this node evaluates the same polynomial FALCON planned
        against. order is asserted rather than trusted: the publisher
        hardcodes degree 3, so a mismatch is a silent wrong-curve failure.
        """
        if msg.order != 3:
            raise ValueError(
                "FALCON published a degree-%d position spline; this follower "
                "rebuilds degree 3. Refusing to fly a curve it would "
                "evaluate incorrectly." % (msg.order,))
        trajectory = BsplineTrajectory.from_falcon(
            order=msg.order,
            knots=list(msg.knots),
            position_points=[(p.x, p.y, p.z) for p in msg.pos_pts],
            yaw_points=list(msg.yaw_pts),
            yaw_dt=msg.yaw_dt,
            start_time_s=msg.start_time.to_sec(),
            traj_id=msg.traj_id)
        if self._servo.set_trajectory(trajectory):
            # A new plan supersedes a stop: FALCON only replans after
            # condemning what it was flying, so a fresh curve is its own
            # statement that it found a way out.
            self._stopped = False

    def _replan_cb(self, msg):
        """FALCON's own verdict on the trajectory it is flying -- a control
        input, not telemetry: msg.data==1 means the EXECUTING trajectory was
        found in collision, and flying it further carries the aircraft's
        momentum into whatever FALCON just found.
        """
        if msg.data == REPLAN_TRAJECTORY_UNSAFE:
            self._stopped = True
            rospy.logwarn("[bspline_follower] FALCON condemned the live trajectory; holding")
        elif msg.data == REPLAN_EXPLORATION_FINISHED:
            self._finished = True
            rospy.loginfo("[bspline_follower] exploration finished; holding station")

    def _demo_mode_cb(self, msg):
        mode = str(msg.data).strip().lower()
        if mode == MODE_EXPLORING and self.current_demo_mode != MODE_EXPLORING:
            # Freshly granted the channel -- start clean, not carrying
            # integrators/hold state accumulated while idle.
            self._servo.reset()
        self.current_demo_mode = mode

    # ── Demo-mode hand-off (mirrors falcon_exploration_follower_node.py) ──
    def _driving(self):
        return self.current_demo_mode == MODE_EXPLORING

    def _request_mode(self, mode):
        if self._requested_mode != mode:
            self._requested_mode = mode
            self._last_request_pub_t = rospy.Time(0)
            rospy.loginfo("rooster_bspline_follower: request demo_mode=%s (current=%s)",
                          mode, self.current_demo_mode)
        if self.current_demo_mode == mode:
            return
        now = rospy.Time.now()
        if (now - self._last_request_pub_t).to_sec() >= self.request_repeat_sec:
            self.demo_req_pub.publish(String(data=mode))
            self._last_request_pub_t = now

    # ── Control loop ─────────────────────────────────────────────────────
    def _tick(self, _evt):
        try:
            self._step()
        except Exception as e:   # noqa: BLE001 -- resilience is the point
            rospy.logwarn_throttle(2.0, "rooster_bspline_follower: tick error (%s: %s)",
                                   type(e).__name__, e)

    def _step(self):
        now = rospy.Time.now()
        now_s = now.to_sec()
        dt = 0.0 if self._prev_tick_t is None else max(1e-3, now_s - self._prev_tick_t)
        self._prev_tick_t = now_s

        if self._pose is None or self._yaw is None:
            return   # no odometry yet -- nothing to track from

        if (self._odom_at is None
                or (now - self._odom_at).to_sec() > self.state_timeout_s):
            # Stale state means no loop: publishing the previous command would
            # fly blind on it. Zero and forget the integrators/hold rather
            # than resume from wherever they were when the state went stale.
            self._publish_cmd(0.0, 0.0, 0.0)
            self._servo.reset()
            return

        # No real attitude yet, or it went stale: assume the worst rather than
        # silently fly as if level (that silent-level default is exactly the
        # bug this reflex was added to fix the first time -- see module
        # docstring's "second correction").
        if (self._attitude_at is None
                or (now - self._attitude_at).to_sec() > self.state_timeout_s):
            self._publish_cmd(0.0, 0.0, 0.0)
            self._servo.reset()
            rospy.logwarn_throttle(
                1.0, "rooster_bspline_follower: no fresh attitude on %s; "
                "cutting drive rather than fly blind on tilt", self.attitude_topic)
            return

        # Attitude reflex, ahead of the demo-mode gate: a contact the depth
        # camera's near clip can't see can tip the aircraft, and past its
        # physical recoverability ceiling it cannot come back. If roll or
        # pitch crosses the margin, cut horizontal drive and hold so it
        # settles back level rather than tipping further under continued
        # translation commands -- added 2026-08-17 after a live capsize this
        # follower's first flight test did not cause but also did not cut
        # short. See module docstring.
        if abs(self._roll_deg) > self.tilt_limit_deg or abs(self._pitch_deg) > self.tilt_limit_deg:
            self._publish_cmd(0.0, 0.0, 0.0)
            self._servo.reset()
            rospy.logwarn_throttle(
                1.0, "rooster_bspline_follower: tilt roll=%.0f pitch=%.0f deg; "
                "cutting drive to let it settle level", self._roll_deg, self._pitch_deg)
            return

        self._request_mode(MODE_EXPLORING)
        if not self._driving():
            return   # not granted the channel -- stay silent, don't fight anyone

        follow = not (self._stopped or self._finished)
        command = self._servo.update(
            self._pose, self._velocity, self._yaw, dt, now_s, follow=follow)
        self._last_command = command

        body_vx, body_vy = command.vx, command.vy
        if self.max_measured_speed_xy > 0.0 and self._velocity is not None:
            measured_speed = math.hypot(self._velocity[0], self._velocity[1])
            if measured_speed >= self.max_measured_speed_xy:
                rospy.logwarn_throttle(
                    2.0, "rooster_bspline_follower: measured speed %.2fm/s >= "
                    "%.2fm/s cap, withholding translation this tick",
                    measured_speed, self.max_measured_speed_xy)
                body_vx, body_vy = 0.0, 0.0

        self._publish_cmd(body_vx, body_vy, command.yaw_rate)

    def _publish_cmd(self, vx, vy, wz):
        m = Twist()
        m.linear.x = float(vx)
        m.linear.y = float(vy)
        m.linear.z = 0.0   # see module docstring: z is not wired downstream yet
        m.angular.z = float(wz)
        self.cmd_pub.publish(m)

    def _hb(self, _evt):
        c = self._last_command
        rospy.loginfo(
            "rooster_bspline_follower hb  demo=%s  stopped=%s  finished=%s  "
            "pos_err=%s  holding=%s  diverged=%s",
            self.current_demo_mode, self._stopped, self._finished,
            "-" if c is None else "%.2fm" % c.position_error_m,
            "-" if c is None else c.holding,
            "-" if c is None else c.diverged)


def main():
    RoosterBsplineFollowerNode()
    rospy.spin()


if __name__ == "__main__":
    main()
