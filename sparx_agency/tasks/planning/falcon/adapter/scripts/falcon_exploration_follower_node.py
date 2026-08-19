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

**2026-08-17 addition**: a tilt-cutoff reflex (``~tilt_limit_deg``) cuts
horizontal drive the instant roll/pitch crosses a threshold -- this node had
none at all before (only ever tracked yaw). Attitude comes from
``~attitude_topic`` (``/R1/attitude_rpy``, a plain ``geometry_msgs/Vector3``),
NOT from ``/odom_world``'s orientation, which is always yaw-only by a
deliberate, shared contract with other consumers
(``rooster_ground_truth_localization.py``) -- reading roll/pitch off it would
silently never see a real tilt. See LESSONS.md's 2026-08-17 entry.

ROS I/O:
  in   ~pos_cmd_topic          quadrotor_msgs/PositionCommand (traj_server)
  in   ~odom_topic             nav_msgs/Odometry (falcon_adapter_node's /odom_world)
  in   ~attitude_topic         geometry_msgs/Vector3 (raw roll/pitch/yaw, rad)
  in   ~demo_mode_topic        std_msgs/String   (granted mode, from the arbiter)
  out  ~demo_mode_request_topic std_msgs/String  (this node's mode request)
  out  ~cmd_vel_topic          geometry_msgs/Twist (gated cmd_vel_raw)

Usage: roslaunch falcon_adapter sphera_exploration.launch
"""
from __future__ import annotations

import math

import rospy
from geometry_msgs.msg import Twist, Vector3
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
        self.attitude_topic = str(G("~attitude_topic", "/R1/attitude_rpy"))
        self.request_repeat_sec = float(G("~request_repeat_sec", 0.5))
        self.ctrl_hz = float(G("~ctrl_hz", 20.0))
        self.state_timeout_s = float(G("~state_timeout_s", 0.5))
        # Same default as falcon_sjtu's bspline_follower_node.py -- cuts
        # horizontal drive the instant roll/pitch crosses this, well before
        # the airframe's physical recoverability ceiling. Added 2026-08-17
        # after a live capsize exposed that this node had no attitude
        # awareness at all (only tracked yaw). See LESSONS.md.
        self.tilt_limit_deg = float(G("~tilt_limit_deg", 15.0))
        #: Tilt the aircraft must fall back below before drive resumes.
        #:
        #: Without hysteresis this reflex chatters, because the threshold sits
        #: inside the range of ORDINARY flight: measured 2026-08-20 at the
        #: current cruise, pitch is p90 21 deg and p99 29 against a 25 deg
        #: limit, and the cut fired 56-196 times in a single run -- once every
        #: 3-7 seconds, each time zeroing translation AND yaw and resetting the
        #: tracker. That is a stop/go stutter generator, not a safety reflex.
        #: Default is 8 deg of margin below the limit.
        self.tilt_resume_deg = float(
            G("~tilt_resume_deg", max(0.0, self.tilt_limit_deg - 8.0)))
        self._tilt_cut = False

        # 2026-08-18: measured that this node was emitting PURE LATERAL demand
        # -- cmd_fwd exactly 0.000 and cmd_wz exactly 0.000 across 2254
        # consecutive samples, while cmd_lat averaged 0.49 m/s. Rooster's
        # lateral axis is its worst (ground-truth measured dead until ~axis
        # 1000, then 30deg+ of roll), and its forward axis is its best (usable
        # from ~620, up to 1.25 m/s). Following FALCON's independently-planned
        # yaw curve leaves the nose across the direction of travel, so every
        # command lands on the bad axis.
        #
        # "course" mode instead points the nose ALONG the commanded velocity,
        # turning sideways demand into forward demand, and holds translation
        # until roughly aligned (turn-then-go) so the aircraft is not sliding
        # sideways mid-turn. "reference" restores the old behaviour of tracking
        # FALCON's own yaw curve -- keep it for comparison, not as the default.
        self.yaw_mode = str(G("~yaw_mode", "course")).strip().lower()
        self.course_min_speed = float(G("~course_min_speed", 0.05))
        # Near 90, not 40: cos(err) fades forward speed out on its own, so this
        # only needs to catch the sign change past 90 deg. See the shaping block
        # in the command path for the measurement behind that.
        self.align_gate_deg = float(G("~align_gate_deg", 85.0))
        #: Ceiling on how fast the COMMANDED course may rotate, deg/s. 0 = off.
        #:
        #: A reference cannot be tracked faster than the plant can follow it.
        #: Measured 2026-08-19: in the stalled half of a run the aircraft yawed
        #: ~3900 deg per minute (65 deg/s, near its own 90 deg/s ceiling) while
        #: travelling 15 m inside a ONE-METRE box -- 230 deg of turning per
        #: metre against 45 deg/m when it was exploring properly. Heading error
        #: never converged (39-147 deg for 300 s straight) because the demand
        #: swung as fast as the aircraft could chase it: FALCON replans at ~68 Hz
        #: and each plan sets off in its own direction, so the nose chased a
        #: course that never held still long enough to fly.
        #:
        #: Half the platform's yaw ceiling leaves the aircraft comfortably
        #: faster than its own reference, which is the condition for the error
        #: to close at all. A genuine 180 deg turn costs 4 s instead of 2.
        self.course_slew_deg_s = float(G("~course_slew_deg_s", 45.0))
        self._course_cmd = None
        # Forward speed floor held through a turn, and the planned speed above
        # which it arms. See the cos-fade block in _step for why a floor beats
        # letting cos() brake the aircraft to zero.
        self.turn_creep_mps = float(G("~turn_creep_mps", 0.18))
        self.turn_creep_arm_mps = float(G("~turn_creep_arm_mps", 0.05))

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

        # <=0.0 disables. This checks MEASURED speed (from odometry), not the
        # commanded reference -- max_speed_xy above only clamps what the
        # tracker asks for, it says nothing about what the aircraft is
        # actually doing. It is a backstop against a genuine runaway, not a
        # speed regulator: the adapter downstream now closes a real velocity
        # loop on truth-derived velocity, so ordinary overshoot is its job.
        #
        # It used to zero translation outright the instant measured speed
        # touched the cap. Measured 2026-08-18: that produced bang-bang -- full
        # command, overspeed, zero, coast, full command -- and the achieved
        # speed histogram showed spikes to 0.99 m/s against a 0.30 m/s demand.
        # A hard cutoff also hands the downstream servo a step to zero, which
        # discards its integrator. So taper instead, and only past the cap.
        # Yaw is unaffected -- turning doesn't add speed.
        self.max_measured_speed_xy = float(G("~max_measured_speed_xy", 0.0))
        # Fraction of the cap over which the taper runs to zero: 0.5 means
        # translation is fully withheld at 1.5x the cap and scaled linearly
        # between. <=0 restores the old hard cutoff at the cap.
        self.measured_speed_taper = float(G("~measured_speed_taper", 0.5))

        # ── Pinned-against-something escape ────────────────────────────────
        # Measured 2026-08-18: the aircraft sat at one point for 400 SECONDS
        # while this node commanded 0.45 m/s forward the whole time, and nothing
        # anywhere reacted. FALCON's own stall guard could not help -- it only
        # fires when the aircraft is SHORT of its target viewpoint, and FALCON
        # believed it had already arrived, because a wall inside the depth near
        # clip (0.45 m) is invisible to the map, so the viewpoint it chose was
        # on the far side of a wall it could not see.
        #
        # This detector deliberately depends on nothing but its own two
        # measurements -- "I am asking for motion" and "the aircraft is not
        # moving". It cannot be fooled by a wrong map or a confident planner,
        # and reverse is the only direction that gets this airframe off a wall
        # (there is no lateral authority worth using).
        self.stall_cmd_mps = float(G("~stall_cmd_mps", 0.15))
        self.stall_speed_mps = float(G("~stall_speed_mps", 0.06))
        self.stall_detect_sec = float(G("~stall_detect_sec", 3.0))
        self.escape_sec = float(G("~escape_sec", 2.5))
        self.escape_speed_mps = float(G("~escape_speed_mps", 0.30))
        self.escape_yaw_rate_deg = float(G("~escape_yaw_rate_deg", 35.0))
        self.escape_cooldown_sec = float(G("~escape_cooldown_sec", 4.0))
        # How many escapes may fire without the aircraft regaining sustained
        # motion before the reflex gives up, and how long counts as regained.
        self.escape_give_up_count = int(G("~escape_give_up_count", 4))
        self.escape_progress_sec = float(G("~escape_progress_sec", 5.0))
        self._stall_since = None
        self._escape_until = None
        self._escape_ready_at = None
        self._escape_sign = 1.0
        self._escapes = 0
        self._escapes_since_progress = 0
        self._moving_since = None
        self._pinned_hold = False

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
        self._roll_deg = 0.0
        self._pitch_deg = 0.0
        self._attitude_at = None   # rospy.Time of the last real attitude message
        self._velocity = None      # (vx, vy, vz), world frame

        self._reference = None         # TrajectoryPoint, last received
        self._reference_stamp = None   # rospy.Time of that reference
        self._reference_ready = False  # trajectory_flag == READY

        self.current_demo_mode = None
        self._requested_mode = None
        self._last_request_pub_t = rospy.Time(0)
        self._last_setpoint = None
        self._last_heading_err = None
        self._prev_tick_t = None

        # ── ROS I/O (publishers before subscribers) ──────────────────
        self.cmd_pub = rospy.Publisher(self.cmd_vel_topic, Twist, queue_size=1)
        self.demo_req_pub = rospy.Publisher(self.demo_mode_request_topic, String,
                                            queue_size=1, latch=True)

        rospy.Subscriber(self.odom_topic, Odometry, self._odom_cb, queue_size=10)
        rospy.Subscriber(self.attitude_topic, Vector3, self._attitude_cb, queue_size=5)
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

    def _attitude_cb(self, msg):
        # Vector3.x/y = raw roll/pitch, radians, sign UNVERIFIED -- magnitude
        # only. Deliberately not read off /odom_world -- see module docstring.
        self._roll_deg = math.degrees(msg.x)
        self._pitch_deg = math.degrees(msg.y)
        self._attitude_at = rospy.Time.now()

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

    def _slew_course(self, desired_yaw, dt):
        """Rate-limit the commanded course so the aircraft can actually catch it.

        Args:
            desired_yaw: Course the tracker wants this tick, radians.
            dt: Seconds since the previous tick.

        Returns:
            The commanded course, radians -- ``desired_yaw`` when the limit is
            disabled or the demand is already within reach this tick.
        """
        if self.course_slew_deg_s <= 0.0 or dt <= 0.0:
            self._course_cmd = desired_yaw
            return desired_yaw
        if self._course_cmd is None:
            self._course_cmd = self._yaw
        delta = math.atan2(math.sin(desired_yaw - self._course_cmd),
                           math.cos(desired_yaw - self._course_cmd))
        step = math.radians(self.course_slew_deg_s) * dt
        if delta > step:
            delta = step
        elif delta < -step:
            delta = -step
        self._course_cmd = math.atan2(math.sin(self._course_cmd + delta),
                                      math.cos(self._course_cmd + delta))
        return self._course_cmd

    def _step(self):
        now = rospy.Time.now()
        now_s = now.to_sec()
        dt = 0.0 if self._prev_tick_t is None else max(1e-3, now_s - self._prev_tick_t)
        self._prev_tick_t = now_s

        if self._pose is None or self._yaw is None:
            return   # no odometry yet -- nothing to track from

        # No real attitude yet, or it went stale: assume the worst rather
        # than silently fly as if level.
        if (self._attitude_at is None
                or (now - self._attitude_at).to_sec() > self.state_timeout_s):
            self._publish_cmd(0.0, 0.0, 0.0)
            self.tracker.reset(yaw=self._yaw)
            self.shaper.reset()
            rospy.logwarn_throttle(
                1.0, "falcon_exploration_follower: no fresh attitude on %s; "
                "cutting drive rather than fly blind on tilt", self.attitude_topic)
            return

        # Attitude reflex, ahead of everything else: a contact the depth
        # camera's near clip can't see can tip the aircraft, and past its
        # physical recoverability ceiling it cannot come back. Cut
        # horizontal drive and hold so it settles back level rather than
        # tipping further under continued translation commands. Added
        # 2026-08-17 after a live capsize exposed that this node tracked
        # yaw only and had no attitude awareness at all -- see LESSONS.md.
        tilt = max(abs(self._roll_deg), abs(self._pitch_deg))
        if self._tilt_cut:
            self._tilt_cut = tilt > self.tilt_resume_deg
        else:
            self._tilt_cut = tilt > self.tilt_limit_deg
            if self._tilt_cut:
                # State is dropped ONCE, on the way in. Doing it every tick of a
                # long cut throws away what the tracker relearns in between.
                self.tracker.reset(yaw=self._yaw)
                self.shaper.reset()
                rospy.logwarn(
                    "falcon_exploration_follower: tilt roll=%.0f pitch=%.0f deg; "
                    "cutting drive until it is back under %.0f deg",
                    self._roll_deg, self._pitch_deg, self.tilt_resume_deg)
        if self._tilt_cut:
            self._publish_cmd(0.0, 0.0, 0.0)
            return

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

        # ── Heading: aim the nose along the path, not along FALCON's own yaw ──
        world_speed = math.hypot(setpoint.vx, setpoint.vy)
        heading_err = None
        if self.yaw_mode == "course" and world_speed > self.course_min_speed:
            desired_yaw = self._slew_course(
                math.atan2(setpoint.vy, setpoint.vx), dt)
            heading_err = math.atan2(math.sin(desired_yaw - self._yaw),
                                     math.cos(desired_yaw - self._yaw))
        else:
            # Not steering: the next course starts from where the nose is, not
            # from a stale demand the aircraft never flew.
            self._course_cmd = None

        # World-frame velocity -> body frame (Rooster's cmd_vel convention).
        cos_y, sin_y = math.cos(self._yaw), math.sin(self._yaw)
        body_vx = setpoint.vx * cos_y + setpoint.vy * sin_y
        body_vy = -setpoint.vx * sin_y + setpoint.vy * cos_y

        if heading_err is not None:
            # Turn while going, and only stop turning-in-place when the nose is
            # so far off that forward thrust would take the aircraft away from
            # the reference. Lateral is dropped either way -- it is what causes
            # the roll excursions, and a course-aligned nose does not need it.
            #
            # cos(err) already does the right thing across the whole range: it
            # fades forward speed out as the nose swings away and goes negative
            # past 90 deg, where "forward" genuinely is the wrong direction. The
            # gate only has to catch that sign change, so it sits near 90 rather
            # than at 40 -- measured 2026-08-18, a 40 deg gate stopped the
            # aircraft dead at heading errors of 45-57 deg, which are completely
            # ordinary mid-turn values, and the reference kept moving while it
            # stood still. That was a large part of a tracking error that
            # sawtoothed to 4.2 m.
            if abs(heading_err) > math.radians(self.align_gate_deg):
                body_vx = 0.0
            else:
                body_vx = world_speed * math.cos(heading_err)
                # ...but never let the cos fade brake the aircraft to a stop
                # inside the gate. Measured 2026-08-18 over a full run: heading
                # error is p50 56deg / p75 88deg, so cos() alone spent 17% of
                # the flight at zero speed across 189 separate stops (median
                # 0.7s). On this platform a stop is not a coast -- PX4 is in
                # Position mode, so a released stick is an active brake -- and
                # re-acceleration then has to climb back through the axis dead
                # band. Creeping through the turn keeps the momentum that
                # braking throws away. Only while the plan actually wants
                # motion: a holding reference must still hold.
                if world_speed > self.turn_creep_arm_mps and body_vx < self.turn_creep_mps:
                    body_vx = self.turn_creep_mps
            body_vy = 0.0

        if self.max_measured_speed_xy > 0.0 and self._velocity is not None:
            measured_speed = math.hypot(self._velocity[0], self._velocity[1])
            overspeed = measured_speed / self.max_measured_speed_xy
            if overspeed > 1.0:
                if self.measured_speed_taper > 0.0:
                    scale = 1.0 - (overspeed - 1.0) / self.measured_speed_taper
                    scale = max(0.0, min(1.0, scale))
                else:
                    scale = 0.0
                rospy.logwarn_throttle(
                    2.0, "falcon_exploration_follower: measured speed %.2fm/s over "
                    "%.2fm/s cap, scaling translation to %.0f%%",
                    measured_speed, self.max_measured_speed_xy, 100.0 * scale)
                body_vx *= scale
                body_vy *= scale

        # Pinned? Then nothing above matters: back off and turn away. Placed
        # after all normal shaping precisely so it overrides it.
        escape_yaw_rate = self._update_escape(body_vx, body_vy)
        if escape_yaw_rate is not None:
            body_vx = -self.escape_speed_mps
            body_vy = 0.0
        elif self._pinned_hold:
            # Pinned with the escape budget spent: stop driving into whatever is
            # holding the aircraft. Yaw is deliberately left untouched.
            body_vx = 0.0
            body_vy = 0.0

        # The tracker gives an absolute heading; a yaw RATE command is what
        # cmd_vel needs. Proportional on the same signed error the tracker
        # already computed (reference heading vs. measured), capped at the
        # platform's own ceiling.
        yaw_error = heading_err if heading_err is not None else setpoint.yaw_error_rad
        yaw_rate = saturate(self.yaw_kp * yaw_error,
                            self.tracker.params.limits.max_yaw_rate)
        if escape_yaw_rate is not None:
            yaw_rate = escape_yaw_rate

        # Altitude was deliberately dropped here for a long time, because
        # rooster_command_unit owns the throttle axis and a second publisher
        # would fight it. It is now forwarded instead of discarded: the adapter
        # turns it into bounded nudges of that node's OWN hold setpoint, so the
        # single-owner guarantee holds and the tuned hold loop still flies it.
        # Without this the aircraft maps one horizontal band at a fixed height.
        # Never during an escape -- climbing while pinned is not an escape.
        body_vz = 0.0 if escape_yaw_rate is not None else float(setpoint.vz)

        self._last_heading_err = heading_err
        self._publish_cmd(body_vx, body_vy, yaw_rate, body_vz)

    def _update_escape(self, body_vx, body_vy):
        """Detect being pinned, and drive the escape while one is running.

        Depends on nothing but the command this node just computed and the
        measured speed -- no map, no planner opinion, no attitude. See the
        stall_* parameter comments for why that independence is the point.

        Args:
            body_vx: Forward velocity about to be commanded, m/s.
            body_vy: Lateral velocity about to be commanded, m/s.

        Returns:
            Yaw rate to command (rad/s) while escaping, or ``None`` when the
            aircraft is flying normally and the caller should keep its own
            command.
        """
        now = rospy.Time.now().to_sec()

        if self._escape_until is not None:
            if now < self._escape_until:
                return self._escape_sign * math.radians(self.escape_yaw_rate_deg)
            self._escape_until = None
            self._escape_ready_at = now + self.escape_cooldown_sec
            self._stall_since = None
            rospy.logwarn("falcon_exploration_follower: escape complete, resuming "
                          "(cooldown %.0fs)", self.escape_cooldown_sec)
            return None

        if self._escape_ready_at is not None and now < self._escape_ready_at:
            return None

        asking = math.hypot(body_vx, body_vy) > self.stall_cmd_mps
        if not asking or self._velocity is None:
            self._stall_since = None
            return None

        moving = math.hypot(self._velocity[0], self._velocity[1]) > self.stall_speed_mps
        if moving:
            # Sustained real motion is the only thing that proves an escape
            # achieved something, so it is what re-arms the budget below.
            if self._moving_since is None:
                self._moving_since = now
            elif now - self._moving_since >= self.escape_progress_sec:
                self._escapes_since_progress = 0
                self._pinned_hold = False
            self._stall_since = None
            return None
        self._moving_since = None

        if self._stall_since is None:
            self._stall_since = now
            return None
        if now - self._stall_since < self.stall_detect_sec:
            return None

        # A reflex for being physically pinned, firing on an aircraft that
        # simply cannot reach 0.06 m/s, burns the flight: one escape plus its
        # cooldown is ~9.5 s, and a run measured 38 of them -- over half the
        # window spent reversing and turning instead of exploring, with no
        # escape ever restoring motion. After a few fruitless attempts, stop:
        # commanding normally and letting FALCON replan is strictly better than
        # a manoeuvre that demonstrably is not working, and the budget re-arms
        # as soon as the aircraft actually moves again.
        if self._escapes_since_progress >= self.escape_give_up_count:
            rospy.logwarn_throttle(
                10.0,
                "falcon_exploration_follower: %d escapes without regaining "
                "motion -- suppressing further escapes until the aircraft moves "
                "for %.0fs. Asked %.2f m/s, measured %.2f m/s.",
                self._escapes_since_progress, self.escape_progress_sec,
                math.hypot(body_vx, body_vy),
                math.hypot(self._velocity[0], self._velocity[1]))
            # Giving up on escaping must ALSO stop pushing. Suppressing the
            # escape while still commanding full drive is what produced the
            # lock-up: the aircraft cannot move, the velocity servo's integrator
            # winds to whatever ceiling exists (1000, then 900 once capped), the
            # airframe pitches 20-35 deg and translates even less. Holding
            # translation lets the integrator unwind and the airframe settle,
            # while yaw is left alone so the aircraft can still turn and let
            # FALCON replan from a different heading.
            self._pinned_hold = True
            return None

        # Alternate the turn direction between escapes: if one side is blocked,
        # repeating the same turn walks the aircraft along the same wall.
        self._escapes += 1
        self._escapes_since_progress += 1
        self._escape_sign = 1.0 if self._escapes % 2 else -1.0
        self._escape_until = now + self.escape_sec
        rospy.logwarn(
            "falcon_exploration_follower: PINNED -- asked for %.2f m/s for %.1fs and "
            "measured %.2f m/s. Reversing at %.2f m/s and turning %+.0f deg/s "
            "(escape #%d)",
            math.hypot(body_vx, body_vy), now - self._stall_since,
            math.hypot(self._velocity[0], self._velocity[1]),
            self.escape_speed_mps, self._escape_sign * self.escape_yaw_rate_deg,
            self._escapes)
        return self._escape_sign * math.radians(self.escape_yaw_rate_deg)

    def _publish_cmd(self, vx, vy, wz, vz=0.0):
        """Publish a body-frame velocity command.

        Args:
            vx: Forward velocity, m/s.
            vy: Lateral velocity, m/s.
            wz: Yaw rate, rad/s.
            vz: World-frame vertical velocity, m/s. Passed through UNSHAPED --
                the pulse shaper exists to beat the horizontal axes' dead band,
                and the vertical path downstream is not an axis at all: the
                adapter integrates this into bounded altitude-target nudges and
                lets rooster_command_unit's own hold loop fly them, so it must
                see the real demand rather than a pulse.
        """
        shaped = self.shaper.shape(ControlCommand.velocity(float(vx), float(vy), 0.0, float(wz)))
        m = Twist()
        m.linear.x = float(shaped.x)
        m.linear.y = float(shaped.y)
        m.linear.z = float(vz)
        m.angular.z = float(shaped.yaw_rate)
        self.cmd_pub.publish(m)

    def _hb(self, _evt):
        sp = self._last_setpoint
        # ref_age, not just ref_ready: the flag only reports the LAST message's
        # trajectory_flag, so it reads True forever once traj_server dies. A run
        # on 2026-08-18 held station for 560 s printing ref_ready=True the whole
        # way, which sent the diagnosis into the controller instead of into the
        # planner. The age is the number that would have said so immediately.
        age = None
        if self._reference_stamp is not None:
            age = (rospy.Time.now() - self._reference_stamp).to_sec()
        rospy.loginfo(
            "falcon_exploration_follower hb  demo=%s  ref_ready=%s  ref_age=%s  "
            "pos_err=%s  dz=%s  holding=%s  hdg_err=%s  escapes=%d%s",
            self.current_demo_mode, self._reference_ready,
            "-" if age is None else "%.1fs" % age,
            "-" if sp is None else "%.2fm" % sp.position_error_m,
            # Vertical error alone, signed: the aircraft has been flying a
            # metre under FALCON's own viewpoints (planned z p50 2.04, flown
            # ~1.1) while the altitude demand sat pinned at the band edge
            # asking to DESCEND. Those two cannot both be right, and no metric
            # here could tell them apart.
            "-" if (self._reference is None or self._pose is None)
            else "%+.2fm" % (self._reference.z - self._pose[2]),
            "-" if sp is None else sp.holding,
            "-" if getattr(self, "_last_heading_err", None) is None
            else "%.0fdeg" % math.degrees(self._last_heading_err),
            self._escapes, " ESCAPING" if self._escape_until else "")
        if age is not None and age > 5.0:
            rospy.logwarn(
                "falcon_exploration_follower: no fresh /planning/pos_cmd for "
                "%.0fs -- traj_server is gone or FALCON is in FINISH. The "
                "aircraft is holding station, not tracking.", age)


def main():
    FalconExplorationFollowerNode()
    rospy.spin()


if __name__ == "__main__":
    main()
