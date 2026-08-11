#!/usr/bin/env python3
"""Fly FALCON's B-spline on the SJTU Gazebo drone.

The one node between the planner and the aircraft. It replaces
``cmd_to_vel.py`` from the previous stack, and the difference is not a tidy-up:

* **It reads the curve, not the samples.** The old node subscribed to
  ``/planning/pos_cmd``, the 100 Hz point stream ``traj_server`` produces by
  sampling the B-spline. That stream carries position, velocity, acceleration
  and yaw -- but it arrives 15-35 ms stale, it cannot be evaluated at any other
  instant, and it carries no jerk. Subscribing to ``/planning/bspline`` instead
  gives the whole curve, evaluated on this node's own clock at whatever instant
  the controller asks for, exactly.
* **It knows what it is flying.** The Gazebo plugin closes its own velocity loop
  with a measured 0.18 s of transport delay and a 0.51 s time constant. The old
  node ignored that and paid roughly 0.4 m of standing position error for it at
  a 0.6 m/s cruise. ``core.control.velocity_servo`` inverts it instead.
* **It closes the heading loop.** The old node fed the plan's yaw rate forward
  and corrected the heading with a P term computed from a stale unstamped pose.
  Heading matters more than it sounds here: FALCON chooses yaw to aim the depth
  camera at the frontier it means to observe next, so a heading that has drifted
  is a map built of the wrong wall.

Everything above the ROS boundary lives in ``core``: this file subscribes,
converts, calls one ``update()`` and publishes. That is deliberate -- the
control law is unit-tested against a simulated airframe in
``core/control/velocity_servo/tests/`` and must never need Gazebo to be
exercised.

Runs inside the ROS1 Noetic FALCON container, on **Python 3.8**.
"""
from __future__ import annotations

import math

import numpy as np
import rospy
from geometry_msgs.msg import Twist
from gazebo_msgs.msg import ContactsState
from nav_msgs.msg import Odometry
from std_msgs.msg import Bool, Empty, Float32MultiArray, Int32
from trajectory.msg import Bspline

from sparx_agency.core.control.velocity_servo import (
    AxisPlant, VelocityLimits, VelocityPlant, VelocityServo, VelocityServoParams,
)
from sparx_agency.core.planning.trajectories.bspline import BsplineTrajectory

REPLAN_TRAJECTORY_UNSAFE = 1
"""FALCON found the trajectory it is flying in collision. Hold, do not continue.

From ``exploration_fsm.cpp``. FALCON replans and says nothing else about it, so
a follower that ignores this keeps flying a curve the planner has already
condemned -- straight at whatever it just found.
"""

REPLAN_EXPLORATION_FINISHED = 2
"""The FSM reached FINISH. No further trajectory is coming."""


def _yaw_from_quaternion(orientation):
    # type: (object) -> float
    """Heading about world +z, radians, from a ROS quaternion."""
    x, y, z, w = orientation.x, orientation.y, orientation.z, orientation.w
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


class BsplineFollowerNode(object):
    """Subscribe the plan and the state, call the servo, publish a twist."""

    def __init__(self):
        # type: () -> None
        self._rate_hz = float(rospy.get_param("~ctrl_rate_hz", 50.0))
        self._odom_timeout_s = float(rospy.get_param("~odom_timeout_s", 0.5))
        self._takeoff_altitude_m = float(rospy.get_param("~takeoff_altitude_m", 1.0))
        # 1.25 turns; 0 disables. See _survey for why this must exist.
        self._survey_revs = float(rospy.get_param("~survey_revs", 1.25))
        self._survey_remaining_rad = 2.0 * math.pi * self._survey_revs
        self._survey_yaw_rate = float(rospy.get_param("~survey_yaw_rate", 0.5))
        self._survey_last_yaw = None
        # Re-survey when FALCON falls silent for this long after it has been
        # planning -- the signature of an exploration_node respawn (its LKH
        # solver segfaults, roslaunch restarts it, and it comes back with an
        # empty map at wherever the aircraft now is). A fresh 360 deg scan is
        # what lets it find frontiers again instead of sitting in "No path".
        self._resurvey_after_s = float(rospy.get_param("~resurvey_after_s", 12.0))
        # Below the re-survey window but above a normal replan gap: hold instead
        # of carrying the last curve on into an obstacle while FALCON is down (its
        # in-process LKH solver corrupts the heap and roslaunch is mid-respawn).
        # A velocity-commanded aircraft flown a stale plan blind is exactly what
        # tips into the one wall the depth camera never saw. 6 s clears FALCON's
        # ~3 s replan cadence so a normal gap is not mistaken for a crash.
        self._hold_silent_s = float(rospy.get_param("~hold_silent_s", 6.0))
        # Exploration stall: airborne and not moving while FALCON churns
        # plan-fails on a frontier whose only viewpoint is the current cell -- it
        # selects "here", cannot plan a zero-length trajectory to it, and loops
        # ("next_pos ... same as current pos and yaw") forever. A crash used to
        # break this by chance (the respawn wipes the map); a fresh 360 deg scan
        # does it deliberately, clearing or re-routing the stuck frontier.
        self._stall_window_s = float(rospy.get_param("~stall_window_s", 15.0))
        # Net displacement, not path length: a stuck aircraft still jitters ~0.5 m
        # as FALCON's degenerate plans nudge it, so the bar is a real leg, not a
        # twitch. 1.5 m in 15 s is well below any genuine exploration move.
        self._stall_move_m = float(rospy.get_param("~stall_move_m", 1.5))
        self._stall_track = []          # type: list  # (t_s, x, y, z)
        self._resurveys = 0
        self._last_bspline_at = None    # type: object
        self._servo = VelocityServo(self._params())

        # ── wedge/contact reflex ─────────────────────────────────────────
        # A velocity-commanded aircraft pinned against a wall keeps being told
        # to drive into it and never moves -- the map may not carry the wall
        # (the depth camera goes blind inside its ~0.95 m near clip), so nothing
        # in the plan chain notices. Two triggers: ground-truth contact from the
        # Gazebo bumper, and an inferred wedge (commanding motion, net-zero
        # travel). Recovery backs out far enough to bring the wall back inside
        # the near clip, so FALCON re-maps it and plans around it next.
        self._wedge_window_s = float(rospy.get_param("~wedge_window_s", 3.0))
        self._wedge_move_m = float(rospy.get_param("~wedge_move_m", 0.2))
        self._wedge_cmd_speed = float(rospy.get_param("~wedge_cmd_speed", 0.08))
        self._contact_hold_s = float(rospy.get_param("~contact_hold_s", 0.4))
        self._retreat_time_s = float(rospy.get_param("~retreat_time_s", 5.0))
        self._retreat_speed = float(rospy.get_param("~retreat_speed", 0.35))
        self._retreat_clear_m = float(rospy.get_param("~retreat_clear_m", 1.3))
        # Attitude reflex: cut horizontal drive if roll or pitch crosses this, a
        # margin below the plugin's ~35 deg clamp past which the model thrusts
        # sideways and cannot recover. A near-clip contact can pitch the aircraft
        # up as it drives into the unseen obstacle; stopping the drive lets it
        # fall back level before it tips over.
        self._tilt_limit_rad = math.radians(
            float(rospy.get_param("~tilt_limit_deg", 22.0)))
        self._track = []                # type: list  # (t_s, x, y, z) history
        self._contact_seen_at = None    # type: object
        self._retreat_until = None      # type: object
        self._retreat_from = None       # type: object  # world pos at retreat start
        self._retreat_dir_world = None  # type: object
        self._retreats = 0

        self._odom = None               # type: object
        self._odom_at = None            # type: object
        self._airborne = False
        self._stopped = False
        self._finished = False

        self._cmd = rospy.Publisher(
            rospy.get_param("~cmd_topic", "/simple_drone/cmd_vel"), Twist, queue_size=1)
        self._takeoff = rospy.Publisher("/simple_drone/takeoff", Empty, queue_size=1)
        self._diagnostics = rospy.Publisher("~tracking", Float32MultiArray, queue_size=10)
        self._flying = rospy.Publisher("~following", Bool, queue_size=1, latch=True)

        rospy.Subscriber(rospy.get_param("~odom_topic", "/simple_drone/odom"),
                         Odometry, self._on_odom, queue_size=4)
        rospy.Subscriber("/planning/bspline", Bspline, self._on_bspline, queue_size=4)
        rospy.Subscriber("/planning/replan", Int32, self._on_replan, queue_size=10)
        rospy.Subscriber(rospy.get_param("~contact_topic", "/simple_drone/bumper_states"),
                         ContactsState, self._on_bumper, queue_size=4)

        self._last_tick = None          # type: object
        rospy.Timer(rospy.Duration(1.0 / self._rate_hz), self._tick)
        rospy.loginfo("[follower] %.0f Hz, plant tau=%.3f s delay=%.3f s",
                      self._rate_hz,
                      self._servo.params.plant.horizontal.time_constant_s,
                      self._servo.params.plant.horizontal.delay_s)

    def _params(self):
        # type: () -> VelocityServoParams
        """Build the servo tuning from the launch parameters.

        The plant numbers are **measured**, not guessed -- see
        ``robots/SJTU/config/airframe.yaml`` for how, and re-measure them if the
        simulator's real-time factor or the plugin's gains change. A wrong time
        constant makes the lead term push the wrong amount and looks exactly
        like a mistuned position gain.
        """
        plant = VelocityPlant(
            horizontal=AxisPlant(
                dc_gain=float(rospy.get_param("~plant_xy_gain", 1.0)),
                time_constant_s=float(rospy.get_param("~plant_xy_tau", 0.51)),
                delay_s=float(rospy.get_param("~plant_xy_delay", 0.18))),
            vertical=AxisPlant(
                dc_gain=float(rospy.get_param("~plant_z_gain", 1.0)),
                time_constant_s=float(rospy.get_param("~plant_z_tau", 0.41)),
                delay_s=float(rospy.get_param("~plant_z_delay", 0.04))),
            yaw=AxisPlant(
                dc_gain=float(rospy.get_param("~plant_yaw_gain", 1.0)),
                time_constant_s=float(rospy.get_param("~plant_yaw_tau", 0.48)),
                delay_s=float(rospy.get_param("~plant_yaw_delay", 0.06))))
        limits = VelocityLimits(
            max_speed_xy=float(rospy.get_param("~max_speed_xy", 1.2)),
            max_speed_up=float(rospy.get_param("~max_speed_up", 0.8)),
            max_speed_down=float(rospy.get_param("~max_speed_down", 0.6)),
            max_accel_xy=float(rospy.get_param("~max_accel_xy", 2.0)),
            max_accel_z=float(rospy.get_param("~max_accel_z", 2.0)),
            max_yaw_rate=float(rospy.get_param("~max_yaw_rate", 1.4)),
            max_yaw_accel=float(rospy.get_param("~max_yaw_accel", 3.0)))
        return VelocityServoParams(
            plant=plant, limits=limits,
            yaw_gain=float(rospy.get_param("~yaw_gain", 1.5)),
            use_feedforward_lead=bool(rospy.get_param("~use_feedforward_lead", True)),
            predict_reference=bool(rospy.get_param("~predict_reference", True)),
            max_overspeed=float(rospy.get_param("~max_overspeed", 0.25)),
            max_catchup_speed=float(rospy.get_param("~max_catchup_speed", 0.15)))

    # ── inputs ───────────────────────────────────────────────────────────

    def _on_odom(self, msg):
        # type: (Odometry) -> None
        """Latch the state. This is the only feedback the loop has."""
        self._odom = msg
        self._odom_at = rospy.Time.now()

    def _on_bspline(self, msg):
        # type: (Bspline) -> None
        """Rebuild FALCON's curve and queue it.

        Rebuilt with FALCON's own construction rules so this node and
        ``traj_server`` evaluate the same polynomial. A disagreement between
        them would appear as a tracking error with no visible cause.

        The order is asserted rather than trusted: the publisher hardcodes 3
        while the real degree comes from a parameter, so a mismatch is a silent
        wrong-curve failure and this repo raises rather than guessing.
        """
        self._last_bspline_at = rospy.Time.now()
        if msg.order != 3:
            raise ValueError(
                "FALCON published a degree-%d position spline; this follower "
                "rebuilds degree 3, as traj_server does. Refusing to fly a "
                "curve it would evaluate incorrectly." % (msg.order,))
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
            # condemning what it was flying, so the arrival of a curve is its
            # statement that it has found a way out.
            self._stopped = False

    def _on_replan(self, msg):
        # type: (Int32) -> None
        """React to FALCON's own verdict on the trajectory it is flying.

        Treated as a control input rather than telemetry. ``1`` means
        ``safetyCallback`` found the *executing* trajectory in collision, and a
        follower that keeps flying it carries the aircraft's momentum into the
        obstacle FALCON just found.
        """
        if msg.data == REPLAN_TRAJECTORY_UNSAFE:
            self._stopped = True
            rospy.logwarn("[follower] FALCON condemned the live trajectory; holding")
        elif msg.data == REPLAN_EXPLORATION_FINISHED:
            self._finished = True
            rospy.loginfo("[follower] exploration finished; holding station")

    # ── the loop ─────────────────────────────────────────────────────────

    def _tick(self, _event):
        # type: (object) -> None
        """One control tick. Never raises into the timer thread."""
        now = rospy.Time.now()
        dt = self._elapsed(now)
        if dt is None:
            return
        if not self._state_is_fresh(now):
            # No state means no loop. Publishing the previous command would fly
            # the aircraft blind; publishing zero at least stops it.
            self._cmd.publish(Twist())
            self._servo.reset()
            return

        position, velocity, yaw = self._state()
        if not self._airborne:
            self._climb(position)
            return

        # Attitude reflex, ahead of everything below (even the retreat, which
        # drives): a near-clip contact can tip the aircraft, and past the plugin's
        # ~35 deg clamp it cannot recover. If roll or pitch crosses the margin,
        # cut horizontal drive and hold so it settles back level rather than over.
        roll, pitch = self._roll_pitch()
        if abs(roll) > self._tilt_limit_rad or abs(pitch) > self._tilt_limit_rad:
            self._cmd.publish(Twist())
            self._servo.reset()
            self._retreat_until = None
            rospy.logwarn_throttle(
                1.0, "[follower] tilt roll=%.0f pitch=%.0f deg; cutting drive to "
                "level before it capsizes", math.degrees(roll), math.degrees(pitch))
            return

        # Re-survey only after a crash: a respawned exploration_node comes back
        # with an EMPTY map and never re-triggers itself, and a fresh 360 deg
        # scan is what gives its frontier finder something to fire on. Position
        # stalls at an unreachable frontier are NOT handled here any more -- FALCON
        # itself now shadows those viewpoints (the dead-end guard in
        # exploration_manager), and yawing in place on every stall only thrashed
        # against that 20 s block and re-scanned a view that never changes.
        silent_s = self._falcon_silent_s(now)
        if (self._survey_remaining_rad <= 0.0 and self._retreat_until is None
                and silent_s > self._resurvey_after_s):
            self._begin_resurvey(now, "silent %.0fs (respawn?)" % silent_s)

        if self._survey_remaining_rad > 0.0:
            self._survey(yaw)
            return

        # Contact recovery owns the aircraft while it is backing out of a wall:
        # the plan that put it there has been dropped, so nothing else may drive.
        if self._retreat_until is not None:
            self._retreat(now, position, yaw)
            return

        # FALCON briefly silent (crashed mid-plan, before the re-survey window):
        # hold station rather than carry the stale curve on blind into the wall
        # it could not see. Retreat above takes priority -- backing out of a real
        # contact must not be interrupted by a hold.
        if silent_s > self._hold_silent_s:
            self._cmd.publish(Twist())
            self._servo.reset()
            return

        follow = not (self._stopped or self._finished)
        command = self._servo.update(position, velocity, yaw, dt, now.to_sec(),
                                     follow=follow)

        # Back out on a real contact or an inferred wedge. A wall inside the
        # depth near clip (~0.95 m) is invisible to FALCON, which then plans
        # straight through it, so the follower is the only thing that can notice.
        if follow and not command.holding and self._should_retreat(now, position, command):
            self._begin_retreat(position, command)
            return

        self._publish(command)

    def _elapsed(self, now):
        # type: (object) -> object
        """Seconds since the previous tick, or None on the first one.

        Measured rather than assumed. The timer's nominal period is not what
        actually elapses once the simulator's real-time factor drops below one,
        and a control law integrating a fictitious dt learns a fictitious bias.
        """
        if self._last_tick is None:
            self._last_tick = now
            return None
        dt = (now - self._last_tick).to_sec()
        self._last_tick = now
        if dt <= 0.0:
            return None
        return dt

    def _state_is_fresh(self, now):
        # type: (object) -> bool
        """Whether the odometry is recent enough to close a loop on."""
        if self._odom is None or self._odom_at is None:
            return False
        return (now - self._odom_at).to_sec() <= self._odom_timeout_s

    def _state(self):
        # type: () -> tuple
        """Measured world position, world velocity and heading.

        The twist on ``/simple_drone/odom`` is expressed in the child frame, so
        it is rotated into the world here. The plugin's ``/simple_drone/gt_vel``
        looks like a shortcut and is not: it applies the rotation the wrong way
        round, which puts a yaw-dependent axis swap into any loop that uses it.
        """
        pose = self._odom.pose.pose
        twist = self._odom.twist.twist
        yaw = _yaw_from_quaternion(pose.orientation)
        cos, sin = math.cos(yaw), math.sin(yaw)
        world = np.array([cos * twist.linear.x - sin * twist.linear.y,
                          sin * twist.linear.x + cos * twist.linear.y,
                          twist.linear.z], dtype=float)
        position = np.array([pose.position.x, pose.position.y, pose.position.z],
                            dtype=float)
        return position, world, yaw

    def _roll_pitch(self):
        # type: () -> tuple
        """Roll and pitch, radians, from the latched odometry orientation."""
        o = self._odom.pose.pose.orientation
        sinr = 2.0 * (o.w * o.x + o.y * o.z)
        cosr = 1.0 - 2.0 * (o.x * o.x + o.y * o.y)
        roll = math.atan2(sinr, cosr)
        sinp = max(-1.0, min(1.0, 2.0 * (o.w * o.y - o.z * o.x)))
        pitch = math.asin(sinp)
        return roll, pitch

    def _survey(self, yaw):
        # type: (float) -> None
        """One full turn on the spot before FALCON is allowed to matter.

        The depth camera sees a 75 degree wedge. An aircraft that has only
        ever pointed one way hands FALCON a map that is one wedge of free
        space; its coverage tour then picks a viewpoint far outside it,
        A* cannot route through the unknown, and the FSM sits in PLAN_TRAJ
        reporting "No path to next viewpoint" forever -- measured on the
        Pegasus deployment before its survey turn existed, and the previous
        SJTU stack carried the same scan (mapping_scan_revs) for the same
        reason. 1.25 revolutions rather than 1.0 so the wedge edges overlap.

        Yaw progress is integrated from the measured heading, wrap-aware,
        rather than assumed from the commanded rate: the plugin's yaw loop
        is not instant and a timed turn undershoots.
        """
        if self._survey_last_yaw is not None:
            step = math.atan2(math.sin(yaw - self._survey_last_yaw),
                              math.cos(yaw - self._survey_last_yaw))
            self._survey_remaining_rad -= abs(step)
        self._survey_last_yaw = yaw
        command = Twist()
        if self._survey_remaining_rad <= 0.0:
            self._cmd.publish(command)          # stop the turn cleanly
            # Start the stall clock fresh: the scan just built new map, so give
            # FALCON a full window to act on it before calling it stuck again --
            # otherwise the in-place turn itself reads as a stall and loops.
            self._stall_track = []
            rospy.loginfo("[bspline_follower] survey turn complete -- "
                          "following FALCON")
            return
        command.angular.z = self._survey_yaw_rate
        self._cmd.publish(command)

    def _falcon_silent_s(self, now):
        # type: (object) -> float
        """Seconds since FALCON last published a plan; 0 before the first one.

        Zero until the first curve arrives so the startup climb and survey own
        that phase -- there is no plan to be stale yet, and a large value there
        would trigger a hold or a re-survey before exploration has even begun.
        """
        if self._last_bspline_at is None:
            return 0.0
        return (now - self._last_bspline_at).to_sec()

    def _track_and_check_stall(self, now, position):
        # type: (object, np.ndarray) -> bool
        """Whether the aircraft has gone nowhere for the whole stall window.

        Net displacement start-to-now, so a drone yawing on the spot at a
        viewpoint it cannot leave reads as stalled while one genuinely creeping
        along a corridor does not. Returns False until the window is full.
        """
        now_s = now.to_sec()
        self._stall_track.append((now_s, float(position[0]), float(position[1]),
                                  float(position[2])))
        cutoff = now_s - self._stall_window_s
        while len(self._stall_track) > 1 and self._stall_track[0][0] < cutoff:
            self._stall_track.pop(0)
        if self._stall_track[0][0] > cutoff:
            return False
        t0, x0, y0, z0 = self._stall_track[0]
        moved = math.sqrt((position[0] - x0) ** 2 + (position[1] - y0) ** 2
                          + (position[2] - z0) ** 2)
        return moved < self._stall_move_m

    def _begin_resurvey(self, now, reason):
        # type: (object, str) -> None
        """Restart the survey turn to rebuild the map and break a dead end.

        ``_last_bspline_at`` is pushed to ``now`` and the stall history cleared
        so the turn is armed once per dead end rather than every tick; the next
        real plan and the motion it produces reset both when exploration resumes.
        """
        self._survey_remaining_rad = 2.0 * math.pi * self._survey_revs
        self._survey_last_yaw = None
        self._last_bspline_at = now
        self._stall_track = []
        self._resurveys += 1
        rospy.logwarn("[follower] FALCON %s (recovery #%d); re-surveying to "
                      "rebuild its map", reason, self._resurveys)

    def _climb(self, position):
        # type: (np.ndarray) -> None
        """Get off the ground before handing over to the servo.

        The servo's integrators are reset at the handover: a bias learned while
        the aircraft was on its skids belongs to a different flight regime, and
        carrying it across puts a step into the first command of the new one.
        """
        if position[2] >= self._takeoff_altitude_m:
            self._airborne = True
            self._servo.reset()
            self._flying.publish(Bool(data=True))
            rospy.loginfo("[follower] airborne at %.2f m; following the plan",
                          position[2])
            return
        self._takeoff.publish(Empty())

    def _on_bumper(self, msg):
        # type: (ContactsState) -> None
        """Latch the time of the most recent ground-truth collision."""
        if len(msg.states) > 0:
            self._contact_seen_at = rospy.Time.now()

    def _should_retreat(self, now, position, command):
        # type: (object, np.ndarray, object) -> bool
        """Whether to back out: a fresh contact, or an inferred wedge."""
        return self._contact_recent(now) or self._is_wedged(now, position, command)

    def _contact_recent(self, now):
        # type: (object) -> bool
        """Whether the bumper reported a contact within the hold window."""
        if self._contact_seen_at is None:
            return False
        return (now - self._contact_seen_at).to_sec() <= self._contact_hold_s

    def _is_wedged(self, now, position, command):
        # type: (object, np.ndarray, object) -> bool
        """Whether the aircraft is pinned: commanded to move, going nowhere.

        Net travel over the window -- start point to now -- not the maximum
        excursion within it, because an aircraft grinding on a wall jitters back
        and forth by more than the threshold while making no actual progress, and
        a max-excursion test reads that jitter as movement and never fires.
        """
        wanted = math.hypot(command.world_vx, command.world_vy)
        now_s = now.to_sec()
        self._track.append((now_s, float(position[0]), float(position[1]),
                            float(position[2])))
        cutoff = now_s - self._wedge_window_s
        while len(self._track) > 1 and self._track[0][0] < cutoff:
            self._track.pop(0)
        if wanted < self._wedge_cmd_speed:
            return False
        if self._track[0][0] > cutoff:
            return False                # window not yet full
        t0, x0, y0, z0 = self._track[0]
        net = math.sqrt((position[0] - x0) ** 2 + (position[1] - y0) ** 2
                        + (position[2] - z0) ** 2)
        return net < self._wedge_move_m

    def _begin_retreat(self, position, command):
        # type: (np.ndarray, object) -> None
        """Start backing out of a wall, opposite the direction it drove in.

        The condemned curve is dropped (``reset``) so following resumes only when
        FALCON publishes a fresh plan. The retreat runs until the aircraft has
        cleared ``retreat_clear_m`` -- enough to bring the wall back outside the
        depth near clip, so FALCON re-maps it and plans around it rather than
        straight back into it.
        """
        self._retreats += 1
        speed = math.hypot(command.world_vx, command.world_vy)
        if speed > 1e-3:
            self._retreat_dir_world = (-command.world_vx / speed,
                                       -command.world_vy / speed)
        else:
            self._retreat_dir_world = None      # fall back to -body-x
        self._retreat_from = np.array(position, dtype=float)
        self._retreat_until = rospy.Time.now() + rospy.Duration(self._retreat_time_s)
        self._contact_seen_at = None
        self._servo.reset()
        self._track = []
        self._cmd.publish(Twist())
        rospy.logwarn("[follower] contact/wedge (retreat #%d); backing out %.1f m",
                      self._retreats, self._retreat_clear_m)

    def _retreat(self, now, position, yaw):
        # type: (object, np.ndarray, float) -> None
        """Fly back out of the wall until clear of it, then hold for a fresh plan.

        Ends on distance (cleared the near clip) or a time cap, whichever first --
        the cap guards against a retreat that is itself blocked, e.g. a corner.
        """
        cleared = float(np.linalg.norm(position - self._retreat_from)) >= self._retreat_clear_m
        if cleared or now >= self._retreat_until:
            self._retreat_until = None
            self._retreat_dir_world = None
            self._retreat_from = None
            self._cmd.publish(Twist())
            rospy.loginfo("[follower] retreat done (%s); holding for a fresh plan",
                          "cleared" if cleared else "timed out")
            return
        twist = Twist()
        if self._retreat_dir_world is None:
            twist.linear.x = -self._retreat_speed         # straight back
        else:
            cos, sin = math.cos(yaw), math.sin(yaw)
            wx, wy = self._retreat_dir_world
            twist.linear.x = self._retreat_speed * (cos * wx + sin * wy)
            twist.linear.y = self._retreat_speed * (-sin * wx + cos * wy)
        self._cmd.publish(twist)

    def _publish(self, command):
        # type: (object) -> None
        """Put the twist on the wire and report how well it is tracking."""
        twist = Twist()
        twist.linear.x = command.vx
        twist.linear.y = command.vy
        twist.linear.z = command.vz
        twist.angular.z = command.yaw_rate
        self._cmd.publish(twist)

        # Flat array rather than a custom message so nothing has to be built
        # into the container to record it. Order is fixed and documented in the
        # package README; `postmortem.py` reads it positionally.
        self._diagnostics.publish(Float32MultiArray(data=[
            command.position_error_m, command.along_track_lag_m,
            command.cross_track_error_m, command.yaw_error_rad,
            command.world_vx, command.world_vy, command.world_vz,
            command.yaw_rate, float(command.trajectory_id),
            command.reference_time_s,
            1.0 if command.saturated else 0.0,
            1.0 if command.holding else 0.0,
            1.0 if command.past_end else 0.0]))


def main():
    # type: () -> None
    """Entry point."""
    rospy.init_node("bspline_follower")
    BsplineFollowerNode()
    rospy.spin()


if __name__ == "__main__":
    main()
