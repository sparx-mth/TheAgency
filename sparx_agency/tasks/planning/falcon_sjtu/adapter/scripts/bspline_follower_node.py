#!/usr/bin/env python
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
        self._survey_remaining_rad = (2.0 * math.pi
                                      * float(rospy.get_param("~survey_revs", 1.25)))
        self._survey_yaw_rate = float(rospy.get_param("~survey_yaw_rate", 0.5))
        self._survey_last_yaw = None
        self._servo = VelocityServo(self._params())

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
        if self._survey_remaining_rad > 0.0:
            self._survey(yaw)
            return

        follow = not (self._stopped or self._finished)
        command = self._servo.update(position, velocity, yaw, dt, now.to_sec(),
                                     follow=follow)
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
            rospy.loginfo("[bspline_follower] survey turn complete -- "
                          "following FALCON")
            return
        command.angular.z = self._survey_yaw_rate
        self._cmd.publish(command)

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
