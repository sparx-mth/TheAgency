#!/usr/bin/env python3
"""Fly a known B-spline on the real Gazebo drone and measure how well it tracks.

The point of this rig is that **the controller does not know it is a rig**. It
is handed a ``BsplineTrajectory`` built the way FALCON builds one -- cubic,
uniform knots, a separate degree-3 yaw curve -- and flies it through exactly the
``VelocityServo`` the ROS1 follower uses. So a number measured here is a
statement about the control layer on the real airframe, not about a simulation
of it, and it can be taken before FALCON is wired up at all.

It exists because every tracking number in ``core/control/README.md`` was
measured against ``LaggingAirframe``, a first-order-plus-delay model. That model
is seeded with measurements off this aircraft and ``test_plant.py`` pins the two
together, but a model that agrees with its own measurements is still not the
aircraft. This closes that gap.

Run the same route twice, with and without the inverse-plant lead, and the
difference is the design's central claim measured on hardware-in-the-loop rather
than argued.

Usage::

    ros2 run ... track_bspline.py --route corner --speed 0.6 --compare

Frames: world ENU, body FLU (REP-103), matching ``core``.
"""
from __future__ import annotations

import argparse
import math
import sys

import numpy as np
import rclpy
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import Empty

from sparx_agency.core.control.velocity_servo import (
    AxisPlant, VelocityLimits, VelocityPlant, VelocityServo, VelocityServoParams,
)
from sparx_agency.core.planning.trajectories.bspline import (
    BsplineTrajectory, NonUniformBspline,
)

SENSOR_QOS = QoSProfile(history=HistoryPolicy.KEEP_LAST, depth=10,
                        reliability=ReliabilityPolicy.BEST_EFFORT)

MEASURED_PLANT = VelocityPlant(
    horizontal=AxisPlant(dc_gain=0.998, time_constant_s=0.510, delay_s=0.181),
    vertical=AxisPlant(dc_gain=1.024, time_constant_s=0.409, delay_s=0.033),
    yaw=AxisPlant(dc_gain=0.999, time_constant_s=0.477, delay_s=0.055))
"""Step-response measurements off this aircraft. See robots/SJTU/config/airframe.yaml."""


def _spline(points, headings, knot_dt, start_time_s, traj_id=1):
    # type: (object, object, float, float, int) -> BsplineTrajectory
    """Build a FALCON-shaped trajectory from control points and headings."""
    return BsplineTrajectory(
        NonUniformBspline(np.asarray(points, dtype=float).reshape(-1, 3), 3, knot_dt),
        NonUniformBspline(np.asarray(headings, dtype=float).reshape(-1, 1), 3, knot_dt),
        start_time_s=start_time_s, traj_id=traj_id)


def build_route(name, speed, altitude, origin, start_time_s, knot_dt=0.5):
    # type: (str, float, float, object, float, float) -> BsplineTrajectory
    """One of the named routes, anchored at the aircraft's current position.

    **Every route is a closed loop**, ending where it began. That is not
    aesthetic: the rig's whole value is that the same route can be flown twice
    and the two runs compared, and an open route leaves the aircraft a route's
    length downrange, so the second run starts somewhere else and the third
    starts somewhere else again. Flown open in the playground world this walked
    the aircraft eleven metres from spawn and pinned it against the scenery, at
    which point the rig faithfully measured a "tracking error" that was really
    an aircraft with a wall in front of it (+0.007 m of travel forward, -1.99 m
    backward, on the same commanded speed).

    That failure has a signature worth recognising anywhere, not just here: the
    along-track lag grows **linearly** while the cross-track error stays near
    zero and the command sits saturated. A mistuned loop does not do that; a
    blocked airframe does.
    """
    step = speed * knot_dt
    base = np.asarray(origin, dtype=float).reshape(3)
    points, headings = [], []

    if name == "circle":
        # **The primary benchmark.** Constant speed, constant curvature and a
        # constant yaw rate, so it is dynamically feasible everywhere and
        # closes on itself -- the aircraft can fly it indefinitely and two runs
        # start from the same place.
        #
        # It is also the only route here that isolates what is being measured.
        # A route with corners asks the airframe for a yaw step it physically
        # cannot make, and then most of the "tracking error" is the aircraft
        # failing to be somewhere no aircraft could be. A circle asks for
        # nothing infeasible, so what is left over IS the controller.
        # The radius is DERIVED from the requested speed, not chosen. A cubic
        # B-spline runs at roughly one control-point spacing per knot interval,
        # so spacing the points `speed * knot_dt` apart around the circle is
        # what makes `--speed` mean what it says. Picking a radius instead
        # silently flies a different speed than the one reported, which would
        # make every number here incomparable with the next run.
        count = 34
        spans = count - 4
        radius = spans * step / (2.0 * math.pi)
        for i in range(count):
            angle = 2.0 * math.pi * i / spans
            points.append((radius * math.sin(angle), radius * (1.0 - math.cos(angle)), 0.0))
            headings.append(angle)
    elif name == "straight":
        # The null case: constant heading, constant speed, zero plan
        # acceleration. The lead term has nothing to do here, so the two
        # configurations MUST come out close -- which is what makes this a
        # control on the experiment rather than a weak test. It does not close
        # on itself; it is short enough not to matter.
        for i in range(14):
            points.append((i * step, 0.0, 0.0))
            headings.append(0.0)
    elif name == "corner":
        # A closed square: four 90-degree turns, the nose following the route.
        side = 6
        for leg, (dx, dy, heading) in enumerate(
                ((1, 0, 0.0), (0, 1, math.pi / 2.0), (-1, 0, math.pi), (0, -1, -math.pi / 2.0))):
            for i in range(side):
                x = (side * step if leg in (1, 2) else 0.0) + (dx * i * step if dx else 0.0)
                y = (side * step if leg in (2, 3) else 0.0) + (dy * i * step if dy else 0.0)
                if leg == 2:
                    x = side * step - i * step
                if leg == 3:
                    y = side * step - i * step
                points.append((x, y, 0.0))
                headings.append(heading)
    elif name == "slalom":
        # Continuous curvature, out and back: the plan is accelerating almost
        # everywhere, so this is where an inverse-plant lead is worth the most.
        amplitude, wavelength = 1.0, 5.0
        for direction in (1.0, -1.0):
            for i in range(10):
                x = (i if direction > 0 else 9 - i) * step
                y = amplitude * math.sin(2.0 * math.pi * x / wavelength)
                slope = (amplitude * 2.0 * math.pi / wavelength) * math.cos(
                    2.0 * math.pi * x / wavelength)
                points.append((x, y, 0.0))
                headings.append(math.atan2(direction * slope, direction))
    elif name == "climb":
        # Vertical motion only: exercises the vertical lead, which has its own
        # much shorter time constant, without moving the aircraft laterally.
        for i in range(18):
            points.append((0.0, 0.0, 0.5 * math.sin(i * 0.5)))
            headings.append(0.0)
    else:
        raise ValueError("unknown route %r" % (name,))

    world = [(base[0] + p[0], base[1] + p[1], altitude + p[2]) for p in points]
    # Unwrapped so the yaw spline interpolates through the turns rather than
    # taking the short way round a 2*pi jump it never actually flies.
    return _spline(world, np.unwrap(np.asarray(headings, dtype=float)),
                   knot_dt, start_time_s)


def _looks_blocked(alongs, speeds, stalled_speed=0.05, fraction=0.3):
    # type: (object, object, float, float) -> bool
    """Whether the aircraft was held against something rather than tracking badly.

    A tracking error and an obstruction produce very different records and it is
    worth refusing to average them together. When the airframe is merely
    behind, its speed stays near the plan's and the lag oscillates about some
    value. When it is **blocked**, the speed collapses to nothing while the
    commanded velocity stays saturated, so the lag grows without bound and
    every summary statistic becomes a statement about how long the run was.

    The test is deliberately crude -- the aircraft spent a large part of the run
    essentially stationary, and finished much further behind than it started --
    because the purpose is to make the report say "blocked" instead of quietly
    printing a number that means nothing.
    """
    if len(alongs) < 20:
        return False
    stalled = sum(1 for s in speeds if s < stalled_speed)
    if stalled < fraction * len(speeds):
        return False
    return float(alongs[-1]) > float(alongs[0]) + 1.0


class TrackingRig(Node):
    """Take off, fly a route through the servo, and record the error."""

    def __init__(self, args):
        # type: (argparse.Namespace) -> None
        super().__init__("tracking_rig")
        self.args = args
        self._cmd = self.create_publisher(Twist, "/simple_drone/cmd_vel", 1)
        self._takeoff = self.create_publisher(Empty, "/simple_drone/takeoff", 1)
        self.create_subscription(Odometry, "/simple_drone/odom", self._on_odom,
                                 SENSOR_QOS)
        self._odom = None

    def _on_odom(self, msg):
        # type: (Odometry) -> None
        self._odom = msg

    # ── state ────────────────────────────────────────────────────────────

    def now(self):
        # type: () -> float
        return self.get_clock().now().nanoseconds * 1e-9

    def state(self):
        # type: () -> tuple
        """Measured world position, world velocity and heading.

        The odom twist is in the child frame, so it is rotated here. The
        plugin's own ``gt_vel`` applies that rotation the wrong way round and
        must not be used as feedback.
        """
        pose = self._odom.pose.pose
        twist = self._odom.twist.twist
        q = pose.orientation
        yaw = math.atan2(2.0 * (q.w * q.z + q.x * q.y),
                         1.0 - 2.0 * (q.y * q.y + q.z * q.z))
        cos, sin = math.cos(yaw), math.sin(yaw)
        return (np.array([pose.position.x, pose.position.y, pose.position.z]),
                np.array([cos * twist.linear.x - sin * twist.linear.y,
                          sin * twist.linear.x + cos * twist.linear.y,
                          twist.linear.z]),
                yaw)

    def attitude(self):
        # type: () -> tuple
        """Measured ``(roll, pitch, yaw)`` in radians."""
        q = self._odom.pose.pose.orientation
        roll = math.atan2(2.0 * (q.w * q.x + q.y * q.z),
                          1.0 - 2.0 * (q.x * q.x + q.y * q.y))
        pitch = math.asin(max(-1.0, min(1.0, 2.0 * (q.w * q.y - q.z * q.x))))
        yaw = math.atan2(2.0 * (q.w * q.z + q.x * q.y),
                         1.0 - 2.0 * (q.y * q.y + q.z * q.z))
        return roll, pitch, yaw

    def upright(self, limit_rad=math.radians(35.0)):
        # type: (float) -> bool
        """Whether the airframe is the right way up and can still fly.

        **Check this before believing any number this rig produces.** The Gazebo
        plugin actuates with ``AddRelativeForce``, i.e. along the **body** z
        axis. A capsized model therefore points its entire thrust sideways: it
        cannot climb, cannot translate and cannot even yaw, while remaining in
        ``FLYING_MODEL`` and continuing to publish perfectly healthy-looking
        odometry at 30 Hz. Nothing in the topic list says anything is wrong.

        An entire measurement campaign here was run against an aircraft lying at
        roll 82 degrees and pitch -68 degrees. It produced plausible,
        self-consistent, completely meaningless tracking numbers -- and, because
        the aircraft could not move at all, it looked exactly like a controller
        that "progressively diverges from the plan". The tilt limit the plugin
        applies to its own attitude command is 0.5 rad, so anything past ~35
        degrees is a crash, not a manoeuvre.
        """
        roll, pitch, _ = self.attitude()
        return abs(roll) < limit_rad and abs(pitch) < limit_rad

    def spin(self, seconds):
        # type: (float) -> None
        """Pump callbacks for a wall-clock interval."""
        end = self.get_clock().now().nanoseconds * 1e-9 + seconds
        while rclpy.ok() and self.get_clock().now().nanoseconds * 1e-9 < end:
            rclpy.spin_once(self, timeout_sec=0.01)

    def wait_for_odom(self, timeout=20.0):
        # type: (float) -> bool
        for _ in range(int(timeout / 0.05)):
            rclpy.spin_once(self, timeout_sec=0.05)
            if self._odom is not None:
                return True
        return False

    def climb_to(self, altitude):
        # type: (float) -> None
        """Get airborne and stable before any measurement begins."""
        for _ in range(40):
            if self._odom is not None and self._odom.pose.pose.position.z >= altitude:
                break
            self._takeoff.publish(Empty())
            self.spin(0.25)
        self.spin(2.0)

    # ── the run ──────────────────────────────────────────────────────────

    def return_to(self, point, tolerance=0.25, timeout_s=25.0):
        # type: (object, float, float) -> bool
        """Fly back to where the run began, then stop.

        Every run must start from the same place or successive runs are not
        comparable -- and worse, an aircraft that ends each run a little further
        downrange eventually ends one inside the scenery, at which point the rig
        measures an obstruction and reports it as a tracking error. That has
        happened twice here and cost a whole set of numbers both times.

        Uses the servo's own station-keeping rather than a Gazebo teleport,
        because ``/simple_drone/reset`` does not move the model and this build's
        ``gz model -p`` does not either.
        """
        target = np.asarray(point, dtype=float).reshape(3)
        servo = VelocityServo(VelocityServoParams(plant=MEASURED_PLANT))
        servo.reset(hold_position=target)
        deadline = self.now() + timeout_s
        last = self.now()
        while rclpy.ok() and self.now() < deadline:
            rclpy.spin_once(self, timeout_sec=0.02)
            now = self.now()
            dt = now - last
            if dt <= 1e-6:
                continue
            last = now
            position, velocity, yaw = self.state()
            if float(np.linalg.norm(position - target)) <= tolerance:
                break
            command = servo.update(position, velocity, yaw, dt, now, follow=False)
            twist = Twist()
            twist.linear.x, twist.linear.y, twist.linear.z = command.body_velocity()
            twist.angular.z = command.yaw_rate
            self._cmd.publish(twist)
        for _ in range(25):
            self._cmd.publish(Twist())
            self.spin(0.02)
        position, _, _ = self.state()
        return float(np.linalg.norm(position - target)) <= tolerance

    def fly(self, use_lead):
        # type: (bool) -> dict
        """Fly the configured route once and return the error record."""
        params = VelocityServoParams(
            plant=MEASURED_PLANT,
            limits=VelocityLimits(max_speed_xy=1.5, max_speed_up=0.8,
                                  max_speed_down=0.6, max_accel_xy=2.0,
                                  max_accel_z=2.0, max_yaw_rate=1.4,
                                  max_yaw_accel=3.0),
            use_feedforward_lead=use_lead,
            predict_reference=use_lead)
        servo = VelocityServo(params)

        if not self.upright():
            roll, pitch, _ = self.attitude()
            raise RuntimeError(
                "refusing to fly: the airframe is already capsized (roll %.0f deg, "
                "pitch %.0f deg). Restart the world." % (math.degrees(roll), math.degrees(pitch)))
        origin, _, _ = self.state()
        # Started half a second in the future, exactly as FALCON does, so the
        # curve is adopted rather than joined mid-flight.
        start = self.now() + 0.5
        route = build_route(self.args.route, self.args.speed, self.args.altitude,
                            origin, start)
        servo.set_trajectory(route)
        servo.reset()
        servo.set_trajectory(route)

        dt_nominal = 1.0 / self.args.rate
        last = self.now()
        gaps, crosses, alongs, yaws, speeds = [], [], [], [], []
        deadline = start + route.duration + 1.0
        while rclpy.ok() and self.now() < deadline:
            rclpy.spin_once(self, timeout_sec=dt_nominal)
            now = self.now()
            dt = now - last
            if dt <= 1e-6:
                continue
            last = now
            position, velocity, yaw = self.state()
            if not self.upright():
                roll, pitch, _ = self.attitude()
                for _ in range(20):
                    self._cmd.publish(Twist())
                    self.spin(0.02)
                raise RuntimeError(
                    "airframe capsized mid-run (roll %.0f deg, pitch %.0f deg). "
                    "The plugin thrusts along BODY z, so it can no longer move at "
                    "all -- every sample after this point would be a measurement "
                    "of a crash, not of the controller."
                    % (math.degrees(roll), math.degrees(pitch)))
            command = servo.update(position, velocity, yaw, dt, now)
            twist = Twist()
            twist.linear.x, twist.linear.y, twist.linear.z = command.body_velocity()
            twist.angular.z = command.yaw_rate
            self._cmd.publish(twist)
            # Skip the settling window. The aircraft starts at rest while a
            # synthetic plan starts at cruise, so the opening seconds are an
            # ACQUISITION transient -- the aircraft closing a gap it began
            # with -- and not a statement about tracking. FALCON does not have
            # this problem, because it plans from the aircraft's measured
            # state, so its curves start at whatever velocity the aircraft
            # already has. This rig is pessimistic by comparison, and `--settle`
            # is how much of that pessimism is discarded.
            if now > start + self.args.settle and not command.holding:
                gaps.append(command.position_error_m)
                crosses.append(abs(command.cross_track_error_m))
                alongs.append(command.along_track_lag_m)
                yaws.append(abs(command.yaw_error_rad))
                speeds.append(float(np.linalg.norm(velocity)))

        self.return_to(origin)
        if not gaps:
            raise RuntimeError("no samples recorded; did the aircraft ever follow?")
        return {"mean_gap": float(np.mean(gaps)), "max_gap": float(np.max(gaps)),
                "mean_cross": float(np.mean(crosses)), "max_cross": float(np.max(crosses)),
                "max_along": float(np.max(np.abs(alongs))),
                "max_yaw_deg": math.degrees(float(np.max(yaws))),
                "max_speed": float(np.max(speeds)), "samples": len(gaps),
                "duration_s": route.duration,
                "blocked": _looks_blocked(alongs, speeds)}


def main():
    # type: () -> int
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--route", default="circle",
                        choices=["circle", "straight", "corner", "slalom", "climb"])
    parser.add_argument("--speed", type=float, default=0.6, help="plan speed, m/s")
    parser.add_argument("--altitude", type=float, default=1.5, help="metres")
    parser.add_argument("--rate", type=float, default=50.0, help="control rate, Hz")
    parser.add_argument("--settle", type=float, default=1.0,
                        help="seconds of acquisition transient to discard")
    parser.add_argument("--compare", action="store_true",
                        help="fly the route twice, with and without the lead term")
    args = parser.parse_args()

    rclpy.init()
    rig = TrackingRig(args)
    try:
        if not rig.wait_for_odom():
            print("no odometry on /simple_drone/odom -- is the world up?", file=sys.stderr)
            return 1
        rig.climb_to(args.altitude)

        runs = [("inverse-plant lead", True)]
        if args.compare:
            runs.append(("P + feedforward", False))

        results = []
        for label, lead in runs:
            print("\n=== %s : route=%s speed=%.2f m/s ===" % (label, args.route, args.speed))
            result = rig.fly(lead)
            results.append((label, result))
            print("  samples %d over %.1f s of plan" % (result["samples"], result["duration_s"]))
            rig.spin(3.0)

        print("\n%-22s %9s %9s %9s %9s %9s" %
              ("configuration", "mean gap", "max gap", "mean X", "max X", "max yaw"))
        for label, r in results:
            print("%-22s %9.4f %9.4f %9.4f %9.4f %7.1f deg%s" %
                  (label, r["mean_gap"], r["max_gap"], r["mean_cross"],
                   r["max_cross"], r["max_yaw_deg"],
                   "   <-- BLOCKED" if r["blocked"] else ""))
        if any(r["blocked"] for _, r in results):
            print("\nAt least one run was BLOCKED: the aircraft was held against "
                  "something\nwhile the command stayed saturated. These numbers "
                  "describe an obstruction,\nnot the controller. Move the "
                  "aircraft to clear space and fly it again.")
            return 2
        if len(results) == 2:
            lead, plain = results[0][1], results[1][1]
            if plain["mean_gap"] > 1e-9:
                print("\nmean gap  %.2fx better with the lead term" %
                      (plain["mean_gap"] / max(lead["mean_gap"], 1e-9)))
                print("max cross %.2fx better with the lead term" %
                      (plain["max_cross"] / max(lead["max_cross"], 1e-9)))
        return 0
    finally:
        rig.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    sys.exit(main())
