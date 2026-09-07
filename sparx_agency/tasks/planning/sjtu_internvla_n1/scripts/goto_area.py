#!/usr/bin/env python3
"""Put the SJTU drone down in a named part of the hospital, ready to be flown.

Five recordings "in different areas" needs a way to *get* to those areas that is
not itself a navigation problem. This is that way, and it is deliberately dumb:

1. take off and climb to a **ferry altitude above the interior walls**;
2. fly straight to the target under a plain world-frame P controller;
3. descend to the cruise altitude the policy will fly at;
4. turn to the requested heading and hand back.

Step 1 is what makes step 2 safe. The hospital's floor-1 walls top out at 3 m
and this world includes no ceiling over them, so a straight line at 4.5 m
crosses the whole building without touching anything -- whereas the same
straight line at 1.2 m goes through several of them. This is a ferry, not a
planner, and it does not pretend otherwise.

**Not** the plugin's own position mode, which looks like exactly the right tool
and is not: `pid_controller.cpp` clamps the setpoint to the controller's
`Limit`, and the drone's `Position XY` limit is **5**. A position setpoint is
therefore silently truncated to +/-5 m of the world origin -- the aircraft flies
confidently to (x, 5.00) and hovers there for ever, with healthy odometry and
no error anywhere. In a 27 x 59 m building that reaches almost nothing.

The areas come from ``config/hospital_areas.yaml``, which records for each one
the clearance measured off the occupancy map, so a start pose that is 0.2 m from
a wall is a fact in the file rather than a surprise in the recording.

Runs on the host, CPU only. It publishes ``/simple_drone/cmd_vel`` directly and
must therefore NEVER overlap the follower -- run it between recordings, not
during one.
"""
from __future__ import annotations

import argparse
import math
import os
import sys
import time

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import rclpy
import yaml
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import Empty, Int8

from sparx_agency.core.common.math.se3 import yaw_from_quaternion

_HERE = os.path.dirname(os.path.abspath(__file__))
_AREAS = os.path.join(_HERE, os.pardir, "config", "hospital_areas.yaml")


def _yaw_from_quat(q):
    """Yaw (radians, CCW from +x) from a geometry_msgs quaternion."""
    return yaw_from_quaternion((q.x, q.y, q.z, q.w))


class Ferry(Node):
    """Drive the drone to a world pose with a velocity command."""

    def __init__(self, ns="/simple_drone"):
        super().__init__("sjtu_goto_area")
        self.pose = None
        self.pose_stamp = 0.0
        self.flying = False
        sensor_qos = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT,
                                history=HistoryPolicy.KEEP_LAST, depth=1)
        self.create_subscription(Odometry, ns + "/odom", self._on_odom, sensor_qos)
        self.create_subscription(Int8, ns + "/state", self._on_state, 10)
        self._takeoff = self.create_publisher(Empty, ns + "/takeoff", 1)
        self._cmd = self.create_publisher(Twist, ns + "/cmd_vel", 1)

    def _on_odom(self, msg):
        p = msg.pose.pose.position
        self.pose = (p.x, p.y, p.z, _yaw_from_quat(msg.pose.pose.orientation))
        self.pose_stamp = time.time()

    def _on_state(self, msg):
        self.flying = int(msg.data) == 1

    # -- primitives ------------------------------------------------------

    def spin(self, seconds):
        end = time.time() + seconds
        while time.time() < end and rclpy.ok():
            rclpy.spin_once(self, timeout_sec=0.05)

    def wait_for_odom(self, timeout=45.0):
        # Generous, because this is usually called seconds after a container
        # restart: the plugin logs "finished loading" as soon as it is up, but
        # DDS discovery to a subscriber that did not exist yet takes longer
        # still, and a short timeout here turns a slow start into "could not
        # reach the area" for the whole run.
        end = time.time() + timeout
        while self.pose is None and time.time() < end and rclpy.ok():
            rclpy.spin_once(self, timeout_sec=0.1)
        if self.pose is None:
            raise RuntimeError("no odometry on /simple_drone/odom -- is the world up, "
                               "and is ROS_DOMAIN_ID / the DDS profile right?")

    def takeoff(self, tries=8):
        """Command takeoff and CONFIRM it against /simple_drone/state.

        The plugin silently drops takeoff from the wrong state, and while landed
        it also ignores `cmd_vel` -- so an unconfirmed takeoff turns every leg
        below into a no-op that burns its whole timeout and then reports arrival
        at the spawn point. That is how a campaign records five flights from one
        place and labels them five different areas.

        Returns:
            ``True`` once the aircraft reports FLYING.
        """
        for _ in range(int(tries)):
            if self.flying:
                return True
            self._takeoff.publish(Empty())
            self.spin(2.0)
        return bool(self.flying)

    def goto(self, x, y, z, tol=0.35, timeout=120.0, speed=1.2):
        """Fly to a world point under a P controller, holding heading.

        Args:
            x: Target world x, metres.
            y: Target world y, metres.
            z: Target altitude, metres.
            tol: Arrival radius, metres.
            timeout: Give up after this long.
            speed: Horizontal speed cap, m/s. Well under the airframe's 2 m/s:
                this drone translates by tilting, and a fast ferry is a tilted
                one.

        Returns:
            ``True`` if the point was reached inside ``tol``.
        """
        end = time.time() + timeout
        while time.time() < end and rclpy.ok():
            if time.time() - self.pose_stamp > 1.0:
                # A frozen pose gives a constant error, so the controller would
                # happily fly open-loop at 1.2 m/s for the rest of the timeout
                # and never know. Silence is a stop.
                self.stop()
                raise RuntimeError("odometry stopped mid-ferry")
            px, py, pz, yaw = self.pose
            ex, ey, ez = x - px, y - py, z - pz
            if math.dist((px, py, pz), (x, y, z)) <= tol:
                self.stop()
                return True
            # World-frame demand, capped, then rotated into the yaw-aligned body
            # frame the plugin reads. Same rotation the flight follower does.
            norm = math.hypot(ex, ey)
            scale = min(1.0, speed / norm) if norm > 1e-6 else 0.0
            vx_w, vy_w = ex * scale, ey * scale
            cmd = Twist()
            cmd.linear.x = math.cos(yaw) * vx_w + math.sin(yaw) * vy_w
            cmd.linear.y = -math.sin(yaw) * vx_w + math.cos(yaw) * vy_w
            cmd.linear.z = max(-0.8, min(0.8, 1.0 * ez))
            self._cmd.publish(cmd)
            self.spin(0.05)
        self.stop()
        return False

    def face(self, yaw, tol=0.08, timeout=30.0):
        """Turn to a world heading with a rate command, holding station."""
        end = time.time() + timeout
        while time.time() < end and rclpy.ok():
            error = math.atan2(math.sin(yaw - self.pose[3]), math.cos(yaw - self.pose[3]))
            if abs(error) < tol:
                break
            cmd = Twist()
            cmd.angular.z = max(-0.8, min(0.8, 1.5 * error))
            self._cmd.publish(cmd)
            self.spin(0.05)
        self._cmd.publish(Twist())
        self.spin(0.3)

    def stop(self):
        for _ in range(5):
            self._cmd.publish(Twist())
            self.spin(0.1)


def load_areas(path=_AREAS):
    """Read the named start areas.

    Raises:
        FileNotFoundError: The areas file is missing -- there is no sensible
            default hospital pose to fall back to, and guessing one would put
            the aircraft inside a wall.
    """
    with open(path, "r") as handle:
        data = yaml.safe_load(handle) or {}
    areas = data.get("areas") or {}
    if not areas:
        raise ValueError("%s lists no areas" % path)
    return areas


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("area", nargs="?", help="named area from hospital_areas.yaml")
    ap.add_argument("--x", type=float, help="world x, instead of a named area")
    ap.add_argument("--y", type=float, help="world y")
    ap.add_argument("--yaw-deg", type=float, default=None, help="heading to end on")
    ap.add_argument("--ferry-alt", type=float, default=4.5,
                    help="altitude to cross the building at, above the walls")
    ap.add_argument("--cruise-alt", type=float, default=1.2,
                    help="altitude to descend to before handing back")
    ap.add_argument("--list", action="store_true", help="list the areas and exit")
    args = ap.parse_args(argv)

    areas = load_areas()
    if args.list:
        for name, spec in sorted(areas.items()):
            print("%-16s x=%7.2f y=%7.2f yaw=%6.1f  clearance %.2f m  %s"
                  % (name, spec["x"], spec["y"], spec.get("yaw_deg", 0.0),
                     spec.get("clearance_m", float("nan")), spec.get("note", "")))
        return 0

    if args.x is not None and args.y is not None:
        target = {"x": args.x, "y": args.y, "yaw_deg": args.yaw_deg or 0.0}
    elif args.area:
        if args.area not in areas:
            print("unknown area %r; known: %s" % (args.area, ", ".join(sorted(areas))),
                  file=sys.stderr)
            return 2
        target = dict(areas[args.area])
        if args.yaw_deg is not None:
            target["yaw_deg"] = args.yaw_deg
    else:
        ap.error("give an area name, or --x and --y")

    rclpy.init()
    node = Ferry()
    try:
        node.wait_for_odom()
        start = node.pose
        print("[goto_area] from (%.2f, %.2f, %.2f) to (%.2f, %.2f) yaw %.0f deg"
              % (start[0], start[1], start[2], target["x"], target["y"],
                 target.get("yaw_deg", 0.0)))
        if not node.takeoff():
            print("[goto_area] the aircraft never reported FLYING; takeoff was "
                  "dropped. A capsized airframe cannot take off and "
                  "/simple_drone/reset does not right it -- restart the world.",
                  file=sys.stderr)
            return 1
        # Climb where it stands, cross above the walls, then descend.
        if not node.goto(start[0], start[1], args.ferry_alt, tol=0.4, timeout=60.0):
            print("[goto_area] never reached the ferry altitude (%.1f m); refusing "
                  "to cross the building below the walls." % args.ferry_alt,
                  file=sys.stderr)
            return 1
        if not node.goto(target["x"], target["y"], args.ferry_alt, tol=0.6, timeout=180.0):
            # Descending here is the one thing that must not happen. The whole
            # safety argument is that 4.5 m clears the 3 m walls and 1.2 m does
            # not, and a crossing that timed out is by definition one that is
            # stuck -- so dropping to cruise altitude and pushing on drives into
            # whatever stopped it.
            print("[goto_area] did not reach the target in time; NOT descending.",
                  file=sys.stderr)
            return 1
        if not node.goto(target["x"], target["y"], args.cruise_alt, tol=0.35, timeout=90.0):
            print("[goto_area] reached the area but not the cruise altitude.",
                  file=sys.stderr)
            return 1
        node.stop()
        node.face(math.radians(target.get("yaw_deg", 0.0)))
        node.stop()
        final = node.pose
        print("[goto_area] at (%.2f, %.2f, %.2f) yaw %.0f deg"
              % (final[0], final[1], final[2], math.degrees(final[3])))
        return 0
    finally:
        try:
            node.stop()
        except Exception:  # noqa: BLE001
            pass
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    sys.exit(main())
