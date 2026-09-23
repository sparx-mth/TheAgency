#!/usr/bin/env python3
"""Take off, and do not return until the aircraft says it is flying.

The SJTU plugin silently drops ``/simple_drone/takeoff`` from the wrong state.
Nothing reports it: the aircraft sits on the floor at state 0 while a policy
commits route after route to it and a follower publishes a twist the plugin
ignores while landed. A 90 s recording has been lost here to exactly that. So
takeoff is *confirmed* against ``/simple_drone/state`` rather than assumed, and
retried until it takes.

It is a script and not a line of bash because ``ros2 topic echo --once`` is not
a reliable reader of this topic -- it answers ``A message was lost!!!`` often
enough that a shell test on its output is a coin flip.

Exit status is the contract: 0 flying, 1 never got airborne, 2 no telemetry at
all. A capsized aircraft cannot take off and ``/simple_drone/reset`` does not
right it -- the world has to be restarted -- so a refusal here says so.
"""
from __future__ import annotations

import argparse
import math
import os
import sys
import time

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import rclpy
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import Empty, Int8

LANDED, FLYING = 0, 1


def _roll_pitch(q):
    roll = math.atan2(2.0 * (q.w * q.x + q.y * q.z), 1.0 - 2.0 * (q.x * q.x + q.y * q.y))
    s = max(-1.0, min(1.0, 2.0 * (q.w * q.y - q.z * q.x)))
    return roll, math.asin(s)


class Takeoff(Node):
    """Command takeoff and watch the flight state and attitude."""

    def __init__(self, ns="/simple_drone"):
        super().__init__("sjtu_ensure_flying")
        self.state = None
        self.attitude = None
        self.altitude = None
        sensor_qos = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT,
                                history=HistoryPolicy.KEEP_LAST, depth=1)
        self.create_subscription(Int8, ns + "/state", self._on_state, 10)
        self.create_subscription(Odometry, ns + "/odom", self._on_odom, sensor_qos)
        self._takeoff = self.create_publisher(Empty, ns + "/takeoff", 1)

    def _on_state(self, msg):
        self.state = int(msg.data)

    def _on_odom(self, msg):
        self.altitude = msg.pose.pose.position.z
        self.attitude = _roll_pitch(msg.pose.pose.orientation)

    def spin(self, seconds):
        end = time.time() + seconds
        while time.time() < end and rclpy.ok():
            rclpy.spin_once(self, timeout_sec=0.05)

    def command(self):
        self._takeoff.publish(Empty())


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tries", type=int, default=8)
    ap.add_argument("--settle", type=float, default=4.0,
                    help="seconds to let the climb settle once flying")
    ap.add_argument("--capsize-deg", type=float, default=35.0)
    args = ap.parse_args(argv)

    rclpy.init()
    node = Takeoff()
    try:
        node.spin(4.0)
        if node.attitude is None:
            # Odometry is the ground truth for "is there a drone there at all".
            # Testing `state is None AND attitude is None` meant a run with odom
            # but no /state published takeoff eight times and then reported a
            # perfectly healthy flying aircraft as "never got airborne".
            print("[ensure_flying] no telemetry from the drone -- is the world up, and "
                  "do ROS_DOMAIN_ID and the DDS profile match it?", file=sys.stderr)
            return 2

        if node.attitude is not None:
            roll, pitch = node.attitude
            if max(abs(roll), abs(pitch)) > math.radians(args.capsize_deg):
                print("[ensure_flying] the aircraft is CAPSIZED (roll %.0f deg, pitch "
                      "%.0f deg). It cannot take off, and /simple_drone/reset does not "
                      "right it. Restart the world."
                      % (math.degrees(roll), math.degrees(pitch)), file=sys.stderr)
                return 1

        for attempt in range(1, args.tries + 1):
            if node.state == FLYING:
                break
            node.command()
            node.spin(2.0)
            print("[ensure_flying] takeoff attempt %d: state=%s altitude=%s"
                  % (attempt, node.state,
                     "?" if node.altitude is None else round(node.altitude, 2)))

        if node.state != FLYING:
            print("[ensure_flying] never reached the FLYING state (state=%s)" % (node.state,),
                  file=sys.stderr)
            return 1
        node.spin(args.settle)
        # Check attitude AGAIN. The first check only proved it was level on the
        # ground; the climb itself is where a drone standing against something
        # goes over, and an aircraft that reports FLYING while lying on its side
        # would otherwise be handed to the policy as ready.
        if node.attitude is not None:
            roll, pitch = node.attitude
            if max(abs(roll), abs(pitch)) > math.radians(args.capsize_deg):
                print("[ensure_flying] CAPSIZED during the climb (roll %.0f deg, "
                      "pitch %.0f deg). Restart the world."
                      % (math.degrees(roll), math.degrees(pitch)), file=sys.stderr)
                return 1
        print("[ensure_flying] airborne: state=1 altitude=%.2f m"
              % (node.altitude if node.altitude is not None else float("nan")))
        return 0
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    sys.exit(main())
