#!/usr/bin/env python3
"""Watch an SJTU-drone flight and call the failure the instant it happens.

Runs inside the sim container (ROS 2 Humble): it is the one process that sees
the *physics* -- the collision contacts PhysX reports and the attitude the
airframe actually reaches -- which the ROS 1 FALCON side, reasoning off its own
map, structurally cannot. FALCON believing a wall is not there and the aircraft
hitting it is exactly the split this catches.

It subscribes to two topics and nothing else:

* ``/simple_drone/odom``          -- pose, attitude, body twist.
* ``/simple_drone/bumper_states`` -- ``gazebo_msgs/ContactsState``. A non-empty
  ``states`` array is a live contact; the collided model names come with it.

and emits, to stdout and to a JSON-lines trace, a status line a few times a
second plus an edge-triggered event whenever the flight crosses a failure
boundary:

* ``CONTACT``   -- a collision began (with the entity, if Gazebo names it).
* ``CAPSIZE``   -- roll or pitch past the plugin's ~35 deg attitude clamp, which
  is unrecoverable: the model thrusts along body +z, so on its side it cannot
  climb, translate or yaw (see the package README's capsize post-mortem).
* ``GROUNDED``  -- fell back below 0.3 m after having been airborne.
* ``WEDGED``    -- airborne but displaced < ``wedge_move_m`` over
  ``wedge_window_s``: pinned against geometry, going nowhere.

Verdict semantics: CAPSIZE and GROUNDED are terminal (the flight is over and the
monitor exits non-zero). CONTACT and WEDGED are reported but not terminal on
their own -- a glancing touch or a brief hold is survivable, and only the
operator/iterate loop decides whether to stop. ``--exit-on-contact`` makes the
first contact terminal too, for a run where any wall strike should halt.
"""
from __future__ import annotations

import argparse
import json
import math
import time

import rclpy
from gazebo_msgs.msg import ContactsState
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy


def _rpy_from_quaternion(x, y, z, w):
    # type: (float, float, float, float) -> tuple
    """Roll, pitch, yaw in radians from a ROS quaternion (ZYX convention)."""
    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = math.atan2(sinr_cosp, cosr_cosp)
    sinp = 2.0 * (w * y - z * x)
    sinp = max(-1.0, min(1.0, sinp))
    pitch = math.asin(sinp)
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = math.atan2(siny_cosp, cosy_cosp)
    return roll, pitch, yaw


class FlightMonitor(Node):
    """Trace the flight and name the failure boundary it crosses."""

    def __init__(self, args):
        # type: (argparse.Namespace) -> None
        super().__init__("flight_monitor")
        self._args = args
        self._trace = open(args.trace, "w") if args.trace else None
        self._start_wall = time.time()

        # best_effort reader accepts both reliable and best_effort writers, so
        # one QoS receives odom (reliable) and the contact topic (either).
        qos = QoSProfile(depth=10, history=HistoryPolicy.KEEP_LAST,
                         reliability=ReliabilityPolicy.BEST_EFFORT)
        self.create_subscription(Odometry, args.odom_topic, self._on_odom, qos)
        self.create_subscription(ContactsState, args.bumper_topic,
                                 self._on_bumper, qos)

        self._pose = None            # (x, y, z)
        self._rpy = None             # (roll, pitch, yaw)
        self._speed = 0.0
        self._airborne_seen = False
        self._in_contact = False
        self._contact_names = []
        self._window = []            # (t, x, y, z) for the wedge test
        self._failed = None          # terminal verdict, set once
        self._last_status = 0.0
        self.create_timer(0.2, self._tick)

    # ── inputs ───────────────────────────────────────────────────────────
    def _on_odom(self, msg):
        # type: (Odometry) -> None
        p = msg.pose.pose.position
        o = msg.pose.pose.orientation
        self._pose = (p.x, p.y, p.z)
        self._rpy = _rpy_from_quaternion(o.x, o.y, o.z, o.w)
        v = msg.twist.twist.linear
        self._speed = math.sqrt(v.x * v.x + v.y * v.y + v.z * v.z)
        if p.z > self._args.airborne_m:
            self._airborne_seen = True

    def _on_bumper(self, msg):
        # type: (ContactsState) -> None
        names = []
        for state in msg.states:
            names.append(getattr(state, "collision2_name", "") or
                         getattr(state, "collision1_name", ""))
        now_contact = len(msg.states) > 0
        if now_contact and not self._in_contact:
            self._event("CONTACT", "began: %s" % (", ".join(sorted(set(names))) or "?",))
            if self._args.exit_on_contact:
                self._failed = "CONTACT"
        self._in_contact = now_contact
        self._contact_names = sorted(set(n for n in names if n))

    # ── the watchdog ─────────────────────────────────────────────────────
    def _tick(self):
        # type: () -> None
        if self._pose is None or self._rpy is None:
            return
        t = time.time() - self._start_wall
        roll, pitch, yaw = self._rpy
        x, y, z = self._pose

        if self._failed is None:
            if abs(roll) > self._args.capsize_deg * math.pi / 180.0 or \
               abs(pitch) > self._args.capsize_deg * math.pi / 180.0:
                self._event("CAPSIZE", "roll=%.0f deg pitch=%.0f deg -- unrecoverable"
                            % (math.degrees(roll), math.degrees(pitch)))
                self._failed = "CAPSIZE"
            elif self._airborne_seen and z < self._args.grounded_m:
                self._event("GROUNDED", "z=%.2f m after flight" % (z,))
                self._failed = "GROUNDED"
            else:
                self._check_wedged(t, x, y, z)

        if t - self._last_status >= self._args.status_period_s:
            self._last_status = t
            self._status(t, x, y, z, roll, pitch)

        if self._trace:
            self._trace.write(json.dumps({
                "t": round(t, 2), "x": round(x, 3), "y": round(y, 3),
                "z": round(z, 3), "roll_deg": round(math.degrees(roll), 1),
                "pitch_deg": round(math.degrees(pitch), 1),
                "yaw_deg": round(math.degrees(yaw), 1),
                "speed": round(self._speed, 3),
                "contact": self._in_contact,
                "contact_names": self._contact_names}) + "\n")
            self._trace.flush()

        if self._failed is not None:
            self._finish()

    def _check_wedged(self, t, x, y, z):
        # type: (float, float, float, float) -> None
        self._window.append((t, x, y, z))
        cutoff = t - self._args.wedge_window_s
        while self._window and self._window[0][0] < cutoff:
            self._window.pop(0)
        if not self._airborne_seen or self._window[0][0] > cutoff:
            return  # not enough history yet
        span = max(math.dist((x, y, z), (wx, wy, wz))
                   for _, wx, wy, wz in self._window)
        if span < self._args.wedge_move_m:
            self._event("WEDGED", "moved %.2f m in %.0f s" %
                        (span, self._args.wedge_window_s))
            if self._args.exit_on_wedge:
                self._failed = "WEDGED"

    # ── output ───────────────────────────────────────────────────────────
    def _event(self, kind, detail):
        # type: (str, str) -> None
        t = time.time() - self._start_wall
        line = "[MONITOR][%s] t=%.1fs %s" % (kind, t, detail)
        print(line, flush=True)
        if self._trace:
            self._trace.write(json.dumps({"event": kind, "t": round(t, 2),
                                          "detail": detail}) + "\n")
            self._trace.flush()

    def _status(self, t, x, y, z, roll, pitch):
        # type: (float, float, float, float, float, float) -> None
        print("[MONITOR] t=%5.1fs pos=(%6.2f,%6.2f,%5.2f) rp=(%4.0f,%4.0f)deg "
              "v=%.2f contact=%s" % (t, x, y, z, math.degrees(roll),
                                     math.degrees(pitch), self._speed,
                                     "YES" if self._in_contact else "no"),
              flush=True)

    def _finish(self):
        # type: () -> None
        self._event("VERDICT", "flight ended: %s" % (self._failed,))
        if self._trace:
            self._trace.close()
        raise SystemExit(2 if self._failed in ("CAPSIZE", "GROUNDED") else 1)


def main():
    # type: () -> None
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--odom-topic", default="/simple_drone/odom")
    ap.add_argument("--bumper-topic", default="/simple_drone/bumper_states")
    ap.add_argument("--trace", default="/tmp/flight_trace.jsonl")
    ap.add_argument("--capsize-deg", type=float, default=35.0)
    ap.add_argument("--airborne-m", type=float, default=0.8)
    ap.add_argument("--grounded-m", type=float, default=0.3)
    ap.add_argument("--wedge-window-s", type=float, default=25.0)
    ap.add_argument("--wedge-move-m", type=float, default=0.4)
    ap.add_argument("--status-period-s", type=float, default=4.0)
    ap.add_argument("--exit-on-contact", action="store_true")
    ap.add_argument("--exit-on-wedge", action="store_true")
    args = ap.parse_args()

    rclpy.init()
    node = FlightMonitor(args)
    print("[MONITOR] watching %s and %s; trace -> %s"
          % (args.odom_topic, args.bumper_topic, args.trace), flush=True)
    try:
        rclpy.spin(node)
    except SystemExit:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()

