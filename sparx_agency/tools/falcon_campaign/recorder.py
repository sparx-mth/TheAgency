#!/usr/bin/env python3
"""ROS 2 flight recorder for one autonomous FALCON exploration run.

MISSION.md section 6 asks every flight to be judged after the fact -- tracking
error, commanded-vs-achieved speed, stop count, coverage -- and none of that can
be reconstructed from a terminal scrollback. This node is the ROS 2 half of that
record: ground truth alongside what the follower actually commanded, sampled at
a fixed cadence into a file the analyzer reads as a time series.

Two choices here were paid for in wasted debugging days. **Sample-and-hold, not
one line per callback**: streams arrive at unrelated rates, and interleaved raw
callbacks must be re-aligned before anything can be compared, whereas one line
per tick is directly differentiable and joinable. **Every stream carries an
``age``**: "velocity is 0.0" and "the velocity publisher died twenty seconds
ago" look identical in a value column, and this project has repeatedly mistaken
one for the other -- ``age`` is seconds since that stream last updated, and
``null`` means it never published at all.

Truth is recorded raw -- no sign flips, no frame corrections, no filtering: a
recording that applied a correction cannot un-apply it once that correction
turns out to be wrong.

Runs inside the ``robotican_dev`` container (ROS 2 Humble, ROS_DOMAIN_ID 9,
CycloneDDS), where the repo is mounted at its host path so output lands straight
in ``config.RUNS_DIR``. Subscribe-only: it never publishes and never actuates.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import pathlib
import signal
import sys
import time

import rclpy
from geometry_msgs.msg import PoseStamped, TwistStamped, Vector3
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from std_msgs.msg import String

try:
    from sparx_agency.tools.falcon_campaign import config
except ImportError:  # invoked as a plain script, without the repo on PYTHONPATH
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import config

try:
    from rooster_manager_interfaces.msg import RoosterState
    from sphera_common_interfaces.msg import SpheraPawnState
except ImportError as exc:
    raise ImportError(
        "recorder.py needs the vendor ROS 2 interfaces (sphera_common_"
        "interfaces, rooster_manager_interfaces), which are only built inside "
        "the '{container}' container -- run it there, not on the host "
        "(docker exec {container} ...). Original error: {err}".format(
            container=config.DEV_CONTAINER, err=exc))

#: Stream keys, in write order; each is a key of :data:`config.ROS2_TOPICS`.
STREAMS = ("truth", "velocity", "localization", "attitude", "cmd_nav", "state")

TRUTH_FILE = "truth.jsonl"
META_FILE = "recorder_meta.json"


def _num(value: float):
    """Return ``value`` as a float, or ``None`` when it is NaN or infinite.

    ``RoosterState.ranger`` is legitimately ``inf`` before the first rangefinder
    sample, and ``json.dumps`` would emit a bare ``Infinity`` that no strict
    JSON reader accepts -- one such value makes the whole run unreadable.
    """
    number = float(value)
    return number if math.isfinite(number) else None


class FlightRecorder(Node):
    """Samples every telemetry stream of one flight into ``truth.jsonl``.

    Attributes:
        finished: Set once ``duration_sec`` elapsed; the main loop exits on it.
        samples: Lines written so far.
    """

    def __init__(self, run_dir: str, rooster_id: str, hz: float,
                 duration_sec: float) -> None:
        """Open the output files and subscribe to every recorded stream.

        Args:
            run_dir: Where the output files go; created if missing.
            rooster_id: Drone id, substituted into :data:`config.ROS2_TOPICS`.
            hz: Sample and flush cadence, must be positive.
            duration_sec: Stop after this long; ``0`` waits for a signal.

        Raises:
            ValueError: If ``hz`` is not positive.
        """
        super().__init__("falcon_campaign_recorder")
        if hz <= 0.0:
            raise ValueError("--hz must be positive, got {}".format(hz))

        self.rooster_id = rooster_id
        self.hz = float(hz)
        self.duration_sec = float(duration_sec)
        self.run_dir = pathlib.Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)

        self.finished = False
        self.samples = 0
        self._latest = dict((name, None) for name in STREAMS)
        self._stamp = dict((name, None) for name in STREAMS)
        self._counts = dict((name, 0) for name in STREAMS)
        self._topics = dict((name, self._topic(name)) for name in STREAMS)

        self._start_wall = time.time()
        self._start_mono = time.monotonic()
        # Append, so a recorder restarted mid-flight never truncates the flight.
        self._fh = open(str(self.run_dir / TRUTH_FILE), "a")

        # Sphera publishes the pawn state BEST_EFFORT; a default (RELIABLE)
        # subscription matches nothing and receives silence.
        best_effort = QoSProfile(depth=10,
                                 reliability=ReliabilityPolicy.BEST_EFFORT)
        for name, msg_type, callback, qos in (
                ("truth", SpheraPawnState, self._on_truth, best_effort),
                ("velocity", TwistStamped, self._on_velocity, 10),
                ("localization", PoseStamped, self._on_localization, 10),
                ("attitude", Vector3, self._on_attitude, 10),
                ("cmd_nav", String, self._on_cmd_nav, 10),
                ("state", RoosterState, self._on_state, 10)):
            self.create_subscription(msg_type, self._topics[name], callback, qos)

        self.create_timer(1.0 / self.hz, self._sample)
        self.get_logger().info(
            "recording {} at {:.1f} Hz -> {}".format(
                rooster_id, self.hz, self.run_dir / TRUTH_FILE))

    def _topic(self, name: str) -> str:
        """Resolve a stream's topic, retargeted at ``rooster_id``."""
        topic = config.ROS2_TOPICS[name]
        return topic.replace("/{}/".format(config.DRONE_ID),
                             "/{}/".format(self.rooster_id), 1)

    def _store(self, name: str, value: dict) -> None:
        """Record the newest value of one stream and stamp its arrival."""
        self._latest[name] = value
        self._stamp[name] = time.monotonic()
        self._counts[name] += 1

    def _on_truth(self, msg):
        """Store Sphera's pawn state raw -- no sign or frame correction."""
        loc, vel, rot = msg.location, msg.velocity, msg.rotation
        self._store("truth", {
            "x": _num(loc.x), "y": _num(loc.y), "z": _num(loc.z),
            "vx": _num(vel.x), "vy": _num(vel.y), "vz": _num(vel.z),
            "roll": _num(rot.roll), "pitch": _num(rot.pitch),
            "yaw": _num(rot.yaw)})

    def _on_velocity(self, msg):
        """Store the world-frame truth-derived velocity."""
        linear, angular = msg.twist.linear, msg.twist.angular
        self._store("velocity", {
            "vx": _num(linear.x), "vy": _num(linear.y), "vz": _num(linear.z),
            "yaw_rate": _num(angular.z)})

    def _on_localization(self, msg):
        """Store the pose the whole stack (and FALCON) actually consumes."""
        position, q = msg.pose.position, msg.pose.orientation
        # Exact yaw for the z/w-only planar contract this topic publishes.
        yaw = math.atan2(2.0 * q.w * q.z, 1.0 - 2.0 * q.z * q.z)
        self._store("localization", {
            "x": _num(position.x), "y": _num(position.y),
            "z": _num(position.z), "yaw": _num(yaw)})

    def _on_attitude(self, msg):
        """Store raw roll/pitch/yaw (radians, sign convention unverified)."""
        self._store("attitude", {
            "roll": _num(msg.x), "pitch": _num(msg.y), "yaw": _num(msg.z)})

    def _on_cmd_nav(self, msg):
        """Store the commanded axes so achieved-vs-commanded is computable."""
        try:
            payload = json.loads(msg.data)
        except ValueError:
            # A malformed publisher must not kill the recording of the flight.
            self._store("cmd_nav", {"action": None, "raw": msg.data})
            return
        axes = payload.get("axes") or {}
        entry = {"action": payload.get("action")}
        for axis in ("x", "y", "r"):
            entry[axis] = _num(axes[axis]) if axis in axes else None
        if "value" in payload:
            entry["value"] = _num(payload["value"])
        self._store("cmd_nav", entry)

    def _on_state(self, msg):
        """Store the Rooster manager's own view of the airframe."""
        self._store("state", {
            "ranger": _num(msg.ranger), "battery": _num(msg.percentage),
            "armed": bool(msg.armed), "airborne": bool(msg.airborne),
            "flight_mode": int(msg.flight_mode)})

    def _sample(self):
        """Write one sample-and-hold snapshot of every stream, then flush."""
        now = time.monotonic()
        elapsed = now - self._start_mono
        row = {"t": round(elapsed, 4), "wall": round(time.time(), 4)}
        for name in STREAMS:
            latest = self._latest[name]
            entry = dict(latest) if latest is not None else {}
            stamp = self._stamp[name]
            # age None == never published; the analyzer must not read that as 0.
            entry["age"] = None if stamp is None else round(now - stamp, 4)
            row[name] = entry
        self._fh.write(json.dumps(row) + "\n")
        self._fh.flush()
        self.samples += 1
        if self.duration_sec > 0.0 and elapsed >= self.duration_sec:
            self.finished = True

    def write_meta(self, ended: bool = False) -> None:
        """Write ``recorder_meta.json``; also called at startup, so a
        hard-killed run still tells a crash apart from a failure to start.

        Args:
            ended: True when the flight is over; stamps ``end_wall``.
        """
        meta = {
            "rooster_id": self.rooster_id, "hz": self.hz,
            "duration_sec": self.duration_sec,
            "start_wall": self._start_wall,
            "end_wall": time.time() if ended else None,
            "samples": self.samples, "topics": self._topics,
            "counts": dict(self._counts),
            "ever_received": dict(
                (name, self._counts[name] > 0) for name in STREAMS)}
        with open(str(self.run_dir / META_FILE), "w") as handle:
            json.dump(meta, handle, indent=2)
            handle.write("\n")

    def close(self):
        """Close the JSONL file. Every line was already flushed."""
        self._fh.close()


def _parse_args(argv):
    """Parse the recorder's command line (``None`` reads ``sys.argv[1:]``)."""
    parser = argparse.ArgumentParser(
        description="Record one FALCON campaign flight to JSONL.")
    parser.add_argument("--run-dir", required=True,
                        help="Run dir, normally under {}".format(config.RUNS_DIR))
    parser.add_argument("--rooster-id", default=config.DRONE_ID)
    parser.add_argument("--hz", type=float, default=20.0,
                        help="Sample and flush cadence (default: 20).")
    parser.add_argument("--duration-sec", type=float, default=0.0,
                        help="0 (default) records until SIGINT/SIGTERM.")
    return parser.parse_args(argv)


def main(argv=None):
    """Run the recorder until the duration elapses or a signal arrives.

    Args:
        argv: Argument list, or ``None`` for ``sys.argv[1:]``.

    Returns:
        Process exit code; 0 on a clean stop.
    """
    args = _parse_args(argv)
    rclpy.init()
    recorder = FlightRecorder(args.run_dir, args.rooster_id, args.hz,
                              args.duration_sec)
    recorder.write_meta()

    stop = {"requested": False}

    def _request_stop(signum, frame):
        stop["requested"] = True

    # Installed after rclpy.init() so these win over rclpy's own SIGINT handler.
    signal.signal(signal.SIGINT, _request_stop)
    signal.signal(signal.SIGTERM, _request_stop)

    try:
        while rclpy.ok() and not stop["requested"] and not recorder.finished:
            rclpy.spin_once(recorder, timeout_sec=0.1)
    finally:
        recorder.write_meta(ended=True)
        recorder.close()
        recorder.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    return 0


if __name__ == "__main__":
    sys.exit(main())
