#!/usr/bin/env python3
"""ROS 2 sidecar recorder: what the Rooster was told, and what it actually did.

A Sphera flight is recorded by two processes that cannot see each other's ROS
graph. The ROS 1 recorder in the ``falcon`` container has the plan, the
reference and the command we ask for. It cannot have the rest: ``bridge.yaml``
carries only the frame paths, ``/R1/localization``, ``/R1/attitude_rpy`` and
``/cmd_vel``, so the actuator topics (``/R1/cmd_nav``, ``/R1/manual_control``),
the achieved velocity (``/R1/velocity_truth``) and Sphera's own ground truth do
not exist on the ROS 1 side at all. This node is the other half: it records the
last hop before the airframe and the only honest yardstick for what that hop
achieved, and the two halves are joined offline on ``wall`` (see
:mod:`sparx_agency.tasks.planning.nav_debug.schema`).

It must run inside the ``it`` container -- ROS 2 **Foxy**, ``ROS_DOMAIN_ID=9``,
``rmw_cyclonedds_cpp`` -- because that is the only place the vendor message
packages (``fcu_driver_interfaces``, ``sphera_common_interfaces``,
``rooster_manager_interfaces``) are built. A missing vendor package disables
just that stream and is named in the manifest, so the recording degrades instead
of vanishing.

Output goes to ``<out_dir>/ros2/``: ``actuator.jsonl`` and ``truth.jsonl``
sample-and-held at ``~record_hz``, ``axis_trace.jsonl`` and ``altitude.jsonl``
appended verbatim as their publishers emit them, plus ``manifest_ros2.json``
with per-stream receive counts -- a stream sitting at zero is how a silent QoS
mismatch is found after the fact rather than never. See
:mod:`nav_debug_ros2_writer` for where the run folder lands and why.

Two conventions are load-bearing. **Sample-and-hold, not one line per callback**:
streams arrive at unrelated rates, and one row per tick is directly joinable
against the ROS 1 side. **Every stream carries an age**: "velocity is 0.0" and
"the velocity publisher died twenty seconds ago" look identical in a value
column, and ``null`` means the stream never published at all.

Subscribe-only, and every callback and every write is guarded: this is a
diagnostic riding along with a live flight and it must never take one down.

Python 3.8 compatible (ROS 2 Foxy). Rosparams are listed at the bottom.
"""
from __future__ import annotations

import os
import signal
import sys
import time

import rclpy
from geometry_msgs.msg import TwistStamped, Vector3
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from std_msgs.msg import String

try:
    from sparx_agency.robots.ROBOTICAN import nav_debug_ros2_imports as resolved
    from sparx_agency.robots.ROBOTICAN import nav_debug_ros2_streams as streams
    from sparx_agency.robots.ROBOTICAN.nav_debug_ros2_writer import (
        RunWriter, resolve_run_dir)
except ImportError:  # started by path, without the repo root on PYTHONPATH
    sys.path.insert(0, os.path.abspath(os.path.join(
        os.path.dirname(os.path.abspath(__file__)), os.pardir, os.pardir,
        os.pardir)))
    from sparx_agency.robots.ROBOTICAN import nav_debug_ros2_imports as resolved
    from sparx_agency.robots.ROBOTICAN import nav_debug_ros2_streams as streams
    from sparx_agency.robots.ROBOTICAN.nav_debug_ros2_writer import (
        RunWriter, resolve_run_dir)

schema = resolved.schema

DEFAULT_DRONE_ID = "R1"
WARN_INTERVAL_S = 10.0

#: Verbatim trace streams and the file each is appended to.
TRACE_FILES = {"axis_trace": schema.AXIS_TRACE_FILE,
               "altitude": schema.ALTITUDE_FILE}


def _cmd_nav_of(msg):
    """``std_msgs/String`` -> the cmd_nav half of ``frame.Actuator``."""
    return streams.cmd_nav_fields(msg.data)


def _status_of(msg):
    """``/R1/rooster_status`` -> ``frame.Truth.status``, kept verbatim."""
    return {"status": msg.data}


class NavDebugRos2Recorder(Node):
    """Sample-and-holds the actuator and ground-truth streams of one flight.

    Attributes:
        finished: Set once ``~duration_sec`` elapsed; the main loop exits on it.
        samples: Sample-and-hold ticks written so far.
        writer: The run folder this recording is going into.
    """

    def __init__(self):
        """Resolve the run folder, subscribe, and start the sampling timer.

        Raises:
            ValueError: If ``~record_hz`` is not positive.
            OSError: If the run folder cannot be created.
        """
        super().__init__("nav_debug_ros2_recorder")
        self.declare_parameter("rooster_id", DEFAULT_DRONE_ID)
        self.declare_parameter("out_dir", "")
        self.declare_parameter("record_hz", 20.0)
        self.declare_parameter("manifest_interval_s", 10.0)
        self.declare_parameter("duration_sec", 0.0)

        self.rooster_id = str(self.get_parameter("rooster_id").value)
        self.record_hz = float(self.get_parameter("record_hz").value)
        if self.record_hz <= 0.0:
            raise ValueError(
                "record_hz must be positive, got {}".format(self.record_hz))
        self.duration_sec = float(self.get_parameter("duration_sec").value)

        self.finished = False
        self.samples = 0
        self._latest, self._stamp, self._counts = {}, {}, {}
        self._topics, self._skipped, self._warned = {}, {}, {}
        self._start_wall = time.time()
        self._start_mono = time.monotonic()

        self.writer = RunWriter(
            resolve_run_dir(str(self.get_parameter("out_dir").value)))
        self._subscribe()
        self.write_manifest()

        self.create_timer(1.0 / self.record_hz, self._sample)
        interval = float(self.get_parameter("manifest_interval_s").value)
        if interval > 0.0:
            self.create_timer(interval, self._bookkeeping)
        self.get_logger().info(
            "nav_debug_ros2_recorder: {} at {:.1f} Hz -> {}".format(
                self.rooster_id, self.record_hz, self.writer.ros2_dir))

    # ── subscriptions ────────────────────────────────────────────────────────
    def _subscribe(self):
        """Subscribe to every stream, skipping ones whose message type is gone.

        A stream whose vendor package is not built here is registered in the
        manifest as skipped instead of aborting the recording -- the other
        streams are still worth having.
        """
        for name, msg_type, topic, convert, qos in self._specs():
            self._counts[name] = 0
            self._latest[name] = None
            self._stamp[name] = None
            self._topics[name] = topic
            if msg_type is None:
                self._skipped[name] = resolved.MISSING.get(name, "unavailable")
                self.get_logger().error(
                    "stream '{}' disabled -- {} (vendor messages only exist in "
                    "the 'it' container)".format(name, self._skipped[name]))
                continue
            callback = (self._holder(name, convert) if convert is not None
                        else self._appender(name, TRACE_FILES[name]))
            self.create_subscription(msg_type, topic, callback, qos)

    def _specs(self):
        """Return ``(name, msg_type, topic, converter, qos)`` rows.

        ``msg_type`` is ``None`` for a vendor package that is not built here,
        and ``converter`` is ``None`` for the two verbatim trace streams. Sphera
        publishes the pawn state BEST_EFFORT; a default (RELIABLE) subscription
        matches nothing and receives silence.
        """
        best_effort = QoSProfile(depth=10,
                                 reliability=ReliabilityPolicy.BEST_EFFORT)
        topic = self._topic
        return (
            ("cmd_nav", String, topic("cmd_nav"), _cmd_nav_of, 10),
            ("manual", resolved.ManualControl, topic("manual_control"),
             streams.manual_fields, 10),
            ("velocity", TwistStamped, topic("velocity_truth"),
             streams.velocity_fields, 10),
            ("attitude", Vector3, topic("attitude_rpy"),
             streams.attitude_fields, 10),
            ("sphera", resolved.SpheraPawnState, topic("sphera/state"),
             streams.sphera_fields, best_effort),
            ("state", resolved.RoosterState, topic("state"),
             streams.state_fields, 10),
            ("status", String, topic("rooster_status"), _status_of, 10),
            ("axis_trace", String, self._retarget(schema.AXIS_TRACE_TOPIC),
             None, 10),
            ("altitude", String, self._retarget(schema.ALTITUDE_TRACE_TOPIC),
             None, 10),
        )

    def _topic(self, suffix):
        """``"cmd_nav"`` -> ``"/<rooster_id>/cmd_nav"``."""
        return "/{}/{}".format(self.rooster_id, suffix)

    def _retarget(self, topic):
        """Point a schema topic constant at this recorder's drone id."""
        return topic.replace("/{}/".format(DEFAULT_DRONE_ID),
                             "/{}/".format(self.rooster_id), 1)

    def _holder(self, name, convert):
        """Build a guarded callback that sample-and-holds one stream."""
        def callback(msg):
            try:
                self._latest[name] = convert(msg)
            except Exception as exc:            # a diagnostic never raises
                self._warn(name, "{} callback failed: {}".format(name, exc))
                return
            self._mark(name)
        return callback

    def _appender(self, name, filename):
        """Build a guarded callback that appends one String trace verbatim."""
        def callback(msg):
            try:
                fields = streams.trace_fields(msg.data)
                self.writer.write(
                    filename, schema.row(self._ros_time(), time.time(), **fields))
            except Exception as exc:            # a diagnostic never raises
                self._warn(name, "{} callback failed: {}".format(name, exc))
                return
            self._mark(name)
        return callback

    def _mark(self, name):
        """Stamp a stream's arrival and count it."""
        self._stamp[name] = time.monotonic()
        self._counts[name] += 1

    # ── sampling ─────────────────────────────────────────────────────────────
    def _sample(self):
        """Write one sample-and-hold row to ``actuator.jsonl`` and ``truth.jsonl``."""
        try:
            now, wall, t = time.monotonic(), time.time(), self._ros_time()
            ages = dict((name, self._age(name, now)) for name in self._stamp)
            self.writer.write(schema.ACTUATOR_FILE,
                              streams.actuator_row(t, wall, self._latest, ages))
            self.writer.write(schema.TRUTH_FILE,
                              streams.truth_row(t, wall, self._latest, ages))
            self.samples += 1
        except Exception as exc:                # a diagnostic never raises
            self._warn("sample", "sample failed: {}".format(exc))
            return
        if self.duration_sec > 0.0 and now - self._start_mono >= self.duration_sec:
            self.finished = True

    def _age(self, name, now):
        """Seconds since ``name`` last arrived, or ``None`` if it never did."""
        stamp = self._stamp.get(name)
        return None if stamp is None else now - stamp

    def _ros_time(self):
        """Node clock in seconds -- the ``t`` half of every schema row."""
        return self.get_clock().now().nanoseconds * 1e-9

    # ── manifest / plumbing ──────────────────────────────────────────────────
    def write_manifest(self, ended=False):
        """Write ``manifest_ros2.json`` with per-stream receive counts.

        Args:
            ended: True at shutdown; stamps ``end_wall``.
        """
        self.writer.write_manifest({
            "schema_version": schema.SCHEMA_VERSION,
            "recorder": "nav_debug_ros2_recorder",
            "ros_distro": os.environ.get("ROS_DISTRO", ""),
            "ros_domain_id": os.environ.get("ROS_DOMAIN_ID", ""),
            "rooster_id": self.rooster_id,
            "record_hz": self.record_hz,
            "duration_sec": self.duration_sec,
            "run_dir": self.writer.run_dir,
            "start_wall": self._start_wall,
            "end_wall": time.time() if ended else None,
            "samples": self.samples,
            "topics": dict(self._topics),
            "counts": dict(self._counts),
            "ever_received": dict((name, count > 0)
                                  for name, count in self._counts.items()),
            "skipped": dict(self._skipped)})

    def _bookkeeping(self):
        """Refresh the manifest and log a heartbeat naming the silent streams."""
        self.write_manifest()
        silent = sorted(n for n, c in self._counts.items()
                        if c == 0 and n not in self._skipped)
        self.get_logger().info(
            "nav_debug hb  samples={} counts={} silent={}".format(
                self.samples, dict(self._counts), silent or "none"))

    def _warn(self, key, text):
        """Log ``text`` at most once per ``WARN_INTERVAL_S`` per ``key``."""
        now = time.monotonic()
        if now - self._warned.get(key, -WARN_INTERVAL_S) < WARN_INTERVAL_S:
            return
        self._warned[key] = now
        self.get_logger().warn(text)

    def close(self):
        """Write the final manifest and close every stream file."""
        self.write_manifest(ended=True)
        self.writer.close()


def main(args=None):
    """Record until ``~duration_sec`` elapses or SIGINT/SIGTERM arrives."""
    rclpy.init(args=args)
    try:
        recorder = NavDebugRos2Recorder()
    except Exception:       # a misconfigured recorder still shuts rclpy down
        if rclpy.ok():
            rclpy.shutdown()
        raise
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
        recorder.close()
        recorder.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    return 0


if __name__ == "__main__":
    sys.exit(main())


# ============================================================================
# ROSPARAMS (defaults in parentheses):
#   rooster_id ("R1")            drone id every topic is built from
#   out_dir ("")                 run folder; "" -> $NAV_DEBUG_RUN_DIR, else a
#                                fresh nav_debug_<stamp> (see the writer module)
#   record_hz (20.0)             sample-and-hold cadence for actuator + truth
#   manifest_interval_s (10.0)   manifest refresh + heartbeat; <=0 disables
#   duration_sec (0.0)           self-terminate after this long; 0 = until signal
# ============================================================================
