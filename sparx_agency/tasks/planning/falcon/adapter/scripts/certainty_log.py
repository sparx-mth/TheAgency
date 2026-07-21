#!/usr/bin/env python3
"""certainty_log.py -- persist AprilTag/drift certainty to a per-tick CSV.

Helper module (imported by ``waypoint_follower_node.py``, not run as a node).
``thought_journal.py`` records WHY the stack acted, in prose, gated to one line
per decision. This module records HOW SURE it was and WHAT IT DID about it, one
row per control tick, so the two numbers that gate every drift_pid decision --
the AprilTag pose's own confidence and the controller's tracking/blockage state
-- can be plotted against the command that was actually sent to the drone.

Design mirrors :mod:`thought_journal` deliberately (same log directory, same
size cap, same in-process-writer reasoning); see that module for why. The
difference is shape: this is dense, un-gated, numeric telemetry, so it is a
CSV -- one row per tick -- rather than an edge-triggered prose line.

Python 3.8 compatible: the FALCON ROS1/Noetic adapter runs these scripts under
3.8 (see ``tasks/planning/falcon/run_falcon.sh``).
"""
import csv
import os
import time

from thought_journal import LOG_DIR_ENV

#: CSV column order. Kept in one place so the header and every row agree.
FIELDNAMES = [
    "wall_clock", "ros_stamp",
    "pos_x", "pos_y", "yaw_deg",
    "confidence", "pos_std_m", "cmd_effectiveness", "coasting", "age_s",
    "target_wp_idx", "num_waypoints", "target_x", "target_y",
    "drift_vx", "drift_vy", "drift_wz",
    "cross_track_m", "along_track_m", "heading_err_deg",
    "effort", "speed_scale", "lead_s", "deadband_extra_m",
    "authority", "blocked_axis", "escape_state",
    "cmd_vx", "cmd_vy", "cmd_wz",
]


def default_certainty_path(root=None, now=None):
    """Build a timestamped certainty-log path under ``root``.

    Args:
        root: Directory to write into. Defaults to ``$FALCON_LOG_DIR`` (the
            same variable ``thought_journal`` uses) and then to
            ``~/.ros/falcon``, so the certainty CSV always lands next to the
            thought journal for a run.
        now: ``time.struct_time`` to stamp the filename with (defaults to now).

    Returns:
        Absolute path of the form ``<root>/certainty_YYYYmmdd_HHMMSS.csv``.
    """
    base = (root or os.environ.get(LOG_DIR_ENV)
            or os.path.join(os.path.expanduser("~"), ".ros", "falcon"))
    stamp = time.strftime("%Y%m%d_%H%M%S", now or time.localtime())
    return os.path.join(base, "certainty_%s.csv" % stamp)


class CertaintyLog(object):
    """Append-only, size-capped, line-flushed CSV of certainty vs. command.

    Args:
        path: File to append to. Parent directories are created.
        max_bytes: Stop writing past this size (0 = unlimited). Checked before
            each write, so the file may exceed it by one row.
        wall_clock: Callable returning epoch seconds. Injected for tests.

    Raises:
        IOError / OSError: If the path cannot be opened. Logging is opt-in, so
            a caller that asked for it deserves to hear that it failed rather
            than fly on believing a log is being written.
    """

    def __init__(self, path, max_bytes=16 * 1024 * 1024, wall_clock=None):
        self.path = str(path)
        self.max_bytes = int(max_bytes)
        self._wall_clock = wall_clock or time.time
        self._written = 0
        self._rows = 0
        self._capped = False
        parent = os.path.dirname(self.path)
        if parent:
            try:
                os.makedirs(parent)
            except OSError:
                if not os.path.isdir(parent):
                    raise
        write_header = not os.path.exists(self.path) or os.path.getsize(self.path) == 0
        self._fh = open(self.path, "a")
        self._writer = csv.DictWriter(self._fh, fieldnames=FIELDNAMES)
        try:
            self._written = os.path.getsize(self.path)
        except OSError:
            self._written = 0
        if write_header:
            self._writer.writeheader()
            self._fh.flush()
            self._written = os.path.getsize(self.path) if os.path.exists(self.path) else self._written

    @property
    def rows(self):
        """Certainty rows written so far (excludes the header)."""
        return self._rows

    @property
    def capped(self):
        """True once ``max_bytes`` was reached and writing stopped."""
        return self._capped

    def write(self, ros_stamp, pose2d, quality, telemetry, target_xy,
              wp_idx, num_waypoints, cmd_vx, cmd_vy, cmd_wz):
        """Append one certainty row. Returns True if it was written.

        Args:
            ros_stamp: ROS time (seconds) of this control tick.
            pose2d: The drone's own ``Pose2D`` (x, y, yaw) this tick -- where it
                believes it is, independent of how much that belief is trusted.
            quality: The ``core.planning.trackers.drift_pid.LocalizationQuality``
                snapshot fed to the controller this tick -- the AprilTag pose's
                confidence, pos_std, coasting flag, cmd_effectiveness (the
                controller's certainty that its commands are reaching the
                world) and age.
            telemetry: The controller's ``DriftTelemetry`` for this tick -- the
                learned drift corrections it is applying, the tracking errors,
                the authority text and the blockage/escape state.
            target_xy: ``(x, y)`` of the waypoint being flown to this tick, or
                ``None`` if there is none (e.g. arrived/idle).
            wp_idx: Index of that waypoint (0-based), or ``None``.
            num_waypoints: Total waypoints in the current route, or ``None``.
            cmd_vx, cmd_vy, cmd_wz: The flight command actually sent this tick
                to reach the target waypoint, so a confidence dip can be
                matched against what the drone was told to do about it.
        """
        if self._capped:
            return False
        if self.max_bytes > 0 and self._written >= self.max_bytes:
            self._capped = True
            self._fh.write("# certainty log capped at %d bytes after %d rows\n"
                           % (self.max_bytes, self._rows))
            self._fh.flush()
            return False
        wall = self._wall_clock()
        target_x, target_y = target_xy if target_xy is not None else ("", "")
        row = {
            "wall_clock": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(wall)),
            "ros_stamp": "%.3f" % float(ros_stamp),
            "pos_x": "%.4f" % float(pose2d.x),
            "pos_y": "%.4f" % float(pose2d.y),
            "yaw_deg": "%.2f" % (float(pose2d.yaw) * 57.29577951308232),
            "confidence": "%.4f" % float(quality.confidence),
            "pos_std_m": "%.4f" % float(quality.pos_std_m),
            "cmd_effectiveness": "%.4f" % float(quality.cmd_effectiveness),
            "coasting": bool(quality.coasting),
            "age_s": "%.3f" % float(quality.age_s),
            "target_wp_idx": "" if wp_idx is None else int(wp_idx),
            "num_waypoints": "" if num_waypoints is None else int(num_waypoints),
            "target_x": "" if target_x == "" else "%.4f" % float(target_x),
            "target_y": "" if target_y == "" else "%.4f" % float(target_y),
            "drift_vx": "%.4f" % float(telemetry.drift_vx),
            "drift_vy": "%.4f" % float(telemetry.drift_vy),
            "drift_wz": "%.4f" % float(telemetry.drift_wz),
            "cross_track_m": "%.4f" % float(telemetry.cross_track_m),
            "along_track_m": "%.4f" % float(telemetry.along_track_m),
            "heading_err_deg": "%.2f" % (float(telemetry.heading_err_rad) * 57.29577951308232),
            "effort": "%.3f" % float(telemetry.effort),
            "speed_scale": "%.3f" % float(telemetry.speed_scale),
            "lead_s": "%.3f" % float(telemetry.lead_s),
            "deadband_extra_m": "%.3f" % float(telemetry.deadband_extra_m),
            "authority": telemetry.authority,
            "blocked_axis": telemetry.blocked_axis,
            "escape_state": telemetry.escape_state,
            "cmd_vx": "%.4f" % float(cmd_vx),
            "cmd_vy": "%.4f" % float(cmd_vy),
            "cmd_wz": "%.4f" % float(cmd_wz),
        }
        self._writer.writerow(row)
        self._fh.flush()
        self._rows += 1
        try:
            self._written = os.path.getsize(self.path)
        except OSError:
            pass
        return True

    def close(self):
        """Flush and close. Safe to call more than once."""
        if self._fh is None:
            return
        try:
            self._fh.close()
        finally:
            self._fh = None
