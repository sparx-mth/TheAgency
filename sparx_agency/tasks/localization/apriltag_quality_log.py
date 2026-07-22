"""apriltag_quality_log.py -- persist per-tag AprilTag quality to a CSV.

The certainty log (FALCON side) records how much the CONTROLLER trusted the pose
and what it did about it. This is its localization-side counterpart: it records,
per camera frame, WHICH tags were seen and HOW GOOD each one was -- so a tag that
is poorly placed (rarely in view, only ever small or grazing) or mis-mapped (its
recorded pose/size is wrong, so it drags every fix) can be found and fixed on the
wall, instead of hiding inside an aggregate confidence number.

Shape: one row per DETECTED tag per frame, with the frame's fix context repeated
on each of its rows -- so the file is tidy for "group by tag_id". A frame that
detected nothing (blind, or coasting) still writes ONE row with the tag columns
blank, so a stretch of lost tracking is visible as itself rather than as a gap.

To read it back, group by ``tag_id`` and look at: row count (how often the tag is
usable at all), ``decision_margin`` and ``apparent_px`` (how readable it is),
``tag_reproj_rms_px`` (how well its map entry matches reality -- the mis-map
signal), and ``used`` / ``in_map`` (whether it is trusted when seen).

Design mirrors ``certainty_log`` deliberately: append-only, size-capped,
line-flushed so a kill still leaves the record. Python 3.8 compatible.
"""
from __future__ import annotations

import csv
import os
import time

#: Env var naming the log directory, shared with the FALCON thought/certainty
#: logs so all of a run's logs land together. Falls back to /tmp/falcon, which is
#: where the operator already reads the other logs.
LOG_DIR_ENV = "FALCON_LOG_DIR"
_DEFAULT_DIR = os.path.join("/tmp", "falcon")

#: CSV column order. Frame context first, then the per-tag columns.
FIELDNAMES = [
    "wall_clock", "stamp",
    "source", "n_detected", "n_used",
    "fix_confidence", "pos_std_m", "geometry", "ambiguity", "reproj_rms_px",
    "tag_id", "in_map", "used",
    "decision_margin", "apparent_px", "center_x", "center_y",
    "dist_m", "tag_reproj_rms_px",
]


def default_apriltag_log_path(root=None, now=None):
    """Build a timestamped apriltag-quality-log path under ``root``.

    Args:
        root: Directory to write into. Defaults to ``$FALCON_LOG_DIR`` and then
            to ``/tmp/falcon``, so the tag log lands next to the flight's other
            logs.
        now: ``time.struct_time`` to stamp the filename with (defaults to now).

    Returns:
        Absolute path of the form ``<root>/apriltag_YYYYmmdd_HHMMSS.csv``.
    """
    base = root or os.environ.get(LOG_DIR_ENV) or _DEFAULT_DIR
    stamp = time.strftime("%Y%m%d_%H%M%S", now or time.localtime())
    return os.path.join(base, "apriltag_%s.csv" % stamp)


def _fmt(value, spec):
    """Format ``value`` with ``spec``, or '' when it is None."""
    return "" if value is None else spec % value


class AprilTagQualityLog(object):
    """Append-only, size-capped, line-flushed CSV of per-tag AprilTag quality.

    Args:
        path: File to append to. Parent directories are created.
        max_bytes: Stop writing past this size (0 = unlimited). Checked before
            each frame, so the file may exceed it by one frame's rows.
        wall_clock: Callable returning epoch seconds. Injected for tests.

    Raises:
        IOError / OSError: If the path cannot be opened. Logging is opt-in, so a
            caller that asked for it deserves to hear it failed rather than fly
            believing a log is being written.
    """

    def __init__(self, path, max_bytes=16 * 1024 * 1024, wall_clock=None):
        self.path = str(path)
        self.max_bytes = int(max_bytes)
        self._wall_clock = wall_clock or time.time
        self._rows = 0
        self._frames = 0
        self._capped = False
        parent = os.path.dirname(self.path)
        if parent:
            try:
                os.makedirs(parent)
            except OSError:
                if not os.path.isdir(parent):
                    raise
        write_header = (not os.path.exists(self.path)
                        or os.path.getsize(self.path) == 0)
        self._fh = open(self.path, "a")
        self._writer = csv.DictWriter(self._fh, fieldnames=FIELDNAMES)
        try:
            self._written = os.path.getsize(self.path)
        except OSError:
            self._written = 0
        if write_header:
            self._writer.writeheader()
            self._fh.flush()
            self._written = (os.path.getsize(self.path)
                             if os.path.exists(self.path) else self._written)

    @property
    def rows(self):
        """Per-tag rows written so far (excludes the header)."""
        return self._rows

    @property
    def frames(self):
        """Frames written so far."""
        return self._frames

    @property
    def capped(self):
        """True once ``max_bytes`` was reached and writing stopped."""
        return self._capped

    def write(self, diag):
        """Append one frame's rows (one per detected tag, or one blank-tag row).

        Args:
            diag: A ``apriltag_frame_diag.FrameDiag`` for this frame.

        Returns:
            Number of rows written (0 if the log is capped).
        """
        if self._capped:
            return 0
        if self.max_bytes > 0 and self._written >= self.max_bytes:
            self._capped = True
            self._fh.write("# apriltag quality log capped at %d bytes after %d "
                           "rows\n" % (self.max_bytes, self._rows))
            self._fh.flush()
            return 0

        wall = time.strftime("%Y-%m-%d %H:%M:%S",
                             time.localtime(self._wall_clock()))
        context = {
            "wall_clock": wall,
            "stamp": "%.3f" % float(diag.stamp_sec),
            "source": diag.source,
            "n_detected": int(diag.n_detected),
            "n_used": int(diag.n_used),
            "fix_confidence": "%.4f" % float(diag.confidence),
            "pos_std_m": "%.4f" % float(diag.pos_std_m),
            "geometry": "%.3f" % float(diag.geometry),
            "ambiguity": "%.3f" % float(diag.ambiguity),
            "reproj_rms_px": "%.3f" % float(diag.reproj_rms_px),
        }
        # A frame with no detected tags still gets a row, so a lost-tracking
        # stretch is a visible run of blank-tag rows, not an absence.
        observations = diag.tags or (None,)
        written = 0
        for obs in observations:
            row = dict(context)
            if obs is None:
                row.update({k: "" for k in (
                    "tag_id", "in_map", "used", "decision_margin", "apparent_px",
                    "center_x", "center_y", "dist_m", "tag_reproj_rms_px")})
            else:
                row.update({
                    "tag_id": int(obs.tag_id),
                    "in_map": int(bool(obs.in_map)),
                    "used": int(bool(obs.used)),
                    "decision_margin": "%.2f" % float(obs.decision_margin),
                    "apparent_px": "%.1f" % float(obs.apparent_px),
                    "center_x": "%.1f" % float(obs.center_x),
                    "center_y": "%.1f" % float(obs.center_y),
                    "dist_m": _fmt(obs.dist_m, "%.3f"),
                    "tag_reproj_rms_px": _fmt(obs.reproj_rms_px, "%.3f"),
                })
            self._writer.writerow(row)
            written += 1
        self._fh.flush()
        self._rows += written
        self._frames += 1
        try:
            self._written = os.path.getsize(self.path)
        except OSError:
            pass
        return written

    def close(self):
        """Flush and close. Safe to call more than once."""
        if self._fh is None:
            return
        try:
            self._fh.close()
        finally:
            self._fh = None
