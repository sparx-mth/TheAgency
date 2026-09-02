"""The frame spine: the per-tick rows every other lane is joined onto.

Two recordings can define it, and which one does is a property of the stack that
flew, not of the recording's completeness:

* the XTEND click-to-fly stack writes a per-tick **certainty CSV** (pose, both
  command sets, drift and localization quality), the richest spine there is;
* the Sphera exploration stack never writes one -- the AprilTag/drift-PID chain
  that produces it does not run under ``nav_mode:=exploration`` -- so the ROS1
  recorder's ``telemetry.jsonl`` (pose + ``/cmd_vel`` at ~15-20 Hz) *is* the
  timeline there, a first-class spine rather than a degraded fallback.

Both produce the same row dict, so :class:`~.session.NavSession` assembles
frames identically either way; the fields a source cannot know (drift, quality,
waypoint target) stay None and the renderer omits those panels.

The CSV's ``wall_clock`` column is a human-readable local-time string with
one-second resolution -- useless as a join key -- so CSV rows carry no ``wall``
and are placed on the host clock through the recorder's estimated offset
instead (see :class:`~.sources.ClockOffset`).
"""
from __future__ import annotations

import csv
import math
import os
from typing import List, Optional, Tuple

from sparx_agency.tasks.planning.nav_debug.frame import Drift, Quality
from sparx_agency.tasks.planning.nav_debug.schema import TELEMETRY_FILE
from sparx_agency.tasks.planning.nav_debug.sources import (
    as_of_index, read_jsonl, to_float,
)

CERTAINTY_CSV = "certainty_csv"
TELEMETRY = "telemetry"

_DEG = math.pi / 180.0
_EMPTY = {"drone": None, "conf": None, "quality": None, "drift": None,
          "wp_idx": None, "num_wp": None, "tx": None, "ty": None}


def load(run_dir: str, csv_path: Optional[str]) -> Tuple[List[dict], str]:
    """Load the spine for a run.

    Args:
        run_dir: The run folder.
        csv_path: A certainty CSV to prefer, or None to use the recorder's own
            telemetry.

    Returns:
        ``(rows, source)`` where ``source`` is :data:`CERTAINTY_CSV` or
        :data:`TELEMETRY`. ``rows`` is time-ordered and may be empty.
    """
    if csv_path and os.path.isfile(csv_path):
        rows = from_csv(csv_path)
        if rows:
            return rows, CERTAINTY_CSV
    return telemetry_rows(run_dir), TELEMETRY


def telemetry_rows(run_dir: str) -> List[dict]:
    """The ROS1 recorder's ``telemetry.jsonl`` as spine rows."""
    out = []
    for r in read_jsonl(os.path.join(run_dir, TELEMETRY_FILE)):
        t, x, y = to_float(r.get("t")), to_float(r.get("x")), to_float(r.get("y"))
        if t is None or x is None or y is None:
            continue
        row = dict(_EMPTY)
        row.update(t=t, wall=to_float(r.get("wall")), x=x, y=y,
                   z=to_float(r.get("z")), yaw=to_float(r.get("yaw")) or 0.0,
                   vx=to_float(r.get("vx")), vy=to_float(r.get("vy")),
                   vz=to_float(r.get("vz")), wz=to_float(r.get("wz")))
        out.append(row)
    out.sort(key=lambda d: d["t"])
    return out


def from_csv(path: str) -> List[dict]:
    """The flight node's per-tick certainty CSV as spine rows."""
    out = []
    try:
        handle = open(path, "r")
    except (OSError, IOError):
        return out
    with handle:
        for r in csv.DictReader(handle):
            row = _csv_row(r)
            if row is not None:
                out.append(row)
    out.sort(key=lambda d: d["t"])
    return out


def _csv_row(r: dict) -> Optional[dict]:
    """One certainty-CSV record -> a spine row, or None if it has no pose."""
    t, x, y = (to_float(r.get("ros_stamp")), to_float(r.get("pos_x")),
               to_float(r.get("pos_y")))
    if t is None or x is None or y is None:
        return None
    fwd = to_float(r.get("axis_forward"))
    return {
        "t": t, "wall": None, "x": x, "y": y, "z": to_float(r.get("pos_z")),
        "yaw": (to_float(r.get("yaw_deg")) or 0.0) * _DEG,
        "vx": to_float(r.get("cmd_vx")), "vy": to_float(r.get("cmd_vy")),
        "vz": None, "wz": to_float(r.get("cmd_wz")),
        "drone": None if fwd is None else (
            int(fwd), int(to_float(r.get("axis_lateral")) or 0),
            int(to_float(r.get("axis_vertical")) or 0),
            int(to_float(r.get("axis_yaw")) or 0)),
        "conf": to_float(r.get("confidence")),
        "quality": _quality(r), "drift": _drift(r),
        "wp_idx": to_float(r.get("target_wp_idx")),
        "num_wp": to_float(r.get("num_waypoints")),
        "tx": to_float(r.get("target_x")), "ty": to_float(r.get("target_y")),
    }


def _quality(r: dict) -> Quality:
    """Localization quality columns of one certainty-CSV record."""
    return Quality(
        confidence=to_float(r.get("confidence")) or 0.0,
        pos_std_m=to_float(r.get("pos_std_m")) or 0.0,
        cmd_effectiveness=to_float(r.get("cmd_effectiveness")) or 0.0,
        coasting=str(r.get("coasting", "")).lower() in ("true", "1"),
        age_s=to_float(r.get("age_s")) or 0.0,
        source=str(r.get("state", "")))


def _drift(r: dict) -> Drift:
    """Drift-PID columns of one certainty-CSV record."""
    return Drift(
        drift_vx=to_float(r.get("drift_vx")) or 0.0,
        drift_vy=to_float(r.get("drift_vy")) or 0.0,
        drift_wz=to_float(r.get("drift_wz")) or 0.0,
        cross_track_m=to_float(r.get("cross_track_m")) or 0.0,
        along_track_m=to_float(r.get("along_track_m")) or 0.0,
        heading_err_deg=to_float(r.get("heading_err_deg")) or 0.0,
        effort=to_float(r.get("effort")) or 0.0,
        speed_scale=to_float(r.get("speed_scale")) or 0.0,
        authority=str(r.get("authority", "")),
        state=str(r.get("state", "")),
        escape_state=str(r.get("escape_state", "")),
        blocked_axis=str(r.get("blocked_axis", "")))


def backfill_cmd_vz(rows: List[dict], telemetry: List[dict]) -> None:
    """Fill each row's missing ``vz`` from the recorder's telemetry, in place.

    The certainty CSV logs the drone's vertical axis count but not our own
    ``cmd_vel.linear.z``, so without this the OURS vertical gauge reads zero
    through a climb.
    """
    if not telemetry:
        return
    stamps = [d["t"] for d in telemetry]
    for row in rows:
        if row.get("vz") is not None:
            continue
        j = as_of_index(stamps, row["t"])
        if j is not None:
            row["vz"] = telemetry[j].get("vz")
