"""The Sphera/FALCON outcome column: is the aircraft flying the plan, and why not.

Where :mod:`.render_panel` follows the command *down* the chain, this column
reads the result *back*: the reference being chased, the tracker's verdict on
how well it is being chased, the vertical lane nobody owns, Sphera's own ground
truth, and the map the plan was made on.

Each lane draws only when its data is present and collapses to a single dim line
when it is not, so a run that recorded three of the five still reads as a
column rather than as a wall of dashes. A lane that raises degrades to that same
line: a debug view must never take a live viewer down with it.
"""
from __future__ import annotations

import math
from typing import Optional

import numpy as np

from sparx_agency.tasks.planning.hud import palette
from sparx_agency.tasks.planning.hud.panel import hline, put_line, spark
from sparx_agency.tasks.planning.nav_debug import render_widgets as w
from sparx_agency.tasks.planning.nav_debug.frame import GaugeScales, NavFrame

LANE_WIDTH = 360
_CANVAS_H = 1600            # generous; the column is cropped to its content

_ERR_FS_M = 1.5             # full scale of the tracking error bars, metres
_ALT_ERR_FS_M = 0.5         # full scale of the altitude error bar, metres
_REF_STALE_S = 0.5          # past this the tracker holds instead of chasing
_LAG = (200, 160, 90)       # lag is benign: never graded red
#: Known open defect: ~14% of flights climb past 2 m. Anything above this is
#: called out as a runaway rather than left to be read off a number.
_RUNAWAY_M = 2.0


def has_lane_data(frame: NavFrame) -> bool:
    """True when the run recorded any Sphera lane, so the column is worth drawing."""
    return any((frame.reference, frame.tracking, frame.altitude, frame.truth,
                frame.map_stats, frame.actuator, frame.axes))


def build_lane_column(frame: NavFrame, scales: GaugeScales,
                      width: int = LANE_WIDTH) -> np.ndarray:
    """Draw the outcome column for ``frame``, cropped to its content."""
    panel = np.full((_CANVAS_H, width, 3), palette.PANEL_BG, dtype=np.uint8)
    x, y = 12, 26
    y = put_line(panel, "SPHERA / FALCON", x, y, palette.MUTED, 0.55)
    y = hline(panel, y, width)
    for title, draw in _LANES:
        y = w.guarded(panel, x, y, width, title, draw, frame, scales) + 8
    return panel[:min(y + 8, panel.shape[0])]


# ── the point being chased ───────────────────────────────────────────────────
def _reference_lane(panel, x, y, width, frame: NavFrame, scales) -> int:
    ref = frame.reference
    if ref is None:
        return w.absent(panel, x, y, width, "REFERENCE", "no /planning/pos_cmd")
    traj = "-" if ref.traj_id is None else str(ref.traj_id)
    y = w.section(panel, x, y, width, "REFERENCE (pos_cmd)",
                  palette.CYAN if ref.moving else palette.GRAY, note="traj %s" % traj)
    y = put_line(panel, "xy %s %s   z %s" % (
        w.num(ref.x, "%+.2f"), w.num(ref.y, "%+.2f"), w.num(ref.z, "%+.2f")),
        x, y, palette.TEXT, 0.45)
    yaw = "--" if ref.yaw is None else "%+.0f deg" % math.degrees(w.finite(ref.yaw))
    y = put_line(panel, "v %s m/s   yaw %s   age %s s" % (
        w.num(ref.speed), yaw, w.num(ref.age_s)), x, y, palette.TEXT, 0.45)
    return w.chips(panel, x, y, [
        ("MOVING", bool(ref.moving), palette.GREEN),
        ("FROZEN ENDPOINT", not ref.moving, palette.GRAY),
        ("STALE", w.finite(ref.age_s) > _REF_STALE_S, palette.RED)], width - x)


# ── how well it is being chased ──────────────────────────────────────────────
def _tracking_lane(panel, x, y, width, frame: NavFrame, scales) -> int:
    tr = frame.tracking
    if tr is None:
        return w.absent(panel, x, y, width, "TRACKING", "no control trace")
    y = w.section(panel, x, y, width, "TRACKING", _track_color(tr),
                  note="ref age %s s" % w.num(tr.reference_age_s))
    y = w.labelled_bar(panel, x, y, width, "pos err", tr.position_error_m, _ERR_FS_M,
                       w.grade_color(tr.position_error_m, 0.3, 0.8),
                       text=w.num(tr.position_error_m), zero=0.0)
    # Lag means late and is benign; cross-track means somewhere else, and only
    # that one flies into walls -- so only cross-track is ever graded red.
    y = w.labelled_bar(panel, x, y, width, "lag late", tr.along_track_lag_m,
                       _ERR_FS_M, _LAG)
    y = w.labelled_bar(panel, x, y, width, "x-track", tr.cross_track_error_m,
                       _ERR_FS_M, w.grade_color(tr.cross_track_error_m, 0.15, 0.4))
    y = put_line(panel, "yaw err %+.0f deg" % math.degrees(w.finite(tr.yaw_error_rad)),
                 x, y, palette.TEXT, 0.45)
    y = w.chips(panel, x, y, [("DIVERGED", bool(tr.diverged), palette.RED),
                              ("HOLDING", bool(tr.holding), palette.AMBER)], width - x)
    return _sparkline(panel, x, y, width, "err", frame.err_history, _track_color(tr))


def _track_color(tr):
    if tr.diverged:
        return palette.RED
    return w.grade_color(tr.position_error_m, 0.3, 0.8)


# ── the vertical lane, which no single process owns ──────────────────────────
def _altitude_lane(panel, x, y, width, frame: NavFrame, scales: GaugeScales) -> int:
    alt = frame.altitude
    if alt is None:
        return w.absent(panel, x, y, width, "ALTITUDE", "no altitude trace")
    rejects = int(alt.guard_rejects_total or 0)
    # A tick the loop skipped leaves error/wanted_z/sent_z null; without naming
    # the reason the lane reads exactly like a healthy hold at zero error.
    skipped = bool(alt.reason) and alt.reason != "held"
    note = alt.reason.replace("_", " ") if skipped else (
        "rejects %d" % rejects if rejects else "nudge %s m"
        % w.num(alt.nudge_m, "%+.2f"))
    y = w.section(panel, x, y, width, "ALTITUDE (ranger)", _alt_color(alt), note=note)
    y = put_line(panel, "target %s   ranger %s m" % (
        w.num(alt.target_m), w.num(alt.ranger_m)), x, y, _alt_color(alt), 0.45)
    y = w.labelled_bar(panel, x, y, width, "err", w.finite(alt.error_m), _ALT_ERR_FS_M,
                       w.grade_color(alt.error_m, 0.1, 0.25),
                       text=w.num(alt.error_m, "%+.2f"))
    y = _altitude_counts(panel, x, y, width, alt, scales)
    return _altitude_alerts(panel, x, y, width, alt, rejects)


def _altitude_counts(panel, x, y, width, alt, scales: GaugeScales) -> int:
    """What the hold loop wanted on the throttle axis vs what the gate let past."""
    stepped = (alt.wanted_z is not None and alt.sent_z is not None
               and abs(w.finite(alt.wanted_z) - w.finite(alt.sent_z)) > 1.0)
    y = put_line(panel, "wanted_z %s   sent_z %s%s" % (
        w.num(alt.wanted_z, "%.0f"), w.num(alt.sent_z, "%.0f"),
        "   [step-gated]" if stepped else ""),
        x, y, palette.AMBER if stepped else palette.TEXT, 0.45)
    w.value_bar(panel, x, y - 12, width - 2 * x, 12,
                [(w.finite(alt.sent_z), palette.CYAN)], scales.drone_vertical,
                markers=[(w.finite(alt.wanted_z), palette.WHITE)], zero=0.0)
    return y + 14


def _altitude_alerts(panel, x, y, width, alt, rejects: int) -> int:
    """Two silent failures made loud: the plausibility gate and the ceiling."""
    if alt.guard_rejected:
        y = w.alert(panel, x, y, width,
                    "RANGER GUARD REJECTED  (%d this run)" % rejects, palette.RED)
    if w.finite(alt.ranger_m, 0.0) > _RUNAWAY_M:
        y = w.alert(panel, x, y, width, "ALTITUDE RUNAWAY  ranger %s m"
                    % w.num(alt.ranger_m), palette.RED)
    return w.chips(panel, x, y, [
        ("AT CEILING", bool(alt.at_ceiling), palette.ORANGE),
        ("GUARD x%d" % rejects, rejects > 0, palette.RED)], width - x)


def _alt_color(alt):
    if alt.reason and alt.reason not in ("held", ""):
        return palette.AMBER        # the hold loop did not run this tick
    if alt.guard_rejected or w.finite(alt.ranger_m, 0.0) > _RUNAWAY_M:
        return palette.RED
    if alt.at_ceiling:
        return palette.ORANGE
    return w.grade_color(alt.error_m, 0.1, 0.25)


# ── what the aircraft actually did ───────────────────────────────────────────
def _truth_lane(panel, x, y, width, frame: NavFrame, scales: GaugeScales) -> int:
    t = frame.truth
    if t is None:
        return w.absent(panel, x, y, width, "TRUTH", "no /R1 ground truth")
    y = w.section(panel, x, y, width, "TRUTH (sphera)", palette.TEXT,
                  note=t.flight_mode or t.status or "-")
    y = _speed_row(panel, x, y, width, frame, scales)
    y = _sparkline(panel, x, y, width, "speed", frame.speed_history, palette.GREEN)
    y = put_line(panel, "roll %s   pitch %s deg" % (
        w.num(_deg(t.roll), "%+.0f"), w.num(_deg(t.pitch), "%+.0f")),
        x, y, _attitude_color(t), 0.45)
    # Graded on how much charge is *gone*, so an unrecorded battery reads green.
    gone = 100.0 - w.finite(t.battery_pct, 100.0)
    y = w.labelled_bar(panel, x, y, width, "battery", w.finite(t.battery_pct), 100.0,
                       w.grade_color(gone, 60.0, 80.0),
                       text=w.num(t.battery_pct, "%.0f%%"), zero=0.0)
    return w.chips(panel, x, y, [("ARMED", bool(t.armed), palette.GREEN),
                                 ("DISARMED", t.armed is False, palette.RED)],
                   width - x)


def _speed_row(panel, x, y, width, frame: NavFrame, scales: GaugeScales) -> int:
    """Achieved speed with the commanded speed marked on the same bar."""
    achieved = frame.truth.speed
    wanted = _commanded_speed(frame)
    full = max(w.finite(scales.our_vx, 1.0), 0.1)
    y = put_line(panel, "speed %s  /  cmd %s m/s" % (w.num(achieved), w.num(wanted)),
                 x, y, _speed_color(achieved, wanted), 0.45)
    markers = [] if wanted is None else [(wanted, palette.WHITE)]
    w.value_bar(panel, x, y - 12, width - 2 * x, 12,
                [(w.finite(achieved), _speed_color(achieved, wanted))], full,
                markers=markers, zero=0.0)
    return y + 14


def _commanded_speed(frame: NavFrame) -> Optional[float]:
    if not frame.our_cmd:
        return None
    return math.hypot(w.finite(frame.our_cmd[0]), w.finite(frame.our_cmd[1]))


def _speed_color(achieved, wanted):
    """Grade the gap between what was asked for and what the airframe did."""
    if achieved is None or wanted is None:
        return palette.TEXT
    return w.grade_color(w.finite(wanted) - w.finite(achieved), 0.15, 0.4)


def _attitude_color(truth):
    worst = max(abs(w.finite(_deg(truth.roll))), abs(w.finite(_deg(truth.pitch))))
    return w.grade_color(worst, 15.0, 25.0)


def _deg(rad) -> Optional[float]:
    return None if rad is None else math.degrees(w.finite(rad))


# ── the map the plan was made on ─────────────────────────────────────────────
def _map_lane(panel, x, y, width, frame: NavFrame, scales) -> int:
    m = frame.map_stats
    if m is None:
        return w.absent(panel, x, y, width, "MAP", "no mapping stats")
    y = w.section(panel, x, y, width, "MAP", palette.MUTED, note=m.gate_state or "-")
    y = put_line(panel, "frames %s  emit %s  drop %s %s" % (
        w.num(m.depth_frames, "%.0f"), w.num(m.emitted, "%.0f"),
        w.num(m.dropped, "%.0f"), m.drop_reason or ""),
        x, y, palette.TEXT, 0.42)
    outside = ("-" if m.outside_bbox_frac is None
               else "%.0f%%" % (100.0 * w.finite(m.outside_bbox_frac)))
    y = put_line(panel, "occ %s  free %s  unk %s  outside %s" % (
        w.num(m.occupied_cells, "%.0f"), w.num(m.free_cells, "%.0f"),
        w.num(m.unknown_cells, "%.0f"), outside), x, y, palette.TEXT, 0.42)
    return put_line(panel, "pose age %s s  depth age %s s  tilt %s deg" % (
        w.num(m.pose_age_s), w.num(m.depth_age_s), w.num(m.tilt_deg, "%.0f")),
        x, y, palette.MUTED, 0.42)


def _sparkline(panel, x, y, width, label, series, color) -> int:
    """A short trailing series under its label; nothing at all when empty."""
    if not series:
        return y
    put_line(panel, label, x, y + 10, palette.MUTED, 0.4)
    spark(panel, series, x + 56, y, width - x - 68, 26, color)
    return y + 34


#: Lane order, top to bottom: what was asked, how it went, the vertical lane,
#: the ground truth, and the map it was all planned on.
_LANES = (("REFERENCE", _reference_lane), ("TRACKING", _tracking_lane),
          ("ALTITUDE", _altitude_lane), ("TRUTH", _truth_lane), ("MAP", _map_lane))
