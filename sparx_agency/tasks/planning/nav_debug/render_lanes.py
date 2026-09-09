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


# ── the point being chased, and where the aircraft actually is ───────────────
#: Column right edges for the target/actual/error table, as offsets from the
#: lane's right margin. Right-aligned because the point of the table is that the
#: digits line up and the eye can subtract down a column.
_COL_TARGET = 160
_COL_ACTUAL = 85
_COL_ERROR = 12

#: (label, reference attribute, state attribute, formatter, converter). ``None``
#: for an attribute means that side does not carry the quantity at all.
_COMPARE = (
    ("x m", "x", "x", "%+.2f", None),
    ("y m", "y", "y", "%+.2f", None),
    ("z m", "z", "z", "%+.2f", None),
    ("yaw deg", "yaw", "yaw", "%+.0f", math.degrees),
    ("vx m/s", "vx", "vx", "%+.2f", None),
    ("vy m/s", "vy", "vy", "%+.2f", None),
    ("vz m/s", "vz", "vz", "%+.2f", None),
    ("yaw' r/s", "yaw_dot", "wz", "%+.2f", None),
)
#: Rows whose error is an angle and must be folded to (-pi, pi] before display.
_ANGULAR = ("yaw deg",)


def _reference_lane(panel, x, y, width, frame: NavFrame, scales) -> int:
    """What the aircraft was told to be doing, beside what it was doing.

    The reference and the measured state name the same six quantities -- where,
    how fast, which way round -- so they belong in one table with the error
    already subtracted, rather than as a number on one panel and a shape on the
    map. Everything here is in the **world** frame, which is the frame both
    ``/planning/pos_cmd`` and the odometry are published in; the ``cmd_vel``
    block on the other column is body frame and the two must not be read across.
    """
    ref, state = frame.reference, frame.state
    if ref is None and state is None:
        return w.absent(panel, x, y, width, "REFERENCE", "no /planning/pos_cmd")
    traj = "-" if ref is None or ref.traj_id is None else "traj %s" % ref.traj_id
    color = palette.CYAN if ref is not None and ref.moving else palette.GRAY
    y = w.section(panel, x, y, width, "REFERENCE vs ACTUAL (pos_cmd)", color,
                  note=traj)
    y = _compare_table(panel, x, y, width, ref, state)
    y = _compare_note(panel, x, y, width, ref, state)
    if ref is None:
        return y
    return w.chips(panel, x, y, [
        ("MOVING", bool(ref.moving), palette.GREEN),
        ("FROZEN ENDPOINT", not ref.moving, palette.GRAY),
        ("STALE", w.finite(ref.age_s) > _REF_STALE_S, palette.RED)], width - x)


def _compare_table(panel, x, y, width, ref, state) -> int:
    """The target/actual/error table, one row per quantity, world frame."""
    rights = (width - _COL_TARGET, width - _COL_ACTUAL, width - _COL_ERROR)
    y = w.table_row(panel, x, y, "", [(rights[0], "target", palette.MUTED),
                                      (rights[1], "actual", palette.MUTED),
                                      (rights[2], "err", palette.MUTED)])
    for label, ref_attr, state_attr, fmt, convert in _COMPARE:
        target = _value(ref, ref_attr, convert)
        actual = _value(state, state_attr, convert)
        error = _error(label, target, actual)
        y = w.table_row(panel, x, y, label, [
            (rights[0], w.num(target, fmt), palette.CYAN),
            (rights[1], w.num(actual, fmt), palette.TEXT),
            (rights[2], w.num(error, fmt), _error_color(label, error))])
    return y + 4


def _value(source, attr: str, convert):
    """One field off the reference or the state, converted for display."""
    if source is None:
        return None
    value = getattr(source, attr, None)
    if value is None:
        return None
    return convert(w.finite(value)) if convert else w.finite(value)


def _error(label: str, target, actual):
    """``actual - target``, folded to (-180, 180] on the heading row."""
    if target is None or actual is None:
        return None
    delta = actual - target
    if label in _ANGULAR:
        delta = math.degrees(math.atan2(math.sin(math.radians(delta)),
                                        math.cos(math.radians(delta))))
    return delta


def _error_color(label: str, error):
    """Grade the position rows tightly, the rest loosely; heading in degrees."""
    if error is None:
        return palette.MUTED
    if label in _ANGULAR:
        return w.grade_color(error, 15.0, 45.0)
    return w.grade_color(error, 0.3, 0.8)


def _compare_note(panel, x, y, width, ref, state) -> int:
    """Reference age and -- crucially -- which source the "actual" column used.

    Three sources can fill it and they are not interchangeable (see
    :mod:`.state_source`); a column that silently changed provenance mid-replay
    would be worse than no column.
    """
    age = "age %s s" % w.num(None if ref is None else ref.age_s)
    source = "actual: %s" % ((state.source or "?") if state is not None else "--")
    return put_line(panel, "%s   %s" % (age, source), x, y, palette.MUTED, 0.4)


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
    """A short trailing series under its label; nothing at all when empty.

    The 26 px plot is drawn from ``y``, so the next baseline must clear
    ``y + 26`` plus the following line's ascender -- at 34 a flat series drew
    straight through the text beneath it.
    """
    if not series:
        return y
    put_line(panel, label, x, y + 10, palette.MUTED, 0.4)
    spark(panel, series, x + 56, y, width - x - 68, 26, color)
    return y + 40


#: Lane order, top to bottom: what was asked, how it went, the vertical lane,
#: the ground truth, and the map it was all planned on.
_LANES = (("REFERENCE", _reference_lane), ("TRACKING", _tracking_lane),
          ("ALTITUDE", _altitude_lane), ("TRUTH", _truth_lane), ("MAP", _map_lane))
