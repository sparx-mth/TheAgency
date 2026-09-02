"""The command-chain column: from the velocity we ask for to the counts sent.

Top to bottom this column is one causal chain, so a wrong command can be blamed
on the stage that chose it:

  * **OURS** -- ``cmd_vel``, on the ROLL/PITCH/YAW gauges;
  * **CONTROL** -- the tracker's own split of that command into feed-forward,
    damping and correction, and what the envelope and rate limiter did to it;
  * **TO DRONE** -- either the XTEND ``cmd_nav`` gauge stack, or, when the run
    recorded per-axis traces, the Rooster actuator lane: requested vs measured
    speed and ``feed_forward + correction -> counts`` for each axis;
  * localization confidence and the history strips.

An XTEND run has neither ``terms`` nor ``axes``, so it renders exactly the two
gauge stacks it always did.
"""
from __future__ import annotations

import math
from typing import Optional, Tuple

import cv2
import numpy as np

from sparx_agency.tasks.planning.hud import gauges, palette
from sparx_agency.tasks.planning.hud.panel import hline, put_line, spark
from sparx_agency.tasks.planning.nav_debug import render_widgets as w
from sparx_agency.tasks.planning.nav_debug.frame import AxisTrace, GaugeScales, NavFrame

PANEL_WIDTH = 340
_CANVAS_H = 1400            # generous; the panel is cropped to its content
_G = gauges.GAUGE_SIZE

_FF = (200, 160, 90)        # the feed-forward segment of a count bar
_CORR = (90, 190, 240)      # the servo correction stacked on top of it
_SENT = palette.WHITE       # the count actually sent
_PRE_SLEW = (150, 150, 150)


def build_panel(frame: NavFrame, scales: GaugeScales,
                width: int = PANEL_WIDTH) -> np.ndarray:
    """Draw the command-chain column for ``frame``, cropped to its content."""
    panel = np.full((_CANVAS_H, width, 3), palette.PANEL_BG, dtype=np.uint8)
    x, y = 12, 26
    y = put_line(panel, "NAV DEBUG", x, y, palette.MUTED, 0.55)
    y = hline(panel, y, width)
    y = _ours(panel, x, y, width, frame, scales)
    y = hline(panel, y, width)
    # The Sphera-only sections are guarded (a malformed diagnostic must cost its
    # own lane and nothing else); the XTEND path stays exactly as it was.
    if frame.terms is not None:
        y = w.guarded(panel, x, y, width, "CONTROL", _control_terms, frame)
        y = hline(panel, y, width)
    # The actuator section owns the cmd_nav-vs-ManualControl comparison, so it
    # must be chosen on the actuator lane too -- not on the axis trace alone,
    # which empties whenever the adapter is stopped.
    y = (w.guarded(panel, x, y, width, "TO DRONE", _actuator, frame, scales)
         if (frame.axes or frame.actuator is not None)
         else _to_drone(panel, x, y, width, frame, scales))
    y = hline(panel, y, width)
    y = _quality(panel, x, y, width, frame)
    y = _strips(panel, x, y, width, frame)
    return panel[:min(y + 8, panel.shape[0])]


# ── the command we send ──────────────────────────────────────────────────────
def _ours(panel, x, y, width, frame: NavFrame, scales: GaugeScales) -> int:
    our = frame.our_cmd
    return _gauge_set(
        panel, x, y, "OURS (cmd_vel)", palette.GREEN,
        roll=(our[1] if our else 0.0), pitch=(our[0] if our else 0.0),
        yaw=(our[3] if our else 0.0),
        roll_fs=scales.our_vy, pitch_fs=scales.our_vx, yaw_fs=scales.our_wz,
        numbers=("vx%+.2f vy%+.2f" % (our[0], our[1]) if our else "no cmd",
                 "wz%+.2f vz%+.2f" % (our[3], our[2]) if our else ""))


def _to_drone(panel, x, y, width, frame: NavFrame, scales: GaugeScales) -> int:
    """The XTEND converter output, on gauges (used when no axis trace exists)."""
    d = frame.drone_cmd
    # Negate lateral & yaw counts so the gauges read the same direction as OURS.
    return _gauge_set(
        panel, x, y, "TO DRONE (cmd_nav)", palette.CYAN,
        roll=(-d[1] if d else 0.0), pitch=(d[0] if d else 0.0),
        yaw=(-d[3] if d else 0.0),
        roll_fs=scales.drone_lateral, pitch_fs=scales.drone_forward,
        yaw_fs=scales.drone_yaw,
        numbers=("fwd%d lat%d" % (d[0], d[1]) if d else "no cmd_nav",
                 "yaw%d vert%d" % (d[3], d[2]) if d else ""))


def _gauge_set(panel, x, y, title, color, roll, pitch, yaw, roll_fs, pitch_fs,
               yaw_fs, numbers) -> int:
    y = put_line(panel, title, x, y, color, 0.5)
    row = [("ROLL", gauges.draw_roll_gauge(roll, roll_fs, color)),
           ("PITCH", gauges.draw_pitch_gauge(pitch, pitch_fs, color)),
           ("YAW", gauges.draw_yaw_gauge(yaw, yaw_fs, color))]
    gap = 8
    gx, gy = x, y
    for label, g in row:
        panel[gy:gy + _G, gx:gx + _G] = g
        put_line(panel, label, gx + 2, gy + _G + 16, palette.MUTED, 0.42)
        gx += _G + gap
    y = gy + _G + 36        # clear the gauge labels before the numbers line
    for line in numbers:
        if line:
            y = put_line(panel, line, x, y, palette.TEXT, 0.45)
    return y


# ── the tracker's own split of that command ──────────────────────────────────
def _control_terms(panel, x, y, width, frame: NavFrame) -> int:
    """Feed-forward vs correction, and whether a limiter chose the output."""
    t = frame.terms
    limits = ",".join(t.limits) if t.limits else "none"
    y = w.section(panel, x, y, width, "CONTROL (tracker)", palette.GREEN,
                  note="limits %s" % limits)
    y = put_line(panel, "ff %s  damp %s  cor %s m/s" % (
        w.num(_mag(t.feed_forward)), w.num(_mag(t.damping)), w.num(_mag(t.correction))),
        x, y, palette.TEXT, 0.42)
    bound = _bound(t)
    y = put_line(panel, "cmd %s -> clamp %s -> out %s m/s" % (
        w.num(_mag(t.commanded)), w.num(_mag(t.clamped)), w.num(_mag(t.smoothed))),
        x, y, palette.AMBER if bound else palette.TEXT, 0.42)
    return y


def _mag(triple: Optional[Tuple[float, float, float]]) -> Optional[float]:
    """Horizontal magnitude of a world (x, y, z) velocity triple."""
    if triple is None:
        return None
    return math.hypot(w.finite(triple[0]), w.finite(triple[1]))


def _bound(terms) -> bool:
    """True when the envelope or the rate limiter -- not the controller -- won."""
    raw, out = _mag(terms.commanded), _mag(terms.smoothed)
    return raw is not None and out is not None and abs(raw - out) > 0.02


# ── the Rooster actuator lane ────────────────────────────────────────────────
def _actuator(panel, x, y, width, frame: NavFrame, scales: GaugeScales) -> int:
    """One row per axis: request vs measurement, and who chose the counts."""
    y = w.section(panel, x, y, width, "TO DRONE (rooster axes)", palette.CYAN)
    y = put_line(panel, "bar ff+corr    ticks pre-slew, sent", x, y - 6,
                 palette.MUTED, 0.38)
    for axis in frame.axes:
        y = _axis_row(panel, x, y, width, axis, scales)
    return _actuator_summary(panel, x, y, width, frame)


def _axis_row(panel, x, y, width, axis: AxisTrace, scales: GaugeScales) -> int:
    name = axis.name or "axis"
    speed_fs, count_fs = _axis_scales(name, scales)
    unit = "rad/s" if "yaw" in name.lower() else "m/s"
    put_line(panel, name.upper(), x, y, palette.CYAN, 0.45)
    w.put_right(panel, "req %+.2f  got %+.2f %s" % (
        w.finite(axis.requested), w.finite(axis.measured), unit), width - x, y,
        w.grade_color(w.finite(axis.requested) - w.finite(axis.measured),
                      0.15 * max(speed_fs, 1e-6), 0.4 * max(speed_fs, 1e-6)), 0.42)
    w.value_bar(panel, x, y + 6, width - 2 * x, 14,
                [(axis.feed_forward, _FF), (axis.correction, _CORR)], count_fs,
                markers=[(axis.pre_slew, _PRE_SLEW), (axis.counts, _SENT)])
    y += 38
    y = put_line(panel, "ff %+.0f  cor %+.0f  pre %+.0f  sent %+.0f" % (
        w.finite(axis.feed_forward), w.finite(axis.correction),
        w.finite(axis.pre_slew), w.finite(axis.counts)), x, y, palette.TEXT, 0.42)
    flags = [("SAT", axis.saturated, palette.ORANGE),
             ("SLEW", axis.slew_limited, palette.AMBER),
             ("CAP", axis.capped, palette.RED),
             ("STALE FB", axis.feedback_stale, palette.RED)]
    if any(active for _, active, _ in flags):
        y = w.chips(panel, x, y, flags, width - x)
    return y + 4


def _axis_scales(name: str, scales: GaugeScales) -> Tuple[float, float]:
    """(speed full-scale, count full-scale) for one named axis."""
    key = (name or "").lower()
    if "lat" in key:
        return scales.our_vy, scales.drone_lateral
    if "yaw" in key:
        return scales.our_wz, scales.drone_yaw
    if "vert" in key or key == "z":
        return scales.our_vz, scales.drone_vertical
    return scales.our_vx, scales.drone_forward


def _actuator_summary(panel, x, y, width, frame: NavFrame) -> int:
    """``cmd_nav`` as requested vs the ManualControl actually published.

    They differ whenever the altitude loop writes the throttle axis (expected)
    or a second publisher injects a command (the failure this catches).
    """
    act = frame.actuator
    if act is None:
        return y
    y = put_line(panel, "cmd_nav %s   age %.2fs" % (
        _triple(act.cmd_nav), w.finite(act.cmd_nav_age_s)), x, y, palette.MUTED, 0.42)
    y = put_line(panel, "manual  %s   age %.2fs" % (
        _triple(act.manual), w.finite(act.manual_age_s)), x, y, palette.TEXT, 0.42)
    return w.chips(panel, x, y, [
        ("MANUAL != CMD_NAV", _mismatch(act), palette.RED),
        ("CMD STALE", w.finite(act.cmd_nav_age_s) > 0.4, palette.ORANGE)], width - x)


def _triple(values) -> str:
    if not values:
        return "--"
    return " ".join("%+.0f" % w.finite(v) for v in values)


def _mismatch(act) -> bool:
    """True when the horizontal/yaw axes sent differ from the ones requested."""
    if not act.cmd_nav or not act.manual or len(act.manual) < 4:
        return False
    # Compare x, y, r only: z is the hold loop's own axis and never requested.
    sent = (act.manual[0], act.manual[1], act.manual[3])
    return any(abs(w.finite(a) - w.finite(b)) > 1.0
               for a, b in zip(act.cmd_nav, sent))


# ── quality + history strips ─────────────────────────────────────────────────
def _quality(panel, x, y, width, frame: NavFrame) -> int:
    q = frame.quality
    if q is None:
        return y
    y = _conf_bar(panel, x, y, width, q.confidence)
    return put_line(panel, "std %.2fm  age %.2fs  eff %.2f" % (
        q.pos_std_m, q.age_s, q.cmd_effectiveness), x, y, palette.MUTED, 0.45)


def _conf_bar(panel, x, y, width, conf) -> int:
    bar_w = width - 2 * x
    cv2.rectangle(panel, (x, y), (x + bar_w, y + 12), (60, 60, 60), 1)
    fill = int(round(max(0.0, min(1.0, w.finite(conf))) * (bar_w - 2)))
    if fill > 0:
        cv2.rectangle(panel, (x + 1, y + 1), (x + 1 + fill, y + 11),
                      w.conf_color(conf), -1)
    cv2.putText(panel, "conf %.2f" % w.finite(conf), (x + bar_w - 78, y + 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, palette.WHITE, 1, cv2.LINE_AA)
    return y + 22


def _strips(panel, x, y, width, frame: NavFrame) -> int:
    y += 4
    if frame.cmd_history:
        put_line(panel, "cmd vx", x, y + 10, palette.MUTED, 0.4)
        spark(panel, frame.cmd_history, x + 70, y, width - 90, 26, palette.GREEN)
        y += 34
    if frame.conf_history:
        put_line(panel, "conf", x, y + 10, palette.MUTED, 0.4)
        spark(panel, frame.conf_history, x + 70, y, width - 90, 26, palette.CYAN,
              lo=0.0, hi=1.0)
        y += 34
    return y
