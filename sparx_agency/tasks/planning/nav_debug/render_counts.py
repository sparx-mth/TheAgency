"""What the drone is actually flown on: the joystick counts, axis by axis.

The last hop before the airframe, and the one that used to have nowhere to be
drawn. ``TO DRONE`` reads ``frame.drone_cmd`` (the counts published) beside
``frame.actuator.cmd_nav`` (the counts requested); ``AXES`` reads the twist
adapter's own per-axis trace, which exists only when the ROS2 recorder ran.

These were previously an either/or inside :mod:`.render_panel`: the count gauges
were drawn only when the axis trace was *absent*, i.e. exactly when there were
no counts to draw either, and were replaced by the servo internals whenever the
ROS2 half was present. They are two different questions -- "what was the drone
told" and "how did the servo arrive at it" -- so they are now two sections, and
the first one is always drawn.
"""
from __future__ import annotations

from typing import Optional, Tuple

from sparx_agency.tasks.planning.hud import gauges, palette
from sparx_agency.tasks.planning.hud.panel import put_line
from sparx_agency.tasks.planning.nav_debug import render_widgets as w
from sparx_agency.tasks.planning.nav_debug.frame import AxisTrace, GaugeScales, NavFrame

_FF = (200, 160, 90)        # the feed-forward segment of a count bar
_CORR = (90, 190, 240)      # the servo correction stacked on top of it
_SENT = palette.WHITE       # the count actually sent
_PRE_SLEW = (150, 150, 150)


# ── the counts the drone is actually flown on ────────────────────────────────
#: ``ManualControl`` axis order, and the label each one flies under. ``z`` is the
#: throttle: the planner never requests it, so its "requested" cell is always a
#: dash -- the altitude hold owns that axis (see the ALTITUDE lane).
_AXES = (("forward", 0, "drone_forward"), ("lateral", 1, "drone_lateral"),
         ("vertical", 2, "drone_vertical"), ("yaw", 3, "drone_yaw"))
#: cmd_nav carries three axes, ``[x, y, r]``; ManualControl carries four. This
#: maps a ManualControl index to its cmd_nav index, or None for the throttle.
_REQ_INDEX = {0: 0, 1: 1, 2: None, 3: 2}


def to_drone(panel, x, y, width, frame: NavFrame, scales: GaugeScales) -> int:
    """The joystick command: the two sticks, then the counts axis by axis.

    The sticks answer "what would a pilot's hands be doing", the table answers
    "how hard, and did what was asked for arrive". Both are drawn on every
    stack: the counts are the last thing that happens before the airframe, so a
    run that recorded them should never have to choose between showing them and
    showing the servo internals.
    """
    if frame.drone_cmd is None and frame.actuator is None:
        return _no_counts(panel, x, y, width, frame)
    y = put_line(panel, "TO DRONE (cmd_nav)", x, y, palette.CYAN, 0.5)
    y = _sticks(panel, x, y, width, frame.drone_cmd, scales)
    return _count_table(panel, x, y, width, frame, scales)


def _sticks(panel, x, y, width, counts, scales: GaugeScales) -> int:
    """The transmitter, Mode 2: throttle/yaw on the left, forward/lateral on the right.

    Four axes on two sticks rather than three axes on three needle gauges. The
    throttle was the axis with no gauge at all, and it is the one the altitude
    hold writes underneath the planner -- so it was the least visible axis on a
    screen whose whole job is visibility.

    Nothing is drawn without counts: a stick resting at centre is a *zero*
    command, which is not what "never recorded" means.
    """
    if not counts or len(counts) < 4:
        return y
    fwd, lat, vert, yaw = (w.finite(v) for v in counts[:4])
    pair = ((gauges.draw_stick(yaw, vert, scales.drone_yaw, scales.drone_vertical),
             "throttle / yaw"),
            (gauges.draw_stick(lat, fwd, scales.drone_lateral, scales.drone_forward),
             "fwd / lateral"))
    size, gap = gauges.STICK_SIZE, 12
    gx = x
    for stick, label in pair:
        panel[y:y + size, gx:gx + size] = stick
        put_line(panel, label, gx + 2, y + size + 15, palette.MUTED, 0.4)
        gx += size + gap
    return y + size + 34


def _no_counts(panel, x, y, width, frame: NavFrame) -> int:
    """Why the counts are missing -- which is never the same as "they were zero".

    On Sphera the counts exist only in the ROS2 half of the recording, so the
    usual answer is that the second recorder was never started. Saying so, and
    naming the script that starts it, is the difference between a debug screen
    and a blank one. Split across lines because the column is 316 px wide and
    :func:`render_widgets.absent` does not wrap.
    """
    y = w.absent(panel, x, y, width, "TO DRONE", "no cmd_nav recorded")
    if frame.reference is None and frame.terms is None:
        return y
    y = put_line(panel, "the counts live in the ros2/ lane,", x, y, palette.MUTED, 0.4)
    return put_line(panel, "recorded by run_nav_debug_recorder.sh",
                    x, y, palette.MUTED, 0.4)


def _count_table(panel, x, y, width, frame: NavFrame, scales: GaugeScales) -> int:
    """One row per axis: requested counts, counts sent, and the fraction of full."""
    req = getattr(frame.actuator, "cmd_nav", None)
    sent = frame.drone_cmd
    cols = (width - 60, width - 12)
    y = w.table_row(panel, x, y, "axis", [
        (cols[0] - 46, "req", palette.MUTED), (cols[0], "sent", palette.MUTED),
        (cols[1], "%FS", palette.MUTED)])
    for name, index, scale_attr in _AXES:
        asked = _req_count(req, index)
        got = None if not sent or index >= len(sent) else w.finite(sent[index])
        full = max(w.finite(getattr(scales, scale_attr), 1.0), 1.0)
        y = w.table_row(panel, x, y, name, [
            (cols[0] - 46, w.num(asked, "%+.0f"), palette.MUTED),
            (cols[0], w.num(got, "%+.0f"), palette.TEXT),
            (cols[1], "--" if got is None else "%+.0f%%" % (100.0 * got / full),
             _count_color(asked, got))])
    return _actuator_summary(panel, x, y, width, frame)


def _req_count(req, index: int) -> Optional[float]:
    """The cmd_nav request for a ManualControl axis, or None (the throttle)."""
    cmd_index = _REQ_INDEX.get(index)
    if cmd_index is None or not req or cmd_index >= len(req):
        return None
    return w.finite(req[cmd_index])


def _count_color(asked, got):
    """Red when the axis was asked for one thing and sent another.

    Only the three axes the planner actually requests can disagree; the throttle
    has no request to disagree with, so it stays neutral rather than alarming.
    """
    if asked is None or got is None:
        return palette.TEXT
    return palette.RED if abs(asked - got) > 1.0 else palette.TEXT



# ── the Rooster velocity servo's own internals ───────────────────────────────
def axis_servo(panel, x, y, width, frame: NavFrame, scales: GaugeScales) -> int:
    """One row per axis: request vs measurement, and who chose the counts."""
    y = w.section(panel, x, y, width, "AXES (velocity servo)", palette.CYAN)
    y = put_line(panel, "bar ff+corr    ticks pre-slew, sent", x, y - 6,
                 palette.MUTED, 0.38)
    for axis in frame.axes:
        y = _axis_row(panel, x, y, width, axis, scales)
    return y


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
