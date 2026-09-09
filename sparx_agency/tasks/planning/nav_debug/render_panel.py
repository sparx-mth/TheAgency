"""The command-chain column: from the velocity we ask for to the counts sent.

Top to bottom this column is one causal chain, so a wrong command can be blamed
on the stage that chose it:

  * **OURS (cmd_vel)** -- the twist handed to the velocity-closing block, on the
    ROLL/PITCH/YAW gauges and as an explicit linear + angular vector, with the
    module that produces it and the module that consumes it named on screen;
  * **CONTROL** -- the tracker's own split of that command into feed-forward,
    damping and correction, and what the envelope and rate limiter did to it;
  * **TO DRONE** -- the joystick counts, axis by axis;
  * **AXES** -- when the twist adapter recorded them, the velocity servo's own
    internals: requested vs measured speed and ``ff + cor -> counts`` per axis;
  * localization confidence and the history strips.

An XTEND run has neither ``terms`` nor ``axes``, so it renders the two gauge
stacks it always did, now with the counts spelled out beneath them.

The command chain, named
------------------------

``OURS (cmd_vel)`` was a bare title for a long time, and "ours" is ambiguous the
moment more than one node can publish a velocity. It is exactly one topic::

    /planning/pos_cmd  (traj_server, 100 Hz reference)
        |
    falcon_exploration_follower_node   ReferenceTracker3D + PulseShaper
        |   command_requested  = the tracker's demand, before the shaper
        |   command            = what it published, after the shaper
        v
    <drone_ns>/cmd_vel_raw
        |
    cmd_vel_gate_node                  the GO gate: passes the twist, or zeroes it
        v
    <drone_ns>/cmd_vel   <-- OURS (cmd_vel): recorded in telemetry.jsonl
        |   (ROS1 -> ROS2, bridge.yaml)
        v
    rooster_twist_control_adapter      THE VELOCITY-CLOSING BLOCK
        expo feed-forward -> PI servo on /R1/velocity_truth -> slew -> cap
        v
    /R1/cmd_nav  ->  rooster_command_unit  ->  /R1/manual_control  ->  Sphera

So ``OURS`` is the **input vector of the velocity loop** -- the last form the
command takes as a velocity, one hop before it becomes stick counts. The three
other copies of it (the tracker's demand, what the follower published, and what
the adapter says it received) are drawn beside it only when they *disagree* with
it, because the disagreement is the whole information: the shaper clipping a
tick, the GO gate swallowing one, or the bridge staling one are otherwise
invisible.
"""
from __future__ import annotations

import math
from typing import Optional, Tuple

import cv2
import numpy as np

from sparx_agency.tasks.planning.hud import palette
from sparx_agency.tasks.planning.hud.panel import hline, put_line, spark
from sparx_agency.tasks.planning.nav_debug import render_counts, render_widgets as w
from sparx_agency.tasks.planning.nav_debug.frame import GaugeScales, NavFrame

PANEL_WIDTH = 340
_CANVAS_H = 1400            # generous; the panel is cropped to its content

def build_panel(frame: NavFrame, scales: GaugeScales,
                width: int = PANEL_WIDTH) -> np.ndarray:
    """Draw the command-chain column for ``frame``, cropped to its content."""
    panel = np.full((_CANVAS_H, width, 3), palette.PANEL_BG, dtype=np.uint8)
    x, y = 12, 26
    y = put_line(panel, "NAV DEBUG", x, y, palette.MUTED, 0.55)
    y = hline(panel, y, width)
    y = w.guarded(panel, x, y, width, "OURS", _ours, frame, scales)
    y = hline(panel, y, width)
    # The Sphera-only sections are guarded (a malformed diagnostic must cost its
    # own lane and nothing else); the XTEND path stays exactly as it was.
    if frame.terms is not None:
        y = w.guarded(panel, x, y, width, "CONTROL", _control_terms, frame)
        y = hline(panel, y, width)
    # TO DRONE is drawn unconditionally. It used to be an either/or with the
    # rooster-axes section, which meant the cmd_nav counts had no home: without
    # the ROS2 half there were none to draw, and with it the block was replaced.
    y = w.guarded(panel, x, y, width, "TO DRONE", render_counts.to_drone,
                  frame, scales)
    y = hline(panel, y, width)
    if frame.axes:
        y = w.guarded(panel, x, y, width, "AXES", render_counts.axis_servo,
                      frame, scales)
        y = hline(panel, y, width)
    y = _quality(panel, x, y, width, frame)
    y = _strips(panel, x, y, width, frame)
    return panel[:min(y + 8, panel.shape[0])]


# ── the command we send ──────────────────────────────────────────────────────
#: Who writes ``<drone_ns>/cmd_vel`` and who reads it. Drawn on screen, because
#: a debug view that names a value but not its endpoints leaves the reader to
#: grep for them (see the module docstring for the full chain).
_CMD_VEL_FROM = "from falcon_exploration_follower (GO gate)"
_CMD_VEL_INTO = "into rooster_twist_control_adapter"

_GATE_EPS = 0.02        # m/s (rad/s) below which two stages are the same tick
_MOVING_EPS = 0.05      # above this the follower was asking for real motion


def _ours(panel, x, y, width, frame: NavFrame, scales: GaugeScales) -> int:
    """The velocity-loop input vector: gauges, the numbers, then who owns it."""
    our = frame.our_cmd
    y = w.gauge_set(
        panel, x, y, "OURS (cmd_vel)", palette.GREEN,
        roll=(our[1] if our else 0.0), pitch=(our[0] if our else 0.0),
        yaw=(our[3] if our else 0.0),
        roll_fs=scales.our_vy, pitch_fs=scales.our_vx, yaw_fs=scales.our_wz,
        numbers=())
    y = _target_vector(panel, x, y, width, our)
    y = _elsewhere(panel, x, y, width, frame, our)
    y = put_line(panel, _CMD_VEL_FROM, x, y, palette.MUTED, 0.4)
    return put_line(panel, _CMD_VEL_INTO, x, y, palette.MUTED, 0.4)


def _target_vector(panel, x, y, width, our) -> int:
    """The target twist in full: the linear vector, then the angular one.

    Split into ``lin`` and ``ang`` rather than run together, because they are
    different physical quantities in different units, and the yaw rate is the
    one a reader most often wants in degrees.
    """
    if not our:
        return w.absent(panel, x, y, width, "target", "no cmd_vel recorded")
    y = put_line(panel, "lin vx%+.2f vy%+.2f vz%+.2f m/s" % (
        w.finite(our[0]), w.finite(our[1]), w.finite(our[2])),
        x, y, palette.TEXT, 0.42)
    wz = w.finite(our[3])
    y = put_line(panel, "ang wz%+.3f rad/s (%+.0f deg/s)" % (wz, math.degrees(wz)),
                 x, y, palette.TEXT, 0.42)
    return put_line(panel, "|v| %.2f m/s   body frame" % math.hypot(
        w.finite(our[0]), w.finite(our[1])), x, y, palette.MUTED, 0.42)


def _elsewhere(panel, x, y, width, frame: NavFrame, our) -> int:
    """The same command as seen at the other three points on the chain.

    ``velocity_requested`` is the tracker's demand, ``velocity_target`` is what
    the follower published on ``cmd_vel_raw``, ``our`` is what came out of the GO
    gate, and ``velocity_received`` is what the velocity servo says it acted on.
    They agree on a healthy tick and this row stays silent; a difference names
    the stage that took the command away.

    The three comparisons are deliberately not symmetric. The shaper and the
    bridge are compared term by term, because both make small changes. The gate
    is not: it either passes a twist or zeroes it, so it is only reported when
    the follower asked for real motion and nothing came out -- a test that
    survives the timing slop of joining lanes recorded at different rates.
    """
    sent = frame.velocity_target
    if sent is None:
        return y
    demand, got = frame.velocity_requested, frame.velocity_received
    clipped = demand is not None and _differs(demand, sent)
    blocked = _blocked(sent, our)
    staled = got is not None and our is not None and _differs(_of(our), got)
    if not (clipped or blocked or staled):
        return y
    y = put_line(panel, "follower vx%+.2f vy%+.2f wz%+.3f" % (
        sent.vx, sent.vy, sent.wz), x, y,
        palette.RED if blocked else palette.AMBER, 0.4)
    return w.chips(panel, x, y, [("SHAPER CLIPPED", clipped, palette.AMBER),
                                 ("GO GATE BLOCKED", blocked, palette.RED),
                                 ("BRIDGE MISMATCH", staled, palette.ORANGE)],
                   width - x)


def _of(our) -> "_Twist":
    """``(vx, vy, vz, wz)`` from the spine, in the shape a trace command has."""
    return _Twist(w.finite(our[0]), w.finite(our[1]), w.finite(our[2]),
                  w.finite(our[3]))


class _Twist(object):
    """The four fields ``_differs`` compares. Not a frame dataclass: this only
    ever wraps the spine's own 4-tuple so it can be compared to a recorded one."""

    __slots__ = ("vx", "vy", "vz", "wz")

    def __init__(self, vx, vy, vz, wz):
        self.vx, self.vy, self.vz, self.wz = vx, vy, vz, wz


def _blocked(sent, our) -> bool:
    """True when the follower commanded motion and the gate let nothing through."""
    if our is None:
        return False
    asked = max(abs(sent.vx), abs(sent.vy), abs(sent.wz))
    got = max(abs(w.finite(our[0])), abs(w.finite(our[1])), abs(w.finite(our[3])))
    return asked > _MOVING_EPS and got <= _GATE_EPS


def _differs(a, b) -> bool:
    """True when two points on the chain did not carry the same command."""
    return (abs(a.vx - b.vx) > _GATE_EPS or abs(a.vy - b.vy) > _GATE_EPS
            or abs(a.vz - b.vz) > _GATE_EPS or abs(a.wz - b.wz) > _GATE_EPS)


# ── the tracker's own split of that command ──────────────────────────────────
def _control_terms(panel, x, y, width, frame: NavFrame) -> int:
    """Feed-forward vs correction, and whether a limiter chose the output."""
    t = frame.terms
    limits = ",".join(t.limits) if t.limits else "none"
    # Title kept short on purpose: `section` drops the right-aligned note rather
    # than overprint the title, and "limits <name>" is the more useful half.
    y = w.section(panel, x, y, width, "CONTROL (world)", palette.GREEN,
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
