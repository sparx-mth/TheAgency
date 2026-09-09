"""ROLL / PITCH / YAW command gauges -- the "closing on a target" indicators.

Three tiny square gauges that show a body-frame command at a glance:

  * :func:`draw_roll_gauge` -- a horizontal arrow (lateral, ``vy``),
  * :func:`draw_pitch_gauge` -- a vertical arrow (forward/back, ``vx``),
  * :func:`draw_yaw_gauge` -- a dial whose needle marks the turn rate/direction.

Each takes a value and the full-scale reference to normalise it against, and
returns a fresh ``GAUGE_SIZE x GAUGE_SIZE`` BGR image. The optional ``color``
lets a caller draw two command channels in different colours (e.g. the command
we send vs. the command the converter sends the drone) from the same code.

Extracted verbatim from the object-approach target-lock overlay so the nav-debug
view renders the identical gauges. ``numpy`` + ``cv2`` + ``math`` only.
"""
from __future__ import annotations

import math

import cv2
import numpy as np

# ── gauge geometry / colours (BGR) ───────────────────────────────────────────
GAUGE_SIZE = 96
GAUGE_GAP = 12
GAUGE_BG = (45, 45, 45)
GAUGE_TRACK = (95, 95, 95)
GAUGE_ARROW = (60, 200, 60)
GAUGE_NEEDLE = (60, 200, 60)
GAUGE_DOT = (60, 190, 250)


def gauge_frac(value: float, full_scale: float) -> float:
    """Signed fraction of ``full_scale`` in ``[-1, 1]`` (0 if ``full_scale`` <= 0)."""
    if full_scale <= 0.0:
        return 0.0
    return max(-1.0, min(1.0, value / full_scale))


def new_gauge(size: int = GAUGE_SIZE, bg=GAUGE_BG) -> np.ndarray:
    """A blank square gauge canvas."""
    return np.full((size, size, 3), bg, dtype=np.uint8)


def draw_roll_gauge(vy: float, full_scale: float, color=GAUGE_ARROW) -> np.ndarray:
    """Horizontal arrow: REP-103 ``+vy`` is left, so it points left/right accordingly."""
    g = new_gauge()
    c = GAUGE_SIZE // 2
    r = c - 10
    cv2.line(g, (c - r, c), (c + r, c), GAUGE_TRACK, 2)
    cv2.circle(g, (c, c), 3, GAUGE_TRACK, -1)
    dx = int(round(-gauge_frac(vy, full_scale) * r))
    if abs(dx) >= 4:
        cv2.arrowedLine(g, (c, c), (c + dx, c), color, 3, tipLength=0.3)
    else:
        cv2.circle(g, (c, c), 4, color, -1)
    return g


def draw_pitch_gauge(vx: float, full_scale: float, color=GAUGE_ARROW) -> np.ndarray:
    """Vertical arrow: ``+vx`` is forward, drawn pointing up."""
    g = new_gauge()
    c = GAUGE_SIZE // 2
    r = c - 10
    cv2.line(g, (c, c - r), (c, c + r), GAUGE_TRACK, 2)
    cv2.circle(g, (c, c), 3, GAUGE_TRACK, -1)
    dy = int(round(-gauge_frac(vx, full_scale) * r))
    if abs(dy) >= 4:
        cv2.arrowedLine(g, (c, c), (c, c + dy), color, 3, tipLength=0.3)
    else:
        cv2.circle(g, (c, c), 4, color, -1)
    return g


def draw_yaw_gauge(yaw_rate: float, full_scale: float, color=GAUGE_NEEDLE) -> np.ndarray:
    """Circle dial: the point sits at 12 o'clock at zero yaw, swinging to 9 o'clock
    at full-scale +yaw_rate (CCW/turn left) and 3 o'clock at full-scale -yaw_rate
    (CW/turn right)."""
    g = new_gauge()
    c = GAUGE_SIZE // 2
    r = c - 10
    cv2.circle(g, (c, c), r, GAUGE_TRACK, 2)
    cv2.circle(g, (c, c), 2, GAUGE_TRACK, -1)
    theta = math.radians(-gauge_frac(yaw_rate, full_scale) * 90.0)
    px = int(round(c + r * math.sin(theta)))
    py = int(round(c - r * math.cos(theta)))
    cv2.line(g, (c, c), (px, py), color, 2)
    cv2.circle(g, (px, py), 6, GAUGE_DOT, -1)
    return g


# ── the transmitter sticks ───────────────────────────────────────────────────
#: One stick gate, in pixels. Two of them side by side fit the 316 px of usable
#: width in the nav-debug command column with room to spare.
STICK_SIZE = 118
STICK_GATE = (95, 95, 95)
STICK_CROSS = (62, 62, 62)
STICK_TRAIL = (120, 120, 120)

#: Mode 2, the standard multirotor layout and the one a Rooster operator flies:
#: left stick is throttle (vertical) and yaw (horizontal), right stick is
#: forward/back (vertical) and lateral (horizontal).
STICK_LEFT = ("throttle", "yaw")
STICK_RIGHT = ("forward", "lateral")


def draw_stick(horizontal: float, vertical: float, h_full: float, v_full: float,
               color=GAUGE_DOT, size: int = STICK_SIZE) -> np.ndarray:
    """One transmitter stick: a square gate with the stick position in it.

    This is the *physical* picture of what the drone was told -- the deflection
    a human hand would have to hold to send the same command -- rather than an
    instrument reading of it. The counts passed in are the raw
    ``ManualControl`` axes, because those counts **are** the stick: the twist
    adapter has already applied its own lateral and yaw inversions by the time
    they exist, so negating them here would draw a stick nobody is holding.

    Args:
        horizontal: Axis counts for the stick's left/right travel; positive
            deflects right.
        vertical: Axis counts for its up/down travel; positive deflects up.
        h_full, v_full: Counts at full deflection on each axis.
        color: Knob colour.
        size: Gate size in pixels.

    Returns:
        A fresh ``size x size`` BGR image.
    """
    g = new_gauge(size)
    pad = 9
    lo, hi = pad, size - pad
    c = size // 2
    cv2.rectangle(g, (lo, lo), (hi, hi), STICK_GATE, 1)
    cv2.line(g, (lo + 2, c), (hi - 2, c), STICK_CROSS, 1)
    cv2.line(g, (c, lo + 2), (c, hi - 2), STICK_CROSS, 1)
    span = (hi - lo) // 2
    px = int(round(c + gauge_frac(horizontal, h_full) * span))
    py = int(round(c - gauge_frac(vertical, v_full) * span))
    # A line from centre to the knob, so a small deflection is still legible at
    # this size -- the knob alone moves only a couple of pixels near neutral.
    cv2.line(g, (c, c), (px, py), STICK_TRAIL, 1)
    cv2.circle(g, (px, py), 7, color, -1)
    cv2.circle(g, (px, py), 7, (20, 20, 20), 1)
    return g
