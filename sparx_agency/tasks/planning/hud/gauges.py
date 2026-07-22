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
