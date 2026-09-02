"""Small drawing widgets shared by the nav-debug telemetry columns.

Every panel column is a stack of *lanes*, and every lane is made of the same
few pieces: a titled section rule, labelled numbers, a bar that puts a number in
the context of its full scale, a row of flag chips, and -- for a fault nobody may
miss -- a filled alert bar. Those pieces live here, deliberately dumb (``cv2`` +
``numpy``, no knowledge of any dataclass), so :mod:`.render_panel` and
:mod:`.render_lanes` read as layout instead of as drawing code.

Convention: every ``y`` is a **text baseline**, and every function returns the
baseline of the next line, so a lane is written as a straight ``y = widget(...)``
chain. Bars are the exception and take their own top edge.

Python 3.8 compatible: the same renderer is meant to drive a live viewer inside
the FALCON Noetic container.
"""
from __future__ import annotations

import math
from typing import Iterable, Optional, Sequence, Tuple

import cv2
import numpy as np

from sparx_agency.tasks.planning.hud import palette

_FONT = cv2.FONT_HERSHEY_SIMPLEX
_TRACK = (45, 45, 45)        # bar background
_ZERO = (95, 95, 95)         # the zero tick inside a bar
_OFF = (90, 90, 90)          # an inactive flag chip
_DIM = (110, 110, 110)       # a collapsed (absent) lane


# ── numbers ──────────────────────────────────────────────────────────────────
def finite(value, default: float = 0.0) -> float:
    """``float(value)`` when it is a real number, else ``default``.

    Guards every arithmetic path in the renderer against ``None``/NaN/inf: a
    diagnostic view must degrade to a dash, never raise on bad telemetry.
    """
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def num(value, fmt: str = "%.2f", dash: str = "--") -> str:
    """Format an optional number, or ``dash`` when it was never recorded."""
    if value is None:
        return dash
    out = finite(value, float("nan"))
    return dash if math.isnan(out) else fmt % out


def grade_color(value, ok: float, warn: float):
    """Green below ``ok``, orange below ``warn``, red beyond -- on ``|value|``."""
    magnitude = abs(finite(value))
    if magnitude < ok:
        return palette.GREEN
    return palette.ORANGE if magnitude < warn else palette.RED


def conf_color(conf):
    """Red-to-green blend for a 0..1 confidence."""
    c = max(0.0, min(1.0, finite(conf)))
    lo = np.asarray(palette.RED, np.float32)
    hi = np.asarray(palette.GREEN, np.float32)
    return tuple(int(v) for v in (lo * (1.0 - c) + hi * c))


# ── text ─────────────────────────────────────────────────────────────────────
def text_width(text: str, scale: float = 0.45) -> int:
    """Pixel width of ``text`` at ``scale``, thickness 1."""
    return int(cv2.getTextSize(str(text), _FONT, scale, 1)[0][0])


def put_right(panel: np.ndarray, text: str, x_right: int, y: int,
              color=palette.TEXT, scale: float = 0.45) -> int:
    """Draw ``text`` ending at ``x_right`` with baseline ``y``; return next ``y``."""
    x = max(2, int(x_right) - text_width(text, scale))
    cv2.putText(panel, str(text), (x, y), _FONT, scale, color, 1, cv2.LINE_AA)
    return y + 20


def section(panel: np.ndarray, x: int, y: int, width: int, title: str,
            color=palette.MUTED, note: str = "") -> int:
    """A lane header: the title, an optional right-aligned note, and a rule.

    The note is dropped rather than drawn over the title when the two do not
    both fit on the line.
    """
    cv2.putText(panel, title, (x, y), _FONT, 0.5, color, 1, cv2.LINE_AA)
    if note and text_width(title, 0.5) + text_width(note, 0.42) + 12 <= width - 2 * x:
        put_right(panel, note, width - x, y, palette.MUTED, 0.42)
    cv2.line(panel, (x, y + 7), (width - x, y + 7), palette.HLINE, 1)
    return y + 24


def absent(panel: np.ndarray, x: int, y: int, width: int, title: str,
           why: str = "not recorded") -> int:
    """Collapse a lane with no data to a single dim line."""
    cv2.putText(panel, "%s  --  %s" % (title, why), (x, y), _FONT, 0.44, _DIM, 1,
                cv2.LINE_AA)
    return y + 20


def guarded(panel: np.ndarray, x: int, y: int, width: int, title: str, draw,
            *args) -> int:
    """Draw one lane; on failure leave a stub line instead of raising.

    The renderer is also meant to drive a live viewer beside a flight, where a
    single malformed diagnostic message must cost that lane and nothing else.
    """
    try:
        return draw(panel, x, y, width, *args)
    except Exception as exc:            # never take a live viewer down
        return absent(panel, x, y, width, title,
                      "render failed (%s)" % type(exc).__name__)


def alert(panel: np.ndarray, x: int, y: int, width: int, text: str,
          color=palette.RED) -> int:
    """A filled full-width bar: for a fault that must be impossible to miss."""
    cv2.rectangle(panel, (x, y - 15), (width - x, y + 6), color, -1)
    cv2.putText(panel, text, (x + 6, y), _FONT, 0.48, palette.WHITE, 1, cv2.LINE_AA)
    return y + 28


def chips(panel: np.ndarray, x: int, y: int,
          items: Sequence[Tuple[str, bool, tuple]], max_x: int,
          scale: float = 0.4) -> int:
    """Flag chips left to right, wrapping at ``max_x``.

    Args:
        items: ``(text, active, colour)``. Active chips are filled in their
            colour; inactive ones stay a dim outline, so the flag's *existence*
            is visible even when it is not firing.
        max_x: Right edge to wrap at.

    Returns:
        The next text baseline (unchanged when ``items`` is empty).
    """
    if not items:
        return y
    cx, cy = x, y
    for text, active, color in items:
        w = text_width(text, scale) + 8
        if cx > x and cx + w > max_x:
            cx, cy = x, cy + 20
        if active:
            cv2.rectangle(panel, (cx, cy - 12), (cx + w, cy + 4), color, -1)
            cv2.putText(panel, text, (cx + 4, cy), _FONT, scale, palette.BLACK, 1,
                        cv2.LINE_AA)
        else:
            cv2.rectangle(panel, (cx, cy - 12), (cx + w, cy + 4), (60, 60, 60), 1)
            cv2.putText(panel, text, (cx + 4, cy), _FONT, scale, _OFF, 1, cv2.LINE_AA)
        cx += w + 6
    return cy + 22


# ── bars ─────────────────────────────────────────────────────────────────────
def value_bar(panel: np.ndarray, x: int, y: int, w: int, h: int,
              segments: Iterable[Tuple[float, tuple]], full_scale: float,
              markers: Iterable[Tuple[float, tuple]] = (),
              zero: float = 0.5) -> None:
    """Stacked signed bar with independent tick markers.

    ``segments`` are laid head-to-tail from the zero line, so each stage's
    contribution is its own length (feed-forward, then the servo correction).
    ``markers`` are drawn as bright ticks at absolute values: when a marker does
    not line up with the end of the segments, something *downstream* of them --
    a rate limiter or a ceiling -- chose the number.

    Args:
        x, y, w, h: Bar box; ``y`` is its top edge, not a text baseline.
        full_scale: Value mapped to the longer side of the bar.
        zero: Where the zero line sits, as a fraction of ``w`` (0.5 = signed,
            0.0 = a plain left-to-right magnitude bar).
    """
    cv2.rectangle(panel, (x, y), (x + w, y + h), _TRACK, -1)
    scale = finite(full_scale)
    zx = x + int(round(max(0.0, min(1.0, zero)) * w))
    span = max(1, (x + w) - zx - 1)     # zero line to the right edge = full scale
    cv2.line(panel, (zx, y), (zx, y + h), _ZERO, 1)
    if scale <= 0.0:
        return

    def _px(value: float) -> int:
        frac = max(-1.0, min(1.0, finite(value) / scale))
        return int(max(x, min(x + w, zx + round(frac * span))))

    total = 0.0
    for value, color in segments:
        a, b = _px(total), _px(total + finite(value))
        total += finite(value)
        if a != b:
            cv2.rectangle(panel, (min(a, b), y + 2), (max(a, b), y + h - 2), color, -1)
    for value, color in markers:
        mx = _px(value)
        cv2.line(panel, (mx, y - 1), (mx, y + h + 1), color, 2)


def labelled_bar(panel: np.ndarray, x: int, y: int, width: int, label: str,
                 value, full_scale: float, color, text: Optional[str] = None,
                 zero: float = 0.5, label_w: int = 62, value_w: int = 64) -> int:
    """One metric on one line: label, bar and number, sharing a baseline."""
    cv2.putText(panel, label, (x, y), _FONT, 0.42, palette.MUTED, 1, cv2.LINE_AA)
    bar_x = x + label_w
    bar_w = max(20, width - x - label_w - value_w)
    value_bar(panel, bar_x, y - 10, bar_w, 12, [(finite(value), color)], full_scale,
              zero=zero)
    put_right(panel, text if text is not None else num(value, "%+.2f"), width - x, y,
              color, 0.42)
    return y + 22
