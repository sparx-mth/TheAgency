"""Text and layout helpers for the HUD side panels.

Small ``cv2`` wrappers shared by the object-approach and nav-debug panels: a
left-aligned text line that returns the next baseline, a section divider, a
filled text tag drawn onto an image, and vertical padding to match a
neighbouring image's height for a clean ``hstack``.
"""
from __future__ import annotations

import cv2
import numpy as np

from sparx_agency.tasks.planning.hud.palette import BLACK, HLINE, TEXT

_FONT = cv2.FONT_HERSHEY_SIMPLEX


def put_line(panel: np.ndarray, text: str, x: int, y: int, color=TEXT,
             scale: float = 0.5) -> int:
    """Draw one text line with its baseline at ``(x, y)``; return the next ``y``."""
    cv2.putText(panel, text, (x, y), _FONT, scale, color, 1, cv2.LINE_AA)
    return y + 20


def hline(panel: np.ndarray, y: int, width: int, color=HLINE) -> int:
    """Draw a thin full-width section divider at ``y``; return the next ``y``."""
    cv2.line(panel, (12, y), (width - 12, y), color, 1)
    return y + 16


def wrap_text(text: str, max_width: int, scale: float = 0.5, thickness: int = 1):
    """Break ``text`` into lines that each fit within ``max_width`` pixels.

    Word-wraps on spaces (a single word longer than the width is left whole
    rather than split mid-word), so a long caption flows onto more lines instead
    of being truncated. Returns a list of line strings.
    """
    lines = []
    cur = ""
    for word in str(text).split():
        trial = (cur + " " + word).strip()
        if cur and cv2.getTextSize(trial, _FONT, scale, thickness)[0][0] > max_width:
            lines.append(cur)
            cur = word
        else:
            cur = trial
    if cur:
        lines.append(cur)
    return lines or [""]


def tag(bgr: np.ndarray, text: str, x: int, y: int, color, scale: float = 0.5) -> None:
    """Draw ``text`` in a filled ``color`` box with its bottom-left near ``(x, y)``."""
    (tw, th), _ = cv2.getTextSize(text, _FONT, scale, 1)
    y0 = max(th + 4, y)
    cv2.rectangle(bgr, (x, y0 - th - 4), (x + tw + 4, y0), color, -1)
    cv2.putText(bgr, text, (x + 2, y0 - 3), _FONT, scale, BLACK, 1, cv2.LINE_AA)


def spark(panel: np.ndarray, series, x: int, y: int, w: int, h: int, color=TEXT,
          lo=None, hi=None) -> None:
    """Draw ``series`` as a small sparkline in the ``(x, y, w, h)`` box.

    Auto-scales to the data unless ``lo``/``hi`` pin the range; a flat/empty
    series draws a baseline. The latest sample is marked with a dot.
    """
    cv2.rectangle(panel, (x, y), (x + w, y + h), (40, 40, 40), 1)
    vals = [float(v) for v in series]
    if not vals:
        return
    lo = min(vals) if lo is None else lo
    hi = max(vals) if hi is None else hi
    span = (hi - lo) or 1.0
    n = len(vals)
    step = w / float(max(1, n - 1))

    def _pt(i, v):
        px = x + int(round(i * step))
        py = y + h - 1 - int(round((min(max(v, lo), hi) - lo) / span * (h - 2)))
        return px, py

    pts = [_pt(i, v) for i, v in enumerate(vals)]
    if len(pts) >= 2:
        cv2.polylines(panel, [np.asarray(pts, np.int32)], False, color, 1, cv2.LINE_AA)
    cv2.circle(panel, pts[-1], 2, color, -1)


def pad_to_height(img: np.ndarray, height: int, value=BLACK) -> np.ndarray:
    """Vertically centre ``img`` in a canvas of ``height`` rows (no-op if tall enough)."""
    if img.shape[0] >= height:
        return img
    pad = height - img.shape[0]
    top = pad // 2
    return cv2.copyMakeBorder(img, top, pad - top, 0, 0, cv2.BORDER_CONSTANT,
                              value=value)
