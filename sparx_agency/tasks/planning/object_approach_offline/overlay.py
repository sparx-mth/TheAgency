"""Draw the target lock on the frame and the mission/command HUD in a side panel.

The camera frame carries the raw detections plus a single **colour-coded lock
indicator** that says, at a glance, *how well we currently know where the target
is*:

  * **green box** -- the detector (YOLO) reports the target this frame: the most
    confident state, drawn on the detector's own box;
  * **orange box** -- tracking only: the detector did not report it (too low a
    confidence, or it fires slower than the camera), but the tracker still holds a
    box on it. Also used while the tracker briefly dead-reckons through a dropout;
  * **red full-frame border (no box)** -- the target is lost (both detector and
    tracker) and the mission is actively re-searching (RECOVER): the first seconds
    of trying to re-acquire. There is no box because we do not know where it is,
    so the *whole frame* is bordered instead;
  * **grey full-frame border (no box)** -- searching from scratch (SEARCH / SCAN):
    not found for many frames, hunting for it again.

Everything else (mission state, offsets, range, and the exact body-frame command
that would be published to ``/cmd_vel``) renders into a separate panel
:func:`render` places to the right of the frame, so captions never obscure the
image. The command is shown both as numbers and as three gauges: a ROLL arrow
(lateral, ``vy``), a PITCH arrow (forward/back, ``vx``), and a YAW dial (a circle
with a point marking the commanded turn rate/direction).

The gauges, the boxed-text tag and the colour palette are the shared HUD
primitives in :mod:`sparx_agency.tasks.planning.hud` (the nav-debug view draws
the identical vocabulary); only the target-lock-specific logic lives here.
"""
from __future__ import annotations

from typing import Optional, Sequence, Tuple

import cv2
import numpy as np

from sparx_agency.core.common.types.perception import Detection2D
from sparx_agency.core.planning.visual_servo import RECOVER
from sparx_agency.tasks.planning.hud import gauges, palette
from sparx_agency.tasks.planning.hud.panel import hline, pad_to_height, put_line, tag
from sparx_agency.tasks.planning.object_approach_offline.pipeline import FrameResult

# ── lock-indicator colors (BGR) ─────────────────────────────────────────────
_GREEN = palette.GREEN       # detector sees the target (confident)
_ORANGE = palette.ORANGE     # tracking only (detector silent this frame)
_RED = palette.RED           # lost, actively re-searching (RECOVER)
_GRAY = palette.GRAY         # searching from scratch (SEARCH/SCAN) + context boxes

# ── panel colors/layout ──────────────────────────────────────────────────────
PANEL_WIDTH = 340
_PANEL_BG = palette.PANEL_BG
_TEXT = palette.TEXT
_MUTED = palette.MUTED
_MODE_COLOR = {
    "SEARCH": palette.GRAY,
    "SCAN": palette.GRAY,
    "APPROACH": palette.ORANGE,
    "HOVER_LOCK": palette.GREEN,
    "RECOVER": palette.RED,
    "LAND": palette.AMBER,       # reached the object -> stopping + landing (amber)
}

_GAUGE_SIZE = gauges.GAUGE_SIZE
_GAUGE_GAP = gauges.GAUGE_GAP


# ── frame overlay: the colour-coded lock indicator ──────────────────────────
def classify_lock(result: FrameResult) -> Tuple[str, tuple,
                                                 Optional[Tuple[float, float, float, float]]]:
    """Pick the lock indicator: ``(caption, BGR color, box_or_None)``.

    A returned box (green/orange) means we know *where* the target is and draw a
    rectangle there; ``None`` means we do not, so a whole-frame border in the
    colour is drawn instead. Priority: a live detector box (green) outranks a
    tracker-only box (orange); with neither, RECOVER is red and everything else
    (SEARCH/SCAN) is grey.
    """
    if result.target_detection is not None:
        return "DETECTED", _GREEN, tuple(float(v) for v in result.target_detection.bbox_xyxy)
    track = result.track
    if track is not None and track.valid:
        caption = "TRACKING (coast)" if track.predicted else "TRACKING"
        return caption, _ORANGE, tuple(float(v) for v in track.bbox_xyxy)
    if result.fsm_mode == RECOVER:
        return "LOST -- RE-SEARCHING", _RED, None
    return "SEARCHING", _GRAY, None


def draw_context_detections(bgr: np.ndarray, detections: Sequence[Detection2D],
                            target_detection: Optional[Detection2D]) -> None:
    """Thin gray boxes for every detection except the locked target (context only)."""
    for d in detections:
        if d is target_detection:
            continue  # the target is drawn as the colour-coded lock indicator, on top
        x1, y1, x2, y2 = (int(v) for v in d.bbox_xyxy)
        cv2.rectangle(bgr, (x1, y1), (x2, y2), _GRAY, 1)
        tag(bgr, "%s %.2f" % (d.label, d.score), x1, y1, _GRAY, 0.4)


def draw_lock(bgr: np.ndarray, result: FrameResult) -> None:
    """Draw the lock indicator: a green/orange target box, or a red/grey border."""
    caption, color, box = classify_lock(result)
    if box is None:
        _draw_frame_border(bgr, color, caption)
        return
    x1, y1, x2, y2 = (int(v) for v in box)
    cv2.rectangle(bgr, (x1, y1), (x2, y2), color, 2)
    score = (result.target_detection.score if result.target_detection is not None
             else (result.track.score if result.track is not None else 0.0))
    tag(bgr, "%s %.2f [%s]" % (result.target or "target", score, caption),
        x1, y1, color, 0.5)


def _draw_frame_border(bgr: np.ndarray, color: tuple, caption: str) -> None:
    """Border the whole image (target position unknown), captioned top-left."""
    h, w = bgr.shape[:2]
    t = 6
    cv2.rectangle(bgr, (t // 2, t // 2), (w - t // 2 - 1, h - t // 2 - 1), color, t)
    tag(bgr, caption, 10, 14, color, 0.5)


# ── panel: mission state + gauges + numbers, no image involved ──────────────
def build_panel(result: FrameResult, width: int = PANEL_WIDTH) -> np.ndarray:
    """Render the mission-state/command panel; height grows to fit its content."""
    height = 340  # enough for every section below; render() pads to match the frame
    panel = np.full((height, width, 3), _PANEL_BG, dtype=np.uint8)
    x = 14
    y = 24

    y = put_line(panel, "OBJECT APPROACH -- TARGET LOCK", x, y, _MUTED, 0.5)
    y = put_line(panel, "target: %r" % result.target, x, y)
    y = put_line(panel, "state: %s" % result.fsm_mode, x, y,
                 _MODE_COLOR.get(result.fsm_mode, _TEXT), 0.6)
    y = put_line(panel, "confirmed: %s (streak %d)" % (result.confirmed, result.streak), x, y)
    y = hline(panel, y, width)

    if result.x_offset is not None:
        rng = "%.2fm" % result.range_m if result.range_m is not None else "-"
        y = put_line(panel, "x_off=%+.2f  y_off=%+.2f" % (result.x_offset, result.y_offset), x, y)
        y = put_line(panel, "area=%.3f  range=%s" % (result.area_frac, rng), x, y)
        y = put_line(panel, "at_target: %s" % result.at_target, x, y)
    else:
        y = put_line(panel, "no track yet", x, y, _MUTED)
    y = hline(panel, y, width)

    vx, vy, yaw_rate = (0.0, 0.0, 0.0) if result.command is None else \
        (result.command.x, result.command.y, result.command.yaw_rate)

    gauge_row = [
        ("ROLL", gauges.draw_roll_gauge(vy, result.gauge_max_vy), "vy=%+.2f" % vy),
        ("PITCH", gauges.draw_pitch_gauge(vx, result.gauge_max_vx), "vx=%+.2f" % vx),
        ("YAW", gauges.draw_yaw_gauge(yaw_rate, result.gauge_max_yaw_rate), "wz=%+.2f" % yaw_rate),
    ]
    gx = x
    gy = y
    for label, gauge, value_text in gauge_row:
        panel[gy:gy + _GAUGE_SIZE, gx:gx + _GAUGE_SIZE] = gauge
        put_line(panel, label, gx, gy + _GAUGE_SIZE + 16, _MUTED, 0.45)
        put_line(panel, value_text, gx, gy + _GAUGE_SIZE + 34, _TEXT, 0.45)
        gx += _GAUGE_SIZE + _GAUGE_GAP
    y = gy + _GAUGE_SIZE + 34 + 16
    y = hline(panel, y, width)

    if result.command is None:
        y = put_line(panel, "not driving -- %s" % result.cmd_source, x, y, _MUTED)
    else:
        c = result.command
        y = put_line(panel, "vx=%+.2f vy=%+.2f vz=%+.2f" % (c.x, c.y, c.z), x, y)
        y = put_line(panel, "yaw_rate=%+.2f  [%s]" % (c.yaw_rate, result.cmd_source), x, y)

    return panel[:min(y + 8, height)]


def render(bgr: np.ndarray, result: FrameResult) -> np.ndarray:
    """Camera frame (context detections + colour-coded lock) beside the HUD panel."""
    frame = bgr.copy()
    draw_context_detections(frame, result.detections, result.target_detection)
    draw_lock(frame, result)

    panel = build_panel(result)
    height = max(frame.shape[0], panel.shape[0])
    return np.hstack([pad_to_height(frame, height), pad_to_height(panel, height)])
