"""Render a run: the drone camera on the left, N1's route top-down on the right.

ROS-free on purpose. The ROS2 recorder node
(:mod:`~sparx_agency.tasks.planning.sjtu_internvla_n1.ros2.n1_run_recorder_node`)
does nothing but pull frames off topics and hand them here; every pixel is drawn
by the pure functions below, so the layout is unit-tested in the plain ``.venv``
with no ROS and no Gazebo -- feed it synthetic frames and it produces the same
video it would in flight.

Two panels, because the request is to see two things at once:

* **left** -- what the drone sees: the front camera, with the instruction, the
  action, the System-1 / System-2 FPS and (when System 2 fired) its pixel goal
  drawn on top;
* **right** -- what the network decided: the route drawn on the hospital's own
  occupancy map, the whole building beside a window that follows the aircraft
  (see :mod:`top_down`). That is the honest way to show a *drone's* route -- an
  aircraft flies at camera height, so its path projects to the horizon in the
  first-person view and only reads clearly from above.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    import cv2
except ImportError:  # pragma: no cover - cv2 is present in the venv and ROS env
    cv2 = None

# The right-hand panel lives in its own module now that it draws the route on
# the building's occupancy map as well as fitting axes to the trail; it is
# re-exported here because this is the name every caller already imports.
from sparx_agency.tasks.planning.sjtu_internvla_n1.top_down import (  # noqa: F401
    TopDownRenderer,
)

_FONT = 0  # cv2.FONT_HERSHEY_SIMPLEX; kept as a literal so this imports without cv2.


class FpsMeter:
    """Exponential-moving-average rate from per-call millisecond durations.

    A rate is what a viewer wants to read, but the model reports a *duration*
    per call, and the two are not the same average -- the mean of the rates is
    not the reciprocal of the mean of the durations. This averages the durations
    (what the model measures) and reciprocates once, so the number on screen is
    the honest throughput of the component.
    """

    def __init__(self, alpha: float = 0.2):
        self.alpha = float(alpha)
        self._ms = None  # type: Optional[float]
        self.last_ms = None  # type: Optional[float]

    def update(self, ms):
        # type: (Optional[float]) -> None
        """Fold one call's duration (ms) into the average; ``None`` is ignored."""
        if ms is None:
            return
        ms = float(ms)
        if ms <= 0.0:
            return
        self.last_ms = ms
        self._ms = ms if self._ms is None else (1.0 - self.alpha) * self._ms + self.alpha * ms

    @property
    def fps(self):
        # type: () -> Optional[float]
        """The smoothed rate, Hz, or ``None`` before the first sample."""
        if self._ms is None or self._ms <= 0.0:
            return None
        return 1000.0 / self._ms


@dataclass
class OverlayInfo:
    """Everything the left (camera) panel writes over the video."""
    instruction: str = ""
    action: str = ""
    status: str = ""
    s1_fps: Optional[float] = None
    s2_fps: Optional[float] = None
    s1_ms: Optional[float] = None
    s2_ms: Optional[float] = None
    pixel_goal: Optional[Tuple[int, int]] = None  # (x, y) in the model's input frame
    pixel_goal_frame: Optional[Tuple[int, int]] = None  # (w, h) that goal was in


def _put(img, text, org, scale=0.6, color=(255, 255, 255), thick=2):
    cv2.putText(img, text, org, _FONT, scale, color, thick, cv2.LINE_AA)


def draw_camera_panel(frame_bgr, info, size):
    # type: (np.ndarray, OverlayInfo, Tuple[int, int]) -> np.ndarray
    """The left panel: the drone camera with the run's telemetry drawn on it.

    Args:
        frame_bgr: HxWx3 BGR camera frame.
        info: the text and markers to overlay.
        size: ``(width, height)`` the panel is rendered at.

    Returns:
        A ``(height, width, 3)`` BGR panel.
    """
    w, h = int(size[0]), int(size[1])
    panel = cv2.resize(frame_bgr, (w, h), interpolation=cv2.INTER_AREA)

    # Pixel goal (System 2), rescaled from the frame it was computed in.
    if info.pixel_goal is not None:
        gx, gy = info.pixel_goal
        if info.pixel_goal_frame:
            fw, fh = info.pixel_goal_frame
            gx = int(gx * w / max(1, fw))
            gy = int(gy * h / max(1, fh))
        gx = max(0, min(int(gx), w - 1))
        gy = max(0, min(int(gy), h - 1))
        cv2.circle(panel, (gx, gy), 16, (0, 0, 255), 3, cv2.LINE_AA)
        cv2.circle(panel, (gx, gy), 3, (0, 0, 255), -1, cv2.LINE_AA)
        _put(panel, "S2 goal", (gx + 20, gy), 0.5, (0, 0, 255), 2)

    # Top banner: status + action.
    cv2.rectangle(panel, (0, 0), (w, 66), (0, 0, 0), -1)
    _put(panel, "DRONE CAMERA", (10, 24), 0.6, (0, 255, 0), 2)
    _put(panel, "action: %s   status: %s" % (info.action or "-", info.status or "-"),
         (10, 52), 0.55, (255, 255, 255), 1)

    # FPS block, the headline the request asks for, bottom-left.
    def _fps(label, fps, ms):
        if fps is None:
            return "%s --" % label
        return "%s %.1f Hz (%.0f ms)" % (label, fps, ms or 0.0)

    cv2.rectangle(panel, (0, h - 58), (330, h), (0, 0, 0), -1)
    _put(panel, _fps("System 1:", info.s1_fps, info.s1_ms), (10, h - 34), 0.6, (0, 255, 180), 2)
    _put(panel, _fps("System 2:", info.s2_fps, info.s2_ms), (10, h - 10), 0.6, (0, 200, 255), 2)

    # Instruction, wrapped, bottom band.
    lines = _wrap(info.instruction, 54)[:2]
    y = h - 58 - 8 - (len(lines) - 1) * 22
    for line in lines:
        (tw, _), _ = cv2.getTextSize(line, _FONT, 0.5, 1)
        cv2.rectangle(panel, (8, y - 16), (8 + tw + 8, y + 6), (0, 0, 0), -1)
        _put(panel, line, (12, y), 0.5, (200, 220, 255), 1)
        y += 22
    return panel


def compose(left, right):
    # type: (np.ndarray, np.ndarray) -> np.ndarray
    """Stack the two panels side by side, matching their heights."""
    if left.shape[0] != right.shape[0]:
        h = min(left.shape[0], right.shape[0])
        left = left[:h]
        right = right[:h]
    return np.hstack((left, right))


def _wrap(text, max_chars):
    # type: (str, int) -> List[str]
    text = (text or "").strip()
    if not text:
        return [""]
    out, cur = [], ""
    for word in text.split():
        nxt = (cur + " " + word).strip()
        if len(nxt) <= max_chars:
            cur = nxt
        else:
            if cur:
                out.append(cur)
            cur = word
    if cur:
        out.append(cur)
    return out

