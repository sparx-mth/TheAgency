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
    pixel_goal_fresh: bool = False           # System 2 chose it on THIS decision
    pixel_goal_age: Optional[int] = None     # decisions since it was chosen
    decision_time: Optional[float] = None    # wall clock of the decision
    from_curve: bool = False        # this decision is System 1's continuous curve
    curve_share_pct: Optional[float] = None  # share of decisions that were curves
    # WHAT THE AIRCRAFT IS DOING RIGHT NOW, republished several times a second
    # rather than once per decision. A decision lasts seconds, so a motionless
    # drone on screen is either thinking, turning, dipping for a look-down or
    # wedged against something -- and a recording that cannot tell those apart
    # is a recording nobody can diagnose from. That is the whole reason seventy
    # seconds of the last hospital flight went unexplained.
    phase: str = ""                 # flying | settling | thinking | turning | dipping | stopped
    think_s: Optional[float] = None  # how long it has been standing still for this one
    blocked: bool = False           # the depth reflex allows no forward speed at all
    traj_m: Optional[float] = None   # length of the prediction this decision produced
    traj_pts: Optional[int] = None   # ...and how many waypoints it has
    turn_deg: Optional[float] = None  # the rotation this decision asked for, if any
    commits: Optional[int] = None    # routes flown so far
    turns: Optional[int] = None      # rotations flown so far
    escapes: Optional[int] = None    # blocked-forward escapes so far


def _put(img, text, org, scale=0.6, color=(255, 255, 255), thick=2):
    cv2.putText(img, text, org, _FONT, scale, color, thick, cv2.LINE_AA)


def _label_beside(img, text, x, y, gap, scale, color, thick):
    """Write ``text`` next to (x, y), flipping to the left near the right edge.

    The System-2 goal is very often near a frame edge -- it is a navigation
    target, and those sit where the corridor leaves the picture -- so a label
    pinned to its right is clipped exactly when it matters most.
    """
    (tw, _), _ = cv2.getTextSize(text, _FONT, scale, thick)
    org_x = x + gap if x + gap + tw <= img.shape[1] - 4 else x - gap - tw
    _put(img, text, (max(4, org_x), y), scale, color, thick)

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
    #
    # DRAWN AS OLD AS IT IS. The agent keeps the last goal alive between
    # System-2 calls, so this marker is non-null on almost every frame -- but it
    # is a pixel in the frame System 2 saw, and the aircraft has been moving
    # since. Drawn identically whether it is this decision's goal or one from
    # eight decisions ago, it reads as a live tracker locked onto a target,
    # which is exactly what it is not. Fresh is a solid red ring; stale is a
    # thin dim one carrying its age, so the eye can tell a decision from a
    # memory.
    if info.pixel_goal is not None:
        gx, gy = info.pixel_goal
        if info.pixel_goal_frame:
            fw, fh = info.pixel_goal_frame
            gx = int(gx * w / max(1, fw))
            gy = int(gy * h / max(1, fh))
        gx = max(0, min(int(gx), w - 1))
        gy = max(0, min(int(gy), h - 1))
        if info.pixel_goal_fresh:
            cv2.circle(panel, (gx, gy), 16, (0, 0, 255), 3, cv2.LINE_AA)
            cv2.circle(panel, (gx, gy), 3, (0, 0, 255), -1, cv2.LINE_AA)
            _label_beside(panel, "S2 goal", gx, gy, 20, 0.5, (0, 0, 255), 2)
        else:
            cv2.circle(panel, (gx, gy), 13, (70, 70, 150), 1, cv2.LINE_AA)
            age = "" if info.pixel_goal_age is None else " +%d" % info.pixel_goal_age
            _label_beside(panel, "S2 goal (stale%s)" % age, gx, gy, 18, 0.42,
                          (90, 90, 170), 1)

    # Top banner: status + action + what the aircraft is doing this instant.
    cv2.rectangle(panel, (0, 0), (w, 66), (0, 0, 0), -1)
    _put(panel, "DRONE CAMERA", (10, 24), 0.6, (0, 255, 0), 2)
    _put(panel, "action: %s   status: %s" % (info.action or "-", info.status or "-"),
         (10, 52), 0.55, (255, 255, 255), 1)
    _draw_phase(panel, info, w)

    # Where this decision came from. System 1's curve is the continuous output
    # the dual-system design exists to produce; a discrete action rendered as a
    # 0.25 m step is the fallback. Showing which, and how often, is the only way
    # to read a recording and know what you actually got.
    # "of decisions", spelled out. The share counts every decision -- turns and
    # STOPs included -- while the run log's [curve]/[action] tag counts
    # COMMITTED ROUTES, and the two legitimately differ: a run whose every route
    # was a curve can still report 38% here because most of its decisions were
    # STOPs. Labelling this one "curves" invited the two to be read as the same
    # number and one of them to look broken.
    source = "S1 CURVE" if info.from_curve else "action step"
    colour = (0, 255, 180) if info.from_curve else (0, 165, 255)
    label = source if info.curve_share_pct is None else (
        "%s   (curve on %.0f%% of decisions)" % (source, info.curve_share_pct))
    (tw, _), _ = cv2.getTextSize(label, _FONT, 0.5, 2)
    cv2.rectangle(panel, (w - tw - 24, 70), (w, 96), (0, 0, 0), -1)
    _put(panel, label, (w - tw - 14, 89), 0.5, colour, 2)

    # What this decision actually produced, under the source tag: a length and a
    # point count. "S1 CURVE" says System 1 ran; only these say whether what it
    # produced is a route or a twitch.
    if info.traj_m is not None:
        if info.turn_deg is not None:
            shape = "rotate %+.0f deg" % info.turn_deg
        else:
            shape = "%.2f m / %d pts" % (info.traj_m, info.traj_pts or 0)
        (tw2, _), _ = cv2.getTextSize(shape, _FONT, 0.45, 1)
        cv2.rectangle(panel, (w - tw2 - 24, 98), (w, 120), (0, 0, 0), -1)
        _put(panel, shape, (w - tw2 - 14, 114), 0.45, (200, 200, 200), 1)

    # FPS block, the headline the request asks for, bottom-left.
    def _fps(label, fps, ms):
        if fps is None:
            return "%s --" % label
        return "%s %.1f Hz (%.0f ms)" % (label, fps, ms or 0.0)

    cv2.rectangle(panel, (0, h - 58), (330, h), (0, 0, 0), -1)
    _put(panel, _fps("System 1:", info.s1_fps, info.s1_ms), (10, h - 34), 0.6, (0, 255, 180), 2)
    _put(panel, _fps("System 2:", info.s2_fps, info.s2_ms), (10, h - 10), 0.6, (0, 200, 255), 2)

    # Instruction, wrapped, bottom band.
    # Three lines, not two. The instruction is the one thing on screen that a
    # viewer has to read in full to judge anything else, and a room-and-table
    # order does not fit in two.
    lines = _wrap(info.instruction, 54)[:3]
    y = h - 58 - 8 - (len(lines) - 1) * 22
    for line in lines:
        (tw, _), _ = cv2.getTextSize(line, _FONT, 0.5, 1)
        cv2.rectangle(panel, (8, y - 16), (8 + tw + 8, y + 6), (0, 0, 0), -1)
        _put(panel, line, (12, y), 0.5, (200, 220, 255), 1)
        y += 22
    return panel


_PHASE_COLOURS = {
    "flying": (0, 220, 0),
    "settling": (0, 200, 255),
    "thinking": (0, 200, 255),
    "turning": (255, 180, 0),
    "dipping": (255, 140, 60),
    "stopped": (160, 160, 160),
}


def _draw_phase(panel, info, w):
    """A pill saying what the aircraft is doing, and for how long.

    Placed top-right and coloured, because it is the field a viewer checks
    first: a stationary drone is fine when it is THINKING and a bug when it is
    FLYING, and the picture is identical either way.
    """
    if not info.phase and not info.blocked:
        return
    text = (info.phase or "").upper()
    if info.think_s and info.phase in ("thinking", "settling", "dipping"):
        text = "%s %.1fs" % (text, info.think_s)
    colour = _PHASE_COLOURS.get(info.phase, (200, 200, 200))
    if info.blocked:
        # BLOCKED outranks everything else on screen. It is the one state in
        # which the policy's decisions cannot be flown at all, and it looks
        # exactly like thinking from the outside.
        text = "BLOCKED  " + text
        colour = (0, 0, 255)
    (tw, _), _ = cv2.getTextSize(text, _FONT, 0.55, 2)
    cv2.rectangle(panel, (w - tw - 24, 4), (w - 4, 34), (0, 0, 0), -1)
    cv2.rectangle(panel, (w - tw - 24, 4), (w - 4, 34), colour, 1)
    _put(panel, text, (w - tw - 14, 26), 0.55, colour, 2)


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

