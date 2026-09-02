"""The nav-debug map pane: the BEV grid with the plan and the aircraft on it.

One picture of *where everything is*: the route layers (XTEND's raw A* ->
corrected -> final, or FALCON's planned path and the path it has actually
flown), the goal, the aim point, the pose with its trail, the learned drift --
and, on the Sphera stack, the **reference**: the point the aircraft is chasing
this instant, its velocity as an arrow, and a dashed line from the aircraft to
it so the position error reads as a geometric gap rather than as a number.

A frozen reference is drawn grey. ``traj_server`` republishes a trajectory's
last point with fresh stamps once it is finished, so a fresh reference does not
imply a moving one, and the two must not look alike.
"""
from __future__ import annotations

import math
from typing import List, Optional, Tuple

import cv2
import numpy as np

from sparx_agency.tasks.planning.hud import palette
from sparx_agency.tasks.planning.hud.panel import put_line
from sparx_agency.tasks.planning.nav_debug.bev_image import render_bev
from sparx_agency.tasks.planning.nav_debug.frame import NavFrame
from sparx_agency.tasks.planning.nav_debug.render_widgets import (
    conf_color, grade_color, num,
)

MAP_TARGET_PX = 720
_FONT = cv2.FONT_HERSHEY_SIMPLEX
_EMPTY_BG = (28, 28, 28)

# Route colours (BGR): raw plan faint, corrected blue, final green, flown peach.
_ASTAR = (120, 120, 120)
_SAFE = (210, 140, 40)
_FINAL = palette.GREEN
_EXECUTED = (150, 200, 255)
_TARGET = (0, 215, 255)      # amber ring on the active waypoint
_GOAL = (200, 60, 200)       # magenta goal
_LOOKAHEAD = (250, 190, 60)  # cyan aim point (XTEND pure pursuit)
_TRAIL = (90, 130, 90)
_DRIFT = palette.ORANGE
_REF = (60, 255, 255)        # the reference being chased (yellow)
_REF_FROZEN = (120, 145, 145)  # ... republished, frozen: not a moving target

_REF_STALE_S = 0.5           # the tracker holds station past this reference age
_VEL_ARROW_S = 0.7           # seconds of reference velocity drawn as the arrow
_DRIFT_ARROW_M_PER_MS = 2.0  # metres drawn per m/s of learned drift, for visibility


def build_map_pane(frame: NavFrame, map_px: int = MAP_TARGET_PX) -> np.ndarray:
    """Draw the map pane for ``frame``; every layer is skipped when absent."""
    if frame.bev is None:
        img = np.full((map_px, map_px, 3), _EMPTY_BG, np.uint8)
        put_line(img, "no BEV map recorded yet", 20, map_px // 2, palette.MUTED, 0.7)
        to_px = None
    else:
        img, to_px = render_bev(frame.bev, frame.bev_conf, map_px)
    if to_px is not None:
        for layer in _LAYERS:
            _guarded(layer, img, to_px, frame)
    _guarded(_banner, img, to_px, frame)
    return img


def _guarded(layer, img, to_px, frame) -> None:
    """Draw one layer; a layer that fails is dropped, never raised.

    The map has nowhere to print an error, and this renderer is meant to drive a
    live viewer beside a flight: losing one overlay beats losing the window.
    """
    try:
        layer(img, to_px, frame)
    except Exception:      # never take a live viewer down over a diagnostic
        pass


# ── geometry helpers ─────────────────────────────────────────────────────────
def _clip(p: Tuple[int, int]) -> Tuple[int, int]:
    return (int(max(-20000, min(20000, p[0]))), int(max(-20000, min(20000, p[1]))))


def _polyline(img, to_px, pts: Optional[List[Tuple[float, float]]], color, thick,
              dotted: bool = False) -> None:
    if not pts:
        return
    arr = [_clip(to_px(x, y)) for x, y in pts]
    if len(arr) == 1:
        cv2.circle(img, arr[0], 3, color, -1)
        return
    if dotted:
        for a, b in zip(arr, arr[1:]):
            cv2.line(img, a, b, color, thick, cv2.LINE_AA)  # cv2 has no dashed line
    else:
        cv2.polylines(img, [np.asarray(arr, np.int32)], False, color, thick, cv2.LINE_AA)


def _dashed(img, p0, p1, color, thick: int = 1, dash: int = 10) -> None:
    """A dashed segment -- cv2 has no dashed line, and the error must not read
    as a route."""
    dist = math.hypot(p1[0] - p0[0], p1[1] - p0[1])
    steps = max(1, min(120, int(dist // dash)))
    if steps <= 1:
        cv2.line(img, p0, p1, color, thick, cv2.LINE_AA)
        return
    for i in range(0, steps, 2):
        a = _lerp(p0, p1, i / float(steps))
        b = _lerp(p0, p1, (i + 1) / float(steps))
        cv2.line(img, a, b, color, thick, cv2.LINE_AA)


def _lerp(p0, p1, t: float) -> Tuple[int, int]:
    return (int(p0[0] + (p1[0] - p0[0]) * t), int(p0[1] + (p1[1] - p0[1]) * t))


# ── layers ───────────────────────────────────────────────────────────────────
def _draw_routes(img, to_px, frame: NavFrame) -> None:
    routes = frame.routes
    _polyline(img, to_px, routes.astar, _ASTAR, 1, dotted=True)
    _polyline(img, to_px, routes.safe, _SAFE, 2)
    _polyline(img, to_px, getattr(routes, "executed", None), _EXECUTED, 2)
    _polyline(img, to_px, routes.final, _FINAL, 3)
    if routes.final:                     # mark the flown waypoints
        for x, y in routes.final:
            cv2.circle(img, _clip(to_px(x, y)), 3, _FINAL, -1)
    if routes.goal:
        cv2.drawMarker(img, _clip(to_px(*routes.goal)), _GOAL, cv2.MARKER_STAR, 16, 2)
    if routes.lookahead:
        cv2.circle(img, _clip(to_px(*routes.lookahead)), 5, _LOOKAHEAD, -1)


def _draw_target(img, to_px, frame: NavFrame) -> None:
    if frame.target is None:
        return
    wp_idx, num_wp, tx, ty = frame.target
    p = _clip(to_px(tx, ty))
    cv2.circle(img, p, 10, _TARGET, 2)
    cv2.circle(img, p, 3, _TARGET, -1)
    label = "wp %d/%d" % (wp_idx + 1, num_wp)
    if frame.advanced:
        label = "reached %d -> %d" % (wp_idx, wp_idx + 1)
    cv2.putText(img, label, (p[0] + 12, p[1] - 8), _FONT, 0.5, _TARGET, 1, cv2.LINE_AA)


def _draw_trail(img, to_px, frame: NavFrame) -> None:
    trail = frame.trail
    if trail and len(trail) >= 2:
        _polyline(img, to_px, trail, _TRAIL, 1)


def _draw_pose(img, to_px, frame: NavFrame) -> None:
    p = _clip(to_px(frame.x, frame.y))
    conf = frame.quality.confidence if frame.quality is not None else 1.0
    color = conf_color(conf)
    head = _clip(to_px(frame.x + 0.5 * math.cos(frame.yaw),
                       frame.y + 0.5 * math.sin(frame.yaw)))
    cv2.arrowedLine(img, p, head, color, 2, tipLength=0.35)
    cv2.circle(img, p, 6, color, -1)
    cv2.circle(img, p, 6, palette.BLACK, 1)


def _draw_drift(img, to_px, frame: NavFrame) -> None:
    d = frame.drift
    if d is None or (abs(d.drift_vx) < 0.01 and abs(d.drift_vy) < 0.01):
        return
    cy, sy = math.cos(frame.yaw), math.sin(frame.yaw)
    k = _DRIFT_ARROW_M_PER_MS
    wx = frame.x + k * (d.drift_vx * cy - d.drift_vy * sy)
    wy = frame.y + k * (d.drift_vx * sy + d.drift_vy * cy)
    cv2.arrowedLine(img, _clip(to_px(frame.x, frame.y)), _clip(to_px(wx, wy)),
                    _DRIFT, 2, tipLength=0.3)


def _draw_reference(img, to_px, frame: NavFrame) -> None:
    """The point being chased: the gap to it, its velocity, and its state."""
    ref = frame.reference
    if ref is None:
        return
    moving = bool(ref.moving)
    color = _REF if moving else _REF_FROZEN
    here = _clip(to_px(frame.x, frame.y))
    there = _clip(to_px(ref.x, ref.y))
    _dashed(img, here, there, _error_color(frame), 1)
    if moving and ref.speed > 0.02:
        tip = _clip(to_px(ref.x + _VEL_ARROW_S * ref.vx, ref.y + _VEL_ARROW_S * ref.vy))
        cv2.arrowedLine(img, there, tip, color, 2, tipLength=0.35)
    cv2.drawMarker(img, there, color, cv2.MARKER_DIAMOND, 18, 2)
    cv2.circle(img, there, 3, color, -1)
    cv2.putText(img, _ref_label(frame, moving), (there[0] + 12, there[1] + 18),
                _FONT, 0.45, color, 1, cv2.LINE_AA)


def _ref_label(frame: NavFrame, moving: bool) -> str:
    ref = frame.reference
    parts = ["ref" if moving else "ref FROZEN"]
    if ref is not None and ref.age_s > _REF_STALE_S:
        parts.append("STALE %.1fs" % ref.age_s)
    if frame.tracking is not None:
        parts.append("err %s m" % num(frame.tracking.position_error_m))
    return "  ".join(parts)


def _error_color(frame: NavFrame):
    """Colour the aircraft-to-reference gap by how bad the tracking is."""
    tr = frame.tracking
    if tr is None:
        return _REF_FROZEN
    if tr.diverged:
        return palette.RED
    return grade_color(tr.position_error_m, 0.3, 0.8)


def _scale_bar(img, to_px, frame=None) -> None:
    p0, p1 = to_px(0.0, 0.0), to_px(1.0, 0.0)
    length = abs(p1[0] - p0[0]) or 1
    x0, y0 = 14, img.shape[0] - 46
    cv2.line(img, (x0, y0), (x0 + length, y0), palette.WHITE, 2)
    cv2.putText(img, "1 m", (x0, y0 - 6), _FONT, 0.45, palette.WHITE, 1, cv2.LINE_AA)


# ── banner ───────────────────────────────────────────────────────────────────
def _strip(img, y0: int, y1: int) -> None:
    """Darken a horizontal band so overlaid text stays readable over the map."""
    y1 = min(y1, img.shape[0])
    img[y0:y1] = (img[y0:y1].astype(np.float32) * 0.35).astype(np.uint8)


def _banner(img, to_px, frame: NavFrame) -> None:
    """Two lines over the map: what the controller is doing, and how it is going.

    XTEND runs read drift state + AprilTag quality; Sphera runs, which have
    neither, read the tracker verdict + Sphera's own ground truth instead.
    """
    _strip(img, 0, 52)
    put_line(img, _banner_mode(frame), 12, 20, palette.TEXT, 0.55)
    text, color = _banner_state(frame)
    if text:
        put_line(img, text, 12, 42, color, 0.5)
    # The replan reason + the "why" render in the full-width caption bar below
    # everything (never truncated), not overlaid on the map -- see render.py.


def _banner_mode(frame: NavFrame) -> str:
    if frame.drift is not None:
        return "MODE: %s" % (frame.drift.state or "-")
    tr = frame.tracking
    if tr is None:
        return "MODE: -"
    state = "DIVERGED" if tr.diverged else ("HOLDING" if tr.holding else "TRACKING")
    return "%s  err %s m  lag %s  xtrack %s" % (
        state, num(tr.position_error_m), num(tr.along_track_lag_m, "%+.2f"),
        num(tr.cross_track_error_m, "%+.2f"))


def _banner_state(frame: NavFrame):
    q = frame.quality
    if q is not None:
        loc = "coast" if q.coasting else "apriltag"
        return ("LOC: %s  conf %.2f  std %.2fm" % (loc, q.confidence, q.pos_std_m),
                conf_color(q.confidence))
    t = frame.truth
    if t is None:
        return ("", palette.TEXT)
    return ("TRUTH: speed %s m/s  %s  batt %s%%" % (
        num(t.speed), t.flight_mode or t.status or "-", num(t.battery_pct, "%.0f")),
        palette.TEXT)


#: Overlay order, bottom to top: the plan first, the aircraft last, so the pose
#: is never hidden by a route drawn over it. Every layer takes the same
#: ``(img, to_px, frame)`` so :func:`_guarded` can drive them all.
_LAYERS = (_draw_routes, _draw_target, _draw_trail, _draw_drift, _draw_reference,
           _draw_pose, _scale_bar)
