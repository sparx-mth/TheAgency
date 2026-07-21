"""Render one :class:`NavFrame` as the nav-debug screen (map + telemetry panel).

Left: the BEV map with the raw / corrected / final routes, the target waypoint,
the drone pose + a localization trail and the drift vector, under a banner naming
the active replan reason and localization state. Right: a telemetry panel with
two ROLL/PITCH/YAW gauge stacks -- the command WE send (``cmd_vel``, green) and
the command the converter sends the DRONE (``cmd_nav`` counts, cyan) -- plus a
confidence bar, the "why", and command/confidence history strips.

The two gauge stacks are drawn so the *same physical motion reads the same way*
in both, despite the converter inverting the lateral and yaw signs: the drone
lateral/yaw counts are negated before the gauge so "drone going left" points left
in both stacks. ``cv2`` + ``numpy`` only; the same renderer serves the offline
player and a future live viewer node.
"""
from __future__ import annotations

import math
from typing import List, Optional, Tuple

import cv2
import numpy as np

from sparx_agency.tasks.planning.hud import gauges, palette
from sparx_agency.tasks.planning.hud.panel import (
    hline, pad_to_height, put_line, spark, wrap_text,
)
from sparx_agency.tasks.planning.nav_debug.bev_image import render_bev
from sparx_agency.tasks.planning.nav_debug.frame import GaugeScales, NavFrame, Routes

PANEL_WIDTH = 340
MAP_TARGET_PX = 720
_G = gauges.GAUGE_SIZE

# route colours (BGR): raw plan faint, corrected blue, final flown green.
_ASTAR = (120, 120, 120)
_SAFE = (210, 140, 40)
_FINAL = palette.GREEN
_TARGET = (0, 215, 255)      # amber ring on the active waypoint
_GOAL = (200, 60, 200)       # magenta goal
_LOOKAHEAD = (250, 190, 60)  # cyan aim point
_TRAIL = (90, 130, 90)
_DRIFT = palette.ORANGE
_KIND_COLOR = {
    "time": palette.GRAY, "rotation": (250, 190, 60), "obstacle": palette.RED,
    "blockage": palette.RED, "boxed_in": palette.RED, "info": palette.GRAY,
}


def render(frame: NavFrame, scales: Optional[GaugeScales] = None,
           map_px: int = MAP_TARGET_PX) -> np.ndarray:
    """The full debug screen for ``frame``: map pane beside the telemetry panel."""
    scales = scales or GaugeScales()
    pane = _map_pane(frame, map_px)
    panel = _panel(frame, scales)
    h = max(pane.shape[0], panel.shape[0])
    top = np.hstack([pad_to_height(pane, h), pad_to_height(panel, h)])
    # The narration goes in a full-width caption BELOW everything, wrapped, so a
    # long line is never overwritten by the map or the panel.
    return np.vstack([top, _caption_bar(top.shape[1], frame)])


# ── left: the BEV map pane ───────────────────────────────────────────────────
def _clip(p: Tuple[int, int]) -> Tuple[int, int]:
    return (int(max(-20000, min(20000, p[0]))), int(max(-20000, min(20000, p[1]))))


def _map_pane(frame: NavFrame, map_px: int) -> np.ndarray:
    if frame.bev is None:
        img = np.full((map_px, map_px, 3), (28, 28, 28), np.uint8)
        put_line(img, "no BEV map recorded yet", 20, map_px // 2, palette.MUTED, 0.7)
        to_px = None
    else:
        img, to_px = render_bev(frame.bev, frame.bev_conf, map_px)
    if to_px is not None:
        _draw_routes(img, to_px, frame.routes)
        _draw_target(img, to_px, frame)
        _draw_trail(img, to_px, frame.trail)
        _draw_drift(img, to_px, frame)
        _draw_pose(img, to_px, frame)
        _scale_bar(img, to_px)
    _banner(img, frame)
    return img


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


def _draw_routes(img, to_px, routes: Routes) -> None:
    _polyline(img, to_px, routes.astar, _ASTAR, 1, dotted=True)
    _polyline(img, to_px, routes.safe, _SAFE, 2)
    _polyline(img, to_px, routes.final, _FINAL, 3)
    if routes.final:                     # mark the flown waypoints
        for x, y in routes.final:
            cv2.circle(img, _clip(to_px(x, y)), 3, _FINAL, -1)
    if routes.goal:
        gp = _clip(to_px(*routes.goal))
        cv2.drawMarker(img, gp, _GOAL, cv2.MARKER_STAR, 16, 2)
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
    cv2.putText(img, label, (p[0] + 12, p[1] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                _TARGET, 1, cv2.LINE_AA)


def _draw_trail(img, to_px, trail) -> None:
    if trail and len(trail) >= 2:
        _polyline(img, to_px, trail, _TRAIL, 1)


def _draw_pose(img, to_px, frame: NavFrame) -> None:
    p = _clip(to_px(frame.x, frame.y))
    conf = frame.quality.confidence if frame.quality is not None else 1.0
    color = _conf_color(conf)
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
    k = 2.0    # metres drawn per (m/s) of learned drift, for visibility
    wx = frame.x + k * (d.drift_vx * cy - d.drift_vy * sy)
    wy = frame.y + k * (d.drift_vx * sy + d.drift_vy * cy)
    cv2.arrowedLine(img, _clip(to_px(frame.x, frame.y)), _clip(to_px(wx, wy)),
                    _DRIFT, 2, tipLength=0.3)


def _scale_bar(img, to_px) -> None:
    p0 = to_px(0.0, 0.0)
    p1 = to_px(1.0, 0.0)
    length = abs(p1[0] - p0[0]) or 1
    h = img.shape[0]
    x0, y0 = 14, h - 46
    cv2.line(img, (x0, y0), (x0 + length, y0), palette.WHITE, 2)
    cv2.putText(img, "1 m", (x0, y0 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                palette.WHITE, 1, cv2.LINE_AA)


def _strip(img, y0: int, y1: int) -> None:
    """Darken a horizontal band so overlaid text stays readable over the map."""
    y1 = min(y1, img.shape[0])
    band = img[y0:y1].astype(np.float32) * 0.35
    img[y0:y1] = band.astype(np.uint8)


def _banner(img, frame: NavFrame) -> None:
    _strip(img, 0, 52)
    state = frame.drift.state if frame.drift is not None else "-"
    put_line(img, "MODE: %s" % (state or "-"), 12, 20, palette.TEXT, 0.55)
    if frame.quality is not None:
        loc = "coast" if frame.quality.coasting else "apriltag"
        put_line(img, "LOC: %s  conf %.2f  std %.2fm" % (
            loc, frame.quality.confidence, frame.quality.pos_std_m),
            12, 42, _conf_color(frame.quality.confidence), 0.5)
    # The replan reason + the drift "why" render in the full-width caption bar
    # below everything (never truncated), not overlaid on the map -- see _caption_bar.


def _caption_bar(width: int, frame: NavFrame, scale: float = 0.55) -> np.ndarray:
    """Full-width caption below everything: the replan reason + the drift 'why'.

    Word-wrapped and never truncated -- a long line flows onto more lines and the
    bar grows to fit, so the narration is never overwritten by the map or panel.
    """
    segs = []   # (text, color)
    if frame.replan is not None:
        txt = frame.replan.text
        for pre in ("REPLAN:", "BLOCKAGE:", "STOP:"):     # kind already says this
            if txt.upper().startswith(pre):
                txt = txt[len(pre):].strip()
                break
        segs.append(("REPLAN [%s]: %s  (%.1fs ago)"
                     % (frame.replan.kind, txt, frame.replan.age_s),
                     _KIND_COLOR.get(frame.replan.kind, palette.GRAY)))
    if frame.why:
        segs.append(("WHY: %s" % frame.why, palette.TEXT))
    if not segs:
        segs.append(("flying the route -- no replan or drift note this tick",
                     palette.MUTED))

    pad = 14
    lines = []   # (text, color)
    for text, color in segs:
        for ln in wrap_text(text, width - 2 * pad, scale):
            lines.append((ln, color))
    bar = np.full((pad + len(lines) * 22 + pad, width, 3), (18, 18, 18), np.uint8)
    cv2.line(bar, (0, 0), (width - 1, 0), (70, 70, 70), 1)   # divider from panes above
    y = pad + 12
    for ln, color in lines:
        put_line(bar, ln, pad, y, color, scale)
        y += 22
    return bar


def _conf_color(conf: float):
    c = max(0.0, min(1.0, conf))
    lo = np.asarray(palette.RED, np.float32)
    hi = np.asarray(palette.GREEN, np.float32)
    return tuple(int(v) for v in (lo * (1.0 - c) + hi * c))


# ── right: the telemetry panel ───────────────────────────────────────────────
def _panel(frame: NavFrame, scales: GaugeScales, width: int = PANEL_WIDTH) -> np.ndarray:
    panel = np.full((780, width, 3), palette.PANEL_BG, dtype=np.uint8)
    x, y = 12, 26
    y = put_line(panel, "NAV DEBUG", x, y, palette.MUTED, 0.55)
    y = hline(panel, y, width)

    our = frame.our_cmd
    y = _gauge_set(panel, x, y, "OURS (cmd_vel)", palette.GREEN,
                   roll=(our[1] if our else 0.0), pitch=(our[0] if our else 0.0),
                   yaw=(our[3] if our else 0.0),
                   roll_fs=scales.our_vy, pitch_fs=scales.our_vx, yaw_fs=scales.our_wz,
                   numbers=("vx%+.2f vy%+.2f" % (our[0], our[1]) if our else "no cmd",
                            "wz%+.2f vz%+.2f" % (our[3], our[2]) if our else ""))
    y = hline(panel, y, width)

    d = frame.drone_cmd
    # negate lateral & yaw counts so the gauges read the same direction as OURS.
    y = _gauge_set(panel, x, y, "TO DRONE (cmd_nav)", palette.CYAN,
                   roll=(-d[1] if d else 0.0), pitch=(d[0] if d else 0.0),
                   yaw=(-d[3] if d else 0.0),
                   roll_fs=scales.drone_lateral, pitch_fs=scales.drone_forward,
                   yaw_fs=scales.drone_yaw,
                   numbers=("fwd%d lat%d" % (d[0], d[1]) if d else "no cmd_nav",
                            "yaw%d vert%d" % (d[3], d[2]) if d else ""))
    y = hline(panel, y, width)

    if frame.quality is not None:
        q = frame.quality
        y = _conf_bar(panel, x, y, width, q.confidence)
        y = put_line(panel, "std %.2fm  age %.2fs  eff %.2f" % (
            q.pos_std_m, q.age_s, q.cmd_effectiveness), x, y, palette.MUTED, 0.45)
    y += 4
    if frame.cmd_history:
        put_line(panel, "cmd vx", x, y + 10, palette.MUTED, 0.4)
        spark(panel, frame.cmd_history, x + 70, y, width - 90, 26, palette.GREEN)
        y += 34
    if frame.conf_history:
        put_line(panel, "conf", x, y + 10, palette.MUTED, 0.4)
        spark(panel, frame.conf_history, x + 70, y, width - 90, 26,
              palette.CYAN, lo=0.0, hi=1.0)
        y += 34
    return panel[:min(y + 8, panel.shape[0])]


def _gauge_set(panel, x, y, title, color, roll, pitch, yaw, roll_fs, pitch_fs,
               yaw_fs, numbers) -> int:
    y = put_line(panel, title, x, y, color, 0.5)
    row = [("ROLL", gauges.draw_roll_gauge(roll, roll_fs, color)),
           ("PITCH", gauges.draw_pitch_gauge(pitch, pitch_fs, color)),
           ("YAW", gauges.draw_yaw_gauge(yaw, yaw_fs, color))]
    gap = 8
    gx, gy = x, y
    for label, g in row:
        panel[gy:gy + _G, gx:gx + _G] = g
        put_line(panel, label, gx + 2, gy + _G + 16, palette.MUTED, 0.42)
        gx += _G + gap
    y = gy + _G + 36        # clear the gauge labels before the numbers line
    for line in numbers:
        if line:
            y = put_line(panel, line, x, y, palette.TEXT, 0.45)
    return y


def _conf_bar(panel, x, y, width, conf) -> int:
    w = width - 2 * x
    cv2.rectangle(panel, (x, y), (x + w, y + 12), (60, 60, 60), 1)
    fill = int(round(max(0.0, min(1.0, conf)) * (w - 2)))
    if fill > 0:
        cv2.rectangle(panel, (x + 1, y + 1), (x + 1 + fill, y + 11),
                      _conf_color(conf), -1)
    cv2.putText(panel, "conf %.2f" % conf, (x + w - 78, y + 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, palette.WHITE, 1, cv2.LINE_AA)
    return y + 22
