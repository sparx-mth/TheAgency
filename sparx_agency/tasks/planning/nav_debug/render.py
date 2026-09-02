"""Render one :class:`NavFrame` as the nav-debug screen.

The screen is a map pane, one or two telemetry columns, and a full-width caption
bar underneath:

  * :mod:`.render_map` -- the BEV map with the routes, the pose and, on the
    Sphera stack, the reference being chased and the gap to it;
  * :mod:`.render_panel` -- the command chain, from the ``cmd_vel`` we ask for
    down to the axis counts the Rooster is actually sent;
  * :mod:`.render_lanes` -- the outcome: reference, tracking, altitude, ground
    truth and map quality. Drawn **only** when the run recorded any of them, so
    an XTEND recording renders the two-pane screen it always did;
  * :func:`_caption_bar` -- the replan reason and the narration, word-wrapped so
    a long line is never overwritten by the panes above it.

Both gauge stacks are drawn so the *same physical motion reads the same way* in
both, despite the converter inverting the lateral and yaw signs. ``cv2`` +
``numpy`` only; the same renderer serves the offline player and a live viewer.
"""
from __future__ import annotations

from typing import Optional

import cv2
import numpy as np

from sparx_agency.tasks.planning.hud import palette
from sparx_agency.tasks.planning.hud.panel import pad_to_height, put_line, wrap_text
from sparx_agency.tasks.planning.nav_debug.frame import GaugeScales, NavFrame
from sparx_agency.tasks.planning.nav_debug.render_lanes import (
    LANE_WIDTH, build_lane_column, has_lane_data,
)
from sparx_agency.tasks.planning.nav_debug.render_map import (
    MAP_TARGET_PX, build_map_pane,
)
from sparx_agency.tasks.planning.nav_debug.render_panel import PANEL_WIDTH, build_panel

#: The Rooster/Sphera envelope, 3-4x the XTEND one: the measured expo curve tops
#: out at 900 counts = 1.566 m/s on both horizontal axes, and yaw reaches
#: 2.589 rad/s at full stick. Vertical is the command unit's own throttle axis
#: (+/-1000 counts), which the planner never commands directly.
ROOSTER_SCALES = GaugeScales(
    our_vx=1.566, our_vy=1.566, our_vz=0.5, our_wz=2.589,
    drone_forward=900.0, drone_lateral=900.0, drone_vertical=1000.0,
    drone_yaw=1000.0)

_KIND_COLOR = {
    "time": palette.GRAY, "rotation": (250, 190, 60), "obstacle": palette.RED,
    "blockage": palette.RED, "boxed_in": palette.RED, "info": palette.GRAY,
}

# Re-exported so callers keep importing the screen's geometry from one module.
__all__ = ["render", "default_scales", "ROOSTER_SCALES", "MAP_TARGET_PX",
           "PANEL_WIDTH", "LANE_WIDTH"]


def render(frame: NavFrame, scales: Optional[GaugeScales] = None,
           map_px: int = MAP_TARGET_PX) -> np.ndarray:
    """The full debug screen for ``frame``: map pane beside the telemetry columns.

    Args:
        frame: The moment to draw. Every optional lane is skipped when absent.
        scales: Gauge full-scales. ``None`` picks :data:`ROOSTER_SCALES` for a
            run with axis traces and the XTEND defaults otherwise.
        map_px: Roughly the longest edge of the map pane.

    Returns:
        An HxWx3 uint8 BGR image.
    """
    scales = scales if scales is not None else default_scales(frame)
    pane = build_map_pane(frame, map_px)
    columns = [build_panel(frame, scales)]
    if has_lane_data(frame):
        columns.append(build_lane_column(frame, scales))
    height = max([pane.shape[0]] + [col.shape[0] for col in columns])
    top = np.hstack([pad_to_height(pane, height)]
                    + [pad_to_height(col, height, palette.PANEL_BG)
                       for col in columns])
    # The narration goes in a full-width caption BELOW everything, wrapped, so a
    # long line is never overwritten by the map or the panels.
    return np.vstack([top, _caption_bar(top.shape[1], frame)])


def default_scales(frame: NavFrame) -> GaugeScales:
    """Rooster full-scales for a Rooster frame, XTEND's otherwise.

    Which airframe flew is a property of the RUN, not of one moment -- but
    ``frame.axes`` empties on ordinary frames (the adapter publishes an empty
    trace while stopped, and any gap wider than the lane freshness window blanks
    it), and flipping between envelopes that differ by 3.5x mid-replay makes
    every command gauge lie. So take evidence from any Rooster-only lane, and
    prefer passing explicit scales chosen once per run (the player does).
    """
    rooster = frame.axes or frame.actuator is not None or frame.altitude is not None
    return ROOSTER_SCALES if rooster else GaugeScales()


def _caption_bar(width: int, frame: NavFrame, scale: float = 0.55) -> np.ndarray:
    """Full-width caption below everything: the replan reason and the "why".

    Word-wrapped and never truncated -- a long line flows onto more lines and the
    bar grows to fit, so the narration is never overwritten by the map or panels.
    """
    segs = _caption_segments(frame)
    pad = 14
    lines = []   # (text, color)
    for text, color in segs:
        for ln in wrap_text(text, width - 2 * pad, scale):
            lines.append((ln, color))
    bar = np.full((pad + len(lines) * 22 + pad, width, 3), (18, 18, 18), np.uint8)
    cv2.line(bar, (0, 0), (width - 1, 0), (70, 70, 70), 1)   # divider from above
    y = pad + 12
    for ln, color in lines:
        put_line(bar, ln, pad, y, color, scale)
        y += 22
    return bar


def _caption_segments(frame: NavFrame):
    """The (text, colour) lines the caption is built from, in reading order."""
    segs = []
    if frame.replan is not None:
        segs.append(("REPLAN [%s]: %s  (%.1fs ago)" % (
            frame.replan.kind, _replan_text(frame.replan), frame.replan.age_s),
            _KIND_COLOR.get(frame.replan.kind, palette.GRAY)))
    if frame.reference is not None and not frame.reference.moving:
        segs.append(("REFERENCE FROZEN: traj_server is republishing the last point "
                     "with fresh stamps -- the aircraft is holding, not tracking.",
                     palette.AMBER))
    if frame.why:
        segs.append(("WHY: %s" % frame.why, palette.TEXT))
    if not segs:
        segs.append(("flying the route -- no replan or drift note this tick",
                     palette.MUTED))
    return segs


def _replan_text(replan) -> str:
    """The planner's own string, minus a prefix the ``kind`` already says."""
    text = replan.text
    for prefix in ("REPLAN:", "BLOCKAGE:", "STOP:"):
        if text.upper().startswith(prefix):
            return text[len(prefix):].strip()
    return text
