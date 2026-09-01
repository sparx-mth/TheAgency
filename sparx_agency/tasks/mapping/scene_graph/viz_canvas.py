"""Palette and panel geometry shared by every layer of the dashboard.

The map panel is an ordinary picture — row 0 at maximum y — while the
``nav_msgs/OccupancyGrid`` layers drawn onto it are y-up, row 0 at minimum y.
:func:`world_to_px` and :func:`warp_grid` are the only two places that flip, so
no caller flips again.

The world->pixel affine mirrors ``sjtu_internvla_n1.map_backdrop`` (the +/-0.5
pixel-centre terms included) so the BEV and room tints land on the same pixels
as the backdrop walls instead of creeping half a cell against them.

Colors are BGR, and the room/object palettes are **not** defined here: they come
from :func:`sparx_agency.core.mapping.topology.room_color` and
:func:`sparx_agency.core.mapping.objects.landmarks.class_color`, the same
functions the mapper puts in the ``/scene_graph`` JSON and the RViz markers, so
every view of the mission agrees on which room is which color.
"""
from __future__ import annotations

from typing import List, Tuple

import cv2
import numpy as np

from sparx_agency.core.mapping.objects.landmarks import class_color
from sparx_agency.core.mapping.topology import room_color

FONT = cv2.FONT_HERSHEY_SIMPLEX

# The backdrop underneath is the map_backdrop one (light walls on near-black
# free space), dimmed, so every overlay is chosen to read on that: light washes
# for free space, saturated marks for the live entities.
PANEL_BG = (24, 24, 26)
SIDE_BG = (30, 30, 34)
TEXT_BGR = (235, 235, 235)
DIM_TEXT_BGR = (150, 150, 155)
BEV_OCC_BGR = (200, 200, 210)     # live walls: bright, over the washed floor
BEV_FREE_BGR = (68, 62, 52)       # a warm wash where the BEV has seen floor
TRAIL_BGR = (0, 200, 0)
DRONE_BGR = (255, 255, 255)
TARGET_BGR = (0, 0, 255)
BAR_BG_BGR = (55, 55, 60)

# The topological graph, in the colors of the old RViz MarkerArray:
# discovered doors orange (1.0, 0.55, 0.0), pending doors a grey ghost
# (0.55, 0.58, 0.65), and the room->door->room links gold (1.0, 0.85, 0.0).
# Doors are deliberately *not* gold any more — they used to be, and a gold door
# sitting on a gold edge made the link invisible exactly where it matters.
DOOR_BGR = (0, 140, 255)
DOOR_UNDISCOVERED_BGR = (166, 148, 140)
DOOR_LABEL_BGR = (215, 245, 255)
EDGE_BGR = (0, 217, 255)
EDGE_OUTLINE_BGR = (18, 24, 30)   # halo: the gold has to read on pale floor too
NODE_RING_BGR = (255, 255, 255)
NODE_OUTLINE_BGR = (18, 24, 30)
LEGEND_BG = (16, 16, 18)
LEGEND_BORDER_BGR = (95, 95, 102)
CHIP_BG = (8, 8, 10)

# The object -> parent-room link, in the color of the old RViz
# ``room_object_edges`` marker: ColorRGBA(0.2, 0.7, 0.2, 0.55). Semi-transparent
# green on purpose — the link says "this object belongs to that room" and must
# read as an underlay beneath the gold door topology, never compete with it.
OBJ_EDGE_BGR = (51, 178, 51)

BACKDROP_DIM = 0.55
BEV_FREE_ALPHA = 0.45
ROOM_TINT_ALPHA = 0.35
OBJ_EDGE_ALPHA = 0.55
LEGEND_ALPHA = 0.80
CHIP_ALPHA = 0.80


def world_to_px(extent, size, x, y):
    # type: (Tuple[float, float, float, float], Tuple[int, int], float, float) -> Tuple[int, int]
    """World metres -> panel pixel (row 0 at max y), clamped into the panel."""
    min_x, min_y, max_x, max_y = extent
    w, h = size
    col = int(np.floor((x - min_x) * w / max(1e-6, max_x - min_x)))
    row = int(np.floor((max_y - y) * h / max(1e-6, max_y - min_y)))
    return (min(max(col, 0), w - 1), min(max(row, 0), h - 1))


def pixels_per_metre(extent, size):
    # type: (Tuple[float, float, float, float], Tuple[int, int]) -> float
    """Panel pixels one world metre spans, for sizing marks and gating text.

    The extent is fitted to the data (or to the backdrop), so this is the
    dashboard's live zoom: a hospital-wide view gives a handful of pixels per
    metre and a room-sized view gives dozens. Anything measured in metres —
    door glyphs, whether a door label can be read at all — is derived from
    this rather than hardcoded to one map.
    """
    min_x, _, max_x, _ = extent
    return float(size[0]) / max(1e-6, float(max_x) - float(min_x))


def compute_extent(state, backdrop, size):
    # type: (dict, object, Tuple[int, int]) -> Tuple[float, float, float, float]
    """The world rectangle the map panel shows, at one uniform scale.

    With a backdrop the map fixes it (letterboxed); without one it is fitted to
    whatever live data exists — BEV/room-grid corners, trail, room centroids —
    so the panel still works before the hospital map is configured.
    """
    w, h = int(size[0]), int(size[1])
    if backdrop is not None:
        return backdrop.whole((w, h)).extent
    pts = []  # type: List[Tuple[float, float]]
    for key in ("bev", "room_grid"):
        layer = state.get(key)
        if layer is not None:
            gh, gw = layer["grid"].shape
            ox, oy = layer["origin"]
            res = layer["resolution"]
            pts += [(ox, oy), (ox + gw * res, oy + gh * res)]
    pts += [(float(p[0]), float(p[1])) for p in state.get("trail") or []]
    graph = state.get("scene_graph") or {}
    pts += [tuple(r["centroid"]) for r in graph.get("rooms", [])]
    if state.get("pose") is not None:
        pts.append((state["pose"][0], state["pose"][1]))
    if not pts:
        pts = [(-1.0, -1.0), (1.0, 1.0)]
    xs, ys = [p[0] for p in pts], [p[1] for p in pts]
    cx, cy = 0.5 * (min(xs) + max(xs)), 0.5 * (min(ys) + max(ys))
    span_x = max(xs) - min(xs) + 3.0
    span_y = max(ys) - min(ys) + 3.0
    mpp = max(span_x / float(w), span_y / float(h))
    return (cx - 0.5 * w * mpp, cy - 0.5 * h * mpp,
            cx + 0.5 * w * mpp, cy + 0.5 * h * mpp)


def warp_grid(grid_u8, resolution, origin_xy, extent, size):
    # type: (np.ndarray, float, Tuple[float, float], Tuple[float, float, float, float], Tuple[int, int]) -> np.ndarray
    """Resample a y-up world grid into panel pixels (nearest, 0-filled)."""
    w, h = size
    min_x, min_y, max_x, max_y = extent
    sx = (max_x - min_x) / float(w) / resolution
    sy = (max_y - min_y) / float(h) / resolution
    transform = np.array([
        [sx, 0.0, (min_x - origin_xy[0]) / resolution + 0.5 * (sx - 1.0)],
        [0.0, -sy, (max_y - origin_xy[1]) / resolution - 0.5 * (sy + 1.0)],
    ], dtype=np.float64)
    return cv2.warpAffine(grid_u8, transform, (w, h),
                          flags=cv2.INTER_NEAREST | cv2.WARP_INVERSE_MAP,
                          borderMode=cv2.BORDER_CONSTANT, borderValue=0)


def blend_color(panel, mask, color, alpha):
    """Alpha-blend a flat color into ``panel`` where ``mask`` is True."""
    if not np.any(mask):
        return
    layer = np.array(color, dtype=np.float32)
    panel[mask] = (panel[mask].astype(np.float32) * (1.0 - alpha)
                   + layer * alpha).astype(np.uint8)


def blend_box(panel, x0, y0, x1, y1, color, alpha):
    """Alpha-blend a flat color over a rectangle of ``panel`` (clipped)."""
    x0, y0 = max(0, int(x0)), max(0, int(y0))
    x1 = min(int(x1), panel.shape[1])
    y1 = min(int(y1), panel.shape[0])
    if x1 <= x0 or y1 <= y0:
        return
    roi = panel[y0:y1, x0:x1].astype(np.float32)
    panel[y0:y1, x0:x1] = (roi * (1.0 - alpha)
                           + np.array(color, dtype=np.float32) * alpha
                           ).astype(np.uint8)


def room_bgr(pid):
    # type: (int) -> Tuple[int, int, int]
    """The stable room palette color, as OpenCV BGR."""
    r, g, b = room_color(int(pid))
    return (int(b * 255), int(g * 255), int(r * 255))


def class_bgr(class_name):
    # type: (str) -> Tuple[int, int, int]
    """The stable object-class palette color, as OpenCV BGR."""
    r, g, b = class_color(str(class_name))
    return (int(b * 255), int(g * 255), int(r * 255))
