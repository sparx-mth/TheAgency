"""Turn a BEV occupancy grid into a display image and place world points on it.

Two concerns, kept out of the renderer so they can be unit-tested on their own:

  * :func:`occupancy_to_bgr` -- colour one HxW int8 grid (free/occupied/unknown,
    with an optional confidence tint on occupied cells);
  * :func:`render_bev` -- scale + vertically flip that image so world ``+y``
    points up and world ``+x`` right (the orientation an operator expects), and
    return a ``to_px(x, y)`` closure that maps a world point onto the flipped,
    scaled image so routes/pose/target all land where they belong.

A small XTEND grid is upscaled by an integer factor (crisp cells). A Sphera
exploration grid is far larger than the display, so it is first reduced by
:func:`block_max`, which keeps the most-occupied cell of each block -- plain
resampling drops one-cell-thick walls, which is exactly what a navigation debug
view must never do.

The Sphera map is also far larger than the thing being debugged: the jail is
~105 m across, so on a 900 px pane a 30 cm cross-track error is two pixels.
:func:`render_bev` therefore takes an optional ``center``/``radius_m`` window
and crops to it, padding with unknown where the window runs off the map so the
aircraft stays centred even at an edge.
"""
from __future__ import annotations

import math
from typing import Callable, Optional, Tuple

import cv2
import numpy as np

from sparx_agency.tasks.planning.nav_debug.frame import BevMap

# BGR shades on the dark HUD: unknown darkest, free mid-dark, walls near-white.
_UNKNOWN = (30, 30, 30)
_FREE = (72, 72, 72)
_OCC = (225, 225, 225)
# Low-confidence occupied cells lean orange (a speckle the planner distrusts),
# high-confidence lean white -- the same signal the A* reroute gate uses.
_OCC_LOWCONF = (60, 150, 235)


def occupancy_to_bgr(grid: np.ndarray, conf: Optional[np.ndarray] = None) -> np.ndarray:
    """Colour an HxW int8 occupancy grid (free=0, occupied=100, unknown<0) as BGR.

    Args:
        grid: HxW int8/int array on the ROS OccupancyGrid convention.
        conf: Optional HxW 0..100 confidence, co-registered with ``grid``. Where
            given, occupied cells are tinted from orange (low) to white (high).

    Returns:
        HxW3 uint8 BGR, unflipped (row 0 = the grid's ``origin_y`` row).
    """
    g = np.asarray(grid)
    h, w = g.shape[:2]
    img = np.empty((h, w, 3), np.uint8)
    img[...] = _UNKNOWN
    free = g == 0
    occ = g >= 50
    img[free] = _FREE
    img[occ] = _OCC
    if conf is not None and conf.shape[:2] == g.shape[:2]:
        c = np.clip(np.asarray(conf, np.float32) / 100.0, 0.0, 1.0)[occ][:, None]
        low = np.asarray(_OCC_LOWCONF, np.float32)
        high = np.asarray(_OCC, np.float32)
        img[occ] = (low * (1.0 - c) + high * c).astype(np.uint8)
    return img


def block_max(grid: np.ndarray, k: int) -> np.ndarray:
    """Reduce ``grid`` by ``k`` in both axes, keeping each block's largest cell.

    On the ROS occupancy convention that ordering is exactly the priority a
    debug view wants: an occupied cell (100) survives a block of free ones (0),
    and free survives unknown (-1), so thin walls do not vanish when a large map
    is scaled down to the display.
    """
    if k <= 1:
        return grid
    h, w = grid.shape[:2]
    pad_h, pad_w = (-h) % k, (-w) % k
    padded = np.pad(grid, ((0, pad_h), (0, pad_w)), mode="edge") if (pad_h or pad_w) \
        else grid
    blocks = padded.reshape(padded.shape[0] // k, k, padded.shape[1] // k, k)
    return blocks.max(axis=(1, 3))


def window(bev: BevMap, conf: Optional[np.ndarray], center: Tuple[float, float],
           radius_m: float):
    """Crop ``bev`` to a square of ``radius_m`` about ``center``.

    The window is padded with unknown wherever it runs off the map, so the
    centre of the returned grid is always ``center`` -- an aircraft flying near
    a map edge stays put in the pane instead of sliding into a corner.

    Args:
        bev: The occupancy snapshot to crop.
        conf: Optional confidence grid, co-registered with ``bev.grid``.
        center: World ``(x, y)`` to centre on.
        radius_m: Half-width of the window, metres.

    Returns:
        ``(grid, conf, origin_x, origin_y)`` for the cropped view.
    """
    grid = np.asarray(bev.grid)
    half = max(1, int(round(radius_m / float(bev.resolution))))
    col, row = bev.world_to_cell(center[0], center[1])
    c0, r0 = int(round(col)) - half, int(round(row)) - half
    size = 2 * half + 1

    out = np.full((size, size), -1, grid.dtype)
    out_c = None if conf is None else np.zeros((size, size), np.asarray(conf).dtype)
    # Overlap between the requested window and the real grid.
    sr0, sc0 = max(0, r0), max(0, c0)
    sr1, sc1 = min(grid.shape[0], r0 + size), min(grid.shape[1], c0 + size)
    if sr1 > sr0 and sc1 > sc0:
        out[sr0 - r0:sr1 - r0, sc0 - c0:sc1 - c0] = grid[sr0:sr1, sc0:sc1]
        if out_c is not None:
            src = np.asarray(conf)
            if src.shape[:2] == grid.shape[:2]:
                out_c[sr0 - r0:sr1 - r0, sc0 - c0:sc1 - c0] = src[sr0:sr1, sc0:sc1]
    return (out, out_c,
            bev.origin_x + c0 * bev.resolution,
            bev.origin_y + r0 * bev.resolution)


def render_bev(bev: BevMap, conf: Optional[np.ndarray] = None,
               target_px: int = 720, center: Optional[Tuple[float, float]] = None,
               radius_m: float = 0.0
               ) -> Tuple[np.ndarray, Callable[[float, float], Tuple[int, int]]]:
    """Display image (``+y`` up) plus a world->pixel mapper onto it.

    Args:
        bev: The occupancy snapshot (grid + geometry).
        conf: Optional confidence grid for the occupied-cell tint.
        target_px: Roughly the longest display edge. A grid smaller than that is
            upscaled by an integer factor so cells stay crisp (nearest); a grid
            larger than it is reduced by an integer factor of :func:`block_max`.
        center: World ``(x, y)`` to centre the view on. Ignored unless
            ``radius_m`` is positive.
        radius_m: Half-width of the view, metres. ``0`` draws the whole map --
            on which, for a 105 m Sphera map, a 30 cm tracking error is about
            two pixels. A few metres is what makes the error legible.

    Returns:
        ``(image, to_px)`` where ``to_px(x, y)`` returns integer ``(px, py)`` on
        ``image`` for a world point in the BEV frame.
    """
    target_px = max(16, int(target_px))
    if radius_m > 0.0 and center is not None:
        grid, conf, origin_x, origin_y = window(bev, conf, center, radius_m)
    else:
        grid, origin_x, origin_y = np.asarray(bev.grid), bev.origin_x, bev.origin_y
    k = max(1, int(math.ceil(max(grid.shape[:2]) / float(target_px))))
    color = occupancy_to_bgr(block_max(grid, k), _reduce_conf(conf, k))
    h, w = color.shape[:2]
    up = max(1, int(target_px // max(h, w)))
    disp = cv2.resize(color, (w * up, h * up), interpolation=cv2.INTER_NEAREST)
    disp = cv2.flip(disp, 0)      # world +y up (grid row 0 is min-y -> image bottom)
    rows = disp.shape[0]
    scale = up / float(k)         # original grid cell -> display pixels
    res = float(bev.resolution)

    def to_px(x: float, y: float) -> Tuple[int, int]:
        col, row = (x - origin_x) / res, (y - origin_y) / res
        return (int(round(col * scale)), int(round((rows - 1) - row * scale)))

    return disp, to_px


def _reduce_conf(conf: Optional[np.ndarray], k: int) -> Optional[np.ndarray]:
    """Reduce the confidence grid alongside the occupancy one, or pass it through."""
    if conf is None or k <= 1:
        return conf
    return block_max(np.asarray(conf), k)
