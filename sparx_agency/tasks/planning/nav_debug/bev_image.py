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


def render_bev(bev: BevMap, conf: Optional[np.ndarray] = None,
               target_px: int = 720) -> Tuple[np.ndarray, Callable[[float, float], Tuple[int, int]]]:
    """Display image (``+y`` up) plus a world->pixel mapper onto it.

    Args:
        bev: The occupancy snapshot (grid + geometry).
        conf: Optional confidence grid for the occupied-cell tint.
        target_px: Roughly the longest display edge. A grid smaller than that is
            upscaled by an integer factor so cells stay crisp (nearest); a grid
            larger than it is reduced by an integer factor of :func:`block_max`.

    Returns:
        ``(image, to_px)`` where ``to_px(x, y)`` returns integer ``(px, py)`` on
        ``image`` for a world point in the BEV frame.
    """
    target_px = max(16, int(target_px))
    grid = np.asarray(bev.grid)
    k = max(1, int(math.ceil(max(grid.shape[:2]) / float(target_px))))
    color = occupancy_to_bgr(block_max(grid, k), _reduce_conf(conf, k))
    h, w = color.shape[:2]
    up = max(1, int(target_px // max(h, w)))
    disp = cv2.resize(color, (w * up, h * up), interpolation=cv2.INTER_NEAREST)
    disp = cv2.flip(disp, 0)      # world +y up (grid row 0 is min-y -> image bottom)
    rows = disp.shape[0]
    scale = up / float(k)         # original grid cell -> display pixels

    def to_px(x: float, y: float) -> Tuple[int, int]:
        col, row = bev.world_to_cell(x, y)
        return (int(round(col * scale)), int(round((rows - 1) - row * scale)))

    return disp, to_px


def _reduce_conf(conf: Optional[np.ndarray], k: int) -> Optional[np.ndarray]:
    """Reduce the confidence grid alongside the occupancy one, or pass it through."""
    if conf is None or k <= 1:
        return conf
    return block_max(np.asarray(conf), k)
