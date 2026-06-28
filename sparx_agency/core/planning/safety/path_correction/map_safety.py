"""Shared map-aware safety post-processing for path correctors.

Every :class:`PathCorrector` strategy, after it has moved waypoints, applies the
same two map-aware steps so a corrected path stays trustworthy. They live here
once (DRY) rather than being re-implemented per strategy:

* :func:`dampen_unknown` -- scale each waypoint's shift by the fraction of KNOWN
  cells around its corrected position, so a push into half-mapped space (where no
  opposing wall balances it) is damped, fading back to full strength as the map
  fills in.
* :func:`clip_to_clear` -- pull any corrected waypoint back toward its input
  position just far enough to keep both adjacent segments clear of inflated
  obstacles, so the corrected path is never less safe than the input.

ROS-free, numpy-only; Python 3.8 compatible (the FALCON Noetic adapter imports
core under 3.8).
"""
from __future__ import annotations

import math
from typing import List, Sequence, Tuple

import numpy as np

from sparx_agency.core.common.types import Pose2D
from sparx_agency.core.planning.environment import OccupancyGrid2D

from .grid_collision import InflatedGridCollisionChecker


def map_confidence(grid: OccupancyGrid2D, x: float, y: float, radius_m: float) -> float:
    """Fraction of KNOWN cells in a disk of ``radius_m`` around world ``(x, y)``.

    1.0 = fully mapped neighbourhood (walls observed on both sides, so corrective
    pushes balance); lower when ``(x, y)`` is in or near unknown space, where the
    push is unbalanced. Used to scale down the correction there.
    """
    rad = max(1, int(round(radius_m / grid.resolution)))
    gx, gy = grid.world_to_grid(x, y)
    x0, x1 = max(0, gx - rad), min(grid.width, gx + rad + 1)
    y0, y1 = max(0, gy - rad), min(grid.height, gy + rad + 1)
    win = grid.grid[y0:y1, x0:x1]
    if win.size == 0:
        return 1.0
    return float(np.count_nonzero(win != grid.values.unknown)) / float(win.size)


def dampen_unknown(
    raw_points: Sequence[Pose2D],
    corrected_points: Sequence[Pose2D],
    grid: OccupancyGrid2D,
    radius_m: float,
) -> Tuple[Pose2D, ...]:
    """Scale each waypoint's correction by the map confidence at its corrected
    position, so a push into/near unknown space (no opposing wall to balance it)
    is damped while a push to the centre of a fully-mapped corridor is kept at full
    strength. ``final = raw + confidence * (corrected - raw)``.
    """
    out: List[Pose2D] = []
    for r, c in zip(raw_points, corrected_points):
        conf = map_confidence(grid, c.x, c.y, radius_m)
        if conf >= 1.0 - 1e-6:
            out.append(c)
        else:
            out.append(Pose2D(r.x + conf * (c.x - r.x),
                              r.y + conf * (c.y - r.y), c.yaw))
    return tuple(out)


def clip_to_clear(
    raw_points: Sequence[Pose2D],
    safe_points: Sequence[Pose2D],
    grid: OccupancyGrid2D,
    inflate_radius_m: float,
) -> Tuple[Pose2D, ...]:
    """Per-waypoint safety clip of a corrected path against inflated obstacles.

    Each interior waypoint is pulled back toward its raw position only as far as
    needed to keep BOTH its adjacent segments clear (bisection), so a single
    corner-cut reverts just that waypoint while the rest stay corrected. Never less
    safe than the input: a waypoint that cannot be cleared falls back to its raw
    position. Endpoints stay pinned.
    """
    checker = InflatedGridCollisionChecker(grid, inflate_radius_m)
    out = list(safe_points)
    n = len(out)
    if n < 3:
        return tuple(out)

    def clear(a: Pose2D, b: Pose2D) -> bool:
        return checker.segment_clear(a, b)

    # Re-evaluate EVERY interior waypoint to its most-centred clear position each
    # sweep (not only colliding ones): pulling one waypoint back can later free a
    # neighbour to return to full correction, so we recompute rather than latch a
    # one-time revert. Converges in a few sweeps; endpoints stay pinned.
    for _sweep in range(3):
        changed = False
        for i in range(1, n - 1):
            full = safe_points[i]                # t=1: full correction
            if clear(out[i - 1], full) and clear(full, out[i + 1]):
                new = full
            else:
                rx, ry = raw_points[i].x, raw_points[i].y
                sx, sy = safe_points[i].x, safe_points[i].y
                lo, hi, best = 0.0, 1.0, raw_points[i]   # t: 0=raw .. 1=corrected
                for _ in range(6):           # bisect for the most-centred clear t
                    t = 0.5 * (lo + hi)
                    cand = Pose2D(rx + t * (sx - rx), ry + t * (sy - ry), full.yaw)
                    if clear(out[i - 1], cand) and clear(cand, out[i + 1]):
                        best, lo = cand, t
                    else:
                        hi = t
                new = best
            if math.hypot(new.x - out[i].x, new.y - out[i].y) > 1e-6:
                out[i] = new
                changed = True
        if not changed:
            break
    return tuple(out)
