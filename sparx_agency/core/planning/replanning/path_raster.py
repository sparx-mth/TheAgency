"""Rasterize a world polyline onto a grid and build a route "corridor" mask.

ROS-free and 3.8-compatible. These primitives support the route-relevance test
used by the replanning policy: "did the newly observed cells fall on / near the
path I am about to fly?". The corridor is the set of grid cells within a fixed
radius of the current path; only map changes inside it are considered relevant,
so a large discovery off to the side never triggers a replan.

Cells are ``(gx, gy)`` integer indices; the grid is indexed ``grid[gy, gx]`` (see
:class:`OccupancyGrid2D`). The Bresenham primitive lives in
:mod:`grid_geometry_2d` (shared with the line-of-sight test) so both agree on
which cells a segment covers.
"""
from __future__ import annotations

from typing import List, Sequence, Tuple

import numpy as np

from sparx_agency.core.common.types import Pose2D
from sparx_agency.core.planning.environment import OccupancyGrid2D
from sparx_agency.core.planning.planners.common.grid_geometry_2d import (
    dilate_mask, line_cells)

Cell = Tuple[int, int]


def rasterize_path(points: Sequence[Pose2D], world: OccupancyGrid2D) -> List[Cell]:
    """Return the in-bounds grid cells crossed by the world polyline ``points``.

    Off-grid cells are dropped (the corridor only exists inside the map). The
    shared vertex between consecutive segments may appear twice; callers that
    need a set should deduplicate.

    Args:
        points: World waypoints (>= 1). Fewer than 2 yields at most one cell.
        world: Grid providing the world<->cell transform and bounds.

    Returns:
        List of ``(gx, gy)`` cells inside the grid.
    """
    cells: List[Cell] = []
    if not points:
        return cells
    grid_pts = [world.world_to_grid(p.x, p.y) for p in points]
    if len(grid_pts) == 1:
        gx, gy = grid_pts[0]
        return [(gx, gy)] if world.in_bounds(gx, gy) else []
    for (x0, y0), (x1, y1) in zip(grid_pts[:-1], grid_pts[1:]):
        for cx, cy in line_cells(x0, y0, x1, y1):
            if world.in_bounds(cx, cy):
                cells.append((cx, cy))
    return cells


def corridor_mask(
    points: Sequence[Pose2D], world: OccupancyGrid2D, radius_cells: int
) -> np.ndarray:
    """Boolean ``(H, W)`` mask of cells within ``radius_cells`` of the polyline.

    The path is rasterized, then dilated by ``radius_cells`` (4-connected, the
    same operator obstacles are inflated with) to form a band of half-width
    ``radius_cells`` around the route.

    Args:
        points: World waypoints of the current path.
        world: Grid providing transform / bounds / shape.
        radius_cells: Corridor half-width in cells (<= 0 = just the path line).

    Returns:
        A fresh boolean array, ``True`` inside the corridor.
    """
    mask = np.zeros((world.height, world.width), dtype=bool)
    for cx, cy in rasterize_path(points, world):
        mask[cy, cx] = True
    if radius_cells > 0 and mask.any():
        mask = dilate_mask(mask, int(radius_cells))
    return mask
