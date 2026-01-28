"""
Collision checking utilities for OccupancyGrid2D.

By default, UNKNOWN is treated as not-free because OccupancyGrid2D.is_free()
only returns True for explicit FREE cells. This is a conservative choice for
collision checking.
"""

from __future__ import annotations

from skimage.draw import line

from sparx_agency.core.planning.environment.occupancy_grid2d import OccupancyGrid2D


def is_state_free(grid: OccupancyGrid2D, x: float, y: float) -> bool:
    """
    Return True if the world position (x, y) maps to a FREE cell.

    UNKNOWN and out-of-bounds are treated as not-free.
    """
    gx, gy = grid.world_to_grid(x, y)
    return grid.is_free(gx, gy)


def is_segment_free(
    grid: OccupancyGrid2D,
    x0: float,
    y0: float,
    x1: float,
    y1: float,
) -> bool:
    """
    Return True if all grid cells intersected by the world segment are FREE.

    UNKNOWN and out-of-bounds are treated as not-free.
    """
    gx0, gy0 = grid.world_to_grid(x0, y0)
    gx1, gy1 = grid.world_to_grid(x1, y1)

    rows, cols = line(gy0, gx0, gy1, gx1)
    for gy, gx in zip(rows, cols):
        if not grid.is_free(gx, gy):
            return False
    return True
