"""
Collision checking utilities for Costmap2D.

Costmap2D is a binary grid (free/occupied). These utilities implement:
    - point collision check
    - segment collision check (Bresenham rasterization)
    - optional collision ratio along a segment (soft cost)
"""

from __future__ import annotations

from skimage.draw import line

from sparx_agency.core.planning.environment.costmap2d import Costmap2D


def is_state_free(costmap: Costmap2D, x: float, y: float) -> bool:
    """
    Return True if world position (x, y) is within bounds and in free space.
    """
    gx, gy = costmap.world_to_grid(x, y)
    return costmap.is_free(gx, gy)


def is_segment_free(
    costmap: Costmap2D,
    x0: float,
    y0: float,
    x1: float,
    y1: float,
) -> bool:
    """
    Return True if all grid cells intersected by the world segment are free.
    """
    gx0, gy0 = costmap.world_to_grid(x0, y0)
    gx1, gy1 = costmap.world_to_grid(x1, y1)

    rows, cols = line(gy0, gx0, gy1, gx1)
    for gy, gx in zip(rows, cols):
        if not costmap.is_free(gx, gy):
            return False
    return True


def segment_collision_ratio(
    costmap: Costmap2D,
    x0: float,
    y0: float,
    x1: float,
    y1: float,
) -> float:
    """
    Return the fraction of rasterized segment cells that are occupied.

    Returns:
        Ratio in [0, 1]. Returns 1.0 for zero-length segments.
    """
    gx0, gy0 = costmap.world_to_grid(x0, y0)
    gx1, gy1 = costmap.world_to_grid(x1, y1)

    rows, cols = line(gy0, gx0, gy1, gx1)
    n = len(rows)
    if n == 0:
        return 1.0

    occupied = 0
    for gy, gx in zip(rows, cols):
        if costmap.is_occupied(gx, gy):
            occupied += 1
    return occupied / n
