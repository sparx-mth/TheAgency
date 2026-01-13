"""Collision checking utilities for Costmap2D."""
from __future__ import annotations

from skimage.draw import line

from .costmap2d import Costmap2D


def is_state_valid(costmap: Costmap2D, x: float, y: float) -> bool:
    """
    Check if world position is collision-free.

    Args:
        costmap: Occupancy grid.
        x, y: World coordinates (meters).

    Returns:
        True if position is in free space and within bounds.
    """
    gx, gy = costmap.world_to_grid(x, y)
    return costmap.is_free(gx, gy)


def is_segment_collision_free(
    costmap: Costmap2D,
    x0: float,
    y0: float,
    x1: float,
    y1: float,
) -> bool:
    """
    Check if line segment is collision-free using Bresenham rasterization.

    Args:
        costmap: Occupancy grid.
        x0, y0: Segment start (world meters).
        x1, y1: Segment end (world meters).

    Returns:
        True if all cells along the segment are free.
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
    Compute fraction of cells along segment that are occupied.

    Useful for soft collision costs in optimization-based planners.

    Args:
        costmap: Occupancy grid.
        x0, y0: Segment start (world meters).
        x1, y1: Segment end (world meters).

    Returns:
        Ratio in [0, 1]. Returns 1.0 for zero-length segments.
    """
    gx0, gy0 = costmap.world_to_grid(x0, y0)
    gx1, gy1 = costmap.world_to_grid(x1, y1)

    rows, cols = line(gy0, gx0, gy1, gx1)
    if len(rows) == 0:
        return 1.0

    occupied = sum(1 for gy, gx in zip(rows, cols) if costmap.is_occupied(gx, gy))
    return occupied / len(rows)