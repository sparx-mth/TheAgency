"""
Collision checking helpers for Costmap2D.

Includes:
- point validity checks
- segment collision checks (grid ray marching)
"""

from __future__ import annotations

from math import hypot
from typing import Tuple

from sparx_agency.core.planning.environment.costmap2d import Costmap2D


def is_state_valid(world: Costmap2D, x: float, y: float) -> bool:
    """Check if a world (x,y) lies in free space."""
    gx, gy = world.world_to_grid(x, y)
    return world.is_free(gx, gy)


def _bresenham(gx0: int, gy0: int, gx1: int, gy1: int):
    """
    Bresenham grid traversal from (gx0,gy0) to (gx1,gy1).
    Yields integer cells along the line including endpoints.
    """
    dx = abs(gx1 - gx0)
    dy = abs(gy1 - gy0)
    x, y = gx0, gy0
    sx = 1 if gx0 < gx1 else -1
    sy = 1 if gy0 < gy1 else -1
    err = dx - dy

    while True:
        yield x, y
        if x == gx1 and y == gy1:
            break
        e2 = 2 * err
        if e2 > -dy:
            err -= dy
            x += sx
        if e2 < dx:
            err += dx
            y += sy


def is_segment_collision_free(
    world: Costmap2D,
    x0: float, y0: float,
    x1: float, y1: float,
) -> bool:
    """
    Check if straight segment between two world points is collision-free.
    """
    gx0, gy0 = world.world_to_grid(x0, y0)
    gx1, gy1 = world.world_to_grid(x1, y1)

    for gx, gy in _bresenham(gx0, gy0, gx1, gy1):
        if not world.is_free(gx, gy):
            return False
    return True


def segment_collision_ratio(
    world: Costmap2D,
    x0: float, y0: float,
    x1: float, y1: float,
) -> float:
    """
    Return fraction of cells along the segment that are occupied.
    Useful as a diagnostic / soft-cost.
    """
    gx0, gy0 = world.world_to_grid(x0, y0)
    gx1, gy1 = world.world_to_grid(x1, y1)

    total = 0
    bad = 0
    for gx, gy in _bresenham(gx0, gy0, gx1, gy1):
        total += 1
        if world.is_occupied(gx, gy):
            bad += 1
    return (bad / total) if total > 0 else 1.0
