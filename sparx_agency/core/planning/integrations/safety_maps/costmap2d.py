"""
Safety tube query adapter for Costmap2D maps.

This module provides the tube query implementation for Costmap2D, a 2D cost-based
map representation commonly used in robotics navigation. The adapter supports
both clearance-field-based queries (fast, if available) and fallback occupancy
sampling.

Functions:
    query_tube_costmap2d: Check if a circular tube is clear in a Costmap2D.

Dependencies:
    - Costmap2D from sparx_agency.core.planning.environment.costmap2d
    - SafetyStatus from ..types
    - ring8_offsets from .common
"""

from __future__ import annotations

from typing import Tuple

from sparx_agency.core.planning.environment.costmap2d import Costmap2D

from sparx_agency.core.planning.safety.types import SafetyStatus
from .common import ring8_offsets


def query_tube_costmap2d(
    costmap: Costmap2D,
    *,
    x: float,
    y: float,
    radius_m: float,
) -> Tuple[SafetyStatus, bool]:
    """
    Perform a tube (circular clearance) query on a Costmap2D.

    This function checks whether a circular region of the given radius centered
    at (x, y) is free of obstacles. It uses two strategies:

    1. **Clearance field (preferred)**: If the costmap has a precomputed clearance
       field, directly compare the clearance at the query point to the required
       radius. This is O(1) and highly efficient.

    2. **Occupancy sampling (fallback)**: If no clearance field exists, check
       occupancy at the center point plus 8 samples around the perimeter
       (ring8 approximation).

    Args:
        costmap: The Costmap2D instance to query.
        x: World x-coordinate (meters) of the tube center.
        y: World y-coordinate (meters) of the tube center.
        radius_m: Radius (meters) of the safety tube to check.

    Returns:
        A tuple of (SafetyStatus, saw_unknown):
            - SafetyStatus.CLEAR: The tube is free of obstacles.
            - SafetyStatus.BLOCKED: An obstacle was detected within the tube.
            - SafetyStatus.OUT_OF_BOUNDS: The query point or a sample is outside
              the map boundaries.
            - saw_unknown: Always False for Costmap2D (does not track unknown cells).

    Example:
        >>> from sparx_agency.core.planning.environment.costmap2d import Costmap2D
        >>> costmap = Costmap2D(...)  # Initialize your costmap
        >>> status, saw_unknown = query_tube_costmap2d(costmap, x=1.0, y=2.0, radius_m=0.3)
        >>> if status == SafetyStatus.CLEAR:
        ...     print("Safe to navigate")

    Notes:
        - Costmap2D does not distinguish unknown vs. occupied cells, so saw_unknown
          is always False.
        - The ring8 fallback provides approximate coverage; for safety-critical
          applications, ensure the costmap has a clearance field or use a finer
          sampling strategy.
        - If radius_m <= 0 and the center is free, returns CLEAR immediately.
    """
    gx, gy = costmap.world_to_grid(x, y)
    if not costmap.in_bounds(gx, gy):
        return SafetyStatus.OUT_OF_BOUNDS, False

    # Strategy 1: Use clearance field if available (fast path)
    if costmap.clearance is not None:
        if costmap.clearance_at(gx, gy) >= radius_m:
            return SafetyStatus.CLEAR, False
        else:
            return SafetyStatus.BLOCKED, False

    # Strategy 2: Fallback to occupancy sampling
    if not costmap.is_free(gx, gy):
        return SafetyStatus.BLOCKED, False

    if radius_m <= 0.0:
        return SafetyStatus.CLEAR, False

    # Check 8 samples around the perimeter
    for dx, dy in ring8_offsets(radius_m):
        gxi, gyi = costmap.world_to_grid(x + dx, y + dy)
        if not costmap.in_bounds(gxi, gyi):
            return SafetyStatus.OUT_OF_BOUNDS, False
        if not costmap.is_free(gxi, gyi):
            return SafetyStatus.BLOCKED, False

    return SafetyStatus.CLEAR, False