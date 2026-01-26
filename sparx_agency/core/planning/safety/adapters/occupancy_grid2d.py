"""
Safety tube query adapter for OccupancyGrid2D maps.

This module provides the tube query implementation for OccupancyGrid2D, a 2D
occupancy grid that distinguishes between free, occupied, and unknown cells.
The adapter respects the unknown_policy parameter to determine how unknown
regions should affect safety decisions.

Functions:
    query_tube_occupancy_grid2d: Check if a circular tube is clear in an OccupancyGrid2D.

Dependencies:
    - OccupancyGrid2D from sparx_agency.core.planning.environment.occupancy_grid2d
    - SafetyStatus, UnknownPolicy from ..types
    - ring8_offsets from .common
"""

from __future__ import annotations

from typing import Tuple

from sparx_agency.core.planning.environment.occupancy_grid2d import OccupancyGrid2D

from ..types import SafetyStatus, UnknownPolicy
from .common import ring8_offsets


def query_tube_occupancy_grid2d(
    grid: OccupancyGrid2D,
    *,
    x: float,
    y: float,
    radius_m: float,
    unknown_policy: UnknownPolicy,
) -> Tuple[SafetyStatus, bool]:
    """
    Perform a tube (circular clearance) query on an OccupancyGrid2D.

    This function checks whether a circular region of the given radius centered
    at (x, y) is free of obstacles, with special handling for unknown cells
    based on the provided policy.

    The function checks the center cell first, then samples 8 points around
    the perimeter (ring8 approximation) if radius_m > 0.

    Args:
        grid: The OccupancyGrid2D instance to query.
        x: World x-coordinate (meters) of the tube center.
        y: World y-coordinate (meters) of the tube center.
        radius_m: Radius (meters) of the safety tube to check.
        unknown_policy: How to treat unknown cells:
            - BLOCK: Unknown cells are treated as obstacles (conservative).
            - ALLOW: Unknown cells are treated as free (permissive).
            - WARN: Continue checking but flag that unknown was encountered.

    Returns:
        A tuple of (SafetyStatus, saw_unknown):
            - SafetyStatus.CLEAR: The tube is free of obstacles (and unknown
              cells if policy is ALLOW, or no unknown cells encountered).
            - SafetyStatus.BLOCKED: An obstacle was detected, or unknown cell
              with BLOCK policy.
            - SafetyStatus.UNKNOWN: Unknown cells were encountered with WARN
              policy and no hard obstacles found.
            - SafetyStatus.OUT_OF_BOUNDS: Query point or sample is outside map.
            - saw_unknown: True if any unknown cell was encountered during the check.

    Example:
        >>> from sparx_agency.core.planning.environment.occupancy_grid2d import OccupancyGrid2D
        >>> from sparx_agency.core.planning.safety.types import UnknownPolicy
        >>> grid = OccupancyGrid2D(...)  # Initialize your grid
        >>> status, saw_unknown = query_tube_occupancy_grid2d(
        ...     grid, x=1.0, y=2.0, radius_m=0.3, unknown_policy=UnknownPolicy.WARN
        ... )
        >>> if status == SafetyStatus.UNKNOWN:
        ...     print("Unexplored region detected - proceed with caution")

    Notes:
        - The function prioritizes detecting hard obstacles (occupied cells) over
          unknown cells. Even with WARN policy, if an occupied cell is found,
          BLOCKED is returned immediately.
        - The ring8 fallback provides approximate coverage; for safety-critical
          applications with high-precision requirements, consider using maps
          with clearance fields.
        - saw_unknown tracks whether any unknown cell was seen, independent of
          the final status.
    """
    gx, gy = grid.world_to_grid(x, y)
    if not grid.in_bounds(gx, gy):
        return SafetyStatus.OUT_OF_BOUNDS, False

    v = grid.value_at(gx, gy)
    if v is None:
        return SafetyStatus.OUT_OF_BOUNDS, False

    saw_unknown = False

    # Check center cell
    if v == grid.values.occupied:
        return SafetyStatus.BLOCKED, False

    if v == grid.values.unknown:
        saw_unknown = True
        if unknown_policy == UnknownPolicy.BLOCK:
            return SafetyStatus.BLOCKED, True
        # ALLOW/WARN: continue checking ring

    # If no radius, return based on center cell result
    if radius_m <= 0.0:
        if saw_unknown and unknown_policy == UnknownPolicy.WARN:
            return SafetyStatus.UNKNOWN, True
        return SafetyStatus.CLEAR, saw_unknown

    # Check 8 samples around the perimeter
    for dx, dy in ring8_offsets(radius_m):
        gxi, gyi = grid.world_to_grid(x + dx, y + dy)
        if not grid.in_bounds(gxi, gyi):
            return SafetyStatus.OUT_OF_BOUNDS, saw_unknown

        vi = grid.value_at(gxi, gyi)
        if vi is None:
            return SafetyStatus.OUT_OF_BOUNDS, saw_unknown

        if vi == grid.values.occupied:
            return SafetyStatus.BLOCKED, saw_unknown

        if vi == grid.values.unknown:
            saw_unknown = True
            if unknown_policy == UnknownPolicy.BLOCK:
                return SafetyStatus.BLOCKED, True

    # All samples checked - determine final status
    if saw_unknown and unknown_policy == UnknownPolicy.WARN:
        return SafetyStatus.UNKNOWN, True
    return SafetyStatus.CLEAR, saw_unknown