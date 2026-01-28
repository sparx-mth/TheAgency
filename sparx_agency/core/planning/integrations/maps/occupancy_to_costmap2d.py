"""
World representation adapters for behaviors.

This module provides converter functions between different world/map
representations used in the planning system. Behaviors may require
specific representations (e.g., Costmap2D for cost-aware navigation),
and these adapters enable interoperability.

Functions:
    costmap_from_occupancy_grid: Convert OccupancyGrid2D to Costmap2D

Example:
    >>> from sparx_agency.core.planning.behaviors.utils import costmap_from_occupancy_grid
    >>> costmap = costmap_from_occupancy_grid(occupancy_grid, unknown_is_occupied=True)
    >>> # Use costmap for wall-following or cost-aware behaviors
"""

from __future__ import annotations

import numpy as np

from sparx_agency.core.planning.environment.costmap2d import Costmap2D, CostmapParams
from sparx_agency.core.planning.environment.occupancy_grid2d import OccupancyGrid2D


def costmap_from_occupancy_grid(
    grid: OccupancyGrid2D,
    *,
    unknown_is_occupied: bool = True,
) -> Costmap2D:
    """
    Convert an OccupancyGrid2D to a binary Costmap2D.

    Creates a costmap view where cells are either free (0) or occupied (1).
    This is useful for behaviors that need simple binary occupancy checks
    (e.g., wall-following, collision detection) rather than probabilistic
    occupancy values.

    Args:
        grid: Source OccupancyGrid2D containing FREE, OCCUPIED, and
            UNKNOWN cell values.
        unknown_is_occupied: If True, UNKNOWN cells are treated as
            occupied (conservative/safe). If False, UNKNOWN cells are
            treated as free (optimistic). Defaults to True for safety.

    Returns:
        A new Costmap2D with binary occupancy values:
        - 0: Free space
        - 1: Occupied (or unknown, if unknown_is_occupied=True)

        The costmap inherits the grid's resolution, origin, and frame_id.

    Example:
        >>> grid = OccupancyGrid2D(...)  # From SLAM or mapping
        >>> # Conservative: unknown = occupied (safe for navigation)
        >>> costmap = costmap_from_occupancy_grid(grid, unknown_is_occupied=True)
        >>>
        >>> # Optimistic: unknown = free (for exploration)
        >>> costmap = costmap_from_occupancy_grid(grid, unknown_is_occupied=False)

    Note:
        The returned Costmap2D has `clearance=None`. If distance-based
        costs are needed, compute the clearance field separately using
        the costmap's inflation utilities.
    """
    g = grid.grid
    occ = np.zeros_like(g, dtype=np.uint8)
    occ[g == grid.values.occupied] = 1
    if unknown_is_occupied:
        occ[g == grid.values.unknown] = 1

    params = CostmapParams(
        resolution=grid.resolution,
        origin_x=grid.origin_x,
        origin_y=grid.origin_y,
        frame_id=grid.frame_id,
    )
    return Costmap2D(occupancy=occ, params=params, clearance=None)