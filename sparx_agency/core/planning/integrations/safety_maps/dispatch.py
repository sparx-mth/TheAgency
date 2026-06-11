"""
Dispatcher for tube queries across supported map types.
"""

from __future__ import annotations

from typing import Any, Tuple

from sparx_agency.core.planning.environment.costmap2d import Costmap2D
from sparx_agency.core.planning.environment.occupancy_grid2d import OccupancyGrid2D
from sparx_agency.core.planning.safety.types import SafetyStatus, UnknownPolicy

from .costmap2d import query_tube_costmap2d
from .occupancy_grid2d import query_tube_occupancy_grid2d
from .voxelmap3d import query_tube_voxelmap3d


def query_tube(
    *,
    local_map: Any,
    x: float,
    y: float,
    z: float,
    radius_m: float,
    unknown_policy: UnknownPolicy,
) -> Tuple[SafetyStatus, bool]:
    """
    Query a safety tube/bubble around a point.

    Returns:
        (status, saw_unknown)
    """
    if isinstance(local_map, Costmap2D):
        return query_tube_costmap2d(local_map, x=x, y=y, radius_m=radius_m)

    if isinstance(local_map, OccupancyGrid2D):
        return query_tube_occupancy_grid2d(
            local_map,
            x=x,
            y=y,
            radius_m=radius_m,
            unknown_policy=unknown_policy,
        )

    # Duck-typed voxel map (3D)
    return query_tube_voxelmap3d(local_map, x=x, y=y, z=z, radius_m=radius_m)
