"""
Map adapter layer for safety tube queries.

This subpackage provides a unified interface for querying safety tubes (clearance
regions) across different map representations. The main entry point is the
`query_tube` function, which automatically dispatches to the appropriate
map-specific adapter.

Supported Map Types:
    - Costmap2D: 2D cost-based map (occupancy + optional clearance field)
    - OccupancyGrid2D: 2D occupancy grid with free/occupied/unknown cells
    - VoxelMap3D: 3D voxel-based map (duck-typed Protocol)

Functions:
    query_tube: Unified tube query dispatcher for all supported map types.

Modules:
    common: Shared geometric utilities (ring8_offsets, shell26_offsets)
    costmap2d: Adapter for Costmap2D queries
    occupancy_grid2d: Adapter for OccupancyGrid2D queries
    voxelmap3d: Adapter for VoxelMap3D queries

Example:
    >>> from sparx_agency.core.planning.safety.adapters import query_tube
    >>> from sparx_agency.core.planning.safety.types import SafetyStatus, UnknownPolicy
    >>>
    >>> # Query a tube on any supported map type
    >>> status, saw_unknown = query_tube(
    ...     local_map=my_map,
    ...     x=1.0, y=2.0, z=0.5,
    ...     radius_m=0.3,
    ...     unknown_policy=UnknownPolicy.WARN,
    ... )
    >>> if status == SafetyStatus.CLEAR:
    ...     print("Region is clear for navigation")
"""

from __future__ import annotations

from typing import Any, Tuple

from sparx_agency.core.planning.environment.costmap2d import Costmap2D
from sparx_agency.core.planning.environment.occupancy_grid2d import OccupancyGrid2D
from sparx_agency.core.planning.environment.voxelmap import VoxelMap3D

from ..types import SafetyStatus, UnknownPolicy

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
    Dispatch a tube (clearance) query to the appropriate map-specific adapter.

    This function provides a unified interface for checking whether a circular
    (2D) or spherical (3D) region is clear of obstacles, regardless of the
    underlying map representation. It automatically detects the map type and
    delegates to the appropriate adapter.

    Args:
        local_map: The map instance to query. Supported types:
            - Costmap2D: 2D cost-based map
            - OccupancyGrid2D: 2D occupancy grid with unknown cell support
            - VoxelMap3D: 3D voxel map (duck-typed)
        x: World x-coordinate (meters) of the tube center.
        y: World y-coordinate (meters) of the tube center.
        z: World z-coordinate (meters) of the tube center. Ignored for 2D maps.
        radius_m: Radius (meters) of the safety tube to check.
        unknown_policy: How to treat unknown cells (applies to OccupancyGrid2D):
            - BLOCK: Unknown cells treated as obstacles.
            - ALLOW: Unknown cells treated as free.
            - WARN: Flag unknown cells but continue checking.

    Returns:
        A tuple of (status, saw_unknown):
            - status (SafetyStatus): The result of the clearance check:
                - CLEAR: The tube is free of obstacles.
                - BLOCKED: An obstacle was detected.
                - UNKNOWN: Unknown cells encountered (with WARN policy).
                - OUT_OF_BOUNDS: Query is outside map boundaries.
            - saw_unknown (bool): True if any unknown cells were encountered.
              Always False for map types that don't track unknown cells.

    Example:
        >>> from sparx_agency.core.planning.safety.adapters import query_tube
        >>> from sparx_agency.core.planning.safety.types import UnknownPolicy, SafetyStatus
        >>>
        >>> # Check a 30cm radius tube at position (1, 2, 0.5)
        >>> status, saw_unknown = query_tube(
        ...     local_map=occupancy_grid,
        ...     x=1.0, y=2.0, z=0.5,
        ...     radius_m=0.30,
        ...     unknown_policy=UnknownPolicy.WARN,
        ... )
        >>>
        >>> if status == SafetyStatus.BLOCKED:
        ...     print("Obstacle detected!")
        >>> elif saw_unknown:
        ...     print("Unknown region - proceed with caution")

    Notes:
        - For 2D maps (Costmap2D, OccupancyGrid2D), the z parameter is ignored.
        - VoxelMap3D detection uses duck-typing since it's defined as a Protocol
          and runtime isinstance checks may not work reliably.
        - If the map type is not recognized, returns (BLOCKED, False) as a
          conservative fallback.
    """
    if isinstance(local_map, Costmap2D):
        return query_tube_costmap2d(local_map, x=x, y=y, radius_m=radius_m)

    if isinstance(local_map, OccupancyGrid2D):
        return query_tube_occupancy_grid2d(
            local_map, x=x, y=y, radius_m=radius_m, unknown_policy=unknown_policy
        )

    # VoxelMap3D is a Protocol; runtime isinstance doesn't work reliably.
    # We detect it by duck-typing the required methods/fields.
    if _looks_like_voxelmap3d(local_map):
        return query_tube_voxelmap3d(local_map, x=x, y=y, z=z, radius_m=radius_m)

    # Unknown map type - return conservative result
    return SafetyStatus.BLOCKED, False


def _looks_like_voxelmap3d(obj: Any) -> bool:
    """
    Check if an object implements the VoxelMap3D protocol via duck-typing.

    Since VoxelMap3D is defined as a Protocol, standard isinstance() checks
    may not work at runtime. This function checks for the presence of
    required methods and attributes.

    Args:
        obj: The object to check.

    Returns:
        True if the object has all required VoxelMap3D interface members,
        False otherwise.
    """
    return (
        hasattr(obj, "world_to_grid")
        and hasattr(obj, "is_free")
        and hasattr(obj, "world_clearance")
        and hasattr(obj, "resolution")
    )