"""
Safety tube query adapters for concrete map types.

This package provides a single entrypoint:
    query_tube(local_map, x, y, z, radius_m, unknown_policy) -> (status, saw_unknown)

It dispatches to the correct implementation based on the concrete map type.
"""

from .dispatch import query_tube

from .costmap2d import query_tube_costmap2d
from .occupancy_grid2d import query_tube_occupancy_grid2d
from .voxelmap3d import query_tube_voxelmap3d

__all__ = [
    "query_tube",
    "query_tube_costmap2d",
    "query_tube_occupancy_grid2d",
    "query_tube_voxelmap3d",
]
