"""
Collision queries for concrete map representations.

This package contains small, fast collision-check utilities that operate on
specific map types (Costmap2D, OccupancyGrid2D, and VoxelMap3D-like objects).

Public API is intentionally consistent across map types:
    - is_state_free(...)
    - is_segment_free(...)
"""

from .costmap2d import is_state_free as is_state_free_costmap2d
from .costmap2d import is_segment_free as is_segment_free_costmap2d
from .costmap2d import segment_collision_ratio as segment_collision_ratio_costmap2d

from .occupancy_grid2d import is_state_free as is_state_free_occupancy_grid2d
from .occupancy_grid2d import is_segment_free as is_segment_free_occupancy_grid2d

from .voxelmap3d import is_state_free as is_state_free_voxelmap3d
from .voxelmap3d import is_segment_free as is_segment_free_voxelmap3d

__all__ = [
    # Costmap2D
    "is_state_free_costmap2d",
    "is_segment_free_costmap2d",
    "segment_collision_ratio_costmap2d",
    # OccupancyGrid2D
    "is_state_free_occupancy_grid2d",
    "is_segment_free_occupancy_grid2d",
    # VoxelMap3D
    "is_state_free_voxelmap3d",
    "is_segment_free_voxelmap3d",
]
