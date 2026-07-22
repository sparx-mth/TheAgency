# sparx_agency/core/planning/environment/__init__.py
"""
World and map representations for planning.

This package defines concrete environment representations used by planners,
behaviors, and integration layers. The classes here are pure data models with
basic geometric and occupancy queries, and contain no planning or safety logic.

Contents:
    - Costmap2D: Binary occupancy grid with optional clearance field
    - OccupancyGrid2D: Discrete grid with FREE / OCCUPIED / UNKNOWN semantics
    - VoxelMap3D: Protocol defining the required 3D voxel map interface
"""

from .costmap2d import Costmap2D, CostmapParams
from .occupancy_grid2d import OccupancyGrid2D, OccupancyGrid2DParams, OccupancyValues
from .voxelmap3d import VoxelMap3D

__all__ = [
    "Costmap2D",
    "CostmapParams",
    "OccupancyGrid2D",
    "OccupancyGrid2DParams",
    "OccupancyValues",
    "VoxelMap3D",
]
