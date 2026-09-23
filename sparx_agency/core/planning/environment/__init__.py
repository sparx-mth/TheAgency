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
    - occupancy_io: save/load an OccupancyGrid2D as a single .npz, and build one
      from a boolean obstacle mask
    - grid_regions: connected components of a boolean grid, and the enclosed
      one that is a building's floor
    - VoxelGrid3D: dense 3D occupancy, with save/load and a 2D slab projection
"""

from .costmap2d import Costmap2D, CostmapParams
from .grid_regions import connected_regions, flood_region, largest_enclosed_region
from .occupancy_grid2d import OccupancyGrid2D, OccupancyGrid2DParams, OccupancyValues
from .occupancy_io import load_occupancy_grid, occupancy_from_mask, save_occupancy_grid
from .voxel_grid_3d import (
    VoxelGrid3D, indoor_mask, landable_mask, load_voxel_grid,
    project_to_occupancy_2d, restrict_to_indoor, save_voxel_grid,
)
from .voxelmap3d import VoxelMap3D

__all__ = [
    "Costmap2D",
    "CostmapParams",
    "connected_regions",
    "flood_region",
    "largest_enclosed_region",
    "OccupancyGrid2D",
    "OccupancyGrid2DParams",
    "OccupancyValues",
    "VoxelGrid3D",
    "VoxelMap3D",
    "indoor_mask",
    "landable_mask",
    "load_voxel_grid",
    "project_to_occupancy_2d",
    "restrict_to_indoor",
    "save_voxel_grid",
    "load_occupancy_grid",
    "occupancy_from_mask",
    "save_occupancy_grid",
]
