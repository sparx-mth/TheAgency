"""
VoxelMap3D interface expected by RRTStarOmpl3DPlanner.

Your voxel map implementation must provide these properties and methods.
"""
from __future__ import annotations

from typing import Protocol, Tuple, Optional
import numpy as np


class VoxelMap3D(Protocol):
    """
    Required interface for 3D voxel maps used with RRT* 3D planner.

    Properties:
        origin_x, origin_y, origin_z: World coordinates of grid origin (meters).
        width, height, depth: Grid dimensions (number of voxels).
        resolution: Voxel size (meters).
        frame_id: Coordinate frame identifier.
        clearance: Optional 3D clearance field (ndarray or None).
    """

    # Grid bounds
    origin_x: float
    origin_y: float
    origin_z: float
    width: int  # x dimension
    height: int  # y dimension
    depth: int  # z dimension
    resolution: float
    frame_id: str
    clearance: Optional[np.ndarray]  # Optional clearance field

    def world_to_grid(self, x: float, y: float, z: float) -> Tuple[int, int, int]:
        """Convert world coordinates to grid indices."""
        ...

    def is_free(self, i: int, j: int, k: int) -> bool:
        """Check if voxel at (i, j, k) is free (not occupied)."""
        ...

    def world_clearance(self, x: float, y: float, z: float) -> float:
        """Return clearance (distance to nearest obstacle) at world position."""
        ...