"""
Safety tube query adapter for 3D voxel maps.

This module provides the tube query implementation for VoxelMap3D, a 3D voxel-based
map representation. The adapter is duck-typed to work with any object that implements
the required interface (world_to_grid, is_free, world_clearance, resolution).

Functions:
    query_tube_voxelmap3d: Check if a spherical tube is clear in a 3D voxel map.

Dependencies:
    - SafetyStatus from ..types
    - shell26_offsets from .common

Notes:
    - VoxelMap3D is defined as a Protocol; runtime isinstance checks may not work.
    - The adapter uses duck-typing to detect compatible map objects.
"""

from __future__ import annotations

from typing import Any, Tuple

from ..types import SafetyStatus
from .common import shell26_offsets


def query_tube_voxelmap3d(
    vox: Any,
    *,
    x: float,
    y: float,
    z: float,
    radius_m: float,
) -> Tuple[SafetyStatus, bool]:
    """
    Perform a tube (spherical clearance) query on a 3D voxel map.

    This function checks whether a spherical region of the given radius centered
    at (x, y, z) is free of obstacles. It uses two strategies:

    1. **Clearance field (preferred)**: If the voxel map has a meaningful
       world_clearance method (e.g., from an ESDF), use it directly. This is
       O(1) and highly efficient.

    2. **Occupancy sampling (fallback)**: If world_clearance raises an exception
       or is not implemented, check voxel occupancy at the center point plus
       26 samples on a spherical shell (shell26 approximation).

    Args:
        vox: A duck-typed VoxelMap3D instance. Must implement:
            - world_to_grid(x, y, z) -> (i, j, k)
            - is_free(i, j, k) -> bool
            - world_clearance(x, y, z) -> float (optional, may raise)
            - resolution (attribute, float)
        x: World x-coordinate (meters) of the tube center.
        y: World y-coordinate (meters) of the tube center.
        z: World z-coordinate (meters) of the tube center.
        radius_m: Radius (meters) of the safety tube (sphere) to check.

    Returns:
        A tuple of (SafetyStatus, saw_unknown):
            - SafetyStatus.CLEAR: The spherical tube is free of obstacles.
            - SafetyStatus.BLOCKED: An obstacle was detected within the tube.
            - SafetyStatus.OUT_OF_BOUNDS: The query point or a sample is outside
              the map boundaries (detected via exception from world_to_grid or is_free).
            - saw_unknown: Always False for VoxelMap3D (unknown cells not tracked).

    Example:
        >>> # Assuming a duck-typed VoxelMap3D instance
        >>> voxel_map = create_voxel_map(...)  # Your map creation
        >>> status, saw_unknown = query_tube_voxelmap3d(
        ...     voxel_map, x=1.0, y=2.0, z=0.5, radius_m=0.3
        ... )
        >>> if status == SafetyStatus.CLEAR:
        ...     print("3D region is clear")

    Notes:
        - The function uses exception handling to detect out-of-bounds conditions
          and to gracefully fall back from clearance to occupancy checking.
        - If world_clearance returns a value less than radius_m, BLOCKED is
          returned without falling back to occupancy sampling (assumes the
          clearance field is authoritative when available).
        - The shell26 sampling covers the 26 directions of the Moore neighborhood,
          providing reasonable coverage for real-time applications.
        - For higher fidelity, ensure the voxel map has a proper ESDF/TSDF
          clearance field.
    """
    # Strategy 1: Use clearance field if available (fast path)
    try:
        clearance = vox.world_clearance(x, y, z)
        if clearance >= radius_m:
            return SafetyStatus.CLEAR, False
        # Clearance exists but is insufficient => blocked
        return SafetyStatus.BLOCKED, False
    except Exception:
        # world_clearance not implemented or raised an error; fall back to sampling
        pass

    # Strategy 2: Fallback to occupancy sampling
    try:
        i, j, k = vox.world_to_grid(x, y, z)
        if not vox.is_free(i, j, k):
            return SafetyStatus.BLOCKED, False
    except Exception:
        return SafetyStatus.OUT_OF_BOUNDS, False

    if radius_m <= 0.0:
        return SafetyStatus.CLEAR, False

    # Check 26 samples on the spherical shell
    for dx, dy, dz in shell26_offsets(radius_m):
        try:
            ii, jj, kk = vox.world_to_grid(x + dx, y + dy, z + dz)
            if not vox.is_free(ii, jj, kk):
                return SafetyStatus.BLOCKED, False
        except Exception:
            return SafetyStatus.OUT_OF_BOUNDS, False

    return SafetyStatus.CLEAR, False