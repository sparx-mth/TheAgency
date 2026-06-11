"""
Collision checking utilities for VoxelMap3D-like objects.

This module is duck-typed to support multiple voxel map APIs.

Supported point-free queries:
    - voxelmap.is_free_world(x, y, z) -> bool
    - voxelmap.world_to_grid(x, y, z) + voxelmap.is_free(i, j, k) -> bool

Segment check uses uniform sampling along the segment.
"""

from __future__ import annotations

from math import sqrt
from typing import Any


def is_state_free(voxelmap: Any, x: float, y: float, z: float) -> bool:
    """
    Return True if world position (x, y, z) is collision-free.

    The function supports either:
        - is_free_world(x, y, z)
        - world_to_grid(x, y, z) + is_free(i, j, k)
    """
    if hasattr(voxelmap, "is_free_world"):
        return bool(voxelmap.is_free_world(x, y, z))

    # Fallback to grid-based interface
    i, j, k = voxelmap.world_to_grid(x, y, z)
    return bool(voxelmap.is_free(i, j, k))


def is_segment_free(
    voxelmap: Any,
    x0: float,
    y0: float,
    z0: float,
    x1: float,
    y1: float,
    z1: float,
    *,
    step_m: float,
) -> bool:
    """
    Return True if all sampled points along the segment are collision-free.

    Args:
        voxelmap: Voxel map object (duck-typed).
        x0,y0,z0: Segment start (meters).
        x1,y1,z1: Segment end (meters).
        step_m: Sampling step in meters (must be > 0).

    Notes:
        - This is a conservative discrete check. Smaller step_m increases safety
          but costs more CPU.
    """
    if step_m <= 0.0:
        raise ValueError(f"step_m must be > 0, got {step_m}")

    dx = x1 - x0
    dy = y1 - y0
    dz = z1 - z0
    length = sqrt(dx * dx + dy * dy + dz * dz)

    if length < 1e-9:
        return is_state_free(voxelmap, x0, y0, z0)

    n = max(1, int(length / step_m))
    inv_n = 1.0 / n

    for i in range(n + 1):
        t = i * inv_n
        x = x0 + t * dx
        y = y0 + t * dy
        z = z0 + t * dz
        if not is_state_free(voxelmap, x, y, z):
            return False

    return True
