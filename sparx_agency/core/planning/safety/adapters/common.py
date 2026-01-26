"""
Common geometric utilities for safety tube queries.

This module provides functions to generate sample offsets for approximating
circular (2D) and spherical (3D) regions around a point. These offsets are
used by the map adapters to check clearance in a "tube" or "bubble" around
trajectory waypoints.

Functions:
    ring8_offsets: Generate 8 evenly-spaced 2D offsets on a circle.
    shell26_offsets: Generate 26 direction offsets on a 3D sphere (Moore neighborhood).

Notes:
    - The sampling approach is a discrete approximation for real-time performance.
    - For higher accuracy, increase the number of samples or use native clearance
      fields (e.g., ESDF) if available in the map representation.
"""

from __future__ import annotations

from math import sqrt
from typing import List, Tuple


def ring8_offsets(r: float) -> Tuple[Tuple[float, float], ...]:
    """
    Generate 8 evenly-spaced 2D offsets approximating a circle of radius r.

    This function returns offset coordinates for 8 points arranged around
    a circle: 4 on the cardinal axes (±x, ±y) and 4 on the diagonals.
    This provides a computationally cheap approximation for checking circular
    clearance in 2D maps.

    Args:
        r: Radius of the circle in meters. Must be positive for non-empty result.

    Returns:
        A tuple of 8 (dx, dy) offset pairs if r > 0, otherwise an empty tuple.
        Each offset represents a point on the circle at distance r from origin.

    Example:
        >>> offsets = ring8_offsets(1.0)
        >>> len(offsets)
        8
        >>> offsets[0]  # Point on positive x-axis
        (1.0, 0.0)

    Notes:
        - The 8-sample ring is a balance between coverage and speed.
        - For a radius of 0 or negative, returns an empty tuple (no samples needed).
        - The diagonal offsets use k ≈ 0.7071 (√(1/2)) to maintain radius r.
    """
    if r <= 0.0:
        return ()
    k = 0.7071067811865476  # sqrt(1/2)
    return (
        (r, 0.0),
        (-r, 0.0),
        (0.0, r),
        (0.0, -r),
        (k * r, k * r),
        (k * r, -k * r),
        (-k * r, k * r),
        (-k * r, -k * r),
    )


def shell26_offsets(r: float) -> Tuple[Tuple[float, float, float], ...]:
    """
    Generate 26 direction offsets approximating a sphere of radius r.

    This function returns offset coordinates for 26 points arranged in a
    3D Moore neighborhood pattern (all combinations of {-1, 0, 1}³ except
    the origin), normalized to lie on a sphere of radius r. This provides
    a computationally efficient approximation for checking spherical
    clearance in 3D voxel maps.

    Args:
        r: Radius of the sphere in meters. Must be positive for non-empty result.

    Returns:
        A tuple of 26 (dx, dy, dz) offset tuples if r > 0, otherwise an empty tuple.
        Each offset represents a point on the sphere at distance r from origin.

    Example:
        >>> offsets = shell26_offsets(1.0)
        >>> len(offsets)
        26
        >>> # All offsets should have magnitude ≈ r
        >>> import math
        >>> all(abs(math.sqrt(dx**2 + dy**2 + dz**2) - 1.0) < 1e-9
        ...     for dx, dy, dz in offsets)
        True

    Notes:
        - The 26-direction shell covers the Moore neighborhood (face, edge,
          and corner neighbors in a 3D grid).
        - Each direction vector is normalized to exactly radius r.
        - For a radius of 0 or negative, returns an empty tuple (no samples needed).
        - This is suitable for real-time checks; for higher fidelity, use native
          clearance fields (ESDF/TSDF) if available.
    """
    if r <= 0.0:
        return ()

    out: List[Tuple[float, float, float]] = []
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            for dz in (-1, 0, 1):
                if dx == 0 and dy == 0 and dz == 0:
                    continue
                n = sqrt(dx * dx + dy * dy + dz * dz)
                out.append((r * dx / n, r * dy / n, r * dz / n))
    return tuple(out)