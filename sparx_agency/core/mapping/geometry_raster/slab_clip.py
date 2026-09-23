"""Clip triangles to a horizontal slab ``z_min <= z <= z_max``.

A 2D occupancy map of a building is a horizontal slice through it, and the
honest way to take that slice is to intersect the collision geometry with the
slab the robot's body sweeps through. Everything below (floors, kerbs, cable
trays) and above (ceilings, light fittings, door lintels) is then excluded by
construction rather than by hoping the mesh happened not to cover it.

Clipping a triangle against the two half-spaces leaves a convex polygon of at
most five vertices: each half-space can add one. The result is padded to five
slots with a per-triangle vertex count; a triangle that misses the slab
entirely comes back with a count of zero.
"""
from __future__ import annotations

from typing import Tuple

import numpy as np

from .halfspace_clip import clip_polygons_to_halfspace

Z_AXIS = 2
MAX_SLAB_POLYGON_VERTICES = 5


def clip_triangles_to_slab(
    triangles: np.ndarray, z_min: float, z_max: float
) -> Tuple[np.ndarray, np.ndarray]:
    """Clip a batch of triangles to the slab ``[z_min, z_max]``.

    Args:
        triangles: ``(N, 3, 3)`` array of triangle vertices in world metres.
        z_min: Lower slab boundary, metres.
        z_max: Upper slab boundary, metres.

    Returns:
        ``(polygons, counts)`` where ``polygons`` is
        ``(N, MAX_SLAB_POLYGON_VERTICES, 3)`` and ``counts`` is ``(N,)``. A
        count of 0 means that triangle does not reach the slab.

    Raises:
        ValueError: If ``triangles`` is not ``(N, 3, 3)`` or the slab is empty.
    """
    triangles = np.asarray(triangles, dtype=np.float64)
    if triangles.ndim != 3 or triangles.shape[1:] != (3, 3):
        raise ValueError("triangles must be (N, 3, 3), got %r" % (triangles.shape,))
    if not float(z_max) >= float(z_min):
        raise ValueError("empty slab: z_max %r < z_min %r" % (z_max, z_min))

    counts = np.full(triangles.shape[0], 3, dtype=np.int64)
    polygons, counts = clip_polygons_to_halfspace(
        triangles, counts, Z_AXIS, float(z_min), True
    )
    polygons, counts = clip_polygons_to_halfspace(
        polygons, counts, Z_AXIS, float(z_max), False
    )
    # A polygon with fewer than three vertices has no area and no edges worth
    # drawing; report it as empty so callers need only test the count.
    counts = np.where(counts >= 3, counts, 0)
    return polygons, counts


def clip_triangle_to_slab(
    triangle: np.ndarray, z_min: float, z_max: float
) -> np.ndarray:
    """Clip one triangle to the slab ``[z_min, z_max]``.

    Args:
        triangle: ``(3, 3)`` array of vertices in world metres.
        z_min: Lower slab boundary, metres.
        z_max: Upper slab boundary, metres.

    Returns:
        ``(M, 3)`` array of the clipped polygon's vertices, wound the same way
        as the input. ``M`` is 0 when the triangle misses the slab, otherwise
        3 to 5.

    Raises:
        ValueError: If ``triangle`` is not ``(3, 3)``.
    """
    triangle = np.asarray(triangle, dtype=np.float64)
    if triangle.shape != (3, 3):
        raise ValueError("triangle must be (3, 3), got %r" % (triangle.shape,))
    polygons, counts = clip_triangles_to_slab(triangle[None, :, :], z_min, z_max)
    return polygons[0, : int(counts[0]), :]
