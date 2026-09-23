"""Turn a triangle mesh into a 2D occupancy slice through a height band.

The public entry point of the package. Given the triangles of a world's
collision geometry, already placed in world coordinates, it answers the only
question a ground-floor navigation map asks: which cells does the geometry
occupy somewhere between ``z_min`` and ``z_max``?

The work is done in three stages -- cull triangles that cannot reach the slab
or the grid, clip the survivors to the slab, rasterise the resulting convex
polygons -- and every stage is batched over triangles, because a building's
collision meshes run to millions of them. Triangles are processed in chunks so
the vectorised intermediates stay bounded no matter how large the mesh is.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from .grid_spec import GridSpec
from .polygon_raster import rasterise_polygons
from .slab_clip import clip_triangles_to_slab

FACE_CHUNK = 200_000


def rasterise_mesh_slab(
    vertices: np.ndarray,
    faces: np.ndarray,
    *,
    z_min: float,
    z_max: float,
    resolution: float,
    origin_x: float,
    origin_y: float,
    width: int,
    height: int,
    out: Optional[np.ndarray] = None,
    face_chunk: int = FACE_CHUNK,
) -> np.ndarray:
    """Rasterise the part of a mesh that lies inside a horizontal slab.

    Args:
        vertices: ``(V, 3)`` float array of vertex positions in world metres.
            float32 and float64 are both accepted.
        faces: ``(F, 3)`` integer array of vertex indices.
        z_min: Lower slab boundary, metres.
        z_max: Upper slab boundary, metres.
        resolution: Metres per cell.
        origin_x: World x of the lower-left corner of cell ``(0, 0)``.
        origin_y: World y of the lower-left corner of cell ``(0, 0)``.
        width: Grid columns.
        height: Grid rows.
        out: Optional existing ``(height, width)`` boolean grid to accumulate
            into, so several meshes can share one map. A fresh grid is
            allocated when omitted.
        face_chunk: How many triangles to process per vectorised pass.

    Returns:
        Boolean array of shape ``(height, width)``, indexed ``[row, col]``
        with **row 0 at minimum y** -- plain grid indexing, not the top-down
        convention of a nav2 PGM image.

    Raises:
        ValueError: If the arrays are misshaped, a face index is out of range,
            or the slab is empty.
    """
    vertices = np.asarray(vertices, dtype=np.float64)
    faces = np.asarray(faces)
    if vertices.ndim != 2 or vertices.shape[1] != 3:
        raise ValueError("vertices must be (V, 3), got %r" % (vertices.shape,))
    if faces.ndim != 2 or faces.shape[1] != 3:
        raise ValueError("faces must be (F, 3), got %r" % (faces.shape,))
    if not np.issubdtype(faces.dtype, np.integer):
        raise ValueError("faces must be an integer array, got %r" % (faces.dtype,))
    if faces.shape[0] and (faces.min() < 0 or faces.max() >= vertices.shape[0]):
        raise ValueError(
            "face index out of range for %d vertices" % (vertices.shape[0],)
        )
    if not float(z_max) >= float(z_min):
        raise ValueError("empty slab: z_max %r < z_min %r" % (z_max, z_min))

    spec = GridSpec(
        resolution=float(resolution),
        origin_x=float(origin_x),
        origin_y=float(origin_y),
        width=int(width),
        height=int(height),
    )
    grid = spec.empty() if out is None else out
    if grid.shape != spec.shape or grid.dtype != np.dtype(bool):
        raise ValueError(
            "out must be a boolean array of shape %r, got %r %r"
            % (spec.shape, grid.shape, grid.dtype)
        )
    if faces.shape[0] == 0:
        return grid

    chunk = max(1, int(face_chunk))
    for start in range(0, faces.shape[0], chunk):
        triangles = vertices[faces[start:start + chunk]]
        triangles = _cull(triangles, float(z_min), float(z_max), spec)
        if triangles.shape[0] == 0:
            continue
        polygons, counts = clip_triangles_to_slab(
            triangles, float(z_min), float(z_max)
        )
        live = counts > 0
        if not live.any():
            continue
        rasterise_polygons(grid, polygons[live], counts[live], spec)
    return grid


def _cull(
    triangles: np.ndarray, z_min: float, z_max: float, spec: GridSpec
) -> np.ndarray:
    """Drop triangles whose bounding box misses the slab or the grid.

    Almost all of a building's triangles are floor, ceiling or geometry off the
    side of the map. Rejecting them on a bounding-box test costs one pass and
    saves the clip and the raster.

    Args:
        triangles: ``(N, 3, 3)`` triangle vertices in world metres.
        z_min: Lower slab boundary, metres.
        z_max: Upper slab boundary, metres.
        spec: The grid's world geometry.

    Returns:
        The subset of ``triangles`` that could contribute, ``(M, 3, 3)``.
    """
    lower = triangles.min(axis=1)
    upper = triangles.max(axis=1)
    max_x = spec.origin_x + spec.width * spec.resolution
    max_y = spec.origin_y + spec.height * spec.resolution
    keep = (
        (upper[:, 2] >= z_min)
        & (lower[:, 2] <= z_max)
        & (upper[:, 0] >= spec.origin_x)
        & (lower[:, 0] <= max_x)
        & (upper[:, 1] >= spec.origin_y)
        & (lower[:, 1] <= max_y)
    )
    return triangles[keep]
