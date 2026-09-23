"""Rasterise 3D triangle geometry into a 2D occupancy slice.

Pure numpy, no scipy, no mesh library: the caller brings vertices and faces
already placed in world coordinates, and this package answers which grid cells
the geometry occupies between two heights.

See :func:`rasterise_mesh_slab` for the entry point and ``README.md`` for the
conventions -- in particular that row 0 is minimum y.
"""
from __future__ import annotations

from .grid_spec import GridSpec
from .mesh_occupancy import rasterise_mesh_slab
from .polygon_raster import rasterise_polygon, rasterise_polygons
from .slab_clip import clip_triangle_to_slab, clip_triangles_to_slab

__all__ = [
    "GridSpec",
    "clip_triangle_to_slab",
    "clip_triangles_to_slab",
    "rasterise_mesh_slab",
    "rasterise_polygon",
    "rasterise_polygons",
]
