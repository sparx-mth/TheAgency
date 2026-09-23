"""Rasterise convex polygons into a boolean occupancy grid.

One polygon becomes occupied cells in two steps that answer two different
failure modes:

* **fill** -- cell centres inside the polygon, which is what draws the
  footprint of anything with area, such as a table top sliced at 0.75 m;
* **edge stamping** -- unbroken cell chains along the polygon's boundary,
  which is what draws anything thin. A vertical wall sliced by a horizontal
  slab projects to a segment of zero area: the fill finds nothing there, and
  the edges are the entire answer.

Doing only the first leaves holes in walls. Doing only the second leaves
hollow furniture. Both together give a grid where "occupied" means "geometry
passes through this cell", which is the only reading a planner can trust.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from .edge_raster import stamp_polygon_edges
from .grid_spec import GridSpec
from .polygon_fill import fill_polygons
from .work_chunks import DEFAULT_WORK_BUDGET


def rasterise_polygons(
    grid: np.ndarray,
    polygons: np.ndarray,
    counts: np.ndarray,
    spec: GridSpec,
    budget: int = DEFAULT_WORK_BUDGET,
) -> None:
    """Draw a batch of padded convex polygons into ``grid`` in place.

    Args:
        grid: ``(height, width)`` boolean array, modified in place. Indexed
            ``[row, col]`` with row 0 at minimum y.
        polygons: ``(N, K, 3)`` padded polygon vertices in world metres. Only
            x and y are read; z is carried along by the slab clip and ignored
            here.
        counts: ``(N,)`` vertex count per polygon.
        spec: The grid's world geometry.
        budget: Maximum intermediate elements held in memory at once.

    Raises:
        ValueError: If ``grid`` does not match ``spec``.
    """
    if grid.shape != spec.shape:
        raise ValueError(
            "grid shape %r does not match spec %r" % (grid.shape, spec.shape)
        )
    polygons = np.asarray(polygons, dtype=np.float64)
    counts = np.asarray(counts, dtype=np.int64)
    if polygons.shape[0] == 0:
        return
    fill_polygons(grid, polygons, counts, spec, budget=budget)
    stamp_polygon_edges(grid, polygons, counts, spec, budget=budget)


def rasterise_polygon(
    polygon: np.ndarray,
    resolution: float,
    origin_x: float,
    origin_y: float,
    width: int,
    height: int,
    grid: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Draw a single convex polygon, the convenient form of the batch call.

    Args:
        polygon: ``(M, 2)`` or ``(M, 3)`` vertices in world metres, in order
            around the polygon. Fewer than three vertices draws nothing but
            the edges.
        resolution: Metres per cell.
        origin_x: World x of the grid's lower-left corner.
        origin_y: World y of the grid's lower-left corner.
        width: Columns.
        height: Rows.
        grid: Optional existing ``(height, width)`` boolean grid to draw into.
            A fresh one is allocated when omitted.

    Returns:
        The boolean grid, shape ``(height, width)``, indexed ``[row, col]``
        with row 0 at minimum y.

    Raises:
        ValueError: If ``polygon`` is not 2D with 2 or 3 components, or if
            ``grid`` does not match the requested width and height.
    """
    polygon = np.asarray(polygon, dtype=np.float64)
    if polygon.ndim != 2 or polygon.shape[1] not in (2, 3):
        raise ValueError("polygon must be (M, 2) or (M, 3), got %r" % (polygon.shape,))

    spec = GridSpec(
        resolution=float(resolution),
        origin_x=float(origin_x),
        origin_y=float(origin_y),
        width=int(width),
        height=int(height),
    )
    if grid is None:
        grid = spec.empty()
    elif grid.shape != spec.shape:
        # A grid of the wrong shape is not rescaled or clipped: the cells land
        # at the indices this spec asks for, in the corner of somebody else's
        # map, and come back looking drawn. The batch call refuses it, so does
        # this one.
        raise ValueError(
            "grid shape %r does not match the requested %r"
            % (grid.shape, spec.shape)
        )

    padded = np.zeros((1, max(polygon.shape[0], 1), 3), dtype=np.float64)
    padded[0, : polygon.shape[0], : polygon.shape[1]] = polygon
    counts = np.array([polygon.shape[0]], dtype=np.int64)
    if polygon.shape[0] >= 3:
        fill_polygons(grid, padded, counts, spec)
    if polygon.shape[0] >= 2:
        stamp_polygon_edges(grid, padded, counts, spec)
    return grid
