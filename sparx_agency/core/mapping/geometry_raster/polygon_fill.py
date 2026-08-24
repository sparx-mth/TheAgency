"""Fill the interior of convex polygons by testing cell centres.

The companion to edge stamping. Edges alone would draw a table top as a hollow
outline; the fill closes it. It is the classic edge-function test -- a point is
inside a convex polygon when it lies on the same side of every directed edge --
evaluated only over each polygon's own bounding box, which is what keeps it
cheap when the polygons are the millimetre-scale triangles of a collision mesh.

Two details are not decoration:

* **Winding is taken from the signed area**, not assumed. A mesh's triangles
  stop being consistently wound the moment a model is placed with a mirroring
  scale, and testing "same side of every edge" without knowing which side is
  inside accepts both answers -- which is fine for a real polygon and
  catastrophic for a degenerate one.
* **Degenerate polygons are not filled at all.** A vertical wall clipped to a
  horizontal slab projects to a segment: every edge function is zero, every
  cell in its bounding box passes the test, and a wall running diagonally
  across a room would fill the entire room. Anything below an area threshold
  is left to the edge stamp, which draws it correctly.
"""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

from .grid_spec import GridSpec
from .work_chunks import DEFAULT_WORK_BUDGET, iter_budget_slices

EDGE_TOLERANCE_CELLS = 1e-9
MIN_AREA_CELLS = 1e-6


def fill_polygons(
    grid: np.ndarray,
    polygons: np.ndarray,
    counts: np.ndarray,
    spec: GridSpec,
    budget: int = DEFAULT_WORK_BUDGET,
) -> None:
    """Fill every polygon's interior into ``grid`` in place.

    Args:
        grid: ``(height, width)`` boolean array, modified in place.
        polygons: ``(N, K, 3)`` padded polygon vertices in world metres.
        counts: ``(N,)`` vertex count per polygon. Polygons with fewer than
            three vertices, or with negligible projected area, have no
            interior and are skipped.
        spec: The grid's world geometry.
        budget: Maximum candidate cells held in memory at once.
    """
    polygons = np.asarray(polygons, dtype=np.float64)
    counts = np.asarray(counts, dtype=np.int64)
    if polygons.shape[0] == 0:
        return

    cells = spec.to_cell_coords(polygons)
    starts, ends = _closed_edges(cells, counts)
    area = _signed_areas(starts, ends)

    eligible = (counts >= 3) & (np.abs(area) >= MIN_AREA_CELLS)
    if not eligible.any():
        return

    box = _bounding_boxes(cells, counts, spec, eligible)
    if box is None:
        return
    keep, col0, row0, cols, rows = box

    starts, ends = starts[keep], ends[keep]
    winding = np.where(area[keep] >= 0.0, 1.0, -1.0)
    extent = cols * rows
    for lo, hi in iter_budget_slices(extent, budget):
        _fill_slice(
            grid,
            starts[lo:hi],
            ends[lo:hi],
            winding[lo:hi],
            col0[lo:hi],
            row0[lo:hi],
            cols[lo:hi],
            rows[lo:hi],
        )


def _closed_edges(
    cells: np.ndarray, counts: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """Build each polygon's directed edges in cell coordinates.

    Padding slots collapse to a zero-length edge at the origin, contributing
    nothing to the area and an edge function of exactly zero.

    Args:
        cells: ``(N, K, 2)`` vertices in continuous cell coordinates.
        counts: ``(N,)`` vertex count per polygon.

    Returns:
        ``(starts, ends)``, each ``(N, K, 2)``.
    """
    slot = np.arange(cells.shape[1], dtype=np.int64)
    live = slot[None, :] < counts[:, None]
    successor = np.where(slot[None, :] + 1 >= counts[:, None], 0, slot[None, :] + 1)
    ends = np.take_along_axis(cells, successor[:, :, None].repeat(2, axis=2), axis=1)
    return (
        np.where(live[:, :, None], cells, 0.0),
        np.where(live[:, :, None], ends, 0.0),
    )


def _signed_areas(starts: np.ndarray, ends: np.ndarray) -> np.ndarray:
    """Shoelace area of each polygon, in cells squared.

    Args:
        starts: ``(N, K, 2)`` directed edge starts.
        ends: ``(N, K, 2)`` directed edge ends.

    Returns:
        ``(N,)`` signed areas; positive for counter-clockwise polygons.
    """
    cross = starts[:, :, 0] * ends[:, :, 1] - ends[:, :, 0] * starts[:, :, 1]
    return 0.5 * cross.sum(axis=1)


def _bounding_boxes(
    cells: np.ndarray,
    counts: np.ndarray,
    spec: GridSpec,
    eligible: np.ndarray,
) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    """Clip each eligible polygon's cell bounding box to the grid.

    Args:
        cells: ``(N, K, 2)`` vertices in continuous cell coordinates.
        counts: ``(N,)`` vertex count per polygon.
        spec: The grid's world geometry.
        eligible: ``(N,)`` boolean, which polygons are worth filling.

    Returns:
        ``(keep, col0, row0, cols, rows)`` for the polygons that also overlap
        the grid, or None when none do.
    """
    slot = np.arange(cells.shape[1], dtype=np.int64)
    # `eligible` is folded into the vertex mask rather than applied only to
    # `keep` below, and the rows it blanks then get a finite box: a row with no
    # live vertex keeps the +/-inf sentinels, and casting inf to int64 is a
    # RuntimeWarning -- an error under -W error -- raised for a polygon that is
    # dropped a moment later anyway. Such a row arrives from any caller that
    # hands the batch API its counts unfiltered.
    live = (slot[None, :] < counts[:, None]) & eligible[:, None]
    blank = ~live.any(axis=1)
    lower = np.where(live[:, :, None], cells, np.inf).min(axis=1)
    upper = np.where(live[:, :, None], cells, -np.inf).max(axis=1)
    lower = np.where(blank[:, None], 0.0, lower)
    upper = np.where(blank[:, None], 0.0, upper)

    first = np.floor(lower).astype(np.int64)
    last = np.floor(upper).astype(np.int64)
    keep = (
        eligible
        & (last[:, 0] >= 0)
        & (first[:, 0] < spec.width)
        & (last[:, 1] >= 0)
        & (first[:, 1] < spec.height)
    )
    if not keep.any():
        return None

    col0 = np.clip(first[keep, 0], 0, spec.width - 1)
    col1 = np.clip(last[keep, 0], 0, spec.width - 1)
    row0 = np.clip(first[keep, 1], 0, spec.height - 1)
    row1 = np.clip(last[keep, 1], 0, spec.height - 1)
    return keep, col0, row0, col1 - col0 + 1, row1 - row0 + 1


def _fill_slice(
    grid: np.ndarray,
    starts: np.ndarray,
    ends: np.ndarray,
    winding: np.ndarray,
    col0: np.ndarray,
    row0: np.ndarray,
    cols: np.ndarray,
    rows: np.ndarray,
) -> None:
    """Test every candidate cell centre of one slice of polygons.

    Args:
        grid: ``(height, width)`` boolean array, modified in place.
        starts: ``(n, K, 2)`` directed edge starts in cell coordinates.
        ends: ``(n, K, 2)`` directed edge ends in cell coordinates.
        winding: ``(n,)`` +1 for counter-clockwise polygons, -1 for clockwise.
        col0: ``(n,)`` first column of each polygon's clipped bounding box.
        row0: ``(n,)`` first row of each polygon's clipped bounding box.
        cols: ``(n,)`` bounding box width in cells.
        rows: ``(n,)`` bounding box height in cells.
    """
    extent = cols * rows
    total = int(extent.sum())
    if total == 0:
        return
    owner = np.repeat(np.arange(extent.shape[0], dtype=np.int64), extent)
    local = np.arange(total, dtype=np.int64) - (np.cumsum(extent) - extent)[owner]
    cell_x = col0[owner] + local % cols[owner]
    cell_y = row0[owner] + local // cols[owner]
    point_x = cell_x + 0.5
    point_y = cell_y + 0.5
    sense = winding[owner]

    inside = np.ones(total, dtype=bool)
    for slot in range(starts.shape[1]):
        start_x = starts[owner, slot, 0]
        start_y = starts[owner, slot, 1]
        edge_x = ends[owner, slot, 0] - start_x
        edge_y = ends[owner, slot, 1] - start_y
        side = edge_x * (point_y - start_y) - edge_y * (point_x - start_x)
        inside &= sense * side >= -EDGE_TOLERANCE_CELLS

    grid[cell_y[inside], cell_x[inside]] = True
