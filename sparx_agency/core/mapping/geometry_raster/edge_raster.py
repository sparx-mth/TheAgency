"""Stamp polygon edges into a boolean grid as unbroken cell chains.

Testing cell centres is not enough. An 11.6 cm wall at 5 cm resolution, seen
at a grazing angle, passes between cell centres for stretches of its length and
rasterises with holes -- and a hole in a wall makes the map lie in the most
dangerous possible direction, because a planner will route a robot straight
through it. Worse, a *vertical* wall clipped to a horizontal slab projects to a
line segment of exactly zero area, so an interior fill alone draws nothing at
all for the one class of geometry that matters most.

So every edge of every clipped polygon is stamped as a line. The chain is
4-connected: the segment is sampled at half-cell steps, which guarantees
consecutive samples land in the same or an adjacent cell, and wherever the
chain takes a diagonal step both corner cells are added. A 4-connected chain
cannot be squeezed through even by a planner that allows diagonal moves.
"""
from __future__ import annotations

from typing import Tuple

import numpy as np

from .grid_spec import GridSpec
from .work_chunks import DEFAULT_WORK_BUDGET, iter_budget_slices

SAMPLES_PER_CELL = 2.0


def stamp_polygon_edges(
    grid: np.ndarray,
    polygons: np.ndarray,
    counts: np.ndarray,
    spec: GridSpec,
    budget: int = DEFAULT_WORK_BUDGET,
) -> None:
    """Draw every edge of every polygon into ``grid`` in place.

    Args:
        grid: ``(height, width)`` boolean array, modified in place.
        polygons: ``(N, K, 3)`` padded polygon vertices in world metres.
        counts: ``(N,)`` vertex count per polygon; polygons with fewer than
            two vertices are skipped.
        spec: The grid's world geometry.
        budget: Maximum sample points held in memory at once.
    """
    starts, ends = _polygon_edges(polygons, counts)
    if starts.shape[0] == 0:
        return
    stamp_segments(grid, starts, ends, spec, budget=budget)


def stamp_segments(
    grid: np.ndarray,
    starts: np.ndarray,
    ends: np.ndarray,
    spec: GridSpec,
    budget: int = DEFAULT_WORK_BUDGET,
) -> None:
    """Draw 4-connected cell chains for a batch of world-space segments.

    Args:
        grid: ``(height, width)`` boolean array, modified in place.
        starts: ``(M, 2+)`` segment start points in world metres.
        ends: ``(M, 2+)`` segment end points in world metres.
        spec: The grid's world geometry.
        budget: Maximum sample points held in memory at once.
    """
    begin = spec.to_cell_coords(starts)
    finish = spec.to_cell_coords(ends)
    span = np.abs(finish - begin).max(axis=1)
    step_count = np.maximum(1, np.ceil(SAMPLES_PER_CELL * span)).astype(np.int64)

    for lo, hi in iter_budget_slices(step_count + 1, budget):
        cell_x, cell_y = _chain_cells(
            begin[lo:hi], finish[lo:hi], step_count[lo:hi]
        )
        _mark(grid, cell_x, cell_y, spec)


def _polygon_edges(
    polygons: np.ndarray, counts: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """Flatten padded polygons into a plain list of closed-loop edges.

    Args:
        polygons: ``(N, K, 3)`` padded vertices.
        counts: ``(N,)`` vertex count per polygon.

    Returns:
        ``(starts, ends)``, each ``(M, 3)``, holding every edge of every
        polygon with at least two vertices.
    """
    polygons = np.asarray(polygons, dtype=np.float64)
    counts = np.asarray(counts, dtype=np.int64)
    if polygons.shape[0] == 0:
        return np.zeros((0, 3)), np.zeros((0, 3))

    slot = np.arange(polygons.shape[1], dtype=np.int64)
    live = (slot[None, :] < counts[:, None]) & (counts[:, None] >= 2)
    successor = np.where(slot[None, :] + 1 >= counts[:, None], 0, slot[None, :] + 1)
    ends = np.take_along_axis(
        polygons, successor[:, :, None].repeat(3, axis=2), axis=1
    )
    return polygons[live], ends[live]


def _chain_cells(
    begin: np.ndarray, finish: np.ndarray, step_count: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """Sample segments densely and close every diagonal step.

    Args:
        begin: ``(M, 2)`` segment starts in continuous cell coordinates.
        finish: ``(M, 2)`` segment ends in continuous cell coordinates.
        step_count: ``(M,)`` number of steps, chosen so no step exceeds half a
            cell in either axis.

    Returns:
        ``(cell_x, cell_y)`` integer arrays of the cells the chains cover.
    """
    sample_count = step_count + 1
    offsets = np.cumsum(sample_count) - sample_count
    segment = np.repeat(np.arange(step_count.shape[0], dtype=np.int64), sample_count)
    along = (np.arange(int(sample_count.sum()), dtype=np.float64)
             - offsets[segment]) / step_count[segment]

    point = begin[segment] + along[:, None] * (finish[segment] - begin[segment])
    cell = np.floor(point).astype(np.int64)
    cell_x, cell_y = cell[:, 0], cell[:, 1]

    same = segment[1:] == segment[:-1]
    delta_x = cell_x[1:] - cell_x[:-1]
    delta_y = cell_y[1:] - cell_y[:-1]
    diagonal = same & (delta_x != 0) & (delta_y != 0)
    if not diagonal.any():
        return cell_x, cell_y

    base_x, base_y = cell_x[:-1][diagonal], cell_y[:-1][diagonal]
    step_x, step_y = delta_x[diagonal], delta_y[diagonal]
    corner_x = np.concatenate([base_x + step_x, base_x])
    corner_y = np.concatenate([base_y, base_y + step_y])
    return (np.concatenate([cell_x, corner_x]), np.concatenate([cell_y, corner_y]))


def _mark(
    grid: np.ndarray, cell_x: np.ndarray, cell_y: np.ndarray, spec: GridSpec
) -> None:
    """Set the in-bounds cells of ``(cell_x, cell_y)`` to True."""
    keep = (
        (cell_x >= 0) & (cell_x < spec.width) & (cell_y >= 0) & (cell_y < spec.height)
    )
    grid[cell_y[keep], cell_x[keep]] = True
