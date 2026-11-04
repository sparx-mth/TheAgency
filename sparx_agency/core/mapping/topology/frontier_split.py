# core/mapping/topology/frontier_split.py
"""Frontier cells cut by the SIDE of the unknown they face: one cluster per way into an unmapped block.

A frontier cell is a free cell 4-adjacent to an unknown one, and the
clusters exploration reasons about are their 8-connected components.
One component is one goal, which is right for the common shape -- an arc
at the edge of the depth range, bowed round the robot -- and wrong for
the other common one: the floor mapped along two or three sides of a
block of unknown (an unmapped room glimpsed past its corners, an island
of unknown in a hall). There the one component wraps round the block,
its goal lands on whichever side its centroid falls nearest, and the
other sides -- each a way in from a different room -- have no goal and
no marker (Corozal/000003, ``c1af8954a378``, 2026-10-22, action 291: the
gap O30 on one side of an unknown square, nothing on the side the agent
could reach). A person sees three ways in.

Two steps, shared by the goal maker
(:func:`~sparx_agency.core.planning.exploration.room_costs.frontier_clusters`)
and the scene graph's count
(:func:`~sparx_agency.core.mapping.topology.room_stats.count_frontier_clusters`)
so the two stay one population:

* :func:`frontier_sides` -- which of the four axis directions the unknown
  near each cell lies in, read from the unknown summed as offsets over a
  window: a frontier wrapping a block changes side at the block's corners,
  a diagonal frontier reads one side along its whole length.
* :func:`split_by_side` -- a component whose cells face different sides is
  cut into one part per side **when it wraps the unknown**: the mean of
  ``(centroid - cell) . normal(cell)`` over its cells is positive, i.e. its
  centre lies on the unknown side of its cells. An arc bulging INTO the
  unknown has its centre behind it (the known floor the robot stands on)
  and stays whole: one gap, one goal. A side's cells that are disconnected
  make one part each; a part under ``min_cells`` -- a doorway's width,
  :data:`SIDE_MIN_M`, so a patch the size of a chair keeps its one goal --
  is folded into the nearest part that can stand alone, and nothing a
  single-component cluster offered is lost.

numpy and scipy; no ROS. Not re-exported from the package facade.
"""
from __future__ import annotations

from typing import List, Tuple

import numpy as np
from scipy.ndimage import convolve
from scipy.ndimage import label as cc_label

#: The four sides a frontier cell's unknown can lie on, as ``(dx, dy)`` in grid cells (``gy`` grows with world y).
FRONTIER_SIDES = ((1, 0), (-1, 0), (0, 1), (0, -1))
#: Shortest side, in metres, that stands as a cluster of its own: a doorway's width. A smaller block of
#: unknown -- the patch behind a chair, the island under the agent's own blind radius -- is resolved from any
#: one point of its ring and keeps its one goal.
SIDE_MIN_M = 0.9
#: ... and in cells, for a caller without a resolution (the 0.1 m grids of the ObjectNav runtime).
SIDE_MIN_CELLS = 9
#: Half-width, in metres, of the window the unknown's direction is read from: under half a body length,
#: so a corridor of unknown on both sides of a known tongue reads as what it is, not as one side.
SIDE_WINDOW_M = 0.45
#: ... and in cells, for a caller without a resolution (the 0.1 m grids of the ObjectNav runtime).
SIDE_WINDOW_CELLS = 4

_EIGHT = np.ones((3, 3), dtype=np.uint8)


def side_window_cells(resolution_m: float) -> int:
    """The window half-width :func:`frontier_sides` reads with on a grid of ``resolution_m`` cells (never under 1)."""
    return max(1, int(round(SIDE_WINDOW_M / float(resolution_m))))


def side_min_cells(resolution_m: float, min_cluster_cells: int = 1) -> int:
    """Fewest cells a side's part needs to stand alone on a grid of ``resolution_m`` cells: ``SIDE_MIN_M``, never under the cluster floor."""
    return max(int(min_cluster_cells), 1, int(round(SIDE_MIN_M / float(resolution_m))))


def frontier_sides(unknown: np.ndarray, window_cells: int = SIDE_WINDOW_CELLS) -> np.ndarray:
    """Which side of each cell its nearby unknown lies on: ``0`` +x, ``1`` -x, ``2`` +y, ``3`` -y, ``-1`` none.

    The unknown within ``window_cells`` of a cell, summed as offsets from
    it, points where the unknown is; the axis with the larger component
    wins, +x on an exact tie.

    Args:
        unknown: ``(H, W)`` bool, True where the grid is unknown.
        window_cells: Half-width of the window the unknown is summed over.

    Returns:
        ``(H, W)`` int8 side index per cell, meaningful on frontier cells.
    """
    r = int(max(1, window_cells))
    offsets = np.arange(-r, r + 1, dtype=np.float32)
    kernel_x = np.tile(offsets[None, :], (2 * r + 1, 1))
    kernel_y = np.tile(offsets[:, None], (1, 2 * r + 1))
    weights = np.asarray(unknown, dtype=np.float32)
    # ``convolve`` flips the kernel, so a kernel of offsets sums ``(cell - neighbour)``; negate for ``(neighbour - cell)``.
    wx = -convolve(weights, kernel_x, mode="constant", cval=0.0)
    wy = -convolve(weights, kernel_y, mode="constant", cval=0.0)
    sides = np.full(weights.shape, -1, dtype=np.int8)
    along_x = np.abs(wx) >= np.abs(wy)
    sides[along_x & (wx > 0)] = 0
    sides[along_x & (wx < 0)] = 1
    sides[~along_x & (wy > 0)] = 2
    sides[~along_x & (wy < 0)] = 3
    return sides


def wraps_unknown(ys: np.ndarray, xs: np.ndarray, side: np.ndarray) -> bool:
    """Whether a component's centre lies on the unknown side of its cells (it wraps a block) rather than behind them (an arc)."""
    decided = side >= 0
    if not decided.any():
        return False
    normals = np.asarray(FRONTIER_SIDES, dtype=float)[side[decided]]
    cy, cx = float(ys.mean()), float(xs.mean())
    inward = (cx - xs[decided]) * normals[:, 0] + (cy - ys[decided]) * normals[:, 1]
    return bool(inward.mean() > 0.0)


def split_by_side(ys: np.ndarray, xs: np.ndarray, side: np.ndarray, min_cells: int = SIDE_MIN_CELLS
                  ) -> List[Tuple[np.ndarray, np.ndarray]]:
    """One component's cells cut into the parts that face different sides of their unknown, or the whole.

    Args:
        ys: Row index of every cell of the component.
        xs: Column index of every cell, same order.
        side: :func:`frontier_sides` read at those cells.
        min_cells: Fewest cells a part needs to stand alone; smaller parts
            (and cells whose side is undecided) join the nearest that can.

    Returns:
        ``[(rows, cols), ...]`` -- one entry per part; the component itself
        when it faces one side, does not wrap its unknown, or no two parts
        can stand alone.
    """
    present = [int(v) for v in np.unique(side) if v >= 0]
    if len(present) <= 1 or not wraps_unknown(ys, xs, side):
        return [(ys, xs)]
    y0, x0 = int(ys.min()), int(xs.min())
    shape = (int(ys.max()) - y0 + 1, int(xs.max()) - x0 + 1)
    local_y, local_x = ys - y0, xs - x0
    large: List[np.ndarray] = []          # index arrays into (ys, xs)
    small: List[np.ndarray] = []
    for value in present:
        members = side == value
        mask = np.zeros(shape, dtype=bool)
        mask[local_y[members], local_x[members]] = True
        lbl, n = cc_label(mask, structure=_EIGHT)
        part_of = lbl[local_y, local_x]
        for j in range(1, n + 1):
            index = np.nonzero(members & (part_of == j))[0]
            (large if index.size >= int(min_cells) else small).append(index)
    small.append(np.nonzero(side < 0)[0])
    if len(large) <= 1:
        return [(ys, xs)]
    groups = [list(index) for index in large]
    centres = [(float(ys[index].mean()), float(xs[index].mean())) for index in large]
    for index in small:
        for i in index:
            distances = [(ys[i] - cy) ** 2 + (xs[i] - cx) ** 2 for cy, cx in centres]
            groups[int(np.argmin(distances))].append(int(i))
    out = []
    for group in groups:
        order = np.asarray(sorted(group), dtype=np.intp)
        out.append((ys[order], xs[order]))
    return out
