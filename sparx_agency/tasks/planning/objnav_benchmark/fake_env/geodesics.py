"""Distances on the fake building's grid: the one legal-move rule, the Dijkstra field, and the clearance weights a planner bends its route with.

A **test rig, not a benchmark.** The environment reads ``l``, ``d0`` and
``dT`` off :func:`geodesic_field`, SPL divides by them, and the oracle walks
down the same field, so the two quiet failures of a grid distance are
designed out here:

* **A geodesic that cuts corners.** Distances come from Dijkstra over
  navigable cells, 8-connected, and a diagonal move needs both cells it
  squeezes between to be navigable -- a corner-cutting distance is shorter
  than any path the agent can walk. The oracle descends the same moves
  (:func:`legal_moves`), so it never takes a step the distance did not count.
* **A route that hugs the walls.** A shortest path runs along the edge of the
  navigable band, and a converter that aims ahead along it cuts the corners,
  into steps the environment refuses. :func:`clearance_costs` weighs the
  cells beside anything non-navigable, so a route keeps a cell off the walls
  wherever a lane exists. The environment's own distances never pass costs:
  a route bends, a score never does.

The repo's grid searches sit in ``core.planning.planners`` and
``core.mapping.topology``, which pull OMPL and networkx at import, so the
Dijkstra here is local.

Python 3.8 syntax, numpy.
"""
from __future__ import annotations

import heapq
import math
from typing import Iterator, List, Optional, Tuple

import numpy as np

from sparx_agency.core.planning.objnav.errors import ObjNavError
from sparx_agency.tasks.planning.objnav_benchmark.checks import is_length
from sparx_agency.tasks.planning.objnav_benchmark.fake_env.world import GridWorld

#: The eight moves between neighbouring cells, ``(drow, dcol, length in
#: cells)``, orthogonal first. The order is the tie-break wherever one move is
#: chosen among equals, so every choice is deterministic.
MOVES = ((0, 1, 1.0), (1, 0, 1.0), (0, -1, 1.0), (-1, 0, 1.0),
         (1, 1, math.sqrt(2.0)), (1, -1, math.sqrt(2.0)),
         (-1, 1, math.sqrt(2.0)), (-1, -1, math.sqrt(2.0)))

#: Extra weight on a navigable cell that touches a non-navigable one (8
#: neighbours, the grid's edge included): walking it costs ``1 + penalty``
#: times its length. Large enough that a lane one cell off the wall always
#: beats hugging it; a doorway or an aim with no such lane is still used.
CLEARANCE_PENALTY = 3.0


def legal_moves(world: GridWorld, row: int, col: int,
                navigable: np.ndarray) -> List[Tuple[int, int, float]]:
    """The moves out of a navigable cell, as ``(row, col, metres)``, in :data:`MOVES` order.

    Onto a navigable neighbour; diagonally only when both orthogonal cells
    it squeezes between are navigable (no corner cutting). The same rule
    :func:`geodesic_field` measures with.

    Args:
        world: The building.
        row: The cell's row.
        col: The cell's column.
        navigable: Boolean mask of the cells a walk may use.

    Returns:
        The moves, each to a navigable neighbour.

    Raises:
        ObjNavError: If ``navigable`` is not a boolean mask of the world's
            shape, or ``(row, col)`` is not a navigable cell of it.
    """
    free = world.check_mask("navigable", navigable)
    if not (world.in_bounds(row, col) and free[row, col]):
        raise ObjNavError("cell %r is not a navigable cell of this world"
                          % ((row, col),))
    resolution = world.resolution_m
    return [(r, c, length * resolution)
            for r, c, length in _moves_from(free, world.shape, row, col)]


def geodesic_field(world: GridWorld, targets: np.ndarray,
                   navigable: np.ndarray,
                   cell_costs: Optional[np.ndarray] = None) -> np.ndarray:
    """Metres from every cell to the nearest target, walking navigable cells.

    Dijkstra, 8-connected, steps of one or ``sqrt(2)`` cells, no corner
    cutting (:func:`legal_moves`). Distances are between cell centres. With
    ``cell_costs`` a step costs its length times the mean of the two cells'
    costs -- a planner's weighted distance, no longer metres; the
    environment's own distances never pass it.

    Args:
        world: The building.
        targets: Boolean mask of the target cells; all navigable.
        navigable: Boolean mask of the cells the walk may use.
        cell_costs: Optional float array of per-cell weights, each finite and
            at least 1; None for the true geodesic.

    Returns:
        A new float64 array: 0 on the targets, ``inf`` where no target is
        reachable (and on every non-navigable cell).

    Raises:
        ObjNavError: On a mask that is not the world's shape, a target the
            agent cannot stand on, or a cost below 1 or not finite.
    """
    targets = world.check_mask("targets", targets)
    free = world.check_mask("navigable", navigable)
    stray = np.argwhere(targets & ~free)
    if len(stray):
        raise ObjNavError(
            "%d target cells are not navigable (first %r); a distance to "
            "a cell the agent cannot stand on is undefined -- intersect "
            "the targets with the navigable mask"
            % (len(stray), tuple(int(i) for i in stray[0])))
    weight = _cell_costs(world, cell_costs)
    return _dijkstra(targets, free, world.resolution_m, weight)


def clearance_costs(navigable: np.ndarray,
                    penalty: float = CLEARANCE_PENALTY) -> np.ndarray:
    """Per-cell planning weights: ``1 + penalty`` on cells next to anything non-navigable, else 1.

    Args:
        navigable: Boolean mask of the navigable cells.
        penalty: The extra weight, at least 0.

    Returns:
        A new float64 array of the mask's shape.

    Raises:
        ObjNavError: On a mask that is not a 2-D boolean array, or a negative
            or non-finite penalty.
    """
    if (not isinstance(navigable, np.ndarray) or navigable.dtype != np.bool_
            or navigable.ndim != 2):
        raise ObjNavError("navigable must be a 2-D boolean array")
    if not is_length(penalty):
        raise ObjNavError("penalty must be a finite non-negative number, got "
                          "%r" % (penalty,))
    rows, cols = navigable.shape
    blocked = np.pad(~navigable, 1, mode="constant", constant_values=True)
    tight = np.zeros(navigable.shape, dtype=bool)
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            tight |= blocked[1 + dr:1 + dr + rows, 1 + dc:1 + dc + cols]
    return np.where(tight & navigable, 1.0 + float(penalty), 1.0)


def _moves_from(free, shape: Tuple[int, int], row: int,
                col: int) -> Iterator[Tuple[int, int, float]]:
    """The legal moves out of ``(row, col)``, ``(row, col, length in cells)``.

    The one definition of a legal move: onto a navigable neighbour, and
    diagonally only when both orthogonal cells it squeezes between are
    navigable. ``free`` is anything indexable as ``free[row][col]`` -- a
    nested list for the Dijkstra's speed, a boolean array elsewhere.
    """
    rows, cols = shape
    for dr, dc, length in MOVES:
        r, c = row + dr, col + dc
        if not (0 <= r < rows and 0 <= c < cols) or not free[r][c]:
            continue
        if dr and dc and not (free[row][c] and free[r][col]):
            continue
        yield r, c, length


def _dijkstra(targets: np.ndarray, free: np.ndarray, resolution_m: float,
              weight: Optional[List[List[float]]]) -> np.ndarray:
    """Dijkstra from every target at once over the ``free`` cells; ``inf`` where none is reachable."""
    rows, cols = (int(n) for n in free.shape)
    open_cells = free.tolist()
    best = [[math.inf] * cols for _ in range(rows)]
    heap = []
    for row, col in np.argwhere(targets):
        best[int(row)][int(col)] = 0.0
        heap.append((0.0, int(row), int(col)))
    heapq.heapify(heap)
    while heap:
        distance, row, col = heapq.heappop(heap)
        if distance > best[row][col]:
            continue
        for r, c, length in _moves_from(open_cells, (rows, cols), row, col):
            step = length * resolution_m
            if weight is not None:
                step *= (weight[row][col] + weight[r][c]) / 2.0
            candidate = distance + step
            if candidate < best[r][c]:
                best[r][c] = candidate
                heapq.heappush(heap, (candidate, r, c))
    return np.array(best, dtype=np.float64)


def _cell_costs(world: GridWorld,
                cell_costs) -> Optional[List[List[float]]]:
    """``cell_costs`` as nested lists once each is known to be finite and at least 1."""
    if cell_costs is None:
        return None
    if not isinstance(cell_costs, np.ndarray):
        raise ObjNavError("cell_costs must be a float numpy array, got %s"
                          % type(cell_costs).__name__)
    if (cell_costs.shape != world.shape
            or not np.issubdtype(cell_costs.dtype, np.floating)):
        raise ObjNavError(
            "cell_costs must be a float array of the world's shape %r, got "
            "shape %r dtype %s" % (world.shape, cell_costs.shape,
                                   cell_costs.dtype))
    if not (np.all(np.isfinite(cell_costs)) and np.all(cell_costs >= 1.0)):
        raise ObjNavError("cell_costs must be finite and at least 1, so a "
                          "weighted step is never shorter than the step")
    return cell_costs.tolist()
