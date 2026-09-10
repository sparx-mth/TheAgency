"""Where the privileged oracle aims: deep enough inside the goal region that arriving at its path's end puts the agent in it.

**Part of an upper bound for testing the pipeline, never a baseline.** Aimed
at the goal region itself, the oracle's last waypoint sits on the region's
edge, and the action converter stops following a path once the agent is
within its goal tolerance of the last waypoint -- which, at the edge, can
leave the agent one cell *outside* the region: arrived as far as the
converter is concerned, not in the goal as far as the oracle is, so it
re-plans the same path, is told it has arrived, and turns in place until the
budget runs out -- a failure at the step limit, not an error.

So it aims at the region eroded by :func:`erosion_cells` from its outer
boundary -- the side that borders navigable floor outside the region; walls
and objects do not erode it, or a region against a wall would empty for
nothing -- and stops the moment the agent enters the *full* region. When
erosion leaves nothing, the full region is the aim.

Python 3.8 syntax, numpy.
"""
from __future__ import annotations

import collections
import math

import numpy as np

from sparx_agency.core.planning.objnav.errors import ObjNavError
from sparx_agency.tasks.planning.objnav_benchmark.checks import (
    is_count,
    is_length,
)
from sparx_agency.tasks.planning.objnav_benchmark.fake_env.geodesics import MOVES

#: Slack on the erosion depth: ``(0.25 + 0.25) / 0.25`` must read 2, not 3.
_CELLS_EPS = 1e-9


def erosion_cells(goal_tolerance_m: float, forward_step_m: float,
                  resolution_m: float) -> int:
    """``ceil((goal_tolerance_m + forward_step_m) / resolution_m)``: how deep the aim sits.

    Deep enough that an agent within the converter's goal tolerance of the
    last waypoint -- or a step past it -- is inside the full region.

    Raises:
        ObjNavError: On a length that is not positive and finite.
    """
    for name, value in (("goal_tolerance_m", goal_tolerance_m),
                        ("forward_step_m", forward_step_m),
                        ("resolution_m", resolution_m)):
        if not (is_length(value) and value > 0):
            raise ObjNavError("%s must be a positive finite number, got %r"
                              % (name, value))
    return int(math.ceil((goal_tolerance_m + forward_step_m) / resolution_m
                         - _CELLS_EPS))


def erode_goal_region(goal: np.ndarray, navigable: np.ndarray,
                      cells: int) -> np.ndarray:
    """The goal cells more than ``cells`` 8-connected steps inside the region's outer boundary.

    A multi-source breadth-first search from every navigable cell *outside*
    the region, stepping through goal cells only: a cell's depth is its step
    count from the nearest such cell. Walls and objects are no source, so the
    region is not eroded where it meets them; a region with no outside
    neighbour at all is kept whole. 8-connected on purpose: a diagonal
    neighbour counts as one step, which erodes at least as far as the
    Euclidean distance would.

    Args:
        goal: Boolean mask of the goal region.
        navigable: Boolean mask of the navigable cells, same shape.
        cells: How many cells to erode; 0 keeps the region.

    Returns:
        A new boolean mask, a subset of ``goal``; possibly empty.

    Raises:
        ObjNavError: On masks that are not boolean arrays of one shape, or a
            negative ``cells``.
    """
    for name, mask in (("goal", goal), ("navigable", navigable)):
        if not isinstance(mask, np.ndarray) or mask.dtype != np.bool_ or mask.ndim != 2:
            raise ObjNavError("%s must be a 2-D boolean array" % name)
    if goal.shape != navigable.shape:
        raise ObjNavError("goal %r and navigable %r differ in shape"
                          % (goal.shape, navigable.shape))
    if not is_count(cells):
        raise ObjNavError("cells must be a non-negative integer, got %r"
                          % (cells,))
    rows, cols = goal.shape
    inside = goal.tolist()
    depth = [[-1] * cols for _ in range(rows)]
    queue = collections.deque()
    for row, col in np.argwhere(navigable & ~goal):
        depth[int(row)][int(col)] = 0
        queue.append((int(row), int(col)))
    while queue:
        row, col = queue.popleft()
        for dr, dc, _ in MOVES:
            r, c = row + dr, col + dc
            if 0 <= r < rows and 0 <= c < cols and inside[r][c] and depth[r][c] < 0:
                depth[r][c] = depth[row][col] + 1
                queue.append((r, c))
    steps = np.array(depth)
    return goal & ((steps > cells) | (steps < 0))
