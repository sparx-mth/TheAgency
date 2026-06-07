"""A* search kernels on 2D grids.

Two entry points:

- :func:`astar_grid_2d` works on any object exposing the ``OccupancyGrid2D``
  query protocol (``in_bounds`` / ``is_occupied`` / ``is_unknown``). It treats
  cells as binary traversable/blocked and is used by the exploration behaviours
  and the windowed local planner.
- :func:`astar_cost_grid_2d` works on a dense ``float`` cost array (1.0 = free,
  ``inf`` = blocked, any finite value = weighted) and adds a bounding-box search
  domain plus an optional turn penalty. This is the fast kernel used by the
  weighted FALCON planner.

Both use proper 8-connected step costs (diagonals cost ``sqrt(2)``) and the
octile heuristic, which is the tightest admissible/consistent heuristic for
8-connected motion. For 4-connectivity the heuristic reduces to Manhattan.
"""
from __future__ import annotations

import heapq
import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from sparx_agency.core.planning.environment import OccupancyGrid2D

Index2D = Tuple[int, int]
BBox = Tuple[int, int, int, int]  # (xmin, xmax, ymin, ymax), half-open on max

SQRT2 = math.sqrt(2.0)


@dataclass(frozen=True)
class AStarSearchResult2D:
    path: Tuple[Index2D, ...]
    expanded: int

    @property
    def ok(self) -> bool:
        return len(self.path) > 0


def _octile(ax: int, ay: int, bx: int, by: int) -> float:
    """Octile distance: admissible/consistent heuristic for 8-connected grids.

    Reduces to Manhattan when the path is restricted to 4-connected moves
    (the diagonal term is non-positive only via ``min(dx, dy)``, so on a pure
    cardinal optimal path it adds nothing).
    """
    dx = abs(ax - bx)
    dy = abs(ay - by)
    return (dx + dy) + (SQRT2 - 2.0) * min(dx, dy)


def _manhattan(ax: int, ay: int, bx: int, by: int) -> int:
    return abs(ax - bx) + abs(ay - by)


# (dx, dy, step_cost)
_MOVES_4 = ((-1, 0, 1.0), (1, 0, 1.0), (0, -1, 1.0), (0, 1, 1.0))
_MOVES_8 = _MOVES_4 + (
    (-1, -1, SQRT2), (-1, 1, SQRT2), (1, -1, SQRT2), (1, 1, SQRT2),
)


def astar_grid_2d(
    grid: OccupancyGrid2D,
    start: Index2D,
    goal: Index2D,
    *,
    allow_unknown: bool,
    connectivity: int,
    max_expansions: int | None,
) -> AStarSearchResult2D:
    """A* over a binary (traversable/blocked) grid.

    Args:
        grid: Any object with ``in_bounds``/``is_occupied``/``is_unknown``.
        start: Start cell ``(gx, gy)``.
        goal: Goal cell ``(gx, gy)``.
        allow_unknown: If False, UNKNOWN cells are blocked.
        connectivity: 4 or 8. 8 uses ``sqrt(2)`` diagonal cost + octile h.
        max_expansions: Safety cap on pops; ``None`` for unlimited.

    Returns:
        ``AStarSearchResult2D`` with the path excluding ``start`` and including
        ``goal`` (empty path if unreachable or capped).
    """
    if start == goal:
        return AStarSearchResult2D(path=(), expanded=0)

    moves = _MOVES_8 if connectivity == 8 else _MOVES_4
    h = _octile if connectivity == 8 else _manhattan

    def traversable(x: int, y: int) -> bool:
        if not grid.in_bounds(x, y):
            return False
        if grid.is_occupied(x, y):
            return False
        if (not allow_unknown) and grid.is_unknown(x, y):
            return False
        return True

    if not traversable(*start) or not traversable(*goal):
        return AStarSearchResult2D(path=(), expanded=0)

    gx, gy = goal
    open_heap: List[Tuple[float, Index2D]] = [(0.0, start)]
    came_from: Dict[Index2D, Index2D] = {}
    g_score: Dict[Index2D, float] = {start: 0.0}

    expanded = 0
    while open_heap:
        _, current = heapq.heappop(open_heap)
        expanded += 1
        if max_expansions is not None and expanded > max_expansions:
            return AStarSearchResult2D(path=(), expanded=expanded)

        if current == goal:
            break

        cx, cy = current
        gc = g_score[current]
        for dx, dy, step in moves:
            nx, ny = cx + dx, cy + dy
            if not traversable(nx, ny):
                continue
            tentative = gc + step
            n = (nx, ny)
            if tentative < g_score.get(n, math.inf):
                came_from[n] = current
                g_score[n] = tentative
                heapq.heappush(open_heap, (tentative + h(nx, ny, gx, gy), n))

    if goal not in came_from:
        return AStarSearchResult2D(path=(), expanded=expanded)

    rev: List[Index2D] = [goal]
    cur = goal
    while cur in came_from:
        cur = came_from[cur]
        if cur == start:
            break
        rev.append(cur)
    rev.reverse()
    return AStarSearchResult2D(path=tuple(rev), expanded=expanded)


def astar_cost_grid_2d(
    cost: np.ndarray,
    start: Index2D,
    goal: Index2D,
    *,
    connectivity: int = 8,
    bbox: Optional[BBox] = None,
    turn_penalty: float = 0.0,
    max_expansions: int | None = None,
) -> AStarSearchResult2D:
    """A* over a dense float cost grid (the fast, weighted kernel).

    Edge cost from a cell to a neighbour is ``step * cost[ny, nx]`` where
    ``step`` is 1.0 (cardinal) or ``sqrt(2)`` (diagonal). A cell with a
    non-finite cost (``inf``) is blocked. An optional ``turn_penalty`` is added
    whenever the travel direction changes, which suppresses staircasing before
    line-of-sight smoothing runs.

    Args:
        cost: ``(H, W)`` float array. 1.0 = free, ``inf`` = blocked, finite
            values > 1 are traversable-at-a-penalty (e.g. UNKNOWN).
        start: Start cell ``(gx, gy)`` (indexed ``cost[gy, gx]``).
        goal: Goal cell ``(gx, gy)``.
        connectivity: 4 or 8.
        bbox: Optional ``(xmin, xmax, ymin, ymax)`` half-open search window in
            grid indices. Cells outside are never expanded — the single largest
            speed win when start and goal are close on a large grid.
        turn_penalty: Extra cost added on a direction change (0 disables).
        max_expansions: Safety cap on pops; ``None`` for unlimited.

    Returns:
        ``AStarSearchResult2D`` with the path including both ``start`` and
        ``goal`` (empty if unreachable or capped). Note: unlike
        :func:`astar_grid_2d`, the start cell IS included, because the weighted
        planner consumes the full cell chain for line-of-sight smoothing.
    """
    H, W = cost.shape
    sx, sy = start
    gx, gy = goal

    if bbox is None:
        xmin, xmax, ymin, ymax = 0, W, 0, H
    else:
        xmin, xmax, ymin, ymax = bbox

    moves = _MOVES_8 if connectivity == 8 else _MOVES_4
    h = _octile if connectivity == 8 else _manhattan

    g = np.full((H, W), np.inf, dtype=np.float64)
    g[sy, sx] = 0.0
    closed = np.zeros((H, W), dtype=bool)
    came: Dict[Index2D, Index2D] = {}

    open_heap: List[Tuple[float, float, int, int]] = [(h(sx, sy, gx, gy), 0.0, sx, sy)]
    expanded = 0
    while open_heap:
        _f, gc, x, y = heapq.heappop(open_heap)
        if closed[y, x]:
            continue
        closed[y, x] = True
        expanded += 1
        if max_expansions is not None and expanded > max_expansions:
            return AStarSearchResult2D(path=(), expanded=expanded)

        if x == gx and y == gy:
            path = [(x, y)]
            while (x, y) in came:
                x, y = came[(x, y)]
                path.append((x, y))
            path.reverse()
            return AStarSearchResult2D(path=tuple(path), expanded=expanded)

        prev = None
        if turn_penalty > 0.0 and (x, y) in came:
            px, py = came[(x, y)]
            prev = (x - px, y - py)

        for dx, dy, step in moves:
            nx, ny = x + dx, y + dy
            if not (xmin <= nx < xmax and ymin <= ny < ymax):
                continue
            if closed[ny, nx]:
                continue
            c = cost[ny, nx]
            if not math.isfinite(c):
                continue
            turn = turn_penalty if (prev is not None and prev != (dx, dy)) else 0.0
            ng = gc + step * c + turn
            if ng < g[ny, nx]:
                g[ny, nx] = ng
                came[(nx, ny)] = (x, y)
                heapq.heappush(open_heap, (ng + h(nx, ny, gx, gy), ng, nx, ny))

    return AStarSearchResult2D(path=(), expanded=expanded)
