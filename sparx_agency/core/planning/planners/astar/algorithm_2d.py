from __future__ import annotations

import heapq
from dataclasses import dataclass
from typing import Dict, List, Tuple

from sparx_agency.core.planning.environment import OccupancyGrid2D

Index2D = Tuple[int, int]


@dataclass(frozen=True)
class AStarSearchResult2D:
    path: Tuple[Index2D, ...]
    expanded: int

    @property
    def ok(self) -> bool:
        return len(self.path) > 0


def _manhattan(a: Index2D, b: Index2D) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def _neighbors_2d(gx: int, gy: int, connectivity: int) -> List[Index2D]:
    if connectivity == 8:
        return [
            (gx - 1, gy), (gx + 1, gy), (gx, gy - 1), (gx, gy + 1),
            (gx - 1, gy - 1), (gx - 1, gy + 1), (gx + 1, gy - 1), (gx + 1, gy + 1),
        ]
    # default 4
    return [(gx - 1, gy), (gx + 1, gy), (gx, gy - 1), (gx, gy + 1)]


def astar_grid_2d(
    grid: OccupancyGrid2D,
    start: Index2D,
    goal: Index2D,
    *,
    allow_unknown: bool,
    connectivity: int,
    max_expansions: int | None,
) -> AStarSearchResult2D:
    if start == goal:
        return AStarSearchResult2D(path=(), expanded=0)

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

    open_heap: List[Tuple[int, Index2D]] = []
    heapq.heappush(open_heap, (0, start))

    came_from: Dict[Index2D, Index2D] = {}
    g_score: Dict[Index2D, int] = {start: 0}

    expanded = 0
    while open_heap:
        _, current = heapq.heappop(open_heap)
        expanded += 1
        if max_expansions is not None and expanded > max_expansions:
            return AStarSearchResult2D(path=(), expanded=expanded)

        if current == goal:
            break

        cx, cy = current
        for n in _neighbors_2d(cx, cy, connectivity):
            if not traversable(*n):
                continue
            tentative = g_score[current] + 1
            if n not in g_score or tentative < g_score[n]:
                came_from[n] = current
                g_score[n] = tentative
                f = tentative + _manhattan(n, goal)
                heapq.heappush(open_heap, (f, n))

    if goal not in came_from:
        return AStarSearchResult2D(path=(), expanded=expanded)

    # Reconstruct path excluding start and including goal
    rev: List[Index2D] = [goal]
    cur = goal
    while cur in came_from:
        cur = came_from[cur]
        if cur == start:
            break
        rev.append(cur)

    rev.reverse()
    return AStarSearchResult2D(path=tuple(rev), expanded=expanded)
