from __future__ import annotations

import heapq
from dataclasses import dataclass
from typing import Dict, List, Tuple

from sparx_agency.core.planning.environment.voxelmap import VoxelMap3D

Index3D = Tuple[int, int, int]


@dataclass(frozen=True)
class AStarSearchResult3D:
    path: Tuple[Index3D, ...]
    expanded: int

    @property
    def ok(self) -> bool:
        return len(self.path) > 0


def _manhattan3(a: Index3D, b: Index3D) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1]) + abs(a[2] - b[2])


def _neighbors_3d(i: int, j: int, k: int, connectivity: int) -> List[Index3D]:
    # 6-neighborhood
    base = [(i - 1, j, k), (i + 1, j, k), (i, j - 1, k), (i, j + 1, k), (i, j, k - 1), (i, j, k + 1)]
    if connectivity == 6:
        return base

    # 18: add edge-adjacent (2 axes change)
    nbrs = set(base)
    for di in (-1, 0, 1):
        for dj in (-1, 0, 1):
            for dk in (-1, 0, 1):
                if di == dj == dk == 0:
                    continue
                changes = (di != 0) + (dj != 0) + (dk != 0)
                if connectivity == 18 and changes <= 2:
                    nbrs.add((i + di, j + dj, k + dk))
                elif connectivity == 26:
                    nbrs.add((i + di, j + dj, k + dk))
    return list(nbrs)


def astar_voxel_3d(
    voxelmap: VoxelMap3D,
    start: Index3D,
    goal: Index3D,
    *,
    allow_unknown: bool,
    connectivity: int,
    max_expansions: int | None,
) -> AStarSearchResult3D:
    if start == goal:
        return AStarSearchResult3D(path=(), expanded=0)

    def in_bounds(a: Index3D) -> bool:
        i, j, k = a
        return 0 <= i < voxelmap.width and 0 <= j < voxelmap.height and 0 <= k < voxelmap.depth

    def traversable(a: Index3D) -> bool:
        if not in_bounds(a):
            return False
        i, j, k = a

        # VoxelMap3D protocol guarantees is_free(i,j,k); unknown support is not guaranteed.
        # If a future voxelmap adds unknown, it can encode unknown as not free.
        ok = bool(voxelmap.is_free(i, j, k))
        if ok:
            return True
        if allow_unknown:
            # If unknown is representable, user can implement is_free() accordingly.
            # We keep allow_unknown here for symmetry with 2D; default False.
            return False
        return False

    if not traversable(start) or not traversable(goal):
        return AStarSearchResult3D(path=(), expanded=0)

    open_heap: List[Tuple[int, Index3D]] = []
    heapq.heappush(open_heap, (0, start))

    came_from: Dict[Index3D, Index3D] = {}
    g_score: Dict[Index3D, int] = {start: 0}

    expanded = 0
    while open_heap:
        _, current = heapq.heappop(open_heap)
        expanded += 1
        if max_expansions is not None and expanded > max_expansions:
            return AStarSearchResult3D(path=(), expanded=expanded)

        if current == goal:
            break

        ci, cj, ck = current
        for n in _neighbors_3d(ci, cj, ck, connectivity):
            if not traversable(n):
                continue
            tentative = g_score[current] + 1
            if n not in g_score or tentative < g_score[n]:
                came_from[n] = current
                g_score[n] = tentative
                f = tentative + _manhattan3(n, goal)
                heapq.heappush(open_heap, (f, n))

    if goal not in came_from:
        return AStarSearchResult3D(path=(), expanded=expanded)

    rev: List[Index3D] = [goal]
    cur = goal
    while cur in came_from:
        cur = came_from[cur]
        if cur == start:
            break
        rev.append(cur)

    rev.reverse()
    return AStarSearchResult3D(path=tuple(rev), expanded=expanded)
