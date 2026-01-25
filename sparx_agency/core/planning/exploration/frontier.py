"""
Frontier utilities for 2D exploration.

A frontier cell is defined as a FREE cell that has at least one UNKNOWN neighbor
(4-connected neighborhood by default). This is a standard definition in
frontier-based exploration.

All outputs are in WORLD coordinates (Pose2D) to match the rest of core planning types.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Set, Tuple

from sparx_agency.core.common.types import Pose2D
from sparx_agency.core.planning.environment.occupancy_grid2d import OccupancyGrid2D


Index2D = Tuple[int, int]


def _neighbors_4(gx: int, gy: int) -> List[Index2D]:
    return [(gx - 1, gy), (gx + 1, gy), (gx, gy - 1), (gx, gy + 1)]


def _neighbors_8(gx: int, gy: int) -> List[Index2D]:
    return [
        (gx - 1, gy), (gx + 1, gy), (gx, gy - 1), (gx, gy + 1),
        (gx - 1, gy - 1), (gx - 1, gy + 1), (gx + 1, gy - 1), (gx + 1, gy + 1),
    ]


@dataclass(frozen=True, slots=True)
class FrontierParams:
    """
    Parameters for frontier extraction.

    Attributes:
        connectivity: 4 or 8 neighborhood used for UNKNOWN adjacency checks.
        min_cluster_size: If > 1, clusters smaller than this will be dropped.
    """
    connectivity: int = 4
    min_cluster_size: int = 1


def extract_frontiers(
    grid: OccupancyGrid2D,
    *,
    params: FrontierParams = FrontierParams(),
) -> Set[Pose2D]:
    """
    Extract frontier cells as Pose2D (world).

    Args:
        grid: OccupancyGrid2D containing FREE/OCCUPIED/UNKNOWN.
        params: Frontier extraction settings.

    Returns:
        Set of Pose2D (x,y in world) representing frontier cell centers.
        yaw is set to 0.0 (unused).
    """
    neigh = _neighbors_8 if params.connectivity == 8 else _neighbors_4

    frontier_cells: List[Index2D] = []
    for gy in range(grid.height):
        for gx in range(grid.width):
            if not grid.is_free(gx, gy):
                continue

            # frontier if has at least one unknown neighbor
            for nx, ny in neigh(gx, gy):
                if grid.in_bounds(nx, ny) and grid.is_unknown(nx, ny):
                    frontier_cells.append((gx, gy))
                    break

    if params.min_cluster_size <= 1:
        return {Pose2D(*grid.grid_to_world(gx, gy), 0.0) for gx, gy in frontier_cells}

    # If clustering is requested, cluster by grid adjacency, then keep clusters >= min size.
    kept: Set[Index2D] = set()
    visited: Set[Index2D] = set()
    frontier_set = set(frontier_cells)

    for cell in frontier_cells:
        if cell in visited:
            continue

        # BFS cluster
        stack = [cell]
        cluster: List[Index2D] = []
        visited.add(cell)

        while stack:
            c = stack.pop()
            cluster.append(c)
            cx, cy = c
            for nb in neigh(cx, cy):
                if nb in frontier_set and nb not in visited:
                    visited.add(nb)
                    stack.append(nb)

        if len(cluster) >= params.min_cluster_size:
            kept.update(cluster)

    return {Pose2D(*grid.grid_to_world(gx, gy), 0.0) for gx, gy in kept}
