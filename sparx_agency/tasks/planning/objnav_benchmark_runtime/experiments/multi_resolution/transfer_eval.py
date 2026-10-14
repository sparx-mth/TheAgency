"""Evaluator-only ground truth for explored maps; the policy must never import this."""
from __future__ import annotations

import numpy as np
from scipy.ndimage import binary_dilation


def navigable_lattice_enu(pathfinder, floor_height_m, start_habitat, step_m):
    """Navigable lattice points on the start floor and island, as ENU (x, y)."""
    low, high = pathfinder.get_bounds()
    start_island = pathfinder.get_island(np.asarray(start_habitat, dtype=np.float32))
    xs = np.arange(low[0] + step_m / 2, high[0], step_m)
    zs = np.arange(low[2] + step_m / 2, high[2], step_m)
    points = []
    for x in xs:
        for z in zs:
            candidate = np.array([x, floor_height_m, z], dtype=np.float32)
            if pathfinder.is_navigable(candidate) and pathfinder.get_island(candidate) == start_island:
                points.append((-z, -x))
    return np.asarray(points, dtype=np.float64).reshape(-1, 2)


def _cells_of(world, points_xy):
    gx = np.floor((points_xy[:, 0] - world.origin_x) / world.resolution).astype(int)
    gy = np.floor((points_xy[:, 1] - world.origin_y) / world.resolution).astype(int)
    inside = (gx >= 0) & (gx < world.width) & (gy >= 0) & (gy < world.height)
    return gx[inside], gy[inside]


def coverage_metrics(world, observed_mask, gt_points_xy, support_radius_m=0.4):
    """Per-lattice-point coverage, so maps of different resolution compare; plus unsupported free cells.

    A free cell is supported when a navigable point lies within support_radius_m of it.
    """
    gx, gy = _cells_of(world, np.asarray(gt_points_xy, dtype=np.float64))
    known = world.grid != world.values.unknown
    free = world.grid == world.values.free
    gt = np.zeros(world.grid.shape, dtype=bool)
    gt[gy, gx] = True
    reach = int(np.ceil(support_radius_m / world.resolution))
    supported = binary_dilation(gt, structure=np.ones((3, 3), dtype=bool), iterations=reach)
    points = max(len(gx), 1)
    free_cells = int(free.sum())
    return {"gt_points": len(gx),
            "observed_coverage": float(observed_mask[gy, gx].sum()) / points,
            "known_coverage": float(known[gy, gx].sum()) / points,
            "free_cells": free_cells,
            "false_free_rate": float((free & ~supported).sum()) / max(free_cells, 1)}
