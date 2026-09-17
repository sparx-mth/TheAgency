"""Evaluator-only 3D geodesics to annotated Gibson success regions.

This is explicitly NOT the SemExp planar validation metric. The reference
semantic dilation defines a goal on its labelled floor; a separate CPU
PathFinder measures paths through the full, original building navmesh. Nothing
here is available to the policy or used to generate its actions.
"""
from __future__ import annotations

import math
import numpy as np

from sparx_agency.tasks.planning.objnav_benchmark_runtime.gibson.distance import GibsonDistanceField
from sparx_agency.tasks.planning.objnav_benchmark_runtime.gibson.protocol import PROTOCOL


class MultiFloorDistance:
    def __init__(self, pathfinder, goals, semantic, origin, category, goal_height, path_factory=None):
        self.pathfinder = pathfinder
        self.goals = np.asarray(goals, dtype=np.float32)
        self.goal_height = float(goal_height)
        self.floor = GibsonDistanceField(semantic, origin, category)
        if self.goals.ndim != 2 or self.goals.shape[1] != 3 or len(self.goals) == 0 or not np.isfinite(self.goals).all():
            raise ValueError("Need finite navigable 3D goal-region samples")
        if path_factory is None:
            import habitat_sim
            path_factory = habitat_sim.MultiGoalShortestPath
        self.path_factory = path_factory
        self._cached_position = self._cached_distance = None

    def success(self, position):
        # XY overlap upstairs must never count as success downstairs.
        if abs(float(position[1]) - self.goal_height) > 0.40:
            return False
        try:
            return self.floor.distance(position) == 0.0
        except ValueError:
            return False

    def distance(self, position, start=False):
        position = tuple(float(v) for v in position)
        if self.success(position):
            return 0.0
        if position == self._cached_position:
            return self._cached_distance
        path = self.path_factory()
        path.requested_start = np.asarray(position, dtype=np.float32)
        path.requested_ends = self.goals
        if not self.pathfinder.find_path(path) or not math.isfinite(path.geodesic_distance):
            raise ValueError("Episode left the connected navigable component; geodesic score unavailable")
        distance = float(path.geodesic_distance)
        if distance <= 0:
            raise ValueError("Zero geodesic outside the height-qualified success region")
        self._cached_position, self._cached_distance = position, distance
        return distance

    def start_uses_sentinel(self, position):
        self.distance(position, start=True)
        return False  # unreachable generated starts are errors, not finite fillers


def goal_region_points(pathfinder, floor, category, spacing_cells=4):
    """Deterministic samples, snapped and rechecked on the labelled floor only."""
    semantic = np.asarray(floor["sem_map"])
    field = GibsonDistanceField(semantic, floor["origin"], category)
    valid = (field.cells == 0) & (semantic[0] > 0)
    yy, xx = np.nonzero(valid)
    keep = (yy % spacing_cells == 0) & (xx % spacing_cells == 0)
    points = []
    for y, x in zip(yy[keep], xx[keep]):
        position = np.array([field.origin_m[1] + (y + 0.5) * PROTOCOL.map_resolution_m,
                             float(floor["floor_height"]),
                             field.origin_m[0] + (x + 0.5) * PROTOCOL.map_resolution_m])
        snapped = np.asarray(pathfinder.snap_point(position), dtype=float)
        if not np.isfinite(snapped).all() or abs(snapped[1] - position[1]) > 0.45:
            continue
        try:
            if field.distance(snapped) != 0 or np.linalg.norm(snapped[[0, 2]] - position[[0, 2]]) > 0.15:
                continue
        except ValueError:
            continue
        if pathfinder.is_navigable(snapped):
            points.append(snapped.tolist())
    if not points:
        raise ValueError("No navigable reference-floor success-region samples")
    return points

