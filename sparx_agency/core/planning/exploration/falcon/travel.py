"""Ground-motion replacement for FALCON's path_cost_evaluator/trajectory layer.

Host-only scipy module, deliberately not imported by exploration/__init__.py.
Uses the existing compact graph and geometry; unknown guidance and executable
free routes are distinct instances and cannot share a cache.
"""
from __future__ import annotations

import math
import numpy as np
from scipy.sparse.csgraph import dijkstra

from sparx_agency.core.common.types import normalize_angle
from sparx_agency.core.planning.exploration.falcon.ordering import PlanningCapacity
from sparx_agency.core.planning.exploration.room_costs import passable_graph
from sparx_agency.core.planning.planners.common.grid_geometry_2d import line_cells


def segment_clear(passable, a, b):
    """Bresenham plus both side cells at diagonal crossings; off-map is blocked."""
    h, w = passable.shape
    cells = line_cells(int(a[0]), int(a[1]), int(b[0]), int(b[1]))
    for i, (x, y) in enumerate(cells):
        if not (0 <= x < w and 0 <= y < h and passable[y, x]):
            return False
        if i:
            px, py = cells[i - 1]
            if x != px and y != py and not (passable[y, px] and passable[py, x]):
                return False
    return True


class GridRoutes:
    """Cached shortest paths, followed by safe any-angle path shortening.

    Dijkstra (zero-heuristic A*) amortizes all destinations from each source.
    Edge weights use both endpoint costs symmetrically. Corner cutting is
    removed from the older room-cost graph, without changing that baseline.
    """

    def __init__(self, cost, resolution, max_nodes=18000):
        self.cost = np.asarray(cost, dtype=float)
        self.resolution = resolution
        self.safe = np.isfinite(self.cost)
        if np.count_nonzero(self.safe) > max_nodes:
            raise PlanningCapacity("FALCON passable-grid node cap exceeded")
        self.ids, graph = passable_graph(self.cost)
        self.y, self.x = np.nonzero(self.safe)
        graph = graph.tocoo()
        ay, ax = self.y[graph.row], self.x[graph.row]
        by, bx = self.y[graph.col], self.x[graph.col]
        valid = self.safe[ay, bx] & self.safe[by, ax]
        graph.data *= (self.cost[ay, ax] + self.cost[by, bx]) * 0.5
        graph.data[~valid] = 0
        graph.eliminate_zeros()
        self.graph = graph.tocsr()
        self.cache = {}
        self.paths = {}

    def contains(self, cell):
        x, y = cell
        return 0 <= y < self.ids.shape[0] and 0 <= x < self.ids.shape[1] and self.ids[y, x] >= 0

    def distances(self, cell, deadline):
        deadline.check()
        if not self.contains(cell):
            return None
        key = int(self.ids[cell[1], cell[0]])
        if key not in self.cache:
            self.cache[key] = dijkstra(self.graph, directed=False, indices=key, return_predecessors=True)
        deadline.check()
        return self.cache[key]

    def path(self, a, b, deadline):
        deadline.check()
        key = (tuple(a), tuple(b))
        if key in self.paths:
            return self.paths[key]
        if not self.contains(b):
            return ()
        data = self.distances(a, deadline)
        if data is None:
            return ()
        target = int(self.ids[b[1], b[0]])
        distances, previous = data
        if not np.isfinite(distances[target]):
            return ()
        source = int(self.ids[a[1], a[0]])
        reverse = [b]
        while target != source:
            target = int(previous[target])
            if target < 0:
                return ()
            reverse.append((int(self.x[target]), int(self.y[target])))
        path = self.shorten(tuple(reversed(reverse)), deadline)
        self.paths[key] = path
        self.paths[(tuple(b), tuple(a))] = tuple(reversed(path))
        return path

    def shorten(self, cells, deadline):
        if len(cells) < 3:
            return tuple(cells)
        kept, i = [cells[0]], 0
        while i < len(cells) - 1:
            deadline.check()
            j = len(cells) - 1
            while j > i + 1 and not segment_clear(self.safe, cells[i], cells[j]):
                j -= 1
            kept.append(cells[j])
            i = j
        return tuple(kept)

    def action_cost(self, a, b, yaw_a, yaw_b, spec, deadline):
        """Serial translation + all path heading changes + final facing.

        UAV max(t_position,t_yaw), acceleration and vertical terms are not
        valid here. This estimates the unchanged discrete converter's actions;
        actual failed moves, turns and recovery are charged by the supervisor.
        """
        path = self.path(a, b, deadline)
        if not path:
            return math.inf
        travel, turns, heading = 0.0, 0.0, yaw_a
        for left, right in zip(path, path[1:]):
            dx, dy = right[0] - left[0], right[1] - left[1]
            if not (dx or dy):
                continue
            nxt = math.atan2(dy, dx)
            turns += abs(normalize_angle(nxt - heading)) / spec.turn_angle_rad
            length = math.hypot(dx, dy) * self.resolution
            samples = line_cells(*left, *right)
            penalty = float(np.mean([self.cost[y, x] for x, y in samples]))
            travel += length * penalty / spec.forward_step_m
            heading = nxt
        if yaw_b is not None:
            turns += abs(normalize_angle(yaw_b - heading)) / spec.turn_angle_rad
        return travel + turns
