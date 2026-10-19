"""FALCON IV-A/B and V-A: connectivity-aware cells, zones, portal graph.

Only the caller's observed, bounded scope is decomposed. Unknown zones represent
uncertainty for CP guidance, never permission to navigate there. Tile and edge
caches are invalidated by actual geometry, not volatile room IDs.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
import numpy as np
from scipy.ndimage import label
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components, dijkstra

from sparx_agency.core.planning.exploration.falcon.travel import GridRoutes
from sparx_agency.core.planning.exploration.falcon.ordering import PlanningCapacity


@dataclass
class Zone:
    key: tuple
    cell: tuple
    unknown: bool
    pixels: np.ndarray


class Connectivity:
    def __init__(self, params):
        self.params = params
        self.tiles, self.edges = {}, {}
        self.updated_tiles = 0

    def update(self, world, safe, scope, deadline):
        self.world = world
        self.safe, self.scope = safe, scope
        self.size = max(1, int(round(self.params.cell_size_m / world.resolution)))
        ys, xs = np.nonzero(scope)
        self.zones = []
        if not len(xs):
            return
        live = set()
        for ty in range(int(ys.min()) // self.size, int(ys.max()) // self.size + 1):
            for tx in range(int(xs.min()) // self.size, int(xs.max()) // self.size + 1):
                deadline.check()
                tile = (tx, ty)
                live.add(tile)
                self.zones.extend(self._tile(tile, deadline))
        self.tiles = {k: v for k, v in self.tiles.items() if k in live}
        self.lookup = np.full(scope.shape, -1, dtype=np.int32)
        for i, zone in enumerate(self.zones):
            self.lookup[zone.pixels[:, 1], zone.pixels[:, 0]] = i
        self._graph(deadline)

    def _tile(self, tile, deadline):
        tx, ty = tile
        x, y = tx * self.size, ty * self.size
        area = np.s_[y:y + self.size, x:x + self.size]
        known = self.safe[area] & self.scope[area]
        unknown = (self.world.grid[area] == self.world.values.unknown) & self.scope[area]
        signature = known.tobytes() + unknown.tobytes()
        cached = self.tiles.get(tile)
        if cached is not None and cached[0] == signature:
            return cached[1]
        self.updated_tiles += 1
        zones = []
        for state, mask in enumerate((known, unknown)):
            labels, count = label(mask)  # cardinal CCL, no corner connectivity
            for k in range(1, count + 1):
                deadline.check()
                yy, xx = np.nonzero(labels == k)
                pixels = np.column_stack((xx + x, yy + y))
                center = pixels.mean(axis=0)
                nearest = int(np.argmin(((pixels - center) ** 2).sum(axis=1)))
                zones.append(Zone((tx, ty, state, k), tuple(map(int, pixels[nearest])), bool(state), pixels))
        self.tiles[tile] = (signature, zones)
        return zones

    def _graph(self, deadline):
        n = len(self.zones)
        if n > 4 * self.params.max_order_nodes:
            raise PlanningCapacity("FALCON decomposition zone cap exceeded")
        matrix = np.full((n, n), np.inf)
        np.fill_diagonal(matrix, 0.0)
        live_edges = {}
        for i, a in enumerate(self.zones):
            for j in range(i + 1, n):
                b = self.zones[j]
                adjacent = abs(a.key[0] - b.key[0]) + abs(a.key[1] - b.key[1])
                same_type = a.unknown == b.unknown
                if not ((adjacent == 0 and not same_type) or (adjacent == 1 and same_type)):
                    continue
                deadline.check()
                signature = (a.key, b.key, self.tiles[a.key[:2]][0], self.tiles[b.key[:2]][0])
                if signature in self.edges:
                    value = self.edges[signature]
                else:
                    value = self._restricted_cost(a, b, deadline)
                live_edges[signature] = value
                matrix[i, j] = matrix[j, i] = value
        self.edges = live_edges
        finite = np.isfinite(matrix) & ~np.eye(n, dtype=bool)
        graph = csr_matrix(np.where(finite, np.maximum(matrix, 1e-9), 0.0))
        _, components = connected_components(graph, directed=False)
        free_components = {components[i] for i, z in enumerate(self.zones) if not z.unknown}
        self.active = {i for i in range(n) if components[i] in free_components}
        self.isolated_unknown = n - len(self.active)
        self.graph = graph
        self.graph_distances = dijkstra(graph, directed=False)
        deadline.check()

    def _restricted_cost(self, a, b, deadline):
        x0, y0 = min(a.key[0], b.key[0]) * self.size, min(a.key[1], b.key[1]) * self.size
        x1, y1 = (max(a.key[0], b.key[0]) + 1) * self.size, (max(a.key[1], b.key[1]) + 1) * self.size
        area = np.s_[y0:y1, x0:x1]
        unknown = (self.world.grid[area] == self.world.values.unknown) & self.scope[area]
        free = self.safe[area] & self.scope[area]
        mask = (unknown if a.unknown else free) if a.unknown == b.unknown else (free | unknown)
        weights = np.where(mask, np.where(unknown, self.params.unknown_penalty, 1.0), np.inf)
        routes = GridRoutes(weights, self.world.resolution, self.params.max_grid_nodes)
        start, goal = (a.cell[0] - x0, a.cell[1] - y0), (b.cell[0] - x0, b.cell[1] - y0)
        data = routes.distances(start, deadline)
        if data is None or not routes.contains(goal):
            return math.inf
        return float(data[0][routes.ids[goal[1], goal[0]]]) * self.world.resolution

    def zone_at(self, cell):
        x, y = cell
        return int(self.lookup[y, x]) if self.world.in_bounds(x, y) else -1

    def guidance_cost(self, a, b, yaw_a, yaw_b, spec, routes, deadline):
        """Near grid search; far connectivity-graph search, with endpoint offsets."""
        if math.dist(a, b) * self.world.resolution < self.params.hybrid_distance_m:
            return routes.action_cost(a, b, yaw_a, yaw_b, spec, deadline)
        ia, ib = self.zone_at(a), self.zone_at(b)
        if ia < 0 or ib < 0:
            return math.inf
        distance = self.graph_distances[ia, ib]
        if not math.isfinite(distance):
            return math.inf
        # Resolve endpoint offsets inside the correct connected zones, not
        # across a wall to a Euclidean-nearest centre.
        za, zb = self.zones[ia], self.zones[ib]
        offset = routes.action_cost(a, za.cell, yaw_a, None, spec, deadline)
        offset += routes.action_cost(zb.cell, b, 0.0, yaw_b, spec, deadline)
        return float(distance) / spec.forward_step_m + offset
