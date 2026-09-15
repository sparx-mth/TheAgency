"""Bounded FALCON 2D/2.5D exploration, not a renamed next-frontier heuristic."""
from __future__ import annotations

from dataclasses import dataclass, field
import math
import time
import numpy as np

from sparx_agency.core.common.types import normalize_angle
from sparx_agency.core.planning.exploration.falcon.connectivity import Connectivity
from sparx_agency.core.planning.exploration.falcon.frontiers import CameraVisibility, FrontierMemory, Viewpoint, sample_viewpoints
from sparx_agency.core.planning.exploration.falcon.ordering import Deadline, PlanningCapacity, PlanningDeadline, refine_layers, solve_order
from sparx_agency.core.planning.exploration.falcon.params import FalconParams, SOURCE
from sparx_agency.core.planning.exploration.falcon.travel import GridRoutes


@dataclass
class FalconPlan:
    status: str
    viewpoints: tuple = ()
    routes: tuple = ()
    diagnostics: dict = field(default_factory=dict)


class FalconPlanner:
    """Persistent frontiers/rejections; a fresh bounded planning transaction.

    Replanning is requested by the orchestrator at viewpoint completion,
    frontier invalidation, scope safety change or blockage, not every RGB frame.
    All stages run before a route can be emitted; a timeout cannot masquerade as
    a successful FALCON plan or silently invoke the frontier baseline.
    """

    def __init__(self, params=None):
        self.params = params or FalconParams()
        self.frontiers = FrontierMemory(self.params)
        self.connectivity = Connectivity(self.params)
        self.visited, self.rejected = [], []
        self.records = []
        self.current = None

    def observe(self, world):
        self.world = world
        # Core map values are configurable; the frontier vocabulary is ternary.
        grid = np.full(world.grid.shape, -1, dtype=np.int8)
        grid[world.grid == world.values.free] = 0
        grid[world.grid == world.values.occupied] = 100
        self.frontiers.update(grid)

    def used(self, cell, yaw):
        xy = self.world.grid_to_world(*cell)
        return any(math.dist(xy, point) < 0.3 and abs(normalize_angle(yaw - angle)) < math.radians(20)
                   for point, angle in self.visited + self.rejected)

    def retire(self, view, rejected=False):
        if view is not None:
            history = self.rejected if rejected else self.visited
            history.append((self.world.grid_to_world(*view.cell), view.yaw))
        self.current = None

    def plan(self, world, cost, scope, observation, body_height):
        started = time.monotonic()
        deadline = Deadline(self.params.planning_deadline_s)
        record = {"backend": SOURCE["implementation"], "step": observation.step,
                  "frontier_revision": self.frontiers.revision, "fallback": None}
        try:
            result = self._plan(world, cost, scope, observation, body_height, deadline, record)
        except PlanningCapacity:
            result = FalconPlan("planning_capacity")
        except PlanningDeadline:
            result = FalconPlan("planning_deadline")
        record.update(status=result.status, planning_ms=(time.monotonic() - started) * 1000)
        result.diagnostics = record
        self.records.append(record)
        return result

    def _plan(self, world, cost, scope, obs, body_height, deadline, record):
        p = self.params
        safe = np.isfinite(cost) & scope & (world.grid == world.values.free)
        start = world.world_to_grid(obs.pose.x, obs.pose.y)
        if not world.in_bounds(*start) or not safe[start[1], start[0]]:
            return FalconPlan("start_not_safe")
        free_routes = GridRoutes(np.where(safe, 1.0, np.inf), world.resolution, p.max_grid_nodes)
        distances = free_routes.distances(start, deadline)[0]
        reachable = safe.copy()
        reachable[safe] = np.isfinite(distances)
        camera = CameraVisibility(world, obs.camera, obs.pose.camera_pitch, body_height, p)
        clusters = self.frontiers.in_scope(scope, world.resolution, deadline)
        record["frontier_clusters"] = len(clusters)
        if not clusters:
            return FalconPlan("frontiers_exhausted")
        ranked = sorted(clusters, key=lambda fid: (np.linalg.norm(clusters[fid].mean(axis=0) - start), fid))
        record["deferred_frontiers"] = max(0, len(ranked) - p.max_frontiers)
        sampled = {fid: clusters[fid] for fid in ranked[:p.max_frontiers]}
        views, dormant = sample_viewpoints(
            sampled, world, reachable, scope, start, camera, p, self.used, deadline,
            start_yaw=obs.pose.yaw, turn_angle=obs.action_spec.turn_angle_rad,
            arrival_m=getattr(obs, "arrival_m", 0.2))
        record.update(dormant_frontiers=dormant, viewpoint_candidates=sum(map(len, views.values())))
        if not views:
            return FalconPlan("planning_capacity" if record["deferred_frontiers"] else "no_useful_reachable_frontiers")
        topology_cost = getattr(obs, "topology_cost", cost)
        topology_free = np.isfinite(topology_cost) & scope & (world.grid == world.values.free)
        self.connectivity.update(world, topology_free, scope, deadline)
        c = self.connectivity
        unknown = (world.grid == world.values.unknown) & scope
        guidance = GridRoutes(np.where(topology_free, 1.0, np.where(unknown, p.unknown_penalty, np.inf)), world.resolution, p.max_grid_nodes)
        representatives = {fid: candidates[0] for fid, candidates in views.items()}
        active = {}
        for fid, view in representatives.items():
            active.setdefault(c.zone_at(view.cell), []).append(fid)
        start_zone = c.zone_at(start)
        cp = [(start_zone, start, obs.pose.yaw, False)]
        for zid in sorted(c.active):
            zone = c.zones[zid]
            if not (zone.unknown or zid in active) or not np.isfinite(c.graph_distances[start_zone, zid]):
                continue
            center = zone.cell
            if zid in active:
                mean = np.mean([representatives[f].cell for f in active[zid]], axis=0)
                center = tuple(map(int, zone.pixels[np.argmin(((zone.pixels - mean) ** 2).sum(axis=1))]))
            cp.append((zid, center, 0.0, zone.unknown))
        record.update(zones=len(c.zones), unknown_zones=sum(item[3] for item in cp),
                      connectivity_edges=int(c.graph.nnz // 2), updated_tiles=c.updated_tiles,
                      isolated_unknown_zones=c.isolated_unknown)
        if len(cp) > p.max_order_nodes:
            return FalconPlan("planning_capacity")
        matrix = self._matrix(cp, guidance, obs, deadline)
        for j, item in enumerate(cp[1:], 1):
            if item[3]:
                matrix[0, j] = math.inf  # source: next zone must be active free
        tour = solve_order(matrix, exact_limit=p.exact_order_limit, beam_width=p.beam_width, deadline=deadline)
        record.update(cp_status=tour.status, cp_order=[cp[i][0] for i in tour.order], cp_cost_actions=round(tour.cost, 3) if tour.order else None)
        if len(tour.order) < 2:
            return FalconPlan("guidance_unreachable")
        chosen_zone = cp[tour.order[1]][0]
        local_ids = sorted(set(active.get(start_zone, []) + active.get(chosen_zone, [])))
        reduced = [cp[i] for i in tour.order if i == 0 or cp[i][0] not in (chosen_zone, start_zone)]
        nodes = reduced + [(c.zone_at(representatives[f].cell), representatives[f].cell,
                            representatives[f].yaw, False) for f in local_ids]
        if len(nodes) > p.max_order_nodes:
            return FalconPlan("planning_capacity")
        sop_cost = self._matrix(nodes, guidance, obs, deadline)
        precedence = [(a, b) for a in range(len(reduced)) for b in range(a + 1, len(reduced))]
        # A physical first action must reach a viewpoint, not an unknown CP
        # centre. Later viewpoints may still be deferred beyond CP elements.
        sop_cost[0, 1:len(reduced)] = math.inf
        sop = solve_order(sop_cost, precedence, exact_limit=p.exact_order_limit, beam_width=p.beam_width, deadline=deadline)
        order = [local_ids[i - len(reduced)] for i in sop.order if i >= len(reduced)]
        record.update(sop_status=sop.status, sop_order=list(sop.order),
                      sop_precedence=[list(pair) for pair in precedence], frontier_order=order)
        if not order:
            return FalconPlan("local_order_unreachable")
        prefix, terminal = [], None
        for index in sop.order[1:]:
            if index < len(reduced):
                terminal = Viewpoint(-2, nodes[index][1], nodes[index][2], 0, 0)
                break
            prefix.append(local_ids[index - len(reduced)])
        record["executed_frontier_prefix"] = prefix
        # Source V-C: optimize alternatives jointly after determining order.
        initial = Viewpoint(-1, start, obs.pose.yaw, 0, 0)
        def edge(a, b):
            routes = guidance if b.frontier == -2 else free_routes
            return routes.action_cost(a.cell, b.cell, a.yaw, b.yaw, obs.action_spec, deadline)
        layers = [[initial]] + [views[f] for f in prefix]
        if terminal is not None:
            layers.append([terminal])
        refined = refine_layers(layers, edge, deadline)
        if terminal is not None:
            refined = refined[:-1]
        if len(refined) < 2:
            return FalconPlan("viewpoints_unreachable")
        routes = tuple(free_routes.path(a.cell, b.cell, deadline) for a, b in zip(refined, refined[1:]))
        record.update(refined_viewpoints=len(refined) - 1, route_segments=len(routes),
                      selected_gain_m2=refined[1].gain * world.resolution ** 2,
                      selected_cell=list(refined[1].cell), selected_yaw=refined[1].yaw)
        return FalconPlan("planned", tuple(refined[1:]), routes)

    def _matrix(self, nodes, routes, obs, deadline):
        matrix = np.zeros((len(nodes), len(nodes)))
        for i, (_, a, ya, _) in enumerate(nodes):
            for j, (_, b, yb, _) in enumerate(nodes):
                if i != j:
                    matrix[i, j] = self.connectivity.guidance_cost(a, b, ya, yb, obs.action_spec, routes, deadline)
        return matrix
