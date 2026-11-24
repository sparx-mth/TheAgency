"""Live frontier valuation and routing. Scores live for an episode; routes live only while their node does."""
from __future__ import annotations

from dataclasses import asdict, replace
import math
import time

from sparx_agency.core.planning.exploration.room_costs import build_instance
from sparx_agency.core.planning.exploration.frontier_ranking import accessible_frontiers
from sparx_agency.core.planning.exploration.room_search_policy import RoomCandidate
from sparx_agency.core.planning.object_search.frontier_views.nodes import NodeSettings, frontier_nodes
from sparx_agency.core.planning.object_search.frontier_views.score_ledger import BLOCKED, SCORED, ScoreLedger
from sparx_agency.core.planning.object_search.frontier_views.scoring import ask_frontier, marked_view
from sparx_agency.core.planning.object_search.frontier_views.selection import select_views
from sparx_agency.core.planning.objnav.errors import ObjNavInternalError

VLM = "frontier_vlm"


class FrontierSearch:
    """Observe and score even when another behavior owns motion. Only ``plan`` may issue a command."""

    def __init__(self, policy):
        self.policy = policy
        self.registry = NodeSettings().registry()
        self.ledger = ScoreLedger()
        self.scores = {}
        self.nodes = {}
        self.blocked_cells = set()
        self.current = None
        self.order = []
        self.calls = 0
        self.inventory = None
        self.inventory_step = -1

    def refresh(self, obs, world, *, on_floor):
        """One extraction from observed occupancy. No room labels or room membership."""
        self.inventory, self.inventory_step = None, int(obs.step)
        if on_floor:
            p = self.policy
            self.inventory = accessible_frontiers(
                world, p.navigation_cost(world), None, (obs.pose.x, obs.pose.y), obs.pose.yaw,
                p.sweep.settings.ranking, preferred_cost=p.preferred_cost(world), frontier_world=p.sight.overlay(world))
        self.observe(obs, world, self.inventory.clusters if self.inventory is not None else ())

    def observe(self, obs, world, clusters):
        p = self.policy
        nodes = frontier_nodes(world, clusters, self.registry, p.mapping.floor_id)
        self.nodes = {n.node_id: n for n in nodes}
        self.ledger.observe(obs.step, self.nodes)
        self.blocked_cells = {cell for n in nodes if self.ledger.nodes[n.node_id].state == BLOCKED for cell in n.goal_cells}
        arrival = p.converter_params.goal_tolerance_m + world.resolution
        rotating = p.warmup_actions < p.warmup_turns
        spin_turns = 0 if rotating else p.settings.arrival_spin_turns
        for node in nodes:
            if math.dist((obs.pose.x, obs.pose.y), node.xy) <= arrival:
                if self.ledger.nodes[node.node_id].visited_step is None:
                    p._visited_frontiers.append(node.xy)  # the geometric fallback shares the visit
                    if spin_turns:
                        p.start_rotation(spin_turns)
                        spin_turns = 0
                self.ledger.mark_visited(node.node_id, obs.step)

        pending = [self.nodes[i] for i in self.ledger.pending()]
        if not pending or p.fallback.service_unavailable(VLM, obs.step):
            return
        views = select_views(world, pending, p.frame_log.records, obs.camera,
                             settled=p.sight.resolved(world), reference_z=obs.pose.z)
        for i in self.ledger.due(views):
            chosen = views[i]
            images = [marked_view(p.frame_log.rgb(view.name), view.window) for view in chosen.views]
            started = time.monotonic()
            self.calls += 1
            try:
                probability = ask_frontier(p.vlm_client, p.target.query, images, depth_limit_m=obs.camera.max_depth_m)
            except ObjNavInternalError:
                raise  # model identity drift or a broken frame log invalidates the run
            except (RuntimeError, OSError, ValueError, KeyError) as exc:
                p.fallback.note_service_failure(obs, VLM, exc)
                break  # leave this and subsequent nodes pending until the service recovers
            finally:
                p.telemetry.latencies[VLM].append((time.monotonic() - started) * 1000)
            p.fallback.note_service_success(VLM)
            self.ledger.mark_scored(i, obs.step, chosen.names)
            self.scores[i] = probability

    def _order(self, obs, world, candidates):
        p = self.policy
        goals = {i: node.xy for i, node in candidates.items()}
        instance, dropped = build_instance(
            world, p.navigation_cost(world), goals, {i: self.scores[i] for i in goals},
            depot_xy=(obs.pose.x, obs.pose.y), preferred_cost=p.preferred_cost(world),
            cruise_speed_mps=p.episode.action_spec.forward_step_m / p.settings.action_time_s)
        if instance.index_to_pid[instance.depot] != -1:
            return []  # no reachable start; do not substitute another node for the agent
        eligible = [RoomCandidate(i, "frontier", self.scores[i], self.scores[i], xy)
                    for i, xy in goals.items() if i not in dropped]
        if not eligible:
            return []
        order = p.solver(eligible, instance)
        record = asdict(p.solver.last)
        p._solver_records.append({k: None if isinstance(v, float) and not math.isfinite(v) else v
                                  for k, v in record.items()})
        self.order = list(order)
        return order

    def plan(self, obs, world):
        """Keep a safe route or solve from this pose over current, scored, unvisited nodes only."""
        p = self.policy
        candidates = {n.node_id: n for n in p.sweep.admissible(obs, world, self.nodes.values())
                      if self.ledger.nodes[n.node_id].state == SCORED
                      and self.ledger.nodes[n.node_id].visited_step is None}
        if self.current is not None and self.current not in candidates:
            if p.route_memory.kind == "frontier/%d" % self.current:
                p.route_memory.clear("frontier_visited_or_gone")
                p._route = p._goal = None
            self.current = None

        if self.current is not None:
            command = self._drive(obs, world, candidates.pop(self.current))
            if command is not None:
                return command
            self.current = None
        if candidates:
            for i in self._order(obs, world, candidates):
                command = self._drive(obs, world, candidates[i])
                if command is not None:
                    return command
        self.order = []
        return None

    def _drive(self, obs, world, node):
        p, i = self.policy, node.node_id
        kind = "frontier/%d" % i
        if p.route_memory.kind == kind and p.route_memory.arrived(obs):
            # Still a candidate: its goal moved as the map grew. Do not idle at the old endpoint.
            p.route_memory.clear("frontier_goal_shifted")
        command = p._navigate(obs, world, node.xy, kind, exact_goal=True)
        if command is None:
            return None
        self.current = i
        return replace(command, info=dict(command.info, frontier_id=i, probability=self.scores[i]))

    def diagnostics(self):
        """No pixels in the recording: the exact input frames are named in the ledger."""
        return {"calls": self.calls, "counts": self.ledger.counts(), "current": self.current,
                "nodes": {i: dict(asdict(r), probability=self.scores.get(i)) for i, r in self.ledger.nodes.items()}}
