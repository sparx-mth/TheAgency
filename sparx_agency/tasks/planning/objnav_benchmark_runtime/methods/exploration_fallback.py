"""Where the search goes when its plan fails: the nearest reachable frontier, then further.

Every failure of the decision pipeline -- an A* that finds no path, an RPT*
instance that cannot be built, a room LLM that times out or answers badly,
or a bug in the loop -- used to end in one of two ways: an exception that
ended the episode as an agent error, or an empty hold that the headless
agent spends as an idle turn, on every action, until the step budget ran
out. The recording of the second looks like a camera spinning in place.

This module is the one answer to all of them. Called by the room-search
loop when no room is in force and by the policy when the decision itself
raised, it produces a command that MOVES, in this order:

1. the nearest reachable frontier anywhere on the floor -- not confined to
   the room in force, so the search leaves a room whose routes all failed;
2. the building coordinator, when the floor is exhausted: the ground-truth
   stairs by the explicit fallback rule (up or down, ``floor_decision``).
   The NORMAL way to the stairs is not this: the room-search loop offers
   every staircase to the oracle and to RPT* as a node beside the rooms,
   and climbs when the order says so. This rung is for the floor where that
   machinery has nothing left -- the room LLM is away, or no node is worth
   anything and the frontier is gone;
3. frontiers already retired -- a goal retired for a transient plan failure
   is still unknown space;
4. a relocation: the farthest reachable known cell, for a new vantage point
   from which the map may show a frontier it does not show from here;
5. only when the agent stands off the passable map altogether, one hold --
   the idle turn is then the only action that changes anything.

A failure is never silent: each one is counted, its type, message and origin
recorded in the episode diagnostics, and logged once per type per episode.
A failed model service (the room LLM, the detector) also arms a doubling
back-off during which it is not asked and this module carries the search, so
a dead service costs a few bounded attempts rather than a timeout per action.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import logging
import math
import traceback

import numpy as np
from scipy.sparse.csgraph import dijkstra

from sparx_agency.core.planning.exploration.frontier_ranking import ranked_frontier_goals
from sparx_agency.core.planning.exploration.room_costs import passable_graph, snap_cell
from sparx_agency.core.planning.objnav.types.command import NavigationCommand
from sparx_agency.core.planning.planners.astar.cost_grid_2d import assemble_cost_grid

LOG = logging.getLogger(__name__)

#: The model services whose failures arm a back-off.
ROOM_LLM = "room_llm"
DETECTOR = "detector"


@dataclass(frozen=True)
class FallbackSettings:
    """Bounds of the fallback. Distances in metres, counts in actions.

    Attributes:
        frontier_attempts: Ranked floor-wide goals the planner is asked about
            per action, for the admissible pass and again for the retired one.
        relocation_min_m: A relocation target must be at least this far along
            the floor, or it is not a new vantage point.
        relocation_candidates: How many of the farthest cells are offered to
            the planner before relocation gives up for this action.
        service_backoff_actions: Actions a failed model service (room LLM,
            detector) is left alone after its first failure; doubles per
            consecutive failure.
        service_backoff_max_actions: Cap on that wait.
    """

    frontier_attempts: int = 6
    relocation_min_m: float = 1.5
    relocation_candidates: int = 5
    service_backoff_actions: int = 25
    service_backoff_max_actions: int = 200

    def __post_init__(self):
        for name in ("frontier_attempts", "relocation_candidates", "service_backoff_actions",
                     "service_backoff_max_actions"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError("%s must be a positive integer" % name)
        if isinstance(self.relocation_min_m, bool) or not math.isfinite(self.relocation_min_m) or self.relocation_min_m <= 0:
            raise ValueError("relocation_min_m must be positive and finite")
        if self.service_backoff_max_actions < self.service_backoff_actions:
            raise ValueError("service_backoff_max_actions cannot be shorter than the first back-off")


class ExplorationFallback:
    """Keeps the agent exploring when the decision pipeline cannot."""

    def __init__(self, policy, settings=None):
        self.policy = policy
        self.settings = settings or FallbackSettings()
        self.failures = []
        self.stats = {"invocations": 0, "frontier": 0, "stairs": 0, "frontier_retired": 0,
                      "relocation": 0, "hold": 0, "failures": 0,
                      ROOM_LLM + "_failures": 0, DETECTOR + "_failures": 0}
        self._logged = set()
        self._consecutive = {}
        self.retry_step = {}
        self._last_relocation = None

    # -- failures ------------------------------------------------------------
    def record_failure(self, obs, kind, exc):
        """Count a pipeline failure and keep enough of it to debug from the recording."""
        frames = traceback.extract_tb(exc.__traceback__)
        origin = "%s:%s" % (frames[-1].filename.rsplit("/", 1)[-1], frames[-1].lineno) if frames else "?"
        entry = {"step": int(obs.step), "kind": kind, "error": "%s: %s" % (type(exc).__name__, exc), "origin": origin}
        self.failures.append(entry)
        self.stats["failures"] += 1
        key = (kind, type(exc).__name__)
        if key not in self._logged:
            self._logged.add(key)
            LOG.warning("ObjectNav %s failure at action %d (%s); exploring the nearest frontier instead: %s",
                        kind, obs.step, origin, entry["error"])

    def note_service_failure(self, obs, service, exc):
        """A model service failed: record it and arm the doubling back-off. Returns the retry action."""
        self.record_failure(obs, service, exc)
        self.stats[service + "_failures"] = self.stats.get(service + "_failures", 0) + 1
        self._consecutive[service] = self._consecutive.get(service, 0) + 1
        wait = min(self.settings.service_backoff_actions * 2 ** (self._consecutive[service] - 1),
                   self.settings.service_backoff_max_actions)
        self.retry_step[service] = int(obs.step) + wait
        return self.retry_step[service]

    def note_service_success(self, service):
        self._consecutive.pop(service, None)
        self.retry_step.pop(service, None)

    def service_unavailable(self, service, step):
        """Is the service inside its back-off window?"""
        retry = self.retry_step.get(service)
        return retry is not None and step < retry

    # -- the command -----------------------------------------------------------
    def plan(self, obs, world, cost=None, reason="no room in force"):
        """A command that moves the agent, by the order in the module docstring."""
        p = self.policy
        self.stats["invocations"] += 1
        if cost is None:
            cost = assemble_cost_grid(p.planner.fields_for(world), p.planner_params, p.settings.body_radius_m)[0]
        ids, graph = passable_graph(cost)                    # one graph for the ranking and the relocation
        inventory = p.graph.frontier_inventory
        ranked = (list(inventory.goals) if inventory is not None else
                  ranked_frontier_goals(world, cost, np.ones(world.grid.shape, bool),
                                        (obs.pose.x, obs.pose.y), obs.pose.yaw, p.sweep.settings.ranking,
                                        ids=ids, graph=graph))
        admissible = p.sweep.admissible(obs, world, ranked)
        command = self._try_goals(obs, world, [g.xy for g in admissible])
        if command is not None:
            return self._tag(command, "frontier", reason)
        if p.building is not None:
            command = p.building.plan(obs, world, exhausted=True)
            if command is not None:
                return self._tag(command, "stairs", reason)
        retired = [g.xy for g in ranked if g not in admissible]
        command = self._try_goals(obs, world, retired)
        if command is not None:
            return self._tag(command, "frontier_retired", reason)
        command = self._relocate(obs, world, ids, graph)
        if command is not None:
            return self._tag(command, "relocation", reason)
        self.stats["hold"] += 1
        return NavigationCommand.hold(info={"kind": "fallback_hold", "fallback": reason,
                                            "reason": "off the observed passable map; the idle turn is the only move"})

    def _try_goals(self, obs, world, goals):
        p = self.policy
        for goal in goals[:self.settings.frontier_attempts]:
            command = p._navigate(obs, world, tuple(goal), "frontier")
            if command is not None:
                return command
        return None

    def _relocate(self, obs, world, ids, graph):
        """Walk to the farthest reachable known cell: a new vantage point, never the last one."""
        gx, gy = world.world_to_grid(obs.pose.x, obs.pose.y)
        snap = max(1, int(round(self.policy.sweep.settings.ranking.snap_radius_m / world.resolution)))
        source = snap_cell(ids, gx, gy, snap)
        if source is None or ids.max() < 0:
            return None
        distances = dijkstra(graph, directed=False, indices=int(ids[source[1], source[0]]))
        reachable = np.isfinite(distances) & (distances * world.resolution >= self.settings.relocation_min_m)
        if not reachable.any():
            return None
        order = np.argsort(-np.where(reachable, distances, -np.inf), kind="stable")
        ys, xs = np.nonzero(ids >= 0)
        node_cells = np.empty((ids.max() + 1, 2), dtype=np.int64)
        node_cells[ids[ys, xs]] = np.stack([xs, ys], axis=1)
        tried = 0
        for node in order:
            if not reachable[node]:
                break
            x, y = (float(v) for v in world.grid_to_world(int(node_cells[node][0]), int(node_cells[node][1])))
            if self._last_relocation is not None and math.dist((x, y), self._last_relocation) < 1.0:
                continue
            command = self.policy._navigate(obs, world, (x, y), "relocate")
            tried += 1
            if command is not None:
                self._last_relocation = (x, y)
                return command
            if tried >= self.settings.relocation_candidates:
                break
        return None

    def _tag(self, command, stage, reason):
        self.stats[stage] += 1
        return replace(command, info=dict(command.info, fallback=reason, fallback_stage=stage))

    def diagnostics(self):
        return {"settings": asdict(self.settings), "stats": dict(self.stats),
                "failures": list(self.failures), "retry_step": dict(self.retry_step)}



