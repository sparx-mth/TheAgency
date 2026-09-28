"""The explicit decision to change floors: which staircase, up or down, and why.

Called by the building coordinator when the floor in force has earned a
change -- its frontier is exhausted, or its search allowance is spent -- and
no connector is committed yet. Deliberately not a solver: with one to three
staircases per house the ranking is a sentence, and a sentence can be read
back from the recording. The rule, in order:

1. **Eligible**: a ground-truth connector touching the storey the agent is
   on, and not on cooldown (recently deferred, or the one it just came down).
2. **Reachable now**: its entry anchor snaps onto the floor's observed
   passable map within :attr:`DecisionSettings.approach_snap_m`, with a
   finite geodesic distance from the agent. An anchor in still-unseen space
   is re-checked after a few actions rather than deferred for a hundred.
3. **Worth it**: the storey at the other end is unvisited, or it is a
   visited storey whose saved search still has frontier left. A searched-out
   storey is not re-entered by the stairs.
4. **Order**: unvisited storeys first, then the nearest entry. Up or down
   falls out of the connector chosen; nothing prefers a direction by itself.

Every call leaves a record naming the winner, its direction and the reason
each other candidate lost.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.sparse.csgraph import dijkstra

from sparx_agency.core.planning.exploration.room_costs import passable_graph, snap_cell


@dataclass(frozen=True)
class DecisionSettings:
    """Bounds of the floor decision. Distances in metres, counts in actions.

    Attributes:
        approach_snap_m: How far an entry anchor may be moved onto the
            observed passable map to become an approach goal.
        recheck_actions: Cooldown for a connector whose anchor is not on the
            observed map yet -- the map grows, so ask again soon.
        agent_snap_m: How far the agent's own cell may be moved onto the
            passable graph before it counts as off-map.
    """

    approach_snap_m: float = 1.5
    recheck_actions: int = 10
    agent_snap_m: float = 1.0


@dataclass(frozen=True)
class FloorChoice:
    """The connector to take, and what the approach aims at.

    Attributes:
        portal: The coordinator's portal record for the connector.
        approach_xy: Known-passable point the A* approach targets -- the
            entry anchor itself when it is already on the map.
        direction: ``+1`` up, ``-1`` down.
        destination_visited: Whether the atlas already has the other storey.
        distance_m: Geodesic distance to the approach point along the map.
    """

    portal: Dict
    approach_xy: Tuple[float, float]
    direction: int
    destination_visited: bool
    distance_m: float


def floor_at(atlas, height: float, tolerance_m: float) -> Optional[int]:
    """The atlas floor id whose elevation matches ``height``, or None if unvisited."""
    matches = [f for f in atlas.floors.values() if abs(f.elevation_m - height) <= tolerance_m]
    return min(matches, key=lambda f: abs(f.elevation_m - height)).id if matches else None


def floor_worth_visiting(context: Optional[Dict]) -> bool:
    """A saved floor context still has something to search in it.

    Never searched (no context, or no time spent) counts as worth it; so does
    any room the scene graph last saw with a frontier cluster left.
    """
    if context is None or float(context.get("_floor_time", 0.0)) <= 0.0:
        return True
    facts = getattr(context.get("graph"), "facts", {}) or {}
    return any(getattr(fact, "frontier_clusters", 0) > 0 for fact in facts.values())


def decide_floor_change(portals: Sequence[Dict], obs, world, cost: np.ndarray, atlas, contexts: Dict,
                        step: int, floor_match_m: float, deferral_actions: int,
                        settings: Optional[DecisionSettings] = None) -> Tuple[Optional[FloorChoice], Dict]:
    """Pick the staircase to take now, or None, and say why.

    Args:
        portals: The coordinator's portal records for the floor in force,
            each with ``entry`` (xyz), ``direction``, ``destination_z``,
            ``cooldown_until`` and ``id``.
        obs: The current observation (pose, step).
        world: The floor's observed occupancy grid.
        cost: The planner's cost array for ``world``, ``inf`` where blocked.
        atlas: The floor atlas (visited storeys by elevation).
        contexts: Saved per-floor policy state, keyed by atlas floor id.
        step: The current action index.
        floor_match_m: Elevation tolerance for matching a storey to a floor.
        deferral_actions: Cooldown for a candidate rejected as not worth it.
        settings: Tuning; defaults to :class:`DecisionSettings`.

    Returns:
        ``(choice, record)``. The record is JSON-serialisable and lists every
        candidate with its verdict; it is appended to the coordinator's events.
    """
    settings = settings or DecisionSettings()
    record: Dict = {"event": "floor_decision", "action": int(step), "candidates": [], "chosen": None}
    eligible = [p for p in portals if step >= p.get("cooldown_until", 0)]
    if not eligible:
        record["reason"] = "no eligible connector on this floor"
        return None, record
    ids, graph = passable_graph(cost)
    ax, ay = world.world_to_grid(obs.pose.x, obs.pose.y)
    source = snap_cell(ids, ax, ay, max(1, int(round(settings.agent_snap_m / world.resolution))))
    if source is None:
        record["reason"] = "agent off the observed passable map"
        return None, record
    distances = dijkstra(graph, directed=False, indices=int(ids[source[1], source[0]]))
    radius = max(1, int(round(settings.approach_snap_m / world.resolution)))
    choices: List[FloorChoice] = []
    for portal in eligible:
        entry = portal["entry"]
        verdict = {"portal_id": portal["id"], "direction": "up" if portal["direction"] > 0 else "down"}
        gx, gy = world.world_to_grid(float(entry[0]), float(entry[1]))
        cell = snap_cell(ids, gx, gy, radius)
        distance = math.inf if cell is None else float(distances[int(ids[cell[1], cell[0]])]) * world.resolution
        if cell is None or not math.isfinite(distance):
            portal["cooldown_until"] = step + settings.recheck_actions
            verdict["verdict"] = "entry not on the observed map yet; re-check in %d actions" % settings.recheck_actions
            record["candidates"].append(verdict)
            continue
        destination = floor_at(atlas, float(portal["destination_z"]), floor_match_m)
        visited = destination is not None
        if visited and not floor_worth_visiting(contexts.get(destination)):
            portal["cooldown_until"] = step + deferral_actions
            verdict["verdict"] = "destination floor %d already searched out" % destination
            record["candidates"].append(verdict)
            continue
        approach = tuple(float(v) for v in world.grid_to_world(cell[0], cell[1]))
        verdict.update(verdict="candidate", destination_visited=visited, distance_m=round(distance, 2))
        record["candidates"].append(verdict)
        choices.append(FloorChoice(portal, approach, int(portal["direction"]), visited, distance))
    if not choices:
        record["reason"] = "no reachable connector leads to a storey worth visiting"
        return None, record
    choices.sort(key=lambda c: (c.destination_visited, c.distance_m, c.portal["id"]))
    chosen = choices[0]
    record["chosen"] = chosen.portal["id"]
    record["direction"] = "up" if chosen.direction > 0 else "down"
    record["reason"] = "%s storey %s, nearest reachable entry %.1f m away" % (
        "unvisited" if not chosen.destination_visited else "visited-with-frontier",
        record["direction"], chosen.distance_m)
    return chosen, record

