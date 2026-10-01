"""The fallback decision to change floors: which staircase, up or down, and why.

**Not the normal path.** The room-search loop decides floor changes: every
staircase is a node of its RPT* instance (:mod:`stair_nodes`), valued by the
LLM beside the rooms and charged for the climb, and the stairs are taken when
the order says so. This module is what the exploration fallback asks when
that machinery has nothing to offer -- the room LLM is backing off, or no
node on the floor is worth anything -- and the floor is otherwise exhausted.
Deliberately not a solver: with one to three staircases per house the ranking
is a sentence, and a sentence can be read back from the recording. The rule,
in order:

1. **Eligible**: a ground-truth connector touching the storey the agent is
   on, and not on cooldown (recently deferred after a failed approach).
2. **Reachable now**: its entry anchor snaps onto the floor's observed
   passable map within :attr:`DecisionSettings.approach_snap_m`, with a
   finite geodesic distance from the agent. An anchor in still-unseen space
   is re-checked after a few actions rather than deferred for a hundred.
3. **Worth it**: the storey at the other end is unvisited, or it is a
   visited storey whose saved search still has frontier left. A searched-out
   storey is not re-entered by the stairs.
4. **Score**: worth per metre. Worth is 1.0 for an unvisited storey and
   :attr:`DecisionSettings.visited_value` for a visited one with frontier
   left; cost is the walk to the entry plus the stairs' own cost -- the
   flight's length and the fixed cost of a storey change, the same
   :func:`stair_cost_m` the RPT* leaf uses. Highest wins; nothing prefers a
   direction by itself.

Every call leaves a record naming the winner, its direction, its score and
the reason each other candidate lost. :func:`approach_points` -- the
"reachable now" step on its own -- is shared with the node builder, so the
loop and the fallback agree on where a staircase is entered from.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.sparse.csgraph import dijkstra

from sparx_agency.core.planning.exploration.room_costs import passable_graph, snap_cell
from sparx_agency.core.planning.planners.common.grid_geometry_2d import line_cells


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
        visited_value: Worth of a visited storey with frontier left,
            against 1.0 for one never stood on (fallback rule only).
    """

    approach_snap_m: float = 1.5
    recheck_actions: int = 10
    agent_snap_m: float = 1.0
    visited_value: float = 0.5


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
        score: Worth per metre, the number the fallback choice was made on.
    """

    portal: Dict
    approach_xy: Tuple[float, float]
    direction: int
    destination_visited: bool
    distance_m: float
    score: float = 0.0


@dataclass(frozen=True)
class Approach:
    """Where a staircase is entered from on the observed map, and how far that is."""

    xy: Tuple[float, float]
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


def connector_length_m(portal: Dict) -> float:
    """The flight's length along its polyline, or 0 when the portal carries none."""
    path = portal.get("path") or ()
    return float(sum(math.dist(a[:3], b[:3]) for a, b in zip(path, path[1:]))) if len(path) > 1 else 0.0


def stair_cost_m(portal: Dict, floor_change_cost_m: float) -> float:
    """What taking this staircase costs beyond reaching its foot: the flight plus the fixed change cost.

    The one definition, read by the RPT* leaf (every arc into and out of the
    stair node) and by the fallback rule's score, so the two never disagree
    about what a storey change costs.
    """
    return connector_length_m(portal) + max(0.0, float(floor_change_cost_m))


def snap_entry(ids: np.ndarray, occupied: np.ndarray, gx: int, gy: int,
               max_radius: int) -> Optional[Tuple[int, int]]:
    """The nearest passable cell to the entry cell ``(gx, gy)`` that the entry can be walked to from in a straight line.

    :func:`~sparx_agency.core.planning.exploration.room_costs.snap_cell`
    takes the Euclidean-nearest passable cell, and on a stair head that is a
    narrow corridor the nearest cell is as likely to lie in the room BEHIND
    the corridor's wall as in the corridor -- in the first recorded campaign
    it did, by one centimetre, and the traversal's straight lead-in to the
    anchor then ran through the wall. A candidate here must also have a clear
    Bresenham line to the entry through the observed map: unknown cells are
    allowed (the anchor usually lies in space not yet seen), occupied cells
    are not. The window grows as ``snap_cell``'s does; within a window the
    candidates are tried nearest first.

    Args:
        ids: The ``(H, W)`` index image from ``passable_graph``; ``-1`` off-graph.
        occupied: ``(H, W)`` boolean, True where the observed map is occupied.
        gx: Entry cell column.
        gy: Entry cell row.
        max_radius: Largest window half-width to try, in cells.

    Returns:
        ``(gx, gy)`` of a passable cell with a clear line to the entry, or None.
    """
    h, w = ids.shape
    if not (0 <= gx < w and 0 <= gy < h):
        return None
    if ids[gy, gx] >= 0:
        return int(gx), int(gy)
    for r in range(1, int(max_radius) + 1):
        y0, y1 = max(0, gy - r), min(h, gy + r + 1)
        x0, x1 = max(0, gx - r), min(w, gx + r + 1)
        ys, xs = np.nonzero(ids[y0:y1, x0:x1] >= 0)
        if ys.size == 0:
            continue
        cy, cx = ys + y0, xs + x0
        for k in np.argsort((cx - gx) ** 2 + (cy - gy) ** 2, kind="stable"):
            # Every cell on the way, the entry cell itself excepted: a stair head's own cell may
            # be marked occupied by the drop beyond it, and that is where the flight begins.
            if not any(occupied[y, x] for x, y in line_cells(int(cx[k]), int(cy[k]), int(gx), int(gy))[:-1]):
                return int(cx[k]), int(cy[k])
    return None


def approach_points(portals: Sequence[Dict], obs, world, cost: np.ndarray,
                    settings: Optional[DecisionSettings] = None) -> Optional[Dict[int, Approach]]:
    """For every portal whose entry is on the observed passable map: its approach point and geodesic distance.

    One passable graph and one Dijkstra from the agent for all of them.
    A portal whose entry does not snap (:func:`snap_entry`: no passable cell
    with a clear line to the anchor), or has no path from the agent, is
    absent from the result -- the map has not reached it yet. None when the
    agent itself stands off the passable map.
    """
    settings = settings or DecisionSettings()
    ids, graph = passable_graph(cost)
    ax, ay = world.world_to_grid(obs.pose.x, obs.pose.y)
    source = snap_cell(ids, ax, ay, max(1, int(round(settings.agent_snap_m / world.resolution))))
    if source is None:
        return None
    distances = dijkstra(graph, directed=False, indices=int(ids[source[1], source[0]]))
    radius = max(1, int(round(settings.approach_snap_m / world.resolution)))
    occupied = np.asarray(world.grid) == world.values.occupied
    out: Dict[int, Approach] = {}
    for portal in portals:
        entry = portal["entry"]
        gx, gy = world.world_to_grid(float(entry[0]), float(entry[1]))
        cell = snap_entry(ids, occupied, gx, gy, radius)
        if cell is None:
            continue
        distance = float(distances[int(ids[cell[1], cell[0]])]) * world.resolution
        if not math.isfinite(distance):
            continue
        xy = tuple(float(v) for v in world.grid_to_world(cell[0], cell[1]))
        out[int(portal["id"])] = Approach(xy, distance)
    return out


def decide_floor_change(portals: Sequence[Dict], obs, world, cost: np.ndarray, atlas, contexts: Dict,
                        step: int, floor_match_m: float, deferral_actions: int,
                        settings: Optional[DecisionSettings] = None,
                        floor_change_cost_m: float = 8.0) -> Tuple[Optional[FloorChoice], Dict]:
    """Pick the staircase to take now, or None, and say why -- the fallback rule.

    Args:
        portals: The coordinator's portal records for the floor in force,
            each with ``entry`` (xyz), ``direction``, ``destination_z``,
            ``cooldown_until`` and ``id``; ``path`` when known.
        obs: The current observation (pose, step).
        world: The floor's observed occupancy grid.
        cost: The planner's cost array for ``world``, ``inf`` where blocked.
        atlas: The floor atlas (visited storeys by elevation).
        contexts: Saved per-floor policy state, keyed by atlas floor id.
        step: The current action index.
        floor_match_m: Elevation tolerance for matching a storey to a floor.
        deferral_actions: Cooldown for a candidate rejected as not worth it.
        settings: Tuning; defaults to :class:`DecisionSettings`.
        floor_change_cost_m: The fixed cost of a storey change, metres.

    Returns:
        ``(choice, record)``. The record is JSON-serialisable and lists every
        candidate with its verdict; it is appended to the coordinator's events.
    """
    settings = settings or DecisionSettings()
    record: Dict = {"event": "floor_decision", "action": int(step), "candidates": [], "chosen": None,
                    "rule": "fallback"}
    eligible = [p for p in portals if step >= p.get("cooldown_until", 0)]
    if not eligible:
        record["reason"] = "no eligible connector on this floor"
        return None, record
    approaches = approach_points(eligible, obs, world, cost, settings)
    if approaches is None:
        record["reason"] = "agent off the observed passable map"
        return None, record
    choices: List[FloorChoice] = []
    for portal in eligible:
        direction = int(portal["direction"])
        verdict = {"portal_id": portal["id"], "direction": "up" if direction > 0 else "down"}
        approach = approaches.get(int(portal["id"]))
        if approach is None:
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
        worth = settings.visited_value if visited else 1.0
        total_cost = approach.distance_m + stair_cost_m(portal, floor_change_cost_m)
        score = worth / max(1e-6, total_cost)
        verdict.update(verdict="candidate", destination_visited=visited, distance_m=round(approach.distance_m, 2),
                       cost_m=round(total_cost, 2), score=round(score, 4))
        record["candidates"].append(verdict)
        choices.append(FloorChoice(portal, approach.xy, direction, visited, approach.distance_m, score))
    if not choices:
        record["reason"] = "no reachable connector leads to a storey worth visiting"
        return None, record
    choices.sort(key=lambda c: (-c.score, c.destination_visited, c.distance_m, c.portal["id"]))
    chosen = choices[0]
    record["chosen"] = chosen.portal["id"]
    record["direction"] = "up" if chosen.direction > 0 else "down"
    record["score"] = round(chosen.score, 4)
    record["reason"] = "%s storey %s, entry %.1f m away, worth %.4f per metre" % (
        "unvisited" if not chosen.destination_visited else "visited-with-frontier",
        record["direction"], chosen.distance_m, chosen.score)
    return chosen, record

