"""Staircases as nodes of the room-search loop: ids, the prompt facts, and what a climb costs.

The loop's RPT* instance holds every place the search could go to next. A
room is a node; so is a staircase, because "search the other storey" is a
choice with a probability and a cost like any other, and the only way the
order can weigh "the unknown room down the hall" against "the bedrooms are
probably upstairs" is to see both on one list. This module makes a portal of
the building coordinator into such a node:

* an **id** far above every room pid (:func:`stair_node_id`), so the
  supervisor, the solver and the recording can carry rooms and stairs in one
  id space without a second code path;
* the **facts** the LLM values it by (:class:`SearchNode` of kind
  ``stairs``): direction, whether the other storey was visited and what was
  found and searched there, whether the robot arrived by these stairs;
* the **cost**: the node sits at the foot of the flight on this floor's map
  (its approach point) at the end of a leaf as long as the flight plus the
  fixed cost of a storey change (:func:`~floor_decision.stair_cost_m`),
  charged on every arc into AND out of it -- going upstairs puts every room
  down here that much further away, which is exactly what RPT* must see.

Nothing here decides anything; the decision is the order the solver returns.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

from sparx_agency.core.mapping.topology.search_node_oracle import STAIRS, SearchContext, SearchNode
from sparx_agency.core.planning.exploration.room_search_policy import RoomOption
from sparx_agency.tasks.planning.objnav_benchmark_runtime.methods.floor_decision import (
    Approach, approach_points, floor_at, stair_cost_m)

#: Room pids count up from zero; stair node ids start here. A registry would
#: need a hundred thousand rooms before the two could meet.
STAIR_NODE_BASE = 100_000


def stair_node_id(portal_id: int) -> int:
    """The loop's node id for the coordinator's portal ``portal_id``."""
    return STAIR_NODE_BASE + int(portal_id)


def is_stair_node(node_id) -> bool:
    """Whether a supervisor room id names a staircase rather than a room."""
    return node_id is not None and int(node_id) >= STAIR_NODE_BASE


def portal_id_of(node_id: int) -> int:
    """The coordinator's portal id behind a stair node id."""
    return int(node_id) - STAIR_NODE_BASE


@dataclass(frozen=True)
class StairOption:
    """One staircase as the loop offers it to the supervisor and the solver.

    Attributes:
        node_id: :func:`stair_node_id` of the portal.
        portal: The coordinator's portal record.
        approach: Where it is entered from on this floor's map, and how far.
        leaf_m: The climb's cost beyond the approach point -- the flight and
            the fixed cost of a storey change -- charged on every arc
            touching the node.
        node: The facts the LLM values it by.
    """

    node_id: int
    portal: Dict
    approach: Approach
    leaf_m: float
    node: SearchNode

    def option(self, prob: float) -> RoomOption:
        """The supervisor's view: a room-shaped option at the foot of the stairs."""
        return RoomOption(room_id=self.node_id, prob=float(prob), xy=self.approach.xy, label=self.node.label)


def summarise_floor(context: Optional[Dict], floor_id: Optional[int] = None) -> str:
    """One line on a saved floor context, in the words the prompt uses.

    ``rooms found: kitchen, living_room, 2 unknown; searched 2min; 3 rooms
    with frontier left``. A floor never stood on reads as unvisited.
    """
    if context is None:
        return "unvisited"
    graph = context.get("graph")
    rooms = getattr(getattr(graph, "registry", None), "rooms", {}) or {}
    labels = getattr(getattr(graph, "label_tracker", None), "labels", {}) or {}
    facts = getattr(graph, "facts", {}) or {}
    named = sorted(label.label for pid, label in labels.items() if pid in rooms and label.label != "unknown")
    unknown = sum(1 for pid in rooms if pid not in labels or labels[pid].label == "unknown")
    parts: List[str] = []
    if named:
        parts.append(", ".join(named))
    if unknown:
        parts.append("%d unknown" % unknown)
    found = "rooms found: %s" % ("; ".join(parts) if parts else "none yet")
    searched = float(context.get("_floor_time", 0.0) or 0.0)
    with_frontier = sum(1 for pid, fact in facts.items() if pid in rooms and getattr(fact, "frontier_clusters", 0) > 0)
    prefix = "" if floor_id is None else "storey F%d: " % floor_id
    return "%s%s; searched %s; %d room%s with frontier left" % (
        prefix, found, _coarse(searched), with_frontier, "" if with_frontier == 1 else "s")


def _coarse(seconds: float) -> str:
    seconds = max(0.0, float(seconds))
    if seconds < 60.0:
        return "%ds" % (int(round(seconds / 10.0)) * 10)
    return "%dmin" % int(round(seconds / 60.0))


def stair_options(building, obs, world, cost, contexts: Dict, action_time_s: float = 1.0) -> List[StairOption]:
    """Every eligible staircase on the floor in force, as a node the loop can offer.

    Eligible: a ground-truth connector's portal on this floor -- a SEEN
    staircase, the coordinator makes portals of no other -- that is not
    cooling (a failed approach defers it for a while), whose entry the
    observed passable map has reached, and that is not the staircase the
    agent arrived by while :meth:`~multifloor_policy.MultiFloorSearch
    .way_back_held`: a storey is looked at for ``arrival_grace_actions``
    before the way back is a choice again (Pomaria: offered on the arrival
    action, it was taken three times in a row on one room's evidence).
    Nothing else here is a clock: a portal the map cannot reach yet simply
    is not a node this action and comes back when the map grows to it.

    Args:
        building: The :class:`~multifloor_policy.MultiFloorSearch`.
        obs: The current observation.
        world: The floor's occupancy grid.
        cost: The planner's cost array for ``world``.
        contexts: Saved per-floor policy state, keyed by atlas floor id
            (``policy.floors.save()``), for the destination summaries.
        action_time_s: Seconds per action, to phrase "arrived N ago".
    """
    if building is None or building.ground_truth is None:
        return []
    if not building.can_leave_floor(obs):
        return []
    held = getattr(building, "way_back_held", lambda step: False)(obs.step)
    portals = [q for q in building.portals if q["floor_id"] == building.floor_id
               and q.get("connector_id") is not None and obs.step >= q.get("cooldown_until", 0)
               and not (held and q.get("connector_id") == building.arrived_by)]
    if not portals:
        return []
    approaches = approach_points(portals, obs, world, cost)
    if not approaches:
        return []
    atlas, params = building.policy.mapping.atlas, building.params
    options: List[StairOption] = []
    for portal in portals:
        approach = approaches.get(int(portal["id"]))
        if approach is None:
            continue
        direction = int(portal["direction"])
        destination = floor_at(atlas, float(portal["destination_z"]), params.floor_match_m)
        visited = destination is not None
        arrived_by = building.arrived_by == portal.get("connector_id")
        node = SearchNode(
            id=stair_node_id(portal["id"]), kind=STAIRS,
            label="stairs %s" % ("up" if direction > 0 else "down"), direction=direction,
            destination_visited=visited,
            destination=summarise_floor(contexts.get(destination), destination) if visited else None,
            arrived_by=arrived_by,
            arrived_ago_s=(obs.step - building.arrived_step) * action_time_s if arrived_by and building.arrived_step is not None else None)
        options.append(StairOption(node_id=node.id, portal=portal, approach=approach,
                                   leaf_m=stair_cost_m(portal, params.floor_change_cost_m), node=node))
    return options


def search_context(building, policy, contexts: Dict) -> SearchContext:
    """What the prompt says about this storey and the others."""
    here = {key: getattr(policy, key) for key in ("graph", "_floor_time")}
    floor_id = policy.mapping.floor_id
    storey = summarise_floor(here, floor_id)
    if building is not None and building.ground_truth is not None:
        elevation = policy.mapping.atlas.elevation_m if policy.mapping.atlas.floors else 0.0
        storey += "; %d storey%s known to the building" % (
            len(building.ground_truth.levels), "" if len(building.ground_truth.levels) == 1 else "s")
        storey += " (this one at %+.1f m)" % (elevation - min(building.ground_truth.levels)) if building.ground_truth.levels else ""
    others = tuple(summarise_floor(context, other) for other, context in sorted(contexts.items()) if other != floor_id)
    return SearchContext(storey=storey, others=others)

