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
  fixed cost of a storey change (:func:`~stair_approach.stair_cost_m`),
  charged on every arc into AND out of it -- going upstairs puts every room
  down here that much further away, which is exactly what RPT* must see.

Nothing here decides anything; the decision is the order the solver returns.

**No staircase is a node today.** A stair node's facts need two things of
its portal: the height of the storey the flight leads to (``destination_z``
-- is that storey visited, and what was found there?) and an identity the
portals at the flight's two ends share (``stairs_id`` -- did the robot just
arrive by these stairs?). The only source that ever supplied them was the
``ground_truth`` stair source: connectors and storey levels read from the
simulator's navmesh, privileged scene geometry, removed on 2026-11-15. The
observed discovery (``world_model/stair_terrain.py``) measures a flight's
treads, not where it ends, so its portals carry neither and
:func:`stair_options` offers none of them; a floor change is then the
building coordinator's allowance rule
(:meth:`~multifloor_policy.MultiFloorSearch.plan`). The loop's side is kept
whole -- ids, facts, cost, commitment -- for an observed source that does
measure them, and the prompt never tells the oracle how many storeys the
building has: only the ones the robot has stood on are summarised.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

from sparx_agency.core.mapping.topology.search_node_oracle import STAIRS, SearchContext, SearchNode
from sparx_agency.core.planning.exploration.room_search_policy import RoomOption
from sparx_agency.core.planning.object_search.multifloor.stair_approach import (
    Approach, approach_points, floor_at, stair_cost_m)

#: Room pids count up from zero; stair node ids start here. A registry would
#: need a hundred thousand rooms before the two could meet. The band above
#: (``opening_nodes.OPENING_NODE_BASE``) belongs to the floor's openings.
STAIR_NODE_BASE = 100_000
STAIR_NODE_LIMIT = 200_000


def stair_node_id(portal_id: int) -> int:
    """The loop's node id for the coordinator's portal ``portal_id``."""
    return STAIR_NODE_BASE + int(portal_id)


def is_stair_node(node_id) -> bool:
    """Whether a supervisor room id names a staircase rather than a room or an opening."""
    return node_id is not None and STAIR_NODE_BASE <= int(node_id) < STAIR_NODE_LIMIT


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
    tracker = getattr(graph, "label_tracker", None)
    labels = getattr(tracker, "labels", {}) or {}
    metadata = getattr(tracker, "metadata", {}) or {}
    facts = getattr(graph, "facts", {}) or {}
    # A weak label (one kind of object) is marked as the node lines mark it,
    # ``kitchen?``: a cabinet alone made a "kitchen" on an upper storey and the
    # oracle's STEP 2 read a ground floor off the summary (Ranchester 2026-10-05).
    named = sorted(label.label + ("?" if (metadata.get(pid) or {}).get("strength") == "weak" else "")
                   for pid, label in labels.items() if pid in rooms and label.label != "unknown")
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


def stair_options(building, obs, world, cost, contexts: Dict, action_time_s: float = 1.0,
                  rooms_left: bool = False) -> List[StairOption]:
    """Every eligible staircase on the floor in force, as a node the loop can offer.

    Eligible: a portal on this floor that says which storey its flight leads
    to (``destination_z``; see the module docstring -- no portal of the
    observed discovery does, so today the list is empty), that is not
    cooling (a failed approach defers it for a while), whose entry the
    observed passable map has reached, and that is not the staircase the
    agent arrived by while the way back is held: for
    ``arrival_grace_actions`` after the arrival
    (:meth:`~multifloor_policy.MultiFloorSearch.way_back_held`; Pomaria:
    offered on the arrival action, it was taken three times in a row on one
    room's evidence), and -- since 2026-10-04 -- while ``rooms_left`` says
    this storey still has a room that is neither finished nor ruled out. A
    storey is looked at before "not here" means anything: the 3B oracle sent
    the Ranchester agent straight back up twice with "living room found; no
    couch here", three unscanned rooms on the list each time. Any OTHER
    staircase is offered at once. Nothing else here is a clock: a portal the
    map cannot reach yet simply is not a node this action and comes back when
    the map grows to it.

    Args:
        building: The :class:`~multifloor_policy.MultiFloorSearch`.
        obs: The current observation.
        world: The floor's occupancy grid.
        cost: The planner's cost array for ``world``.
        contexts: Saved per-floor policy state, keyed by atlas floor id
            (``policy.floors.save()``), for the destination summaries.
        action_time_s: Seconds per action, to phrase "arrived N ago".
        rooms_left: Whether a room of this storey is still a node of the search.
    """
    if building is None:
        return []
    if not building.can_leave_floor(obs):
        return []
    held = getattr(building, "way_back_held", lambda step: False)(obs.step) or bool(rooms_left)
    portals = [q for q in building.portals if q["floor_id"] == building.floor_id
               and q.get("destination_z") is not None and obs.step >= q.get("cooldown_until", 0)
               and not (held and _arrived_by(building, q))]
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
        arrived_by = _arrived_by(building, portal)
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


def _arrived_by(building, portal: Dict) -> bool:
    """Whether ``portal`` is an end of the staircase the agent last arrived by (a portal with no identity never is)."""
    stairs = portal.get("stairs_id")
    return stairs is not None and stairs == building.arrived_by


def search_context(policy, contexts: Dict, house: str = "") -> SearchContext:
    """What the prompt says about this home: the HOUSE line, and the storeys the robot has stood on.

    ``house`` is the big-picture line (:func:`~house_context.house_line`);
    the storey lines are kept for the multi-storey development protocol and
    are shown only when another storey has been visited or a staircase is
    offered (the standard single-storey benchmarks never see them). Only
    storeys the robot has stood on are named: how many storeys the building
    has, and where this one ranks among them, is not something it has
    observed, and until 2026-11-15 those two facts came from the
    simulator's navmesh.
    """
    here = {key: getattr(policy, key) for key in ("graph", "_floor_time")}
    floor_id = policy.mapping.floor_id
    storey = summarise_floor(here, floor_id)
    others = tuple(summarise_floor(context, other) for other, context in sorted(contexts.items()) if other != floor_id)
    return SearchContext(house=house or storey, storey=storey, others=others)

