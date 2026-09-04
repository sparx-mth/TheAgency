"""The wire seam of the object-search loop: JSON and grids in, an instance out.

The same discipline as
:mod:`sparx_agency.tasks.mapping.scene_graph.ros2.room_search_payloads`, and
for the same reason: every shape here can be wrong in a way nothing raises
about, so it is unit-tested in the plain ``.venv`` with no rclpy context.

Four jobs:

* **decode** -- turn ``/scene_graph``'s room list into the per-room facts the
  supervisor's done-tests read, and turn ``/scene_graph/room_labels_grid``
  plus its ``grid_pid_map`` back into one room's boolean mask. That mask is
  the only thing that keeps the in-room sweep inside the room, so its
  correctness is a flight-safety property, not a formatting one;
* **assemble** -- build the HPP-PT instance from the live BEV, the centroids
  and the oracle's ranking, so the solver seam is fed from the wire;
* **publish** -- the operator payloads for ``/object_search/costs`` and
  ``/object_search/info``. The cost payload exists so the arc weights an
  "optimal" order was computed from are visible in a recording; an optimality
  claim nobody can audit is not a claim;
* **re-export** -- ``grid_from_bev`` and friends come FROM
  ``room_search_payloads`` rather than being copied. ``grid_from_bev`` is the
  only correct way to wrap ``/falcon/bev_2d``, because it carries
  ``BEV_VALUES``: a grid built with the ``OccupancyValues`` default
  (``occupied=1``) has ZERO occupied cells, A* then plans straight through
  every wall, and nothing anywhere reports an error.

Every emitted value is a plain builtin, list or str-keyed dict. numpy scalars
survive ``json.dumps`` on some types and not others, and JSON has no integer
keys -- which is why ``grid_pid_map`` stringifies its own.
"""
from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from sparx_agency.core.planning.environment import OccupancyGrid2D
from sparx_agency.core.planning.exploration.object_search_supervisor import (
    ObjectSearchState, RoomFacts)
from sparx_agency.core.planning.exploration.room_costs import (
    HppPtInstance, build_instance, in_room_frontier_goals)
from sparx_agency.tasks.mapping.scene_graph.ros2.room_search_payloads import (
    BEV_VALUES, centroids_from_scene_graph, grid_from_bev, room_options,
    route_points)

__all__ = [
    "BEV_VALUES", "centroids_from_scene_graph", "grid_from_bev",
    "room_options", "route_points",
    "costs_payload", "facts_from_scene_graph", "in_room_frontier_goals",
    "instance_from_wire", "room_mask_from_labels", "search_info_payload",
]


# -- decode ---------------------------------------------------------------
def facts_from_scene_graph(payload):
    # type: (Mapping[str, Any]) -> Dict[int, RoomFacts]
    """``{pid: RoomFacts}`` from a ``/scene_graph`` payload's ``rooms`` list.

    Args:
        payload: The decoded ``/scene_graph`` JSON.

    Returns:
        One entry per room with a well-formed id. A room whose id cannot be
        read is dropped; its other fields default rather than raising, so a
        mapper that has not yet credited any dwell time to a new room still
        produces a usable fact for it.
    """
    facts = {}  # type: Dict[int, RoomFacts]
    for room in payload.get("rooms") or []:
        try:
            pid = int(room["id"])
        except (KeyError, TypeError, ValueError):
            continue
        try:
            clusters = int(room.get("frontier_clusters", 0))
        except (TypeError, ValueError):
            clusters = 0
        try:
            dwell = float(room.get("time_in_room_s", 0.0))
        except (TypeError, ValueError):
            dwell = 0.0
        try:
            cells = int(room.get("cells", 0))
        except (TypeError, ValueError):
            cells = 0
        facts[pid] = RoomFacts(room_id=pid, frontier_clusters=clusters,
                               time_in_room_s=dwell, cells=cells)
    return facts


def room_mask_from_labels(label_data, height, width, grid_pid_map, pid):
    # type: (Sequence[int], int, int, Mapping[str, int], int) -> Optional[np.ndarray]
    """One room's boolean mask, from the label grid and its pid indirection.

    ``/scene_graph/room_labels_grid`` is ``int8`` and carries a small GRID
    VALUE per room, not the pid -- pids grow without bound and 130 written
    into an int8 cell wraps negative. The payload's ``grid_pid_map`` resolves
    a grid value back to its pid, keyed by the decimal STRING of that value
    because JSON has no integer keys.

    Args:
        label_data: The message's ``data`` field, row-major.
        height: Rows.
        width: Columns.
        grid_pid_map: ``{str(grid value): pid}`` from the ``/scene_graph``
            payload published in the SAME tick. A stale pairing silently
            returns another room's cells, which is why the caller must take
            both from one tick.
        pid: The room wanted.

    Returns:
        ``(H, W)`` bool, row 0 at minimum y -- the same lattice as
        ``/falcon/bev_2d``, so it indexes the planner's grid directly. None
        when this pid holds no grid value, which is how a renumbered room
        announces itself rather than silently masking nothing.

    Raises:
        ValueError: If ``label_data`` is not ``height * width`` long. A short
            array is what a truncated bridge delivers, and reshaping it
            silently would put the mask's rows out of phase with the map.
    """
    if height <= 0 or width <= 0:
        raise ValueError("label grid shape must be positive, got %dx%d"
                         % (height, width))
    cells = np.asarray(label_data, dtype=np.int16)
    if cells.size != height * width:
        raise ValueError("label grid is %d cells, expected %d (%dx%d)"
                         % (cells.size, height * width, height, width))
    value = None
    for key, mapped in (grid_pid_map or {}).items():
        try:
            if int(mapped) == int(pid):
                value = int(key)
                break
        except (TypeError, ValueError):
            continue
    if value is None:
        return None
    return cells.reshape(int(height), int(width)) == np.int16(value)


# -- assemble -------------------------------------------------------------
def instance_from_wire(world, cost, scene_graph, ranked, depot_xy=None,
                       snap_radius_m=2.0, cruise_speed_mps=0.30,
                       search_time_s=0.0, frontier_weight=0.0):
    # type: (OccupancyGrid2D, np.ndarray, Mapping[str, Any], Sequence[Mapping[str, Any]], Optional[Tuple[float, float]], float, float, float, float) -> Tuple[Optional[HppPtInstance], List[int]]
    """Build the solver's instance from the two latched topics and the BEV.

    Args:
        world: The BEV, wrapped by :func:`grid_from_bev`.
        cost: ``WeightedAStarPlanner2D.cost_for(world)[0]``.
        scene_graph: The decoded ``/scene_graph`` payload.
        ranked: The ``rooms`` list of ``/llm_oracle/probabilities``.
        depot_xy: Where the aircraft is.
        snap_radius_m: How far a blocked room centre may be moved.
        cruise_speed_mps: Converts metres to seconds; 0 leaves metres.
        search_time_s: The per-room budget folded onto entering arcs.
        frontier_weight: Blend of unexplored-space into the probabilities.

    Returns:
        ``(instance, dropped_pids)``, or ``(None, [])`` when the scene graph
        has no room with a centroid yet -- which is the first ~60 s of every
        flight and is not an error.
    """
    centroids = centroids_from_scene_graph(scene_graph)
    if not centroids:
        return None, []
    probs = {}  # type: Dict[int, float]
    for entry in ranked or []:
        try:
            probs[int(entry["id"])] = float(entry.get("prob", 0.0))
        except (KeyError, TypeError, ValueError):
            continue
    counts = dict((pid, f.frontier_clusters)
                  for pid, f in facts_from_scene_graph(scene_graph).items())
    return build_instance(
        world, cost, centroids, probs, depot_xy=depot_xy,
        snap_radius_m=snap_radius_m, cruise_speed_mps=cruise_speed_mps,
        search_time_s=search_time_s, frontier_weight=frontier_weight,
        frontier_counts=counts)


# -- publish --------------------------------------------------------------
def costs_payload(instance, scene_graph_stamp, stamp, dropped_pids=(),
                  prob_source="unknown"):
    # type: (HppPtInstance, float, float, Sequence[int], str) -> Dict[str, Any]
    """The ``/object_search/costs`` payload -- the graph the solver was given.

    This is the audit trail for an optimality claim. Everything RPT* consumed
    is here: the vertex order, the arc weights, the probabilities, which
    vertex was the depot, and which rooms were withheld because the map could
    not reach them.

    Args:
        instance: What the solver was handed.
        scene_graph_stamp: The stamp of the ``/scene_graph`` it was built
            from, so a consumer can tell whether the two agree.
        stamp: Publish time, the node's clock.
        dropped_pids: Rooms withheld this tick.
        prob_source: ``'llm'`` or ``'uniform_fallback'``, echoed from the
            oracle. A whole run of uniform_fallback means the LLM never
            answered and the ordering carries no commonsense at all.

    Returns:
        A dict of plain builtins, safe for ``json.dumps``.
    """
    return {
        "stamp": float(stamp),
        "scene_graph_stamp": float(scene_graph_stamp),
        "units": str(instance.units),
        "rooms": [int(p) for p in instance.index_to_pid],
        "centres": [[float(n.xy[0]), float(n.xy[1])] for n in instance.nodes],
        "snapped": [bool(n.snapped) for n in instance.nodes],
        "depot": int(instance.depot),
        "cost": [[float(v) for v in row] for row in instance.C],
        "p": [float(v) for v in instance.p],
        "dropped": [int(p) for p in dropped_pids],
        "build_ms": float(instance.build_ms),
        "prob_source": str(prob_source),
    }


def search_info_payload(stamp, state, target, fly, planned, route_length,
                        note, stats, room_facts=None, backend="host_sweep"):
    # type: (float, ObjectSearchState, str, bool, bool, int, str, Mapping[str, int], Optional[RoomFacts], str) -> Dict[str, Any]
    """The ``/object_search/info`` payload -- what the loop is doing and why.

    Deliberately a superset of ``/room_search/info`` under the same key names
    (``room_id``, ``label``, ``prob``, ``goal``, ``candidates`` with
    ``prob_renorm``), so a dashboard written against that topic reads this one
    unchanged. What is added is what the older payload could not say: the
    committed visit order and how far into it the loop is, the mapping budget
    remaining, the room's live frontier count, and the verdict on the tick a
    room's turn ends.

    Args:
        stamp: Seconds, the node's clock.
        state: The supervisor's state this tick.
        target: What is being searched for.
        fly: Whether flight is armed.
        planned: Whether the current goal has a route.
        route_length: How many vertices that route has.
        note: The action's human reason.
        stats: The supervisor's running counters.
        room_facts: The scene graph's facts for the room in force.
        backend: Which mapping backend the SEARCH state is using.

    Returns:
        A dict of plain builtins, safe for ``json.dumps``.
    """
    return {
        "stamp": float(stamp),
        "state": str(state.state),
        "target": str(target),
        "fly": bool(fly),
        "backend": str(backend),
        "room_id": None if state.room_id is None else int(state.room_id),
        "label": None if state.label is None else str(state.label),
        "prob": None if state.prob is None else float(state.prob),
        "goal": None if state.goal_xy is None else [float(state.goal_xy[0]),
                                                    float(state.goal_xy[1])],
        "planned": bool(planned),
        "route_length": int(route_length),
        "elapsed_s": float(state.elapsed_s),
        "search_left_s": float(state.search_left_s),
        "order": [int(r) for r in state.order],
        "order_index": int(state.order_index),
        "rooms_done": int(state.rooms_done),
        "frontier_clusters": (None if state.frontier_clusters is None
                              else int(state.frontier_clusters)),
        "time_in_room_s": (None if room_facts is None
                           else float(room_facts.time_in_room_s)),
        "room_cells": None if room_facts is None else int(room_facts.cells),
        "completed": (None if state.completed is None
                      else {"room_id": int(state.completed[0]),
                            "verdict": str(state.completed[1])}),
        "note": str(note),
        "candidates": [
            {"id": int(c.room_id), "label": str(c.label),
             "prob": float(c.prob), "prob_renorm": float(c.prob_renorm)}
            for c in state.candidates],
        "stats": {str(k): int(v) for k, v in stats.items()},
    }
