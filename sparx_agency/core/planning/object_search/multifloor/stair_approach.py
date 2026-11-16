"""Where a staircase is entered from on the observed map, and what taking it costs.

The geometry and the arithmetic of a staircase as a node of the room-search
loop (:mod:`~sparx_agency.core.planning.object_search.rooms.stair_nodes`),
and nothing that decides anything:

* :func:`approach_points` -- the point on the floor's observed passable map
  a portal's entry is reached from, and how far the agent has to walk to it.
  An entry the map has not reached has no approach point this action.
* :func:`stair_cost_m` -- what the climb costs beyond reaching its foot: the
  flight's own length plus the fixed cost of a storey change.
* :func:`floor_at` -- which storey of the atlas a height belongs to.

Until 2026-11-15 this module (``floor_decision.py`` then) also held the
explicit up/down rule of the ground-truth stair source's exploration
fallback; it went with that source.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Dict, Optional, Sequence, Tuple

import numpy as np
from scipy.sparse.csgraph import dijkstra

from sparx_agency.core.planning.exploration.room_costs import passable_graph, snap_cell
from sparx_agency.core.planning.object_search.world_model.standing_floor import nearest_cell_in_sight


@dataclass(frozen=True)
class ApproachSettings:
    """Bounds of the approach-point search. Distances in metres.

    Attributes:
        approach_snap_m: How far an entry anchor may be moved onto the
            observed passable map to become an approach goal.
        agent_snap_m: How far the agent's own cell may be moved onto the
            passable graph before it counts as off-map.
    """

    approach_snap_m: float = 1.5
    agent_snap_m: float = 1.0


@dataclass(frozen=True)
class Approach:
    """Where a staircase is entered from on the observed map, and how far that is."""

    xy: Tuple[float, float]
    distance_m: float


def floor_at(atlas, height: float, tolerance_m: float) -> Optional[int]:
    """The atlas floor id whose elevation matches ``height``, or None if unvisited."""
    matches = [f for f in atlas.floors.values() if abs(f.elevation_m - height) <= tolerance_m]
    return min(matches, key=lambda f: abs(f.elevation_m - height)).id if matches else None


def flight_length_m(portal: Dict) -> float:
    """The flight's length along its polyline, or 0 when the portal carries none."""
    path = portal.get("path") or ()
    return float(sum(math.dist(a[:3], b[:3]) for a, b in zip(path, path[1:]))) if len(path) > 1 else 0.0


def stair_cost_m(portal: Dict, floor_change_cost_m: float) -> float:
    """What taking this staircase costs beyond reaching its foot: the flight plus the fixed change cost.

    The one definition of what a storey change costs, read by the RPT* leaf
    (every arc into and out of the stair node).
    """
    return flight_length_m(portal) + max(0.0, float(floor_change_cost_m))


def approach_points(portals: Sequence[Dict], obs, world, cost: np.ndarray,
                    settings: Optional[ApproachSettings] = None) -> Optional[Dict[int, Approach]]:
    """For every portal whose entry is on the observed passable map: its approach point and geodesic distance.

    One passable graph and one Dijkstra from the agent for all of them.
    A portal whose entry does not snap (``standing_floor.nearest_cell_in_sight``:
    no passable cell with a clear line to the anchor -- every cell on the
    way, the entry cell itself excepted: a stair head's own cell may be
    marked occupied by the drop beyond it, and that is where the flight
    begins), or has no path from the agent, is absent from the result -- the
    map has not reached it yet. None when the agent itself stands off the
    passable map.
    """
    settings = settings or ApproachSettings()
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
        cell = nearest_cell_in_sight(ids, occupied, gx, gy, radius)
        if cell is None:
            continue
        distance = float(distances[int(ids[cell[1], cell[0]])]) * world.resolution
        if not math.isfinite(distance):
            continue
        xy = tuple(float(v) for v in world.grid_to_world(cell[0], cell[1]))
        out[int(portal["id"])] = Approach(xy, distance)
    return out
