"""The wire seam of the room-search loop: JSON and grids in, JSON and routes out.

Pure functions, no ROS imports -- the same discipline as
:mod:`sparx_agency.tasks.mapping.scene_graph.ros2.payloads`, and for the same
reason: everything here is a shape that can be wrong in a way nothing raises
about, so it is unit-tested in the plain ``.venv`` with no rclpy context.

Three jobs, all of them seams:

* **decode** -- turn a ``nav_msgs/OccupancyGrid``'s fields into the planner's
  :class:`~sparx_agency.core.planning.environment.OccupancyGrid2D`. The BEV
  encoding is ``-1`` unknown / ``0`` free / ``100`` occupied, which is NOT the
  ``OccupancyValues`` default (occupied ``1``), and a grid built without saying
  so has no occupied cells at all: A* then plans straight through every wall
  and nothing anywhere reports an error. That single argument is the reason
  this function exists rather than being three lines in a callback;
* **merge** -- the ranking arrives on ``/llm_oracle/probabilities`` and the
  centroids on ``/scene_graph``, keyed by the same room pid, and the policy
  needs them as one list. Note that pids RESTART whenever the BEV geometry
  changes, so this merge must be redone from the latest of both every tick;
  never cache it across a reshape;
* **assemble** -- the route the follower flies, and the operator payload that
  mirrors the old stack's ``/goal_sampler/info``.
"""
from __future__ import annotations

from typing import Any, Dict, List, Mapping, Sequence, Tuple

import numpy as np

from sparx_agency.core.planning.environment import (
    OccupancyGrid2D, OccupancyGrid2DParams, OccupancyValues)
from sparx_agency.core.planning.exploration.room_search_policy import (
    RoomOption, RoomSearchState)

BEV_VALUES = OccupancyValues(free=0, occupied=100, unknown=-1)
"""The FALCON BEV encoding, named once so no caller retypes it wrongly."""

MIN_ROUTE_STEP_M = 1e-3
"""Consecutive route points closer than this collapse; see ``route_points``."""


def grid_from_bev(data, height, width, resolution, origin_x, origin_y,
                  frame_id="world"):
    # type: (Sequence[int], int, int, float, float, float, str) -> OccupancyGrid2D
    """Wrap a ``nav_msgs/OccupancyGrid``'s fields as a planner grid.

    Row 0 is minimum y, exactly as the message carries it -- no flip. The BEV,
    the room-label grid and :class:`OccupancyGrid2D` all agree on that
    convention; only the rendered panels do not, and the flip belongs there.

    Args:
        data: The message's ``data`` field, row-major, ``height * width`` long.
        height: Rows.
        width: Columns.
        resolution: Metres per cell.
        origin_x: World x of cell ``(0, 0)``.
        origin_y: World y of cell ``(0, 0)``.
        frame_id: The frame the grid is expressed in.

    Returns:
        A NEW grid object carrying :data:`BEV_VALUES`. Build one per tick:
        :class:`~sparx_agency.core.planning.planners.astar.WeightedAStarPlanner2D`
        caches its cost field on grid object IDENTITY, so a reused object flies
        a stale map.

    Raises:
        ValueError: If ``data`` is not ``height * width`` long, or the shape is
            not positive. A short ``data`` is what a truncated bridge delivers,
            and reshaping it silently would put the map's rows out of phase.
    """
    if height <= 0 or width <= 0:
        raise ValueError("bev shape must be positive, got %dx%d" % (height, width))
    cells = np.asarray(data, dtype=np.int8)
    if cells.size != height * width:
        raise ValueError("bev data is %d cells, expected %d (%dx%d)"
                         % (cells.size, height * width, height, width))
    params = OccupancyGrid2DParams(
        resolution=float(resolution), origin_x=float(origin_x),
        origin_y=float(origin_y), frame_id=str(frame_id))
    return OccupancyGrid2D(cells.reshape(height, width), params,
                           values=BEV_VALUES)


def centroids_from_scene_graph(payload):
    # type: (Mapping[str, Any]) -> Dict[int, Tuple[float, float]]
    """``{pid: (x, y)}`` from a ``/scene_graph`` payload's ``rooms`` list.

    Args:
        payload: The decoded ``/scene_graph`` JSON.

    Returns:
        One entry per room that has a well-formed centroid. A room without one
        is not a place the aircraft can be sent, so it is left out rather than
        given a guessed position.
    """
    centroids = {}  # type: Dict[int, Tuple[float, float]]
    for room in payload.get("rooms") or []:
        try:
            centroids[int(room["id"])] = (float(room["centroid"][0]),
                                          float(room["centroid"][1]))
        except (KeyError, IndexError, TypeError, ValueError):
            continue
    return centroids


def room_options(ranked, centroids):
    # type: (Sequence[Mapping[str, Any]], Mapping[int, Tuple[float, float]]) -> List[RoomOption]
    """Join the oracle's ranking to the scene graph's centroids.

    Malformed entries are skipped rather than raised on, and that is a
    deliberate exception to this repo's raise-loudly rule, documented here
    because it is one: the ranking is produced by a small language model's
    post-processing, one room with a null probability is a normal Tuesday, and
    a search loop that dies on it grounds the aircraft over a cosmetic fault.
    What is NOT tolerated is silence -- the count of candidates survives into
    ``/room_search/info``, so a ranking that is quietly losing rooms is visible
    on the dashboard.

    Args:
        ranked: The ``rooms`` list of a ``/llm_oracle/probabilities`` payload.
        centroids: From :func:`centroids_from_scene_graph`.

    Returns:
        One :class:`RoomOption` per usable entry, centroid attached where the
        scene graph has one (the policy drops the rest).
    """
    options = []  # type: List[RoomOption]
    for entry in ranked or []:
        try:
            room_id = int(entry["id"])
            prob = float(entry.get("prob", 0.0))
        except (KeyError, TypeError, ValueError):
            continue
        options.append(RoomOption(room_id=room_id, prob=prob,
                                  xy=centroids.get(room_id),
                                  label=str(entry.get("label", "?"))))
    return options


def route_points(start_xy, waypoints, altitude, min_step_m=MIN_ROUTE_STEP_M):
    # type: (Tuple[float, float], Sequence[Tuple[float, float]], float, float) -> List[Tuple[float, float, float]]
    """The flyable route: the aircraft's own position, then the plan.

    The leading point is where the aircraft IS. A pure-pursuit tracker searches
    forward of a monotone progress index, so a route whose first vertex is
    already metres ahead makes the aircraft's position read as off the path --
    and A*'s ``start_skip_m`` deliberately drops exactly those leading
    waypoints, so without this the two conventions fight.

    Args:
        start_xy: Where the aircraft is, world metres.
        waypoints: The planned ``(x, y)`` vertices, in order.
        altitude: The z every point is stamped at -- one cruise height, so the
            3D pursuit commands a vertical rate toward it while tracking xy.
        min_step_m: Collapse a vertex this close to its predecessor.

    Returns:
        ``[(x, y, z), ...]``, at least the start. A caller that gets fewer than
        two points has nothing to fly and should say so rather than command.
    """
    points = [(float(start_xy[0]), float(start_xy[1]), float(altitude))]
    for point in waypoints:
        x, y = float(point[0]), float(point[1])
        if abs(x - points[-1][0]) > min_step_m or abs(y - points[-1][1]) > min_step_m:
            points.append((x, y, float(altitude)))
    return points


def search_info_payload(stamp, state, target, fly, planned, route_length,
                        note, stats):
    # type: (float, RoomSearchState, str, bool, bool, int, str, Mapping[str, int]) -> Dict[str, Any]
    """The ``/room_search/info`` payload -- the old ``/goal_sampler/info`` grown up.

    Every key the flown payload had is here under the same name (``room_id``,
    ``label``, ``prob``, ``goal``, and ``candidates`` carrying
    ``prob_renorm``), so a dashboard written against the old topic reads this
    one. What is added is what the old one could not say: which state the
    machine is in, whether a route to the chosen room actually exists, and
    whether the loop is armed to fly at all.

    Args:
        stamp: Seconds, the node's clock.
        state: The policy's state this tick.
        target: What is being searched for, echoed from the oracle payload.
        fly: Whether flight is armed (the ``fly`` parameter).
        planned: Whether the current goal has a route.
        route_length: How many vertices that route has.
        note: The action's human reason, for the operator.
        stats: The policy's running counters.

    Returns:
        A dict of plain Python scalars, safe for ``json.dumps``.
    """
    return {
        "stamp": float(stamp),
        "state": str(state.state),
        "target": str(target),
        "fly": bool(fly),
        "room_id": None if state.room_id is None else int(state.room_id),
        "label": None if state.label is None else str(state.label),
        "prob": None if state.prob is None else float(state.prob),
        "goal": None if state.goal_xy is None else [float(state.goal_xy[0]),
                                                    float(state.goal_xy[1])],
        "planned": bool(planned),
        "route_length": int(route_length),
        "elapsed_s": float(state.elapsed_s),
        "dwell_left_s": float(state.dwell_left_s),
        "note": str(note),
        "candidates": [
            {"id": int(c.room_id), "label": str(c.label),
             "prob": float(c.prob), "prob_renorm": float(c.prob_renorm)}
            for c in state.candidates],
        "stats": {str(k): int(v) for k, v in stats.items()},
    }
