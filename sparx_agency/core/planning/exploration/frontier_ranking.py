"""Rank a room's frontier goals by what they are worth from where the robot stands.

:func:`~sparx_agency.core.planning.exploration.room_costs.in_room_frontier_goals`
orders a room's unscanned boundaries by size alone, and size alone is the wrong
order from a robot's point of view. On the Ranchester Gibson recording
``e7c4f2ad5402`` (step 297) it sent the agent through a 180-degree turn and a
3.8 m walk back across a room it had just swept, toward a large cluster behind
it, while a smaller boundary sat two steps ahead. The order here is the classic
frontier utility -- gain over cost -- with two deliberate choices:

* **Cost is geodesic, not Euclidean.** One single-source Dijkstra over the
  same unit-weight passable graph the room arc weights use
  (:func:`~sparx_agency.core.planning.exploration.room_costs.passable_graph`),
  so a boundary across a wall costs the walk around it, and a boundary with no
  path through known free space is *dropped* rather than proposed. Proposing
  it costs a failed A* and an idle action; the recording shows three of those
  (``route unavailable``) in its first thirty actions.
* **Facing is a discount, not a veto.** A goal behind the robot costs the turns
  to face it and is discounted by :attr:`FrontierRankingParams.heading_weight`,
  but a large boundary behind still beats a speck ahead. A veto would make the
  robot walk away from the biggest unknown in the room because it happened to
  arrive facing the other way.

Gain is ``size ** gain_exponent`` -- sub-linear by default, because a cluster
twice as long is not twice as informative once the camera's field of view
covers it, and a linear gain reproduces the size-only order at any distance.

numpy and scipy, host side only, like :mod:`room_costs`: not imported by the
Noetic FALCON facade.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
from scipy.sparse.csgraph import dijkstra

from sparx_agency.core.common.types import normalize_angle
from sparx_agency.core.planning.environment import OccupancyGrid2D
from sparx_agency.core.planning.exploration.room_costs import (
    frontier_cluster_cells, passable_graph, snap_cell)


@dataclass(frozen=True)
class FrontierRankingParams:
    """How gain and cost combine into one utility per frontier cluster.

    Attributes:
        gain_exponent: Power applied to the cluster size in cells. 0.5 by
            default; 1.0 recovers a size-proportional gain, 0.0 makes every
            cluster equal so the order is nearest-first.
        distance_floor_m: Added to the geodesic distance before dividing, so
            a boundary under the robot's feet does not get an infinite
            utility and a metre of difference between two near goals still
            matters. Around one forward step.
        heading_weight: How much a goal straight behind is discounted:
            utility is divided by ``1 + heading_weight * |error| / pi``. 1.0
            halves a goal directly behind; 0.0 ignores facing.
        max_geodesic_m: Search horizon for the Dijkstra sweep. A cluster
            farther than this along the floor is dropped this tick; it comes
            back as the robot approaches. Bounds the per-action cost of the
            sweep on a large map.
        snap_radius_m: How far the robot's own cell may be moved onto the
            passable graph -- the robot routinely reads as standing inside an
            inflation skirt.

    Raises:
        ValueError: On a non-finite or out-of-range value.
    """

    gain_exponent: float = 0.5
    distance_floor_m: float = 0.25
    heading_weight: float = 1.0
    max_geodesic_m: float = 30.0
    snap_radius_m: float = 1.0

    def __post_init__(self):
        for name in ("gain_exponent", "heading_weight"):
            value = getattr(self, name)
            if isinstance(value, bool) or not math.isfinite(value) or value < 0.0:
                raise ValueError("%s must be finite and non-negative" % name)
        for name in ("distance_floor_m", "max_geodesic_m", "snap_radius_m"):
            value = getattr(self, name)
            if isinstance(value, bool) or not math.isfinite(value) or value <= 0.0:
                raise ValueError("%s must be positive and finite" % name)


@dataclass(frozen=True)
class FrontierGoal:
    """One ranked frontier cluster.

    Attributes:
        xy: World ENU metres of the cluster's snapped, passable cell.
        cell: That cell, ``(gx, gy)``.
        size_cells: How many frontier cells the cluster holds.
        geodesic_m: Walking distance from the robot along known free space.
        heading_error_rad: Signed turn from the robot's heading to face the
            goal, ``(-pi, pi]``, positive to the left.
        utility: The score the list is sorted by, descending.
    """

    xy: Tuple[float, float]
    cell: Tuple[int, int]
    size_cells: int
    geodesic_m: float
    heading_error_rad: float
    utility: float


def ranked_frontier_goals(world: OccupancyGrid2D,
                          cost: np.ndarray,
                          room_mask: np.ndarray,
                          origin_xy: Tuple[float, float],
                          yaw: float,
                          params: Optional[FrontierRankingParams] = None,
                          min_cluster_cells: int = 4,
                          ids: Optional[np.ndarray] = None,
                          graph=None) -> List[FrontierGoal]:
    """Frontier goals inside ``room_mask``, best first, from ``origin_xy``.

    Args:
        world: The BEV grid.
        cost: The planner's cost array for ``world`` -- ``inf`` where
            blocked. Passed in so the ranking and the route the robot then
            plans share one passable set.
        room_mask: ``(H, W)`` bool, True inside the room; all-True for a
            floor-wide frontier.
        origin_xy: Where the robot is, world metres.
        yaw: The robot's heading, radians, counter-clockwise from world ``+x``.
        params: Tuning. Defaults to :class:`FrontierRankingParams`.
        min_cluster_cells: Clusters smaller than this are noise and dropped.
        ids: The index image from :func:`passable_graph`, if already built.
        graph: The matching CSR graph, if already built. Both or neither.

    Returns:
        Reachable clusters as :class:`FrontierGoal`, utility descending.
        Empty when the room has no unscanned boundary the robot can walk to
        -- including when the robot itself stands off the passable graph.

    Raises:
        ValueError: If exactly one of ``ids`` and ``graph`` is given, or the
            origin or yaw is not finite.
    """
    params = params or FrontierRankingParams()
    if (ids is None) != (graph is None):
        raise ValueError("pass both ids and graph from passable_graph, or neither")
    if not all(math.isfinite(float(v)) for v in (origin_xy[0], origin_xy[1], yaw)):
        raise ValueError("origin_xy and yaw must be finite")
    if ids is None:
        ids, graph = passable_graph(cost)
    clusters = frontier_cluster_cells(world, room_mask, ids, min_cluster_cells)
    if not clusters:
        return []

    ox, oy = world.world_to_grid(float(origin_xy[0]), float(origin_xy[1]))
    radius = max(1, int(round(params.snap_radius_m / world.resolution)))
    source = snap_cell(ids, ox, oy, radius)
    if source is None:
        return []
    limit_cells = params.max_geodesic_m / world.resolution
    dist = dijkstra(graph, directed=False, indices=int(ids[source[1], source[0]]),
                    limit=limit_cells)

    goals: List[FrontierGoal] = []
    for size, (gx, gy) in clusters:
        steps = float(dist[int(ids[gy, gx])])
        if not math.isfinite(steps):
            continue
        geodesic = steps * float(world.resolution)
        x, y = (float(v) for v in world.grid_to_world(gx, gy))
        error = normalize_angle(math.atan2(y - float(origin_xy[1]),
                                           x - float(origin_xy[0])) - float(yaw))
        facing = 1.0 + params.heading_weight * abs(error) / math.pi
        utility = (float(size) ** params.gain_exponent
                   / ((geodesic + params.distance_floor_m) * facing))
        goals.append(FrontierGoal(xy=(x, y), cell=(int(gx), int(gy)),
                                  size_cells=int(size), geodesic_m=geodesic,
                                  heading_error_rad=float(error),
                                  utility=float(utility)))
    goals.sort(key=lambda g: (-g.utility, g.geodesic_m))
    return goals

