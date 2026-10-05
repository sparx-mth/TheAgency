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

:func:`frontier_goals_by_room` is the same ranking for EVERY room at once --
one extraction, one Dijkstra -- with each cluster credited to a room by the
majority vote :func:`~sparx_agency.core.mapping.topology.room_stats.count_frontier_clusters`
uses, so the goals a room is offered and the ``frontier_clusters`` the scene
graph reports for it are the same population. The room-search loop needs
this per action in transit: the entry point into a room is its nearest
frontier, and asking for it room by room would run one Dijkstra per room.

numpy and scipy, host side only, like :mod:`room_costs`: not imported by the
Noetic FALCON facade.
"""
from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass, replace
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.ndimage import label as connected_components
from scipy.sparse.csgraph import dijkstra

from sparx_agency.core.common.types import normalize_angle
from sparx_agency.core.planning.environment import OccupancyGrid2D
from sparx_agency.core.planning.planners.common.grid_geometry_2d import line_of_sight_clear
from sparx_agency.core.planning.exploration.room_costs import (
    FrontierCluster, frontier_clusters, passable_graph, snap_cell)


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


@dataclass(frozen=True)
class FrontierInventory:
    """Reachable frontier goals, room ownership, and display cells on one map.

    Counts describe accessibility, not whether a goal was recently retired.
    Unassigned frontiers remain available to floor-wide exploration. Distances
    are metres from the current pose; infinity means not currently accessible.
    """

    goals: Tuple[FrontierGoal, ...]
    by_room: Dict[int, List[FrontierGoal]]
    cells: np.ndarray
    distance_m: np.ndarray


def accessible_frontiers(world, cost, room_labels, origin_xy, yaw, params=None, preferred_cost=None):
    """Build the inventory without confusing a planning horizon with reachability.

    Uses the planner's clearance-qualified passable graph. A goal snapped across
    a disconnected wall is rejected rather than credited to an unseen room.

    Args:
        preferred_cost: Optional cost array at the planner's PREFERRED
            standoff (``WeightedAStarPlanner2D.cost_for``), beside ``cost``
            at the body radius. A* plans at the preferred standoff and
            relaxes to the body radius only when nothing else gets
            through, so the route it will actually fly is the preferred
            one wherever that exists: with this given, every distance is
            measured at the preferred standoff where the cell is reachable
            there and at the body radius otherwise. Without it a threshold
            3.5 m away through a 0.5 m squeeze read as 3.5 m while the
            route around the squeeze was 12.5 m (Hanson 2026-10-04).
    """
    params = replace(params or FrontierRankingParams(),
                     max_geodesic_m=max(1.0, world.grid.size * world.resolution * math.sqrt(2.0)))
    labels = np.asarray(room_labels)
    if labels.shape != world.grid.shape:
        raise ValueError("Room labels must match the occupancy grid")
    distances = np.full(world.grid.shape, np.inf, dtype=float)
    cells = np.zeros(world.grid.shape, dtype=bool)
    components, _ = connected_components(world.grid == world.values.free, structure=np.ones((3, 3)))
    ox, oy = world.world_to_grid(*origin_xy)
    if not world.in_bounds(ox, oy) or components[oy, ox] == 0:
        return FrontierInventory((), {}, cells, distances)
    source_component = components == components[oy, ox]
    qualified_cost = np.where(source_component, cost, np.inf)
    ids, graph = _graph(qualified_cost, None, None, origin_xy, yaw)
    dist = _distances(world, ids, graph, origin_xy, params)
    if dist is None:
        return FrontierInventory((), {}, cells, distances)
    on_graph = ids >= 0
    distances[on_graph] = dist[ids[on_graph]] * world.resolution
    if preferred_cost is not None:
        preferred = np.asarray(preferred_cost, dtype=float)
        if preferred.shape != cost.shape:
            raise ValueError("preferred_cost %s is not shaped like cost %s" % (preferred.shape, cost.shape))
        distances = _prefer_distances(world, np.where(source_component, preferred, np.inf), origin_xy, yaw,
                                      params, distances)
    clusters = frontier_clusters(world, source_component, ids)
    blocked = world.grid != world.values.free
    goals, by_room = [], {}
    for cluster in clusters:
        gx, gy = cluster.cell
        members = distances[cluster.rows, cluster.cols]
        if not math.isfinite(distances[gy, gx]):
            if not np.isfinite(members).any():
                continue
            nearest = int(np.argmin(members))
            gx, gy = int(cluster.cols[nearest]), int(cluster.rows[nearest])
            cluster = replace(cluster, cell=(gx, gy))
        nearest = int(np.argmin((cluster.cols - gx) ** 2 + (cluster.rows - gy) ** 2))
        if not line_of_sight_clear(blocked, gx, gy, int(cluster.cols[nearest]), int(cluster.rows[nearest])):
            continue
        goal = _rank_by_distance(world, [cluster], distances, origin_xy, yaw, params)[0]
        goals.append(goal)
        cells[cluster.rows, cluster.cols] = True
        room = _room_of(labels, cluster)
        if room is not None:
            by_room.setdefault(room, []).append(goal)
    goals.sort(key=lambda g: (-g.utility, g.geodesic_m))
    for members in by_room.values():
        members.sort(key=lambda g: (-g.utility, g.geodesic_m))
    return FrontierInventory(tuple(goals), by_room, cells, distances)


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
    ids, graph = _graph(cost, ids, graph, origin_xy, yaw)
    clusters = frontier_clusters(world, room_mask, ids, min_cluster_cells)
    if not clusters:
        return []
    dist = _distances(world, ids, graph, origin_xy, params)
    if dist is None:
        return []
    return _rank(world, clusters, dist, ids, origin_xy, yaw, params)


def frontier_goals_by_room(world: OccupancyGrid2D,
                           cost: np.ndarray,
                           room_labels: np.ndarray,
                           origin_xy: Tuple[float, float],
                           yaw: float,
                           params: Optional[FrontierRankingParams] = None,
                           min_cluster_cells: int = 4,
                           ids: Optional[np.ndarray] = None,
                           graph=None) -> Dict[int, List[FrontierGoal]]:
    """Every room's reachable frontier goals in one pass, keyed by room label.

    One extraction over the whole grid and one Dijkstra from the robot, then
    each cluster is credited to ONE room by majority vote over the room
    labels at its cells -- label-0 cells abstain and a cluster with no
    labelled cell is dropped -- which is exactly the rule
    :func:`~sparx_agency.core.mapping.topology.room_stats.count_frontier_clusters`
    applies. A cluster straddling a doorway therefore goes to the room the
    count credits it to, and a room's goal list is empty exactly when its
    count is zero (up to reachability: a cluster with no path through known
    free space is dropped here and still counted there).

    Args:
        world: The BEV grid.
        cost: The planner's cost array for ``world``, ``inf`` where blocked.
        room_labels: ``(H, W)`` int label image, 0 = no room, > 0 = a room.
            Keys of the result are the labels exactly as they appear here;
            whether they are pids or pid+1 is the caller's convention.
        origin_xy: Where the robot is, world metres.
        yaw: The robot's heading, radians.
        params: Tuning. Defaults to :class:`FrontierRankingParams`.
        min_cluster_cells: Clusters smaller than this are noise and dropped.
        ids: The index image from :func:`passable_graph`, if already built.
        graph: The matching CSR graph, if already built. Both or neither.

    Returns:
        ``{room_label: [FrontierGoal, ...]}``, each list utility descending
        and never empty -- a room with nothing reachable is simply absent.
        Empty when the robot stands off the passable graph.

    Raises:
        ValueError: If exactly one of ``ids`` and ``graph`` is given, the
            origin or yaw is not finite, or ``room_labels`` is not shaped
            like the grid.
    """
    params = params or FrontierRankingParams()
    labels = np.asarray(room_labels)
    if labels.shape != world.grid.shape:
        raise ValueError("room_labels %s is not shaped like the grid %s"
                         % (labels.shape, world.grid.shape))
    ids, graph = _graph(cost, ids, graph, origin_xy, yaw)
    everywhere = np.ones(world.grid.shape, dtype=bool)
    clusters = frontier_clusters(world, everywhere, ids, min_cluster_cells)
    if not clusters:
        return {}
    dist = _distances(world, ids, graph, origin_xy, params)
    if dist is None:
        return {}
    by_room: Dict[int, List[FrontierCluster]] = {}
    for cluster in clusters:
        room = _room_of(labels, cluster)
        if room is not None:
            by_room.setdefault(room, []).append(cluster)
    out: Dict[int, List[FrontierGoal]] = {}
    for room, members in by_room.items():
        goals = _rank(world, members, dist, ids, origin_xy, yaw, params)
        if goals:
            out[room] = goals
    return out


def _room_of(labels: np.ndarray, cluster: FrontierCluster) -> Optional[int]:
    """The room label most of the cluster's labelled cells carry, or None."""
    here = labels[cluster.rows, cluster.cols]
    here = here[here > 0]
    if here.size == 0:
        return None
    return int(Counter(here.tolist()).most_common(1)[0][0])


def _graph(cost, ids, graph, origin_xy, yaw) -> Tuple[np.ndarray, Any]:
    """Validate the inputs and build the passable graph if it was not passed."""
    if (ids is None) != (graph is None):
        raise ValueError("pass both ids and graph from passable_graph, or neither")
    if not all(math.isfinite(float(v)) for v in (origin_xy[0], origin_xy[1], yaw)):
        raise ValueError("origin_xy and yaw must be finite")
    if ids is None:
        ids, graph = passable_graph(cost)
    return ids, graph


def _distances(world, ids, graph, origin_xy, params) -> Optional[np.ndarray]:
    """Geodesic steps from the robot to every graph node, or None if it is off-graph."""
    ox, oy = world.world_to_grid(float(origin_xy[0]), float(origin_xy[1]))
    radius = max(1, int(round(params.snap_radius_m / world.resolution)))
    source = snap_cell(ids, ox, oy, radius)
    if source is None:
        return None
    limit_cells = params.max_geodesic_m / world.resolution
    return dijkstra(graph, directed=False, indices=int(ids[source[1], source[0]]),
                    limit=limit_cells)


def _prefer_distances(world, preferred_cost, origin_xy, yaw, params, fallback) -> np.ndarray:
    """Per-cell metres at the preferred standoff where reachable there, else ``fallback``."""
    ids, graph = _graph(preferred_cost, None, None, origin_xy, yaw)
    dist = _distances(world, ids, graph, origin_xy, params)
    if dist is None:
        return fallback
    out = np.array(fallback, dtype=float, copy=True)
    on_graph = ids >= 0
    preferred = dist[ids[on_graph]] * world.resolution
    reached = np.isfinite(preferred)
    rows, cols = np.nonzero(on_graph)
    out[rows[reached], cols[reached]] = preferred[reached]
    return out


def _rank(world, clusters: Sequence[FrontierCluster], dist, ids, origin_xy, yaw,
          params) -> List[FrontierGoal]:
    """Score reachable clusters -- gain over geodesic cost, facing as a discount."""
    distances = np.full(world.grid.shape, np.inf, dtype=float)
    on_graph = ids >= 0
    distances[on_graph] = dist[ids[on_graph]] * float(world.resolution)
    return _rank_by_distance(world, clusters, distances, origin_xy, yaw, params)


def _rank_by_distance(world, clusters: Sequence[FrontierCluster], distances: np.ndarray, origin_xy, yaw,
                      params) -> List[FrontierGoal]:
    """Score reachable clusters from a per-cell metre field -- gain over geodesic cost, facing as a discount."""
    goals: List[FrontierGoal] = []
    for cluster in clusters:
        gx, gy = cluster.cell
        geodesic = float(distances[gy, gx])
        if not math.isfinite(geodesic):
            continue
        x, y = (float(v) for v in world.grid_to_world(gx, gy))
        error = normalize_angle(math.atan2(y - float(origin_xy[1]),
                                           x - float(origin_xy[0])) - float(yaw))
        facing = 1.0 + params.heading_weight * abs(error) / math.pi
        utility = (float(cluster.size) ** params.gain_exponent
                   / ((geodesic + params.distance_floor_m) * facing))
        goals.append(FrontierGoal(xy=(x, y), cell=(int(gx), int(gy)),
                                  size_cells=int(cluster.size), geodesic_m=geodesic,
                                  heading_error_rad=float(error),
                                  utility=float(utility)))
    goals.sort(key=lambda g: (-g.utility, g.geodesic_m))
    return goals

