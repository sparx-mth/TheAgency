"""Candidate viewpoints on a reachable map, and what reaching one costs in discrete actions.

A viewpoint planner asks two questions this module answers without a
policy: *where could the camera stand* -- the passable cells the agent
can walk to, thinned to a lattice so a room of a thousand cells is a few
dozen candidates -- and *what does standing there cost* in the benchmark's
own currency, actions: forward steps along the floor plus the turns that
face the way and face the view. Distance is geodesic (the caller's
Dijkstra field over the planner's passable graph, as
:func:`~frontier_ranking.accessible_frontiers` builds it), never
Euclidean: a point across a wall costs the walk around it.

numpy and scipy (the Dijkstra of :func:`geodesic_field`), host side only like
:mod:`frontier_ranking`; Python 3.8 syntax.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np
from scipy.sparse.csgraph import dijkstra

from sparx_agency.core.common.types import normalize_angle
from sparx_agency.core.planning.environment.occupancy_grid2d import OccupancyGrid2D
from sparx_agency.core.planning.exploration.room_costs import passable_graph, snap_cell


def geodesic_field(world: OccupancyGrid2D, cost: np.ndarray, origin_xy: Tuple[float, float],
                   max_geodesic_m: float, snap_radius_m: float = 1.0) -> np.ndarray:
    """``(H, W)`` metres from ``origin_xy`` to every passable cell along the floor, ``inf`` beyond reach.

    One single-source Dijkstra over the planner's passable graph
    (:func:`~room_costs.passable_graph` of ``cost``, ``inf`` where
    blocked), limited to ``max_geodesic_m``; the agent's own cell is
    snapped onto the graph within ``snap_radius_m`` (it routinely reads as
    standing inside an inflation skirt). All ``inf`` when it cannot be.
    """
    distances = np.full(world.grid.shape, np.inf, dtype=float)
    cost = np.asarray(cost, dtype=float)
    if cost.shape != world.grid.shape:
        raise ValueError("cost %s is not shaped like the grid %s" % (cost.shape, world.grid.shape))
    ids, graph = passable_graph(cost)
    if graph.shape[0] == 0:
        return distances
    ox, oy = world.world_to_grid(float(origin_xy[0]), float(origin_xy[1]))
    source = snap_cell(ids, ox, oy, max(1, int(round(float(snap_radius_m) / float(world.resolution)))))
    if source is None:
        return distances
    steps = dijkstra(graph, directed=False, indices=int(ids[source[1], source[0]]),
                     limit=float(max_geodesic_m) / float(world.resolution))
    on_graph = ids >= 0
    distances[on_graph] = steps[ids[on_graph]] * float(world.resolution)
    return distances


@dataclass(frozen=True)
class ActionPricing:
    """The discrete action set as a price list.

    Attributes:
        forward_step_m: How far one MOVE_FORWARD advances.
        turn_rad: How far one TURN_LEFT / TURN_RIGHT turns.
        path_turn_per_m: Turns the route itself is expected to cost per
            metre walked, beyond the turn that faces its start: a lattice
            route through a doorway bends once or twice. 0.5 by default
            (one 30-degree turn every two metres); a caller with the exact
            plan prices it instead.

    Raises:
        ValueError: On a non-positive step or turn.
    """

    forward_step_m: float
    turn_rad: float
    path_turn_per_m: float = 0.5

    def __post_init__(self) -> None:
        for name in ("forward_step_m", "turn_rad"):
            value = getattr(self, name)
            if isinstance(value, bool) or not math.isfinite(value) or value <= 0.0:
                raise ValueError("%s must be positive and finite, got %r" % (name, value))
        if (isinstance(self.path_turn_per_m, bool) or not math.isfinite(self.path_turn_per_m)
                or self.path_turn_per_m < 0.0):
            raise ValueError("path_turn_per_m must be finite and non-negative, got %r" % (self.path_turn_per_m,))

    def turns(self, angle_rad: float) -> int:
        """In-place turns that cover ``angle_rad`` (its magnitude), at least 0."""
        return int(math.ceil(abs(float(angle_rad)) / self.turn_rad - 1e-9)) if abs(float(angle_rad)) > 1e-9 else 0

    def travel_actions(self, geodesic_m: float, heading_error_rad: float) -> float:
        """Expected actions to walk ``geodesic_m`` after turning ``heading_error_rad`` to face the way."""
        steps = float(geodesic_m) / self.forward_step_m
        return steps + self.turns(heading_error_rad) + self.path_turn_per_m * float(geodesic_m)


@dataclass(frozen=True)
class Viewpoint:
    """One candidate standing point.

    Attributes:
        xy: World metres of the cell's centre.
        cell: ``(gx, gy)``.
        geodesic_m: Walking distance from the agent along the passable map.
        heading_error_rad: Signed turn from the agent's heading to the
            straight-line bearing of the point, ``(-pi, pi]``.
    """

    xy: Tuple[float, float]
    cell: Tuple[int, int]
    geodesic_m: float
    heading_error_rad: float


def sample_viewpoints(world: OccupancyGrid2D, distance_m: np.ndarray, origin_xy: Tuple[float, float], yaw: float,
                      spacing_m: float, max_geodesic_m: float, min_distance_m: float = 0.0,
                      avoid: Sequence[Tuple[float, float]] = (), avoid_radius_m: float = 0.0,
                      within: Optional[np.ndarray] = None) -> List[Viewpoint]:
    """Reachable cells on a lattice of ``spacing_m``, as candidate viewpoints.

    Args:
        world: The occupancy grid (for the cell geometry).
        distance_m: ``(H, W)`` geodesic metres from the agent to every
            cell along the passable map, ``inf`` where unreachable.
        origin_xy: The agent's position; the lattice is phased on its cell.
        yaw: The agent's heading, for the facing cost.
        spacing_m: Lattice pitch; at least one cell.
        max_geodesic_m: Cells farther along the floor are not offered.
        min_distance_m: Cells nearer than this (geodesic) are not offered:
            a move shorter than a step or two shows nothing new.
        avoid: Points no candidate may lie within ``avoid_radius_m`` of
            (Euclidean): the spots already looked around from.
        avoid_radius_m: See ``avoid``; 0 disables.
        within: Optional ``(H, W)`` bool restricting the candidates (a
            room's mask).

    Returns:
        The candidates, nearest first.

    Raises:
        ValueError: On a distance field not shaped like the grid.
    """
    dist = np.asarray(distance_m, dtype=float)
    if dist.shape != world.grid.shape:
        raise ValueError("distance_m %s is not shaped like the grid %s" % (dist.shape, world.grid.shape))
    pitch = max(1, int(round(float(spacing_m) / float(world.resolution))))
    ox, oy = world.world_to_grid(float(origin_xy[0]), float(origin_xy[1]))
    h, w = dist.shape
    lattice = np.zeros((h, w), dtype=bool)
    lattice[(oy % pitch)::pitch, (ox % pitch)::pitch] = True
    eligible = lattice & np.isfinite(dist) & (dist <= float(max_geodesic_m)) & (dist >= float(min_distance_m))
    if within is not None:
        mask = np.asarray(within, dtype=bool)
        if mask.shape != dist.shape:
            raise ValueError("within %s is not shaped like the grid %s" % (mask.shape, dist.shape))
        eligible &= mask
    ys, xs = np.nonzero(eligible)
    if not len(xs):
        return []
    points = np.array([world.grid_to_world(int(gx), int(gy)) for gx, gy in zip(xs, ys)], dtype=float)
    keep = np.ones(len(xs), dtype=bool)
    if avoid and avoid_radius_m > 0.0:
        r2 = float(avoid_radius_m) ** 2
        for ax, ay in avoid:
            keep &= (points[:, 0] - float(ax)) ** 2 + (points[:, 1] - float(ay)) ** 2 > r2
    out: List[Viewpoint] = []
    for i in np.nonzero(keep)[0]:
        x, y = float(points[i, 0]), float(points[i, 1])
        error = normalize_angle(math.atan2(y - float(origin_xy[1]), x - float(origin_xy[0])) - float(yaw))
        out.append(Viewpoint(xy=(x, y), cell=(int(xs[i]), int(ys[i])), geodesic_m=float(dist[ys[i], xs[i]]),
                             heading_error_rad=float(error)))
    out.sort(key=lambda v: v.geodesic_m)
    return out
