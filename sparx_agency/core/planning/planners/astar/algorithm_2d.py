"""A* search kernels on 2D grids.

Two entry points:

- :func:`astar_grid_2d` works on any object exposing the ``OccupancyGrid2D``
  query protocol (``in_bounds`` / ``is_occupied`` / ``is_unknown``). It treats
  cells as binary traversable/blocked and is used by the exploration behaviours
  and the windowed local planner.
- :func:`astar_cost_grid_2d` works on a dense ``float`` cost array (1.0 = free,
  ``inf`` = blocked, any finite value = weighted) and adds a bounding-box search
  domain plus an optional turn penalty. This is the fast kernel used by the
  weighted FALCON planner.

Both use proper 8-connected step costs (diagonals cost ``sqrt(2)``) and the
octile heuristic, which is the tightest admissible/consistent heuristic for
8-connected motion. For 4-connectivity the heuristic reduces to Manhattan.
"""
from __future__ import annotations

import heapq
import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from sparx_agency.core.planning.environment import OccupancyGrid2D

Index2D = Tuple[int, int]
BBox = Tuple[int, int, int, int]  # (xmin, xmax, ymin, ymax), half-open on max

SQRT2 = math.sqrt(2.0)


@dataclass(frozen=True)
class AStarSearchResult2D:
    path: Tuple[Index2D, ...]
    expanded: int

    @property
    def ok(self) -> bool:
        return len(self.path) > 0


def _octile(ax: int, ay: int, bx: int, by: int) -> float:
    """Octile distance: admissible/consistent heuristic for 8-connected grids.

    Reduces to Manhattan when the path is restricted to 4-connected moves
    (the diagonal term is non-positive only via ``min(dx, dy)``, so on a pure
    cardinal optimal path it adds nothing).
    """
    dx = abs(ax - bx)
    dy = abs(ay - by)
    return (dx + dy) + (SQRT2 - 2.0) * min(dx, dy)


def _manhattan(ax: int, ay: int, bx: int, by: int) -> int:
    return abs(ax - bx) + abs(ay - by)


# (dx, dy, step_cost)
_MOVES_4 = ((-1, 0, 1.0), (1, 0, 1.0), (0, -1, 1.0), (0, 1, 1.0))
_MOVES_8 = _MOVES_4 + (
    (-1, -1, SQRT2), (-1, 1, SQRT2), (1, -1, SQRT2), (1, 1, SQRT2),
)


def astar_grid_2d(
    grid: OccupancyGrid2D,
    start: Index2D,
    goal: Index2D,
    *,
    allow_unknown: bool,
    connectivity: int,
    max_expansions: int | None,
) -> AStarSearchResult2D:
    """A* over a binary (traversable/blocked) grid.

    Args:
        grid: Any object with ``in_bounds``/``is_occupied``/``is_unknown``.
        start: Start cell ``(gx, gy)``.
        goal: Goal cell ``(gx, gy)``.
        allow_unknown: If False, UNKNOWN cells are blocked.
        connectivity: 4 or 8. 8 uses ``sqrt(2)`` diagonal cost + octile h.
        max_expansions: Safety cap on pops; ``None`` for unlimited.

    Returns:
        ``AStarSearchResult2D`` with the path excluding ``start`` and including
        ``goal`` (empty path if unreachable or capped).
    """
    if start == goal:
        return AStarSearchResult2D(path=(), expanded=0)

    moves = _MOVES_8 if connectivity == 8 else _MOVES_4
    h = _octile if connectivity == 8 else _manhattan

    def traversable(x: int, y: int) -> bool:
        if not grid.in_bounds(x, y):
            return False
        if grid.is_occupied(x, y):
            return False
        if (not allow_unknown) and grid.is_unknown(x, y):
            return False
        return True

    if not traversable(*start) or not traversable(*goal):
        return AStarSearchResult2D(path=(), expanded=0)

    gx, gy = goal
    open_heap: List[Tuple[float, Index2D]] = [(0.0, start)]
    came_from: Dict[Index2D, Index2D] = {}
    g_score: Dict[Index2D, float] = {start: 0.0}

    expanded = 0
    while open_heap:
        _, current = heapq.heappop(open_heap)
        expanded += 1
        if max_expansions is not None and expanded > max_expansions:
            return AStarSearchResult2D(path=(), expanded=expanded)

        if current == goal:
            break

        cx, cy = current
        gc = g_score[current]
        for dx, dy, step in moves:
            nx, ny = cx + dx, cy + dy
            if not traversable(nx, ny):
                continue
            tentative = gc + step
            n = (nx, ny)
            if tentative < g_score.get(n, math.inf):
                came_from[n] = current
                g_score[n] = tentative
                heapq.heappush(open_heap, (tentative + h(nx, ny, gx, gy), n))

    if goal not in came_from:
        return AStarSearchResult2D(path=(), expanded=expanded)

    rev: List[Index2D] = [goal]
    cur = goal
    while cur in came_from:
        cur = came_from[cur]
        if cur == start:
            break
        rev.append(cur)
    rev.reverse()
    return AStarSearchResult2D(path=tuple(rev), expanded=expanded)


def _backward_weight(a: Index2D, b: Index2D) -> float:
    """How much move ``b`` reverses direction ``a``: 0 for forward/perpendicular
    (angle <= 90 deg), rising to 1.0 for a full 180 deg reversal (``-cos`` of the
    angle, clamped at 0). Lets a heading penalty leave forward and 90 deg turns
    free while heavily penalising turning around."""
    la = math.hypot(a[0], a[1])
    lb = math.hypot(b[0], b[1])
    if la < 1e-9 or lb < 1e-9:
        return 0.0
    cos = (a[0] * b[0] + a[1] * b[1]) / (la * lb)
    return max(0.0, -cos)


def _turn_angle(a: Index2D, b: Index2D) -> float:
    """Unsigned angle between two grid moves, radians in ``[0, pi]``.

    The rotation-time model's counterpart to :func:`_backward_weight`: that one
    asks "is this a reversal?", this one asks "how far must the robot rotate?".
    A robot that must stop and yaw before it can fly a corner accurately spends
    time in proportion to *this*, not to how backward the corner is."""
    la = math.hypot(a[0], a[1])
    lb = math.hypot(b[0], b[1])
    if la < 1e-9 or lb < 1e-9:
        return 0.0
    cos = (a[0] * b[0] + a[1] * b[1]) / (la * lb)
    return math.acos(max(-1.0, min(1.0, cos)))


def astar_cost_grid_2d(
    cost: np.ndarray,
    start: Index2D,
    goal: Index2D,
    *,
    connectivity: int = 8,
    bbox: Optional[BBox] = None,
    turn_penalty: float = 0.0,
    start_dir: Optional[Index2D] = None,
    start_turn_penalty: float = 0.0,
    start_turn_cost_rad: float = 0.0,
    start_turn_radius: float = 0.0,
    max_expansions: int | None = None,
) -> AStarSearchResult2D:
    """A* over a dense float cost grid (the fast, weighted kernel).

    Edge cost from a cell to a neighbour is ``step * cost[ny, nx]`` where
    ``step`` is 1.0 (cardinal) or ``sqrt(2)`` (diagonal). A cell with a
    non-finite cost (``inf``) is blocked. An optional ``turn_penalty`` is added
    whenever the travel direction changes, which suppresses staircasing before
    line-of-sight smoothing runs.

    HEADING AWARENESS: pass ``start_dir`` (the drone's facing as a grid move
    ``(dx, dy)``) to seed the start's incoming direction, so the first move is a
    "turn" like any other. Two costs may then be charged for setting off across
    the robot's heading, and they model different things -- pass either, both, or
    neither:

    * ``start_turn_penalty`` scales by :func:`_backward_weight`: nothing up to a
      90 deg turn, rising to the full penalty at a 180 deg reversal. This asks
      "is the route making me turn *around*?", and leaves a robot free to set off
      sideways. It is what the FALCON stack flies.
    * ``start_turn_cost_rad`` scales linearly with the turn angle from zero. This
      asks "how long will I sit here rotating before I can fly this?", which is
      the right question for a follower that stops and yaws on the spot before
      taking a corner it cannot glide. Set it to the cost of a radian of
      rotation -- cruise speed over yaw rate, in the same units as the grid --
      and the search trades turning against flying in real seconds.

    Either way the route flies the way the drone already looks and only turns
    when it truly must, instead of spinning in place because the shortest path
    happens to run backward.

    ``start_turn_radius`` is what gives either cost any reach, and without it the
    whole mechanism is close to decorative. Charged on the first move, a heading
    cost only chooses between eight neighbours of one cell -- and a single 10 cm
    step cannot steer a 30 m route, which is measurable: over 40 office routes,
    every non-zero first-move heading cost moved the mean start turn by under 4
    degrees and the flight-time proxy by 0.2 s, and the effect saturated
    immediately because it is really just picking one of eight branches. With a
    radius the cost is instead charged once, on the edge that leaves a disc of
    that many cells around the start, against the bearing from the start to
    wherever the route *exits* that disc. Set it to the distance the aircraft
    covers while turning (a few metres) and the search has to commit to leaving
    the area roughly the way it is pointing, or pay for it.

    That formulation stays a proper edge cost -- the charge depends only on the
    cell being entered -- so it neither breaks the closed set nor makes the
    octile heuristic inadmissible (a non-negative addition to edge costs only
    ever makes an unchanged heuristic more conservative). A route that doubles
    back into the disc and leaves again is charged twice, which no shortest route
    does.

    Note all of this applies to *leaving the start* only. A per-cell turn cost
    that scales with angle is deliberately not offered: on a fine grid a straight
    line that is not axis-aligned must staircase, and charging every 45 deg
    staircase step drives the search onto longer L-shaped detours. Measured on
    the office map, raising the flat ``turn_penalty`` from 0.3 to 3.0 lengthened
    routes from 42.5 to 44.7 m and *increased* total turning from 204 to 329
    degrees. Turn awareness along the whole route belongs in a state lattice
    whose motion primitives are long enough for a corner to be a real corner.

    Args:
        cost: ``(H, W)`` float array. 1.0 = free, ``inf`` = blocked, finite
            values > 1 are traversable-at-a-penalty (e.g. UNKNOWN).
        start: Start cell ``(gx, gy)`` (indexed ``cost[gy, gx]``).
        goal: Goal cell ``(gx, gy)``.
        connectivity: 4 or 8.
        bbox: Optional ``(xmin, xmax, ymin, ymax)`` half-open search window in
            grid indices. Cells outside are never expanded — the single largest
            speed win when start and goal are close on a large grid.
        turn_penalty: Extra cost added on a direction change (0 disables).
        start_dir: Drone facing as a grid move ``(dx, dy)`` in ``{-1,0,1}`` (not
            ``(0,0)``); ``None`` disables heading awareness (start move is free).
        start_turn_penalty: Extra cost for the first move going against
            ``start_dir``, scaled by :func:`_backward_weight` (0 disables).
        start_turn_cost_rad: Extra cost per radian of turn away from
            ``start_dir``, scaled by :func:`_turn_angle` (0 disables).
        start_turn_radius: Radius in cells of the run-up disc whose exit bearing
            ``start_turn_cost_rad`` is charged against. 0 charges the first move
            instead, which has almost no effect on the route -- see above.
        max_expansions: Safety cap on pops; ``None`` for unlimited.

    Returns:
        ``AStarSearchResult2D`` with the path including both ``start`` and
        ``goal`` (empty if unreachable or capped). Note: unlike
        :func:`astar_grid_2d`, the start cell IS included, because the weighted
        planner consumes the full cell chain for line-of-sight smoothing.
    """
    H, W = cost.shape
    sx, sy = start
    gx, gy = goal

    if bbox is None:
        xmin, xmax, ymin, ymax = 0, W, 0, H
    else:
        xmin, xmax, ymin, ymax = bbox

    moves = _MOVES_8 if connectivity == 8 else _MOVES_4
    h = _octile if connectivity == 8 else _manhattan

    # A zero radius makes the start cell the whole disc, so its exit edge IS the
    # first move -- the run-up charge degenerates to the first-move charge rather
    # than needing a second code path.
    charge_runup = start_dir is not None and (start_turn_cost_rad > 0.0
                                              or start_turn_penalty > 0.0)
    run_up_sq = start_turn_radius * start_turn_radius

    g = np.full((H, W), np.inf, dtype=np.float64)
    g[sy, sx] = 0.0
    closed = np.zeros((H, W), dtype=bool)
    came: Dict[Index2D, Index2D] = {}

    open_heap: List[Tuple[float, float, int, int]] = [(h(sx, sy, gx, gy), 0.0, sx, sy)]
    expanded = 0
    while open_heap:
        _f, gc, x, y = heapq.heappop(open_heap)
        if closed[y, x]:
            continue
        closed[y, x] = True
        expanded += 1
        if max_expansions is not None and expanded > max_expansions:
            return AStarSearchResult2D(path=(), expanded=expanded)

        if x == gx and y == gy:
            path = [(x, y)]
            while (x, y) in came:
                x, y = came[(x, y)]
                path.append((x, y))
            path.reverse()
            return AStarSearchResult2D(path=tuple(path), expanded=expanded)

        # Incoming direction: the parent's move for an interior cell, else the
        # drone's facing (start_dir) for the start cell -- so the first move is a
        # turn like any other and the heading penalty can bias it forward.
        at_start = (x, y) not in came
        if at_start:
            prev = start_dir
        else:
            px, py = came[(x, y)]
            prev = (x - px, y - py)

        # Is this cell still inside the run-up disc? Only an edge that leaves it
        # is charged the run-up heading cost, and only once.
        in_runup = charge_runup and ((x - sx) ** 2 + (y - sy) ** 2) <= run_up_sq

        for dx, dy, step in moves:
            nx, ny = x + dx, y + dy
            if not (xmin <= nx < xmax and ymin <= ny < ymax):
                continue
            if closed[ny, nx]:
                continue
            c = cost[ny, nx]
            if not math.isfinite(c):
                continue
            if prev is None or prev == (dx, dy):
                turn = 0.0
            else:
                turn = turn_penalty
            # The edge that leaves the run-up disc pays for the bearing the route
            # departs on -- the angle the aircraft must actually rotate through
            # before it can fly this route. Charged once, on one edge, and only
            # against the cell being entered, so it stays a proper edge cost.
            if in_runup and (nx - sx) ** 2 + (ny - sy) ** 2 > run_up_sq:
                away = (nx - sx, ny - sy)
                turn += (start_turn_penalty * _backward_weight(start_dir, away)
                         + start_turn_cost_rad * _turn_angle(start_dir, away))
            ng = gc + step * c + turn
            if ng < g[ny, nx]:
                g[ny, nx] = ng
                came[(nx, ny)] = (x, y)
                heapq.heappush(open_heap, (ng + h(nx, ny, gx, gy), ng, nx, ny))

    return AStarSearchResult2D(path=(), expanded=expanded)
