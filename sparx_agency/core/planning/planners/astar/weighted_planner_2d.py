"""Weighted A* planner on a 2D OccupancyGrid2D, with smoothing.

This is the ROS-free core behind the FALCON ``astar_planner`` node. It turns a
2D occupancy grid (FREE / OCCUPIED / UNKNOWN) plus start/goal world poses into a
clean, corner-preserving set of world waypoints by composing the reusable core
primitives:

    occupancy --build_cost_grid--> float cost map (lethal inflation + clearance
                                   centering cost + UNKNOWN weight)
              --astar_cost_grid_2d--> grid cell path (bbox-restricted, octile)
              --simplify_path_cells--> reduced corners (Douglas-Peucker; keeps
                                   the centred shape and its clearance)
              --split_long_segments_2d--> spaced world waypoints

The planner is stateful only to cache the cost map per input grid (keyed on
object identity), so a plan followed by a collision re-check on the same grid
does not inflate obstacles twice. It owns no ROS or world-IO concepts.
"""
from __future__ import annotations

from math import hypot
from typing import List, Optional, Sequence, Tuple

import numpy as np

from sparx_agency.core.common.types import Pose2D, Path2D, PlanResult, PlanStatus
from sparx_agency.core.planning.environment import OccupancyGrid2D
from sparx_agency.core.planning.interfaces.planner import PlanRequest

from .params import WeightedAStarParams
from .algorithm_2d import astar_cost_grid_2d
from ..common.clearance_2d import clearance_field
from ..common.grid_geometry_2d import (
    line_of_sight_clear,
    simplify_path_cells,
    snap_to_free_cell,
)
from ..common.utils_2d import split_long_segments_2d


def build_cost_grid(
    grid: OccupancyGrid2D, params: WeightedAStarParams
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build the float cost map, lethal mask and clearance field from a grid.

    The cost map encodes:

    - ``inf`` for lethal cells: real obstacles and anything within
      ``inflate_radius_m`` of one (exact Euclidean distance), so gaps narrower
      than the robot are never threaded. UNKNOWN is *not* lethal unless
      ``unknown_blocked``.
    - ``unknown_cost`` (flat) for traversable UNKNOWN ("gray") cells. Keep it
      above ``1 + clearance_weight`` so the planner prefers any known-free route
      over driving blind through gray.
    - a baseline of ``1.0`` for known-free cells, plus a soft *clearance
      penalty* that fades from ``clearance_weight`` to ``0`` over
      ``clearance_margin_m``. The penalty is measured against the distance to
      the nearest *boundary of known-free space* — i.e. an obstacle **or the
      gray frontier**. The middle of a known-free corridor is the farthest point
      from both, so it is the cheapest lane: the route is pulled to the centre,
      away from walls, and does **not** drift toward (or cut through) unknown
      space.

    Two distance fields are used because the two jobs differ: collision cares
    only about real obstacles, while centring cares about staying inside the
    known-free region.

    Args:
        grid: Source occupancy grid (uses ``grid.values`` for OCC/UNKNOWN).
        params: Weighting / inflation / clearance parameters.

    Returns:
        ``(cost, lethal, clearance)`` — ``cost`` is an ``(H, W)`` float array
        (``inf`` = blocked), ``lethal`` is the boolean collision mask (obstacles
        + inscribed radius) used for line-of-sight checks, and ``clearance`` is
        the per-cell distance (m) to the nearest known-free boundary
        (obstacle or gray).
    """
    data = grid.grid
    res = grid.resolution
    occupied = data == grid.values.occupied
    unknown = data == grid.values.unknown

    # Lethal mask from REAL obstacles only (so the drone may still approach the
    # gray frontier). `occupied |` keeps obstacles blocked even at inflate 0.
    if params.inflate_radius_m > 0.0:
        obstacle_clear = clearance_field(occupied, res, params.inflate_radius_m + res)
        lethal = occupied | (obstacle_clear <= params.inflate_radius_m)
    else:
        lethal = occupied.copy()

    # Centring field: distance to the nearest boundary of KNOWN-FREE space —
    # an obstacle OR an unknown cell. This is what keeps the route centred
    # *within* the explored corridor instead of hugging the gray frontier.
    band = params.inflate_radius_m + max(params.clearance_margin_m, 0.0)
    clearance = clearance_field(occupied | unknown, res, band + res)

    cost = np.ones(data.shape, dtype=np.float64)

    # Soft clearance penalty on known-free cells: linear ramp, peak at the
    # boundary -> 0 at inflate_radius_m + clearance_margin_m. Minimum at maximum
    # clearance, so A* settles on the centre of the known-free corridor.
    if params.clearance_weight > 0.0 and params.clearance_margin_m > 0.0:
        ramp = (band - clearance) / params.clearance_margin_m
        np.clip(ramp, 0.0, 1.0, out=ramp)
        cost += params.clearance_weight * ramp

    # Gray cells get a flat, deliberately-high traversal cost (not the centring
    # penalty), so they are used only as a last resort. Then stamp lethal as inf.
    if params.unknown_blocked:
        cost[unknown] = np.inf
    else:
        cost[unknown] = float(params.unknown_cost)
    cost[lethal] = np.inf
    return cost, lethal, clearance


class WeightedAStarPlanner2D:
    """Weighted, bbox-restricted A* with line-of-sight smoothing.

    Implements the ``BasePlanner`` protocol: ``plan(request, world)`` where
    ``world`` is an :class:`OccupancyGrid2D`.
    """

    name: str = "weighted_astar_2d"

    def __init__(self, params: Optional[WeightedAStarParams] = None) -> None:
        self.params = params or WeightedAStarParams()
        self._cache_grid: Optional[OccupancyGrid2D] = None
        self._cache_cost: Optional[np.ndarray] = None
        self._cache_lethal: Optional[np.ndarray] = None
        self._cache_clearance: Optional[np.ndarray] = None

    # ------------------------------------------------------------------
    # Cost cache (keyed on grid object identity)
    # ------------------------------------------------------------------
    def cost_for(
        self, grid: OccupancyGrid2D
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return ``(cost, lethal, clearance)``, rebuilding only when ``grid`` changes."""
        if self._cache_grid is not grid or self._cache_cost is None:
            (self._cache_cost, self._cache_lethal,
             self._cache_clearance) = build_cost_grid(grid, self.params)
            self._cache_grid = grid
        return self._cache_cost, self._cache_lethal, self._cache_clearance

    def invalidate_cache(self) -> None:
        """Drop the cached cost map (e.g. on an explicit replan request)."""
        self._cache_grid = None
        self._cache_cost = None
        self._cache_lethal = None
        self._cache_clearance = None

    def _simplify_tol_cells(self, res: float) -> float:
        """Douglas–Peucker tolerance in cells (auto = ~1 cell when <= 0)."""
        p = self.params
        if p.path_simplify_m > 0.0:
            return p.path_simplify_m / res
        return 1.0

    # ------------------------------------------------------------------
    # Planning
    # ------------------------------------------------------------------
    def plan(self, request: PlanRequest, world: OccupancyGrid2D) -> PlanResult:
        p = self.params
        res = world.resolution
        cost, lethal, _clearance = self.cost_for(world)
        h, w = cost.shape

        sx, sy = world.world_to_grid(request.start.x, request.start.y)
        gx, gy = world.world_to_grid(request.goal.x, request.goal.y)
        if not (0 <= sx < w and 0 <= sy < h):
            return PlanResult(
                status=PlanStatus.INVALID_START,
                message=f"start cell ({sx},{sy}) outside grid {w}x{h}",
            )
        if not (0 <= gx < w and 0 <= gy < h):
            return PlanResult(
                status=PlanStatus.INVALID_GOAL,
                message=f"goal cell ({gx},{gy}) outside grid {w}x{h}",
            )

        # Snap a blocked goal onto the nearest free cell.
        if not np.isfinite(cost[gy, gx]):
            snap_r = int(p.goal_snap_radius_m / res)
            snapped = snap_to_free_cell(cost, gx, gy, snap_r) if snap_r > 0 else None
            if snapped is None:
                return PlanResult(
                    status=PlanStatus.INVALID_GOAL,
                    message=f"goal blocked, no free cell within {p.goal_snap_radius_m:.1f}m",
                )
            gx, gy = snapped

        # Start is always passable (it may sit inside the inflation skirt). Only
        # copy the cost map when we actually need to override, to keep the cache
        # clean and avoid a per-plan allocation in the common case.
        if not np.isfinite(cost[sy, sx]):
            cost = cost.copy()
            cost[sy, sx] = 1.0

        bbox = self._bbox(sx, sy, gx, gy, w, h, res)
        search = astar_cost_grid_2d(
            cost,
            (sx, sy),
            (gx, gy),
            connectivity=p.connectivity,
            bbox=bbox,
            turn_penalty=p.turn_penalty,
            max_expansions=p.max_expansions,
        )
        if not search.ok:
            return PlanResult(
                status=PlanStatus.NO_PATH,
                message=f"A* unreachable (expanded={search.expanded})",
                artifacts={"expanded": search.expanded},
            )

        cells: Sequence[Tuple[int, int]] = search.path
        if p.los_smoothing:
            # Simplify rather than string-pull: A*'s cost already centres the
            # route, so we keep its shape (and clearance) and only drop the
            # redundant near-collinear cells. String-pulling would make the path
            # taut and undo the centring at corners.
            cells = simplify_path_cells(cells, lethal, self._simplify_tol_cells(res))

        pts = [Pose2D(*world.grid_to_world(cx, cy)) for cx, cy in cells]
        pts = split_long_segments_2d(pts, p.waypoint_spacing_m)
        pts = self._trim_start(pts, request.start, p.start_skip_m)
        if len(pts) < 2:
            # Start and goal share a cell (or trimming consumed the path):
            # the robot is effectively already there. Emit a valid 2-point path.
            pts = [request.start, request.goal]

        return PlanResult(
            status=PlanStatus.SUCCESS,
            path=Path2D(
                points=tuple(pts),
                frame_id=world.frame_id,
                metadata={
                    "planner": self.name,
                    "expanded": search.expanded,
                    "corners": len(cells),
                },
            ),
            message=f"A* success, expanded={search.expanded}, waypoints={len(pts)}",
        )

    # ------------------------------------------------------------------
    # Collision re-check (for lazy replanning by the caller)
    # ------------------------------------------------------------------
    def path_collides(self, world: OccupancyGrid2D, points: Sequence[Pose2D]) -> bool:
        """True if any segment of ``points`` crosses an inflated obstacle.

        Reuses the cached cost map for ``world``; intended for a node that
        wants to replan only when its current path becomes blocked.
        """
        if len(points) < 2:
            return False
        _, lethal, _ = self.cost_for(world)
        h, w = lethal.shape
        cells = [world.world_to_grid(pt.x, pt.y) for pt in points]
        for (x0, y0), (x1, y1) in zip(cells[:-1], cells[1:]):
            if not (0 <= x0 < w and 0 <= y0 < h and 0 <= x1 < w and 0 <= y1 < h):
                continue
            if not line_of_sight_clear(lethal, x0, y0, x1, y1):
                return True
        return False

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _bbox(
        self, sx: int, sy: int, gx: int, gy: int, w: int, h: int, res: float
    ) -> Tuple[int, int, int, int]:
        margin = max(1, int(round(self.params.search_margin_m / res)))
        xmin = max(0, min(sx, gx) - margin)
        xmax = min(w, max(sx, gx) + margin + 1)
        ymin = max(0, min(sy, gy) - margin)
        ymax = min(h, max(sy, gy) + margin + 1)
        return xmin, xmax, ymin, ymax

    @staticmethod
    def _trim_start(
        pts: List[Pose2D], start: Pose2D, skip_m: float
    ) -> List[Pose2D]:
        if skip_m <= 0:
            return pts
        while len(pts) > 1 and hypot(pts[0].x - start.x, pts[0].y - start.y) < skip_m:
            pts.pop(0)
        return pts
