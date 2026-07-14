"""Weighted A* planner on a 2D OccupancyGrid2D, with smoothing.

This is the ROS-free core behind the FALCON ``astar_planner`` node. It turns a
2D occupancy grid (FREE / OCCUPIED / UNKNOWN) plus start/goal world poses into a
clean, corner-preserving set of world waypoints by composing the reusable core
primitives:

    occupancy --build_cost_grid--> float cost map (inflation + UNKNOWN weight)
              --astar_cost_grid_2d--> grid cell path (bbox-restricted, octile)
              --los_smooth_cells----> any-angle corners
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
from ..common.corner_rounding_2d import round_corners_2d
from ..common.grid_geometry_2d import (
    dilate_mask,
    line_of_sight_clear,
    los_smooth_cells,
    snap_to_free_cell,
)
from ..common.utils_2d import split_long_segments_2d


def build_cost_grid(
    grid: OccupancyGrid2D, params: WeightedAStarParams
) -> Tuple[np.ndarray, np.ndarray]:
    """Build the float cost map and inflated occupancy mask from a grid.

    Args:
        grid: Source occupancy grid (uses ``grid.values`` for OCC/UNKNOWN).
        params: Weighting / inflation parameters.

    Returns:
        ``(cost, occ)`` where ``cost`` is an ``(H, W)`` float array
        (1.0 free, ``inf`` blocked, ``unknown_cost`` for traversable UNKNOWN)
        and ``occ`` is the inflated boolean obstacle mask used for LOS checks.
    """
    data = grid.grid
    occ = data == grid.values.occupied
    n = max(0, int(round(params.inflate_radius_m / grid.resolution)))
    if n > 0:
        occ = dilate_mask(occ, n)

    cost = np.ones(data.shape, dtype=np.float64)
    cost[occ] = np.inf
    unk = (data == grid.values.unknown) & ~occ
    cost[unk] = np.inf if params.unknown_blocked else float(params.unknown_cost)
    return cost, occ


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
        self._cache_occ: Optional[np.ndarray] = None

    # ------------------------------------------------------------------
    # Cost cache (keyed on grid object identity)
    # ------------------------------------------------------------------
    def cost_for(self, grid: OccupancyGrid2D) -> Tuple[np.ndarray, np.ndarray]:
        """Return ``(cost, occ)`` for ``grid``, rebuilding only when it changes."""
        if self._cache_grid is not grid or self._cache_cost is None:
            self._cache_cost, self._cache_occ = build_cost_grid(grid, self.params)
            self._cache_grid = grid
        return self._cache_cost, self._cache_occ

    def invalidate_cache(self) -> None:
        """Drop the cached cost map (e.g. on an explicit replan request)."""
        self._cache_grid = None
        self._cache_cost = None
        self._cache_occ = None

    # ------------------------------------------------------------------
    # Planning
    # ------------------------------------------------------------------
    def plan(self, request: PlanRequest, world: OccupancyGrid2D) -> PlanResult:
        p = self.params
        res = world.resolution
        cost, occ = self.cost_for(world)
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
            cells = los_smooth_cells(cells, occ)

        pts = [Pose2D(*world.grid_to_world(cx, cy)) for cx, cy in cells]
        # Round corners BEFORE splitting: chamfering needs the true leg lengths,
        # which segment-splitting would hide behind collinear midpoints.
        if p.corner_round:
            pts = round_corners_2d(pts, p, clear_fn=self._clear_fn(world, occ))
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
    def path_collides(
        self,
        world: OccupancyGrid2D,
        points: Sequence[Pose2D],
        passable_start: Optional[Pose2D] = None,
    ) -> bool:
        """True if any segment of ``points`` crosses an inflated obstacle.

        Reuses the cached cost map for ``world``; intended for a node that
        wants to replan only when its current path becomes blocked.

        Args:
            world: The occupancy grid to test against.
            points: World waypoints of the path to check.
            passable_start: If given, the drone's own footprint around this world
                point is treated as free for the duration of the check. The drone
                routinely sits inside the obstacle-inflation *skirt* (its inscribed
                radius overlaps the inflated cells), so without this exemption the
                cells the drone occupies read as blocked and the path "collides" on
                every frame -- a false positive that would force an endless replan.
                Only the inflated *skirt* within ``inflate_radius_m`` of the point
                is cleared; genuinely occupied wall cells are never cleared, so a
                real obstacle right next to (or ahead of) the drone is still
                detected. This mirrors the planner's own start-passable override.

        Returns:
            True iff some segment crosses an inflated (lethal) cell.
        """
        if len(points) < 2:
            return False
        _, occ = self.cost_for(world)
        if passable_start is not None:
            occ = self._exempt_footprint(world, occ, passable_start)
        h, w = occ.shape
        cells = [world.world_to_grid(pt.x, pt.y) for pt in points]
        for (x0, y0), (x1, y1) in zip(cells[:-1], cells[1:]):
            if not (0 <= x0 < w and 0 <= y0 < h and 0 <= x1 < w and 0 <= y1 < h):
                continue
            if not line_of_sight_clear(occ, x0, y0, x1, y1):
                return True
        return False

    def _exempt_footprint(self, world, occ, center):
        """Return a COPY of ``occ`` with the drone's own footprint skirt cleared.

        The drone's own body inflates the cells around it, so those cells read as
        lethal even though the drone is physically standing there. Within the
        robot's inscribed radius of ``center`` we clear only cells that are
        inflated but NOT truly ``occupied``: genuine walls stay lethal (so a real
        obstacle next to or ahead of the drone is still detected), while the
        drone's own skirt is ignored. Works on a copy so the cached mask A* shares
        is never mutated.
        """
        occ = occ.copy()
        n = int(round(self.params.inflate_radius_m / world.resolution))
        if n <= 0:
            return occ
        h, w = occ.shape
        cx, cy = world.world_to_grid(center.x, center.y)
        true_occ = world.grid == world.values.occupied
        for yy in range(max(0, cy - n), min(h, cy + n + 1)):
            dy2 = (yy - cy) ** 2
            for xx in range(max(0, cx - n), min(w, cx + n + 1)):
                if (xx - cx) ** 2 + dy2 > n * n:
                    continue
                if occ[yy, xx] and not true_occ[yy, xx]:
                    occ[yy, xx] = False
        return occ

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
    def _clear_fn(world: OccupancyGrid2D, occ: np.ndarray):
        """Closure ``(a, b) -> bool``: is the world segment a->b obstacle-free?

        Lets the ROS-free corner rounder reject a cut that would clip an
        obstacle, without it knowing anything about the grid.
        """
        h, w = occ.shape

        def clear(a: Pose2D, b: Pose2D) -> bool:
            x0, y0 = world.world_to_grid(a.x, a.y)
            x1, y1 = world.world_to_grid(b.x, b.y)
            if not (0 <= x0 < w and 0 <= y0 < h and 0 <= x1 < w and 0 <= y1 < h):
                return False
            return line_of_sight_clear(occ, x0, y0, x1, y1)

        return clear

    @staticmethod
    def _trim_start(
        pts: List[Pose2D], start: Pose2D, skip_m: float
    ) -> List[Pose2D]:
        if skip_m <= 0:
            return pts
        while len(pts) > 1 and hypot(pts[0].x - start.x, pts[0].y - start.y) < skip_m:
            pts.pop(0)
        return pts
