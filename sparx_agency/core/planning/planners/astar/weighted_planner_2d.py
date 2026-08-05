"""Weighted A* planner on a 2D OccupancyGrid2D, with smoothing.

This is the ROS-free core behind the FALCON ``astar_planner`` node. It turns a
2D occupancy grid (FREE / OCCUPIED / UNKNOWN) plus start/goal world poses into a
clean, corner-preserving set of world waypoints by composing the reusable core
primitives:

    occupancy --build_cost_grid------> cost + lethal mask + clearance field
              --astar_cost_grid_2d---> grid cell path (bbox-restricted, octile)
              --simplify_path_cells--> fewer waypoints, same centred shape
              --split_long_segments_2d--> spaced world waypoints

Cost-map construction (inflation, clearance shaping, confidence-weighted
lethality) is a separate responsibility and lives in :mod:`.cost_grid_2d`; it is
re-exported here so ``from ...weighted_planner_2d import build_cost_grid`` keeps
working.

The path is thinned with Douglas-Peucker rather than string-pulled. String-
pulling makes a route taut, which would drag a clearance-centred path straight
back onto the wall it was just pushed off; DP only deletes points that are
near-collinear with their kept neighbours, so the centring survives.

The planner is stateful only to cache the cost map per input grid (keyed on
object identity), so a plan followed by a collision re-check on the same grid
does not inflate obstacles twice. It owns no ROS or world-IO concepts.
"""
from __future__ import annotations

from math import cos, hypot, sin
from typing import List, Optional, Sequence, Tuple

import numpy as np

from sparx_agency.core.common.types import Pose2D, Path2D, PlanResult, PlanStatus
from sparx_agency.core.planning.environment import OccupancyGrid2D
from sparx_agency.core.planning.interfaces.planner import PlanRequest

from .params import WeightedAStarParams
from .algorithm_2d import astar_cost_grid_2d
from .cost_grid_2d import (
    CostFields,
    assemble_cost_grid,
    build_collision_mask,
    build_cost_fields,
    build_cost_grid,
    split_by_confidence,
)
from ..common.corner_rounding_2d import round_corners_2d
from ..common.grid_geometry_2d import (
    line_of_sight_clear,
    simplify_path_cells,
    snap_to_free_cell,
)
from ..common.utils_2d import split_long_segments_2d

__all__ = ["WeightedAStarPlanner2D", "build_cost_grid"]


def _yaw_to_move(yaw: float) -> Tuple[int, int]:
    """Quantise a world heading (rad) to one of the 8 grid moves ``(dx, dy)``.

    The BEV grid's +x/+y match world +x/+y (``world_to_grid`` is an affine shift),
    so a heading ``(cos yaw, sin yaw)`` rounds directly to a grid step. Snapping to
    the nearest 45 deg first guarantees a unit move in ``{-1,0,1}`` (never
    ``(0,0)``)."""
    from math import pi
    q = round(yaw / (pi / 4.0)) * (pi / 4.0)
    return int(round(cos(q))), int(round(sin(q)))


class WeightedAStarPlanner2D:
    """Weighted, bbox-restricted A* with line-of-sight smoothing.

    Implements the ``BasePlanner`` protocol: ``plan(request, world)`` where
    ``world`` is an :class:`OccupancyGrid2D`.
    """

    name: str = "weighted_astar_2d"

    def __init__(self, params: Optional[WeightedAStarParams] = None) -> None:
        self.params = params or WeightedAStarParams()
        self._confidence: Optional[np.ndarray] = None
        # Standoff the last successful plan was flown at. Collision detection must
        # test the route against the radius it was PLANNED at: checking a relaxed
        # route against the preferred radius would report a collision every frame.
        self._last_inflate_m: float = self.params.inflate_radius_m
        self._cache_grid: Optional[OccupancyGrid2D] = None
        self._cache_fields: Optional[CostFields] = None
        self._cache_cost: Optional[np.ndarray] = None
        self._cache_lethal: Optional[np.ndarray] = None
        self._cache_collision: Optional[np.ndarray] = None
        self._cache_collision_grid: Optional[OccupancyGrid2D] = None
        self._cache_collision_radius: Optional[float] = None

    # ------------------------------------------------------------------
    # Cost cache (keyed on grid object identity)
    # ------------------------------------------------------------------
    def set_confidence(self, confidence: Optional[np.ndarray]) -> None:
        """Install the per-cell OCCUPIED confidence grid and drop the cache.

        Feeds ``params.lethal_confidence``: an OCCUPIED cell below that
        confidence is costly rather than blocking, so single-frame depth speckle
        cannot make the map infeasible. Pass None to go back to treating every
        OCCUPIED cell as a confirmed obstacle.

        Args:
            confidence: ``(H, W)`` array in ``[0, 1]`` co-registered with the
                grids subsequently passed to :meth:`plan`, or None.
        """
        self._confidence = confidence
        self.invalidate_cache()

    def fields_for(self, grid: OccupancyGrid2D) -> CostFields:
        """Cached distance transforms for ``grid`` (independent of the standoff)."""
        if self._cache_grid is not grid or self._cache_fields is None:
            self._cache_fields = build_cost_fields(grid, self.params, self._confidence)
            self._cache_cost = None
            self._cache_lethal = None
            self._cache_grid = grid
        return self._cache_fields

    def cost_for(
        self, grid: OccupancyGrid2D
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """``(cost, lethal, clearance)`` at the PREFERRED standoff, cached."""
        fields = self.fields_for(grid)
        if self._cache_cost is None:
            self._cache_cost, self._cache_lethal = assemble_cost_grid(
                fields, self.params, self.params.inflate_radius_m)
        return self._cache_cost, self._cache_lethal, fields.clearance

    @property
    def last_inflate_m(self) -> float:
        """Standoff the most recent successful :meth:`plan` achieved (meters)."""
        return self._last_inflate_m

    def collision_mask_for(self, grid: OccupancyGrid2D) -> np.ndarray:
        """Strict lethal mask for collision DETECTION, cached on its own.

        Differs from the planning mask in two deliberate ways. It ignores
        ``params.lethal_confidence``, so an unconfirmed cell stays *passable to
        the search* (depth speckle cannot make the map infeasible) while
        remaining *visible to detection*, leaving the caller's own confirm and
        ceiling gates to decide whether to act. And it tests against
        :attr:`last_inflate_m` rather than the preferred radius, so a route that
        was deliberately squeezed through a pinch is judged by the standoff it
        was actually planned at.

        Cached independently of the cost map because the collision re-check runs
        once per map frame while planning runs on a much slower timer: this way a
        per-frame check costs one bounded transform, not a whole cost map.
        """
        if (self._cache_collision_grid is not grid
                or self._cache_collision_radius != self._last_inflate_m
                or self._cache_collision is None):
            self._cache_collision = build_collision_mask(
                grid, self.params, self._last_inflate_m)
            self._cache_collision_grid = grid
            self._cache_collision_radius = self._last_inflate_m
        return self._cache_collision

    def invalidate_cache(self) -> None:
        """Drop the cached cost map (e.g. on an explicit replan request)."""
        self._cache_grid = None
        self._cache_fields = None
        self._cache_cost = None
        self._cache_lethal = None
        self._cache_collision = None
        self._cache_collision_grid = None
        self._cache_collision_radius = None

    # ------------------------------------------------------------------
    # Planning
    # ------------------------------------------------------------------
    def plan(self, request: PlanRequest, world: OccupancyGrid2D) -> PlanResult:
        """Plan at the preferred standoff, relaxing only as far as necessary.

        Walks :meth:`standoff_ladder` from ``inflate_radius_m`` down to
        ``inflate_floor_m`` and returns the FIRST success -- which is by
        construction the safest feasible route, since clearance only decreases
        down the ladder. The common case costs one search; only a pinched
        corridor pays for more. The ladder always restarts at the preferred
        standoff, so a relaxation is never carried over between calls.

        On total failure the phantom probe runs (see :class:`WeightedAStarParams`)
        and its verdict is attached to the returned failure.

        Returns:
            A :class:`PlanResult` whose ``artifacts["inflate_used_m"]`` records
            the standoff achieved (also in ``path.metadata``).
        """
        fields = self.fields_for(world)
        failed = None
        for inflate_m in self.standoff_ladder():
            if inflate_m == self.params.inflate_radius_m:
                cost, lethal, _ = self.cost_for(world)      # cached preferred rung
            else:
                cost, lethal = assemble_cost_grid(fields, self.params, inflate_m)
            result = self._plan_at(request, world, cost, lethal, inflate_m)
            if result.ok:
                self._last_inflate_m = inflate_m
                return result
            failed = result
        return self._probe_for_phantom(request, world, fields, failed)

    def standoff_ladder(self) -> List[float]:
        """Standoffs to attempt, preferred first, descending to the flyable floor."""
        p = self.params
        ladder = [p.inflate_radius_m]
        if p.inflate_floor_m >= p.inflate_radius_m or p.relax_step_m <= 0.0:
            return ladder
        k = 1
        while True:
            nxt = round(p.inflate_radius_m - k * p.relax_step_m, 6)
            if nxt <= p.inflate_floor_m + 1e-9:
                break
            ladder.append(nxt)
            k += 1
        ladder.append(p.inflate_floor_m)
        return ladder

    def _probe_for_phantom(
        self, request: PlanRequest, world: OccupancyGrid2D,
        fields: CostFields, failed: PlanResult,
    ) -> PlanResult:
        """Tag a fully-failed ladder with whether the blockage looks spurious.

        A route that appears only at a sub-airframe clearance means whatever is
        blocking is thin -- far more likely a mis-detected voxel than a wall. The
        probe path is NEVER returned (it cannot be flown); the caller is simply
        told to re-observe rather than to conclude the mission is dead.
        """
        p = self.params
        if p.probe_radius_m <= 0.0 or p.probe_radius_m >= p.inflate_floor_m:
            return failed
        cost, lethal = assemble_cost_grid(fields, p, p.probe_radius_m)
        probe = self._plan_at(request, world, cost, lethal, p.probe_radius_m)
        if not probe.ok:
            return failed
        # Where the probe route is least clear is where the obstruction is: the
        # cells that closed every flyable rung. Reporting it turns "something is
        # in the way" into "look here again".
        pinch, pinch_xy = self._tightest_point(
            world, fields.clearance, probe.path.points)
        artifacts = dict(failed.artifacts)
        artifacts["phantom_suspected"] = True
        artifacts["probe_radius_m"] = p.probe_radius_m
        artifacts["pinch_clearance_m"] = pinch
        artifacts["pinch_xy"] = pinch_xy
        where = (" at (%.2f, %.2f)" % pinch_xy) if pinch_xy else ""
        return PlanResult(
            status=failed.status,
            message=("%s; but the whole route re-derives at %.2fm clearance, pinched "
                     "to %.2fm%s -- thinner than the airframe, so most likely a "
                     "mis-detected voxel rather than a wall. Re-observe before "
                     "giving up."
                     % (failed.message, p.probe_radius_m, pinch, where)),
            artifacts=artifacts,
        )

    @staticmethod
    def _tightest_point(world, clearance, points):
        """``(clearance_m, (x, y))`` at the least-clear waypoint of ``points``."""
        h, w = clearance.shape
        best, best_xy = float("inf"), None
        for pt in points:
            gx, gy = world.world_to_grid(pt.x, pt.y)
            if 0 <= gx < w and 0 <= gy < h and float(clearance[gy, gx]) < best:
                best, best_xy = float(clearance[gy, gx]), (pt.x, pt.y)
        return best, best_xy

    def _plan_at(
        self, request: PlanRequest, world: OccupancyGrid2D,
        cost: np.ndarray, lethal: np.ndarray, inflate_m: float,
    ) -> PlanResult:
        """Run one full search at a single standoff ``inflate_m``."""
        p = self.params
        res = world.resolution
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

        # Start relaxation. On a noisy map the drone often reads as sitting inside
        # an obstacle: pressed against a wall (inside its inflation skirt) or, when
        # a depth frame paints an occupied cell under it, on an occupied cell. The
        # drone is PHYSICALLY at the start, so that space is passable -- never fail
        # to plan because the start reads blocked. Clear the drone's own inscribed
        # footprint (its own cell + its inflation skirt) so A* can step out toward
        # the free space just beyond the skirt. Genuine obstacles are NOT cleared,
        # so A* can never thread a real wall; if the drone is truly walled in, A*
        # returns NO_PATH and the caller falls back to STOP + a reactive planner.
        # Only copy the shared cost cache when we actually override.
        start_blocked = not np.isfinite(cost[sy, sx])
        if start_blocked:
            cost = cost.copy()
            self._clear_start_footprint(cost, lethal, world, sx, sy, inflate_m)

        # Heading awareness: seed the search with the drone's facing (quantised to
        # a grid move) so the FIRST step is a turn like any other, and charge for
        # turning off it -- as a reversal penalty, as a per-radian rotation cost,
        # or both. Costs are converted to cells, the unit the search works in.
        # Off when both are 0 (the search then treats the start move as free).
        start_dir = None
        start_turn_penalty = 0.0
        start_turn_cost_rad = 0.0
        start_turn_radius = 0.0
        if p.heading_penalty_m > 0.0 or p.start_turn_cost_m_per_rad > 0.0:
            start_dir = _yaw_to_move(request.start.yaw)
            start_turn_penalty = p.heading_penalty_m / res
            start_turn_cost_rad = p.start_turn_cost_m_per_rad / res
            start_turn_radius = p.start_turn_radius_m / res

        bbox = self._bbox(sx, sy, gx, gy, w, h, res)
        search = astar_cost_grid_2d(
            cost,
            (sx, sy),
            (gx, gy),
            connectivity=p.connectivity,
            bbox=bbox,
            turn_penalty=p.turn_penalty,
            start_dir=start_dir,
            start_turn_penalty=start_turn_penalty,
            start_turn_cost_rad=start_turn_cost_rad,
            start_turn_radius=start_turn_radius,
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
            # Douglas-Peucker, NOT string-pulling: it only deletes points that are
            # near-collinear with their kept neighbours, so a clearance-centred
            # route keeps its offset instead of being pulled taut onto the wall.
            eps_cells = (p.path_simplify_m / res) if p.path_simplify_m > 0.0 else 1.0
            cells = simplify_path_cells(cells, lethal, eps_cells)

        pts = [Pose2D(*world.grid_to_world(cx, cy)) for cx, cy in cells]
        # Round corners BEFORE splitting: chamfering needs the true leg lengths,
        # which segment-splitting would hide behind collinear midpoints.
        if p.corner_round:
            pts = round_corners_2d(pts, p, clear_fn=self._clear_fn(world, lethal))
        pts = split_long_segments_2d(pts, p.waypoint_spacing_m)
        pts = self._trim_start(pts, request.start, p.start_skip_m)
        if len(pts) < 2:
            # Start and goal share a cell (or trimming consumed the path):
            # the robot is effectively already there. Emit a valid 2-point path.
            pts = [request.start, request.goal]

        # Safety net for a relaxed start: the drone's OWN cell is passable, but the
        # flown route must never cross a real (truly-occupied) cell. Skirt-only
        # footprint clearing already guarantees this (A* cannot traverse a genuine
        # obstacle), but validate the emitted waypoints against TRUE occupancy so a
        # degenerate on-wall start whose only route is straight through fails here
        # (-> NO_PATH -> the caller STOPs and hands off to a reactive planner)
        # rather than commanding the drone through a wall.
        if start_blocked and self._path_crosses_true_obstacle(world, pts, sx, sy):
            return PlanResult(
                status=PlanStatus.NO_PATH,
                message="start walled in: no route out without crossing an obstacle",
            )

        relaxed = inflate_m < p.inflate_radius_m
        return PlanResult(
            status=PlanStatus.SUCCESS,
            path=Path2D(
                points=tuple(pts),
                frame_id=world.frame_id,
                metadata={
                    "planner": self.name,
                    "expanded": search.expanded,
                    "corners": len(cells),
                    "inflate_used_m": inflate_m,
                },
            ),
            artifacts={"expanded": search.expanded, "inflate_used_m": inflate_m},
            message=("A* success, expanded=%d, waypoints=%d%s"
                     % (search.expanded, len(pts),
                        (", standoff RELAXED to %.2fm (preferred %.2fm)"
                         % (inflate_m, p.inflate_radius_m)) if relaxed else "")),
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
        lethal = self.collision_mask_for(world)
        if passable_start is not None:
            lethal = self._exempt_footprint(
                world, lethal, passable_start, self._last_inflate_m)
        h, w = lethal.shape
        cells = [world.world_to_grid(pt.x, pt.y) for pt in points]
        for (x0, y0), (x1, y1) in zip(cells[:-1], cells[1:]):
            if not (0 <= x0 < w and 0 <= y0 < h and 0 <= x1 < w and 0 <= y1 < h):
                continue
            if not line_of_sight_clear(lethal, x0, y0, x1, y1):
                return True
        return False

    def _clear_start_footprint(self, cost: np.ndarray, lethal: np.ndarray,
                               world: OccupancyGrid2D, sx: int, sy: int,
                               inflate_m: float) -> None:
        """Relax the drone's own footprint in ``cost`` (in place) so a blocked start
        can escape -- WITHOUT ever opening a real wall.

        Two things are cleared to free cost:

          * the drone's own cell -- it is physically there, so it is passable even
            when a noisy depth frame paints that single cell OCCUPIED;
          * cells in the inscribed disc that are inflated but NOT truly occupied
            (the drone's own inflation *skirt*).

        Genuine (truly-occupied) cells are left blocked. Because the disc radius is
        the inflation radius, the skirt the drone sits in is fully cleared and A*
        reaches the free space just past it -- while a real obstacle, or any
        occupied blob larger than the drone's own cell, stays lethal so A* can
        never thread a wall. The ``lethal`` mask used for path simplification / the
        collision re-check is untouched.
        """
        cost[sy, sx] = 1.0                       # the drone's own cell: it is there
        n = int(round(inflate_m / world.resolution))
        if n <= 0:
            return
        h, w = cost.shape
        y0, y1 = max(0, sy - n), min(h, sy + n + 1)
        x0, x1 = max(0, sx - n), min(w, sx + n + 1)
        ys, xs = np.ogrid[y0:y1, x0:x1]
        disc = (xs - sx) ** 2 + (ys - sy) ** 2 <= n * n
        true_occ = world.grid == world.values.occupied
        # inflation skirt only: lethal but not a genuine obstacle
        skirt = disc & lethal[y0:y1, x0:x1] & ~true_occ[y0:y1, x0:x1]
        cost[y0:y1, x0:x1][skirt] = 1.0

    def _path_crosses_true_obstacle(self, world: OccupancyGrid2D,
                                    points: Sequence[Pose2D], sx: int, sy: int) -> bool:
        """True if any segment of ``points`` crosses a TRULY-occupied cell (real
        obstacle), exempting only the drone's own cell ``(sx, sy)``.

        Checked against raw occupancy (no inflation): a relaxed start is allowed to
        sit on / next to its own inflation skirt, but the flown route must never run
        through a genuine wall. Used only when the start was relaxed.

        "Genuine" honours ``params.lethal_confidence``: a cell the map flags but
        has not confirmed is exactly the depth speckle the relaxation exists to
        fly through, so gating on raw occupancy here would veto every route the
        relaxation just made possible -- and it fires precisely when the drone is
        boxed in, which is when it most needs one."""
        occupied = world.grid == world.values.occupied
        confirmed, _ = split_by_confidence(
            occupied, self._confidence, self.params.lethal_confidence)
        true_occ = confirmed.copy()
        h, w = true_occ.shape
        if 0 <= sx < w and 0 <= sy < h:
            true_occ[sy, sx] = False
        cells = [world.world_to_grid(pt.x, pt.y) for pt in points]
        for (x0, y0), (x1, y1) in zip(cells[:-1], cells[1:]):
            if not (0 <= x0 < w and 0 <= y0 < h and 0 <= x1 < w and 0 <= y1 < h):
                continue
            if not line_of_sight_clear(true_occ, x0, y0, x1, y1):
                return True
        return False

    def _exempt_footprint(self, world, lethal, center, inflate_m):
        """Return a COPY of ``lethal`` with the drone's own footprint cleared.

        The drone's own body inflates the cells around it, so those cells read as
        lethal even though the drone is physically standing there. Two exemptions:

          * the drone's OWN cell is always cleared -- you cannot collide with the
            cell you occupy, even when a noisy frame paints it truly ``occupied``
            (this is what lets an escape path off a wall-hugging start not read as
            an instant collision at waypoint 0);
          * within the inscribed radius of ``center`` we additionally clear cells
            that are inflated but NOT truly ``occupied`` -- genuine walls in the
            skirt stay lethal, so a real obstacle next to or ahead of the drone is
            still detected, while the drone's own inflation skirt is ignored.

        Works on a copy so the cached mask A* shares is never mutated.
        """
        lethal = lethal.copy()
        h, w = lethal.shape
        cx, cy = world.world_to_grid(center.x, center.y)
        if 0 <= cx < w and 0 <= cy < h:
            lethal[cy, cx] = False
        n = int(round(inflate_m / world.resolution))
        if n <= 0:
            return lethal
        y0, y1 = max(0, cy - n), min(h, cy + n + 1)
        x0, x1 = max(0, cx - n), min(w, cx + n + 1)
        ys, xs = np.ogrid[y0:y1, x0:x1]
        disc = (xs - cx) ** 2 + (ys - cy) ** 2 <= n * n
        true_occ = world.grid == world.values.occupied
        skirt = disc & lethal[y0:y1, x0:x1] & ~true_occ[y0:y1, x0:x1]
        lethal[y0:y1, x0:x1][skirt] = False
        return lethal

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
