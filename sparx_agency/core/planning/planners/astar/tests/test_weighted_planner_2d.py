"""Tests for the weighted 2D A* planner and its kernels.

Run:
    .venv/bin/python -m pytest \
        sparx_agency/core/planning/planners/astar/tests/test_weighted_planner_2d.py
"""
from __future__ import annotations

import math

import numpy as np

from sparx_agency.core.common.types import Pose2D, PlanStatus
from sparx_agency.core.planning.environment import (
    OccupancyGrid2D,
    OccupancyGrid2DParams,
    OccupancyValues,
)
from sparx_agency.core.planning.interfaces.planner import PlanRequest
from sparx_agency.core.planning.planners.astar.algorithm_2d import astar_cost_grid_2d
from sparx_agency.core.planning.planners.astar.params import WeightedAStarParams
from sparx_agency.core.planning.planners.astar.weighted_planner_2d import (
    WeightedAStarPlanner2D,
    build_cost_grid,
)
from sparx_agency.core.planning.planners.common.grid_geometry_2d import (
    dilate_mask,
    line_of_sight_clear,
    los_smooth_cells,
    simplify_path_cells,
    snap_to_free_cell,
)

# BEV value convention published by the FALCON bev node.
VALUES = OccupancyValues(free=0, occupied=100, unknown=-1)


def _grid(arr: np.ndarray, res: float = 0.1) -> OccupancyGrid2D:
    return OccupancyGrid2D(
        arr.astype(np.int16),
        OccupancyGrid2DParams(resolution=res, origin_x=0.0, origin_y=0.0, frame_id="world"),
        values=VALUES,
    )


def _free(h: int, w: int) -> np.ndarray:
    return np.zeros((h, w), dtype=np.int16)


# --------------------------------------------------------------------------
# kernel
# --------------------------------------------------------------------------
def test_cost_kernel_diagonal_is_cheaper_than_cardinal():
    cost = np.ones((5, 5))
    res = astar_cost_grid_2d(cost, (0, 0), (4, 4), connectivity=8)
    assert res.ok
    # 8-connected straight diagonal: 5 cells (incl. both endpoints), 4 steps.
    assert res.path[0] == (0, 0) and res.path[-1] == (4, 4)
    assert len(res.path) == 5


def test_cost_kernel_blocks_inf_and_routes_around():
    cost = np.ones((5, 5))
    cost[:, 2] = np.inf  # vertical wall at x=2
    cost[0, 2] = 1.0      # leave a gap at the top
    res = astar_cost_grid_2d(cost, (0, 2), (4, 2), connectivity=8)
    assert res.ok
    # Must not pass through any inf cell.
    assert all(math.isfinite(cost[y, x]) for x, y in res.path)


def test_cost_kernel_bbox_can_make_goal_unreachable():
    cost = np.ones((10, 10))
    # Tiny window around start only: goal far outside -> no path.
    res = astar_cost_grid_2d(cost, (0, 0), (9, 9), connectivity=8, bbox=(0, 2, 0, 2))
    assert not res.ok


# --------------------------------------------------------------------------
# grid geometry helpers
# --------------------------------------------------------------------------
def test_dilate_mask_grows_one_ring():
    m = np.zeros((5, 5), dtype=bool)
    m[2, 2] = True
    d = dilate_mask(m, 1)
    assert d[1, 2] and d[3, 2] and d[2, 1] and d[2, 3]
    assert not d[0, 2]  # only one ring


def test_line_of_sight_and_smoothing():
    occ = np.zeros((5, 9), dtype=bool)
    cells = [(0, 0), (1, 0), (2, 0), (3, 0), (4, 0)]  # a straight staircase-free run
    assert line_of_sight_clear(occ, 0, 0, 4, 0)
    smoothed = los_smooth_cells(cells, occ)
    assert smoothed == [(0, 0), (4, 0)]  # collapses to endpoints


def test_simplify_collapses_straight_run_and_keeps_corner():
    lethal = np.zeros((5, 9), dtype=bool)
    # An L: straight run east, then a real 90-degree corner, then north.
    cells = [(0, 0), (1, 0), (2, 0), (3, 0), (3, 1), (3, 2)]
    out = simplify_path_cells(cells, lethal, epsilon_cells=0.5)
    assert out == [(0, 0), (3, 0), (3, 2)]  # the corner survives, collinear runs go


def test_simplify_preserves_offset_unlike_string_pulling():
    # A path that detours away from the straight chord (mimics a centred route).
    lethal = np.zeros((9, 9), dtype=bool)
    cells = [(0, 0), (1, 3), (2, 4), (3, 4), (4, 4), (5, 4), (6, 3), (8, 0)]
    simplified = simplify_path_cells(cells, lethal, epsilon_cells=0.5)
    # The mid waypoints stay up at the detour height (y~4), not pulled to the chord.
    mid = [c for c in simplified if 1 <= c[0] <= 7]
    assert mid and max(y for _, y in mid) >= 4
    # String-pulling, by contrast, would collapse the whole thing to the chord.
    pulled = los_smooth_cells(cells, lethal)
    assert pulled == [(0, 0), (8, 0)]


def test_snap_to_free_cell():
    cost = np.full((5, 5), np.inf)
    cost[0, 0] = 1.0
    assert snap_to_free_cell(cost, 2, 2, max_radius=1) is None
    assert snap_to_free_cell(cost, 1, 1, max_radius=2) == (0, 0)


# --------------------------------------------------------------------------
# cost build
# --------------------------------------------------------------------------
def test_build_cost_grid_inflation_and_unknown():
    arr = _free(7, 7)
    arr[3, 3] = 100   # occupied
    arr[0, 0] = -1    # unknown
    grid = _grid(arr, res=0.1)
    # clearance_weight=0 isolates the inflation/unknown logic from the soft layer.
    params = WeightedAStarParams(
        inflate_radius_m=0.1, unknown_cost=5.0, unknown_blocked=False, clearance_weight=0.0
    )
    cost, lethal, clearance = build_cost_grid(grid, params)
    assert math.isinf(cost[3, 3])         # occupied -> inf
    assert math.isinf(cost[2, 3])         # inflated by 1 cell (Euclidean)
    assert cost[0, 0] == 5.0              # unknown weighted
    assert lethal[3, 3] and lethal[2, 3]
    assert clearance[3, 3] == 0.0         # obstacle cell has zero clearance


def test_build_cost_grid_unknown_blocked():
    arr = _free(5, 5)
    arr[1, 1] = -1
    grid = _grid(arr)
    cost, _, _ = build_cost_grid(
        grid, WeightedAStarParams(unknown_blocked=True, inflate_radius_m=0.0)
    )
    assert math.isinf(cost[1, 1])


# --------------------------------------------------------------------------
# planner end-to-end
# --------------------------------------------------------------------------
def _request(sx, sy, gx, gy):
    return PlanRequest(start=Pose2D(sx, sy, 0.0), goal=Pose2D(gx, gy, 0.0), frame_id="world")


def test_planner_straight_path_is_smoothed_to_two_points():
    grid = _grid(_free(40, 40), res=0.1)
    planner = WeightedAStarPlanner2D(
        WeightedAStarParams(inflate_radius_m=0.0, waypoint_spacing_m=10.0, start_skip_m=0.0)
    )
    res = planner.plan(_request(0.5, 0.5, 3.5, 0.5), grid)
    assert res.ok
    # Open straight line, large spacing -> just the two endpoints survive.
    assert len(res.path.points) == 2
    assert res.path.points[-1].distance_to(Pose2D(3.5, 0.5)) < 0.2


def test_planner_routes_around_wall():
    arr = _free(40, 40)
    arr[:30, 20] = 100  # wall from bottom leaving a gap near the top
    grid = _grid(arr, res=0.1)
    planner = WeightedAStarPlanner2D(
        WeightedAStarParams(inflate_radius_m=0.0, search_margin_m=4.0, start_skip_m=0.0)
    )
    res = planner.plan(_request(0.5, 0.5, 3.5, 0.5), grid)
    assert res.ok
    # The smoothed path must detour upward (above the wall gap).
    assert max(pt.y for pt in res.path.points) > 2.5


def test_planner_snaps_blocked_goal():
    arr = _free(40, 40)
    arr[5, 5] = 100
    grid = _grid(arr, res=0.1)
    planner = WeightedAStarPlanner2D(
        WeightedAStarParams(inflate_radius_m=0.0, goal_snap_radius_m=0.5, start_skip_m=0.0)
    )
    # Goal world (0.55, 0.55) -> cell (5, 5), which is blocked: must snap + succeed.
    res = planner.plan(_request(0.5, 0.5, 0.55, 0.55), grid)
    assert res.status == PlanStatus.SUCCESS


def test_planner_no_path_when_goal_walled_in():
    arr = _free(40, 40)
    arr[8:13, 8:13] = 100
    arr[10, 10] = 0  # free pocket fully enclosed by occupied cells
    grid = _grid(arr, res=0.1)
    planner = WeightedAStarPlanner2D(
        WeightedAStarParams(inflate_radius_m=0.0, goal_snap_radius_m=0.0, search_margin_m=6.0)
    )
    res = planner.plan(_request(0.5, 0.5, 1.05, 1.05), grid)
    assert res.status == PlanStatus.NO_PATH


def test_planner_path_collides_detects_new_obstacle():
    grid = _grid(_free(40, 40), res=0.1)
    planner = WeightedAStarPlanner2D(WeightedAStarParams(inflate_radius_m=0.0))
    clear_pts = (Pose2D(0.5, 0.5), Pose2D(3.5, 0.5))
    assert not planner.path_collides(grid, clear_pts)

    arr = _free(40, 40)
    arr[5, 20] = 100  # drop an obstacle on the straight line
    blocked = _grid(arr, res=0.1)
    assert planner.path_collides(blocked, clear_pts)


def _mean_clearance(planner, grid, points, x_lo=None, x_hi=None):
    """Mean obstacle clearance (m) at path waypoints, optionally within [x_lo, x_hi]."""
    _, _, clearance = planner.cost_for(grid)
    h, w = clearance.shape
    vals = []
    for pt in points:
        if x_lo is not None and not (x_lo < pt.x < x_hi):
            continue
        gx, gy = grid.world_to_grid(pt.x, pt.y)
        if 0 <= gx < w and 0 <= gy < h:
            vals.append(float(clearance[gy, gx]))
    return sum(vals) / len(vals) if vals else 0.0


# --------------------------------------------------------------------------
# clearance shaping: centering, narrow-gap rejection, wall standoff
# --------------------------------------------------------------------------
def test_route_runs_through_corridor_middle():
    # Horizontal corridor 11 cells tall (rows 0 and 10 are walls), 60 wide.
    # Inner free rows 1..9; centre row 5 has the most clearance (~0.5 m).
    arr = _free(11, 60)
    arr[0, :] = 100
    arr[10, :] = 100
    grid = _grid(arr, res=0.1)
    planner = WeightedAStarPlanner2D(WeightedAStarParams(
        inflate_radius_m=0.1, clearance_weight=4.0, clearance_margin_m=0.6,
        search_margin_m=2.0, start_skip_m=0.0, waypoint_spacing_m=0.5,
    ))
    # Start and goal deliberately offset to the bottom lane (row 2, clearance 0.2 m).
    res = planner.plan(_request(0.25, 0.25, 5.75, 0.25), grid)
    assert res.ok
    # Across the long middle stretch the route must ride near the centre line,
    # not the offset bottom lane it started/ended on.
    assert _mean_clearance(planner, grid, res.path.points, 1.0, 5.0) > 0.4


def test_centering_beats_binary_planner_on_clearance():
    arr = _free(11, 60)
    arr[0, :] = 100
    arr[10, :] = 100
    grid = _grid(arr, res=0.1)
    req = _request(0.25, 0.25, 5.75, 0.25)

    centered = WeightedAStarPlanner2D(WeightedAStarParams(
        inflate_radius_m=0.1, clearance_weight=4.0, clearance_margin_m=0.6,
        search_margin_m=2.0, start_skip_m=0.0, waypoint_spacing_m=0.5))
    binary = WeightedAStarPlanner2D(WeightedAStarParams(
        inflate_radius_m=0.1, clearance_weight=0.0,
        search_margin_m=2.0, start_skip_m=0.0, waypoint_spacing_m=0.5))

    r_c = centered.plan(req, grid)
    r_b = binary.plan(req, grid)
    assert r_c.ok and r_b.ok
    # The centered route keeps clearly more clearance from the walls on average.
    assert _mean_clearance(centered, grid, r_c.path.points) > \
           _mean_clearance(binary, grid, r_b.path.points) + 0.1


def test_narrow_gap_is_not_threaded():
    # Two walls leaving a single free column (a one-cell gap) at x=20.
    arr = _free(40, 40)
    arr[:, 20] = 0          # the gap column is free...
    arr[:, :20] = 0
    arr[15:25, 19] = 100    # ...but flanked by obstacles on both sides
    arr[15:25, 21] = 100
    grid = _grid(arr, res=0.1)
    # Robot radius 0.3 m >> half the 0.1 m gap: the gap must be treated as closed.
    planner = WeightedAStarPlanner2D(WeightedAStarParams(
        inflate_radius_m=0.3, clearance_weight=3.0, search_margin_m=4.0,
        goal_snap_radius_m=0.0, start_skip_m=0.0))
    _, lethal, _ = planner.cost_for(grid)
    assert lethal[20, 20]  # the lone free cell in the gap is now lethal (inflated shut)


def test_gray_is_a_flat_cost_and_a_centering_boundary():
    # Right half of the map is gray (UNKNOWN); no real obstacles.
    arr = _free(20, 20)
    arr[:, 10:] = -1
    grid = _grid(arr, res=0.1)
    p = WeightedAStarParams(inflate_radius_m=0.0, clearance_weight=3.0,
                            clearance_margin_m=0.5, unknown_cost=5.0)
    cost, lethal, _ = build_cost_grid(grid, p)
    assert cost[5, 15] == 5.0 and not lethal[5, 15]   # gray: flat high cost, not lethal
    assert cost[5, 9] > 1.0                            # known-free next to gray is penalised
    assert cost[5, 2] == 1.0                           # known-free far from gray is cheap


def test_route_stays_in_known_free_not_gray():
    # Known-free band (rows 1..6) between a wall (row 0) and gray (rows 7+).
    arr = _free(20, 40)
    arr[0, :] = 100      # bottom wall
    arr[7:, :] = -1      # gray above the explored band
    grid = _grid(arr, res=0.1)
    planner = WeightedAStarPlanner2D(WeightedAStarParams(
        inflate_radius_m=0.0, clearance_weight=4.0, clearance_margin_m=0.6,
        unknown_cost=8.0, search_margin_m=2.0, start_skip_m=0.0, waypoint_spacing_m=0.4))
    # Start/goal sit near the wall (row 1).
    res = planner.plan(_request(0.25, 0.15, 3.75, 0.15), grid)
    assert res.ok
    # No waypoint may land on a gray cell — the route must not dive into unknown.
    for pt in res.path.points:
        gx, gy = grid.world_to_grid(pt.x, pt.y)
        assert arr[gy, gx] != -1, "path entered gray at (%d,%d)" % (gx, gy)
    # The middle stretch rides the centre of the KNOWN-FREE band (~row 3-4),
    # not the wall (row 1) and not the gray frontier (row 6).
    mid_rows = [grid.world_to_grid(p.x, p.y)[1] for p in res.path.points if 1.0 < p.x < 3.0]
    assert mid_rows and 2.0 < (sum(mid_rows) / len(mid_rows)) < 5.5


def test_clearance_layer_off_recovers_binary_behaviour():
    # With clearance_weight=0 the cost map is the classic free(1.0)/blocked(inf).
    arr = _free(10, 10)
    arr[5, 5] = 100
    grid = _grid(arr, res=0.1)
    cost, _, _ = build_cost_grid(grid, WeightedAStarParams(
        inflate_radius_m=0.0, clearance_weight=0.0))
    finite = cost[np.isfinite(cost)]
    assert np.all(finite == 1.0)  # every passable cell is exactly free cost


def test_path_collides_with_inflation_detects_obstacle_on_route():
    # Reproduces the FALCON collision-replan path: plan across open space, then a
    # cell on the route becomes occupied on a *fresh* grid object (as every BEV
    # frame is). path_collides must flag it so the node can replan.
    planner = WeightedAStarPlanner2D(WeightedAStarParams(
        inflate_radius_m=0.5, clearance_weight=3.0, clearance_margin_m=0.8,
        start_skip_m=0.0, waypoint_spacing_m=1.5))
    res = planner.plan(_request(0.5, 3.0, 5.5, 3.0), _grid(_free(60, 60), res=0.1))
    assert res.ok
    arr = _free(60, 60)
    arr[30, 30] = 100   # obstacle on the straight route (y ~ 3.0 -> row 30)
    assert planner.path_collides(_grid(arr, res=0.1), res.path.points)
    # And a fresh grid with no new obstacle must NOT report a collision.
    assert not planner.path_collides(_grid(_free(60, 60), res=0.1), res.path.points)


def test_cost_cache_reused_until_grid_changes():
    grid = _grid(_free(20, 20))
    planner = WeightedAStarPlanner2D()
    c1, _, _ = planner.cost_for(grid)
    c2, _, _ = planner.cost_for(grid)
    assert c1 is c2  # same grid object -> cached
    planner.invalidate_cache()
    c3, _, _ = planner.cost_for(grid)
    assert c3 is not c1
