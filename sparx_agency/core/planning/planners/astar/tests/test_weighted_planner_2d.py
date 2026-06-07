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
    params = WeightedAStarParams(inflate_radius_m=0.1, unknown_cost=5.0, unknown_blocked=False)
    cost, occ = build_cost_grid(grid, params)
    assert math.isinf(cost[3, 3])         # occupied -> inf
    assert math.isinf(cost[2, 3])         # inflated by 1 cell
    assert cost[0, 0] == 5.0              # unknown weighted
    assert occ[3, 3] and occ[2, 3]


def test_build_cost_grid_unknown_blocked():
    arr = _free(5, 5)
    arr[1, 1] = -1
    grid = _grid(arr)
    cost, _ = build_cost_grid(grid, WeightedAStarParams(unknown_blocked=True, inflate_radius_m=0.0))
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


def test_cost_cache_reused_until_grid_changes():
    grid = _grid(_free(20, 20))
    planner = WeightedAStarPlanner2D()
    c1, _ = planner.cost_for(grid)
    c2, _ = planner.cost_for(grid)
    assert c1 is c2  # same grid object -> cached
    planner.invalidate_cache()
    c3, _ = planner.cost_for(grid)
    assert c3 is not c1
