import math

import numpy as np

from sparx_agency.tasks.planning.objnav_benchmark_runtime.room_dataset.geometry import visible_cells
from sparx_agency.tasks.planning.objnav_benchmark_runtime.room_dataset.grid_views import (
    OccupancyGrid, entrance_pose, leg_actions, plan_views, ray_ranges, synthetic_depth)

CELL = 0.1
WIDTH, HEIGHT, HFOV = 64, 48, 79.0


def room_grid(size_m=4.0, wall_x=None):
    """A square room, free inside; with ``wall_x`` a blocked column at that x."""
    n = int(size_m / CELL)
    free = np.zeros((n + 2, n + 2), dtype=bool)
    free[1:-1, 1:-1] = True
    if wall_x is not None:
        free[:, int(wall_x / CELL) + 1] = False
    return OccupancyGrid(free, origin_xz=(-CELL, -CELL), cell_m=CELL)


FACING_PLUS_X = -math.pi / 2


def test_a_ray_ends_at_the_wall_it_hits():
    grid = room_grid(4.0)
    ranges = ray_ranges(grid, (1.0, 2.0), FACING_PLUS_X, np.array([0.0]), max_range_m=10.0)
    assert abs(ranges[0] - 3.0) < 0.15


def test_a_ray_stops_at_max_range_in_open_space():
    grid = room_grid(40.0)
    assert ray_ranges(grid, (1.0, 20.0), FACING_PLUS_X, np.array([0.0]), max_range_m=5.0)[0] == 5.0


def test_a_blocked_column_in_the_way_shortens_the_ray():
    grid = room_grid(4.0, wall_x=2.0)
    assert abs(ray_ranges(grid, (0.5, 2.0), FACING_PLUS_X, np.array([0.0]), max_range_m=10.0)[0] - 1.5) < 0.15


def test_a_camera_outside_the_free_cells_sees_nothing():
    grid = room_grid(4.0)
    assert ray_ranges(grid, (-5.0, 2.0), FACING_PLUS_X, np.array([0.0]), max_range_m=10.0)[0] == 0.0


def test_the_synthetic_depth_gives_the_same_cells_as_a_straight_wall_would():
    grid = room_grid(4.0)
    depth = synthetic_depth(grid, (1.0, 2.0), FACING_PLUS_X, WIDTH, HEIGHT, HFOV, max_range_m=10.0)
    cells = np.array([[2.0, 0.0, 2.0], [3.5, 0.0, 2.0], [-1.0, 0.0, 2.0]])
    seen = visible_cells(depth, (1.0, 0.0, 2.0, FACING_PLUS_X), cells, WIDTH, HEIGHT, HFOV)
    assert seen.tolist() == [True, True, False]


def test_a_wall_hides_the_cells_behind_it_in_the_synthetic_depth():
    grid = room_grid(4.0, wall_x=2.0)
    depth = synthetic_depth(grid, (0.5, 2.0), FACING_PLUS_X, WIDTH, HEIGHT, HFOV, max_range_m=10.0)
    cells = np.array([[1.2, 0.0, 2.0], [3.0, 0.0, 2.0]])
    assert visible_cells(depth, (0.5, 0.0, 2.0, FACING_PLUS_X), cells, WIDTH, HEIGHT, HFOV).tolist() == [True, False]


def test_a_turn_costs_one_action_per_thirty_degrees_rounded_up():
    assert leg_actions([(0, 0), (0, 0)], 0.0, math.radians(100)) == 4


def test_the_short_way_round_is_taken_for_a_turn():
    assert leg_actions([(0, 0), (0, 0)], math.radians(10), math.radians(350)) == 1


def test_walking_costs_a_forward_step_per_quarter_metre_plus_the_turns_to_face_and_to_finish():
    yaw_plus_x = FACING_PLUS_X
    walk_plus_x = [(0.0, 0.0), (1.0, 0.0)]
    assert leg_actions(walk_plus_x, yaw_plus_x, yaw_plus_x) == 4
    assert leg_actions(walk_plus_x, yaw_plus_x + math.pi / 2, yaw_plus_x + math.pi / 2) == 4 + 3 + 3


def test_the_entrance_is_where_the_path_first_reaches_the_room_and_it_faces_along_the_path():
    path = [(-3.0, 1.0), (-1.0, 1.0), (0.5, 1.0), (2.0, 1.0)]
    point, yaw = entrance_pose(path, lambda p: p[0] >= 0.0)
    assert point == (0.5, 1.0) and abs(yaw - FACING_PLUS_X) < 1e-9


def test_the_entrance_is_none_when_the_path_never_enters():
    assert entrance_pose([(-3.0, 1.0), (-1.0, 1.0)], lambda p: p[0] >= 0.0) is None


def view(seen, cell=(0.0, 0.0), yaw=0.0, near=0.0, inside=1.0, cells=6):
    return {"cells": np.array([i in seen for i in range(cells)]), "near": near, "inside": inside, "cell": cell, "yaw": yaw}


def free_cost(entry, candidate_index, current):
    return 1


def test_with_no_budget_limit_the_plan_is_the_greedy_coverage_order():
    views = [view({0, 1}), view({0, 1, 2, 3}), view({4, 5})]
    assert plan_views(views, 2, None, free_cost) == [1, 2]


def test_a_view_too_costly_for_the_budget_is_skipped_for_a_cheaper_one():
    views = [view({0, 1, 2, 3, 4}), view({0, 1})]
    cost = lambda entry, i, current: 50 if i == 0 else 3
    assert plan_views(views, 1, 10, cost) == [1]


def test_the_budget_is_spent_over_all_legs_and_stops_when_nothing_is_affordable():
    views = [view({0, 1}), view({2, 3}), view({4, 5})]
    assert plan_views(views, 3, 5, lambda e, i, c: 3) == [0]


def test_the_cost_of_a_leg_is_measured_from_the_previous_pick():
    views = [view({0, 1}), view({2, 3}), view({4})]
    distance_from = {None: {0: 1, 1: 9, 2: 9}, 0: {1: 2, 2: 9}, 1: {2: 5}}
    cost = lambda e, i, current: distance_from[current][i]
    assert plan_views(views, 3, 4, cost) == [0, 1]


def test_views_that_are_close_ups_or_look_out_of_the_room_are_never_planned():
    views = [view({0, 1, 2, 3}, near=0.9), view({0, 1, 2}, inside=0.2), view({4})]
    assert plan_views(views, 3, None, free_cost) == [2]


def test_a_ray_that_reaches_the_sensor_range_has_no_depth_so_it_blocks():
    grid = room_grid(4.0)
    depth = synthetic_depth(grid, (1.0, 2.0), FACING_PLUS_X, WIDTH, HEIGHT, HFOV, max_range_m=2.0)
    assert not visible_cells(depth, (1.0, 0.0, 2.0, FACING_PLUS_X), np.array([[1.5, 0.0, 2.0]]), WIDTH, HEIGHT, HFOV).any()
