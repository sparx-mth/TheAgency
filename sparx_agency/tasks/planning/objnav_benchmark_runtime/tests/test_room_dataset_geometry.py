import numpy as np
import pytest

from sparx_agency.tasks.planning.objnav_benchmark_runtime.room_dataset.geometry import (
    UNASSIGNED, assign_cells, compass_yaws, spread_points)


def objects(*rows):
    return [{"region": region, "center": (x, 1.0, z)} for region, x, z in rows]


def test_a_cell_takes_the_region_of_the_objects_around_it():
    cells = np.array([[0.2, 0.0, 0.0], [10.2, 0.0, 0.0]])
    found = assign_cells(cells, objects((1, 0, 0), (1, 0.5, 0), (1, 0, 0.5), (2, 10, 0), (2, 10.5, 0), (2, 10, 0.5)), max_gap_m=2.0)
    assert list(found) == [1, 2]


def test_a_cell_far_from_every_object_belongs_to_no_region():
    found = assign_cells(np.array([[50.0, 0.0, 50.0]]), objects((1, 0, 0), (1, 1, 0), (1, 0, 1)), max_gap_m=2.0)
    assert list(found) == [UNASSIGNED]


def test_objects_on_another_floor_do_not_count():
    upstairs = [{"region": 3, "center": (0.0, 4.5, 0.0)}] * 3
    assert list(assign_cells(np.array([[0.0, 0.0, 0.0]]), upstairs, max_gap_m=2.0)) == [UNASSIGNED]


def test_the_majority_of_the_nearest_objects_decides():
    near = objects((1, 0.5, 0), (2, 0.6, 0), (2, 0.7, 0), (1, 5, 0))
    assert list(assign_cells(np.array([[0.0, 0.0, 0.0]]), near, max_gap_m=2.0)) == [2]


def test_spread_points_start_in_the_middle_and_then_go_far_apart():
    points = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0], [4.0, 0.0]])
    chosen = spread_points(points, 3)
    assert chosen[0] == 2 and set(chosen[1:]) == {0, 4}


def test_spread_points_never_returns_more_than_there_are():
    assert len(spread_points(np.array([[0.0, 0.0], [1.0, 0.0]]), 5)) == 2


@pytest.mark.parametrize("count", [4, 8])
def test_the_headings_are_evenly_spaced_around_the_circle(count):
    yaws = compass_yaws(count)
    assert len(yaws) == count and np.allclose(np.diff(yaws), 2 * np.pi / count)


from sparx_agency.tasks.planning.objnav_benchmark_runtime.room_dataset.geometry import greedy_views, visible_cells

CAMERA = dict(width=64, height=48, hfov_deg=90.0)


def seen(depth_m, cells_xz, camera=(0.0, 0.0, 0.0, 0.0)):
    cells = np.array([[x, 0.0, z] for x, z in cells_xz])
    return list(visible_cells(np.full((48, 64), depth_m), camera, cells_xyz=cells, **CAMERA))


def test_a_floor_cell_ahead_inside_the_view_is_seen():
    assert seen(5.0, [(0.0, -3.0)]) == [True]


def test_cells_behind_or_beside_the_camera_are_not_seen():
    assert seen(5.0, [(0.0, 3.0), (10.0, -3.0)]) == [False, False]


def test_a_cell_behind_a_nearer_surface_is_hidden():
    assert seen(2.0, [(0.0, -3.0)]) == [False]


def test_the_view_turns_with_the_yaw():
    assert seen(5.0, [(-3.0, 0.0)], camera=(0.0, 0.0, 0.0, np.pi / 2)) == [True]


def masks(*seen_sets):
    return [np.array([i in s for i in range(5)]) for s in seen_sets]


def test_greedy_views_take_the_view_that_adds_the_most_unseen_cells_each_time():
    assert greedy_views(masks({0, 1, 2}, {2, 3}, {0, 1}, {4}), count=3, min_new=1) == [0, 1, 3]


def test_greedy_views_stop_when_nothing_new_is_added():
    assert greedy_views(masks({0, 1}, {0, 1}), count=5, min_new=1) == [0]


def test_floor_close_to_the_camera_still_counts_as_in_front_of_it():
    assert seen(5.0, [(0.0, -0.5)]) == [True]


def test_a_view_of_a_wall_a_metre_away_sees_nothing_beyond_it_and_a_long_view_sees_the_room():
    near, far = np.full((48, 64), 1.0), np.full((48, 64), 6.0)
    cells = np.array([[0.0, 0.0, -0.5], [0.0, 0.0, -4.0]])
    assert list(visible_cells(near, (0.0, 0.0, 0.0, 0.0), cells_xyz=cells, **CAMERA)) == [True, False]
    assert list(visible_cells(far, (0.0, 0.0, 0.0, 0.0), cells_xyz=cells, **CAMERA)) == [True, True]


def test_a_pixel_without_depth_blocks_the_view_because_it_is_a_wall_nearer_than_the_sensor_can_see():
    assert seen(0.0, [(0.0, -3.0)]) == [False]
    assert seen(np.nan, [(0.0, -3.0)]) == [False]


from sparx_agency.tasks.planning.objnav_benchmark_runtime.room_dataset.geometry import near_share


def test_the_near_share_counts_pixels_closer_than_the_limit_and_pixels_without_depth():
    depth = np.array([[0.5, 3.0], [0.0, np.nan]])
    assert near_share(depth, near_m=1.0) == 0.75
    assert near_share(np.full((2, 2), 3.0), near_m=1.0) == 0.0
