import numpy as np

from sparx_agency.tasks.planning.objnav_benchmark_runtime.room_dataset.report_grid_vs_depth import compare_room, coverage, mean_overlap


def masks(*rows):
    return np.array(rows, dtype=bool)


def test_coverage_is_the_share_of_cells_any_pick_sees():
    assert coverage(masks([1, 1, 0, 0], [0, 1, 1, 0]), [0, 1]) == 0.75
    assert coverage(masks([1, 1, 0, 0]), []) == 0.0


def test_overlap_is_intersection_over_union_averaged_over_views_that_see_anything():
    a = masks([1, 1, 0, 0], [0, 0, 0, 0], [1, 0, 0, 0])
    b = masks([1, 0, 0, 0], [0, 0, 0, 0], [1, 0, 0, 0])
    assert mean_overlap(a, b) == (0.5 + 1.0) / 2


def arrays(depth, grid, near_depth=None, near_grid=None):
    n = len(depth)
    return {"depth_cells": masks(*depth), "grid_cells": masks(*grid), "depth_near": np.array(near_depth or [0.0] * n), "grid_near": np.array(near_grid or [0.0] * n),
            "depth_inside": np.ones(n), "grid_inside": np.ones(n)}


def test_picks_made_on_the_grid_are_scored_by_what_the_depth_says_they_see():
    depth = [[1, 1, 1, 1], [1, 1, 0, 0], [0, 0, 1, 1]]
    grid = [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 1]]
    result = compare_room(arrays(depth, grid), count=1, rng=np.random.default_rng(0))
    assert result["depth_picks"] == 1.0
    assert result["grid_picks"] == 0.5


def test_a_view_the_grid_calls_usable_but_the_depth_does_not_counts_against_its_precision():
    depth = [[1, 1, 0, 0], [0, 0, 1, 1]]
    result = compare_room(arrays(depth, depth, near_depth=[0.0, 0.9], near_grid=[0.0, 0.0]), count=1, rng=np.random.default_rng(0))
    assert result["usable_precision"] == 0.5 and result["usable_recall"] == 1.0
