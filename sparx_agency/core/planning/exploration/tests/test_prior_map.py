"""Sparse coarse-to-fine map seeding: one fine cell stands for each coarse cell."""
from __future__ import annotations

import numpy as np
import pytest

from sparx_agency.core.planning.environment.occupancy_grid2d import (
    OccupancyGrid2D,
    OccupancyGrid2DParams,
    OccupancyValues,
)
from sparx_agency.core.planning.exploration.prior_map import sparsify_to_fine

VALUES = OccupancyValues(free=0, occupied=100, unknown=-1)


def coarse_grid(cells, resolution=0.5, origin=(-10.0, -10.0)):
    params = OccupancyGrid2DParams(resolution, origin[0], origin[1], "world")
    return OccupancyGrid2D(np.array(cells, dtype=np.int16), params, values=VALUES)


def test_each_known_coarse_cell_seeds_exactly_one_fine_cell():
    coarse = coarse_grid([[0, 100, -1], [-1, 0, 0]])
    fine = sparsify_to_fine(coarse, 0.25)
    assert np.count_nonzero(fine.grid != VALUES.unknown) == 4


def test_fine_grid_covers_the_same_world_extent():
    coarse = coarse_grid([[0, 100, -1], [-1, 0, 0]])
    fine = sparsify_to_fine(coarse, 0.25)
    assert (fine.height, fine.width) == (4, 6)
    assert (fine.origin_x, fine.origin_y, fine.resolution) == (-10.0, -10.0, 0.25)


def test_seeded_cell_lies_inside_its_coarse_cell():
    coarse = coarse_grid(np.full((4, 4), -1))
    coarse.grid[2, 3] = VALUES.occupied
    fine = sparsify_to_fine(coarse, 0.25)
    ys, xs = np.nonzero(fine.grid == VALUES.occupied)
    assert (ys[0] // 2, xs[0] // 2) == (2, 3)


def test_center_rule_picks_the_upper_right_subcell_when_the_center_is_a_corner():
    coarse = coarse_grid([[0]])
    fine = sparsify_to_fine(coarse, 0.25, representative="center")
    assert fine.grid[1, 1] == VALUES.free
    assert np.count_nonzero(fine.grid == VALUES.free) == 1


def test_state_is_copied_not_inferred():
    coarse = coarse_grid([[100, 0]])
    fine = sparsify_to_fine(coarse, 0.25)
    assert sorted(fine.grid[fine.grid != VALUES.unknown].tolist()) == [0, 100]


def test_random_rule_is_reproducible_and_stays_inside_the_coarse_cell():
    coarse = coarse_grid(np.zeros((6, 6)))
    a = sparsify_to_fine(coarse, 0.25, representative="random", seed=3)
    b = sparsify_to_fine(coarse, 0.25, representative="random", seed=3)
    assert np.array_equal(a.grid, b.grid)
    for gy in range(6):
        for gx in range(6):
            block = a.grid[2 * gy:2 * gy + 2, 2 * gx:2 * gx + 2]
            assert np.count_nonzero(block != VALUES.unknown) == 1


def test_non_nesting_ratio_still_seeds_one_cell_per_coarse_cell():
    coarse = coarse_grid(np.zeros((4, 4)))
    fine = sparsify_to_fine(coarse, 0.2)
    assert fine.height == fine.width == 10
    assert np.count_nonzero(fine.grid != VALUES.unknown) == 16


def test_fine_resolution_coarser_than_source_is_rejected():
    with pytest.raises(ValueError):
        sparsify_to_fine(coarse_grid([[0]]), 1.0)


def test_unknown_representative_rule_is_rejected():
    with pytest.raises(ValueError):
        sparsify_to_fine(coarse_grid([[0]]), 0.25, representative="corner")


def test_three_to_one_ratio_nests_exactly_on_a_79_8_m_map():
    coarse = coarse_grid(np.zeros((266, 266)), resolution=0.3)
    fine = sparsify_to_fine(coarse, 0.1)
    assert fine.height == fine.width == 798
    assert np.count_nonzero(fine.grid != VALUES.unknown) == 266 * 266
    ys, xs = np.nonzero(fine.grid != VALUES.unknown)
    assert np.all((ys % 3 == 1) & (xs % 3 == 1))
