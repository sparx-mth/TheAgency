"""Seeding a log-odds grid with a prior that a few fresh observations overturn."""
from __future__ import annotations

import numpy as np

from sparx_agency.core.mapping.costmap.log_odds_grid import LogOddsGridConfig, LogOddsGridCostmap

FREE_MAX_PERCENT, OCCUPIED_MIN_PERCENT = 45, 65


def make_grid():
    return LogOddsGridCostmap(LogOddsGridConfig(size_m=2.0, resolution_m=0.5))


def test_seeded_cells_read_back_as_free_and_occupied():
    grid = make_grid()
    free, occupied = np.zeros((4, 4), bool), np.zeros((4, 4), bool)
    free[0, 0], occupied[1, 1] = True, True
    grid.seed(free, occupied)
    _, percent = grid.get_grid()
    assert 0 <= percent[0, 0] <= FREE_MAX_PERCENT
    assert percent[1, 1] >= OCCUPIED_MIN_PERCENT


def test_unseeded_cells_stay_unknown():
    grid = make_grid()
    mask = np.zeros((4, 4), bool)
    mask[0, 0] = True
    grid.seed(mask, np.zeros((4, 4), bool))
    _, percent = grid.get_grid()
    assert np.count_nonzero(percent != grid.cfg.unknown_value) == 1


def test_three_free_observations_overturn_a_seeded_occupied_cell():
    grid = make_grid()
    occupied = np.zeros((4, 4), bool)
    occupied[1, 1] = True
    grid.seed(np.zeros((4, 4), bool), occupied)
    for _ in range(3):
        grid.apply_free_mask(occupied)
    _, percent = grid.get_grid()
    assert percent[1, 1] <= FREE_MAX_PERCENT


def test_seeding_is_not_counted_as_observation():
    grid = make_grid()
    mask = np.zeros((4, 4), bool)
    mask[0, 0] = True
    grid.seed(mask, np.zeros((4, 4), bool))
    assert not grid.observed_mask.any()


def test_a_free_observation_marks_the_cell_observed():
    grid = make_grid()
    mask = np.zeros((4, 4), bool)
    mask[2, 1] = True
    grid.apply_free_mask(mask)
    assert grid.observed_mask[2, 1] and grid.observed_mask.sum() == 1


def test_seed_rejects_a_mask_of_the_wrong_shape():
    grid = make_grid()
    try:
        grid.seed(np.zeros((3, 3), bool), np.zeros((4, 4), bool))
    except ValueError:
        return
    raise AssertionError("shape mismatch accepted")
