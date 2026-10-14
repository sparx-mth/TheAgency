"""A coarse prior seeded into the observed map, in a frame fixed before the first pose."""
import numpy as np
import pytest

from sparx_agency.core.planning.environment import OccupancyGrid2D, OccupancyGrid2DParams, OccupancyValues
from sparx_agency.tasks.planning.objnav_benchmark_runtime.methods.observed_map import ObservedMap
from sparx_agency.tasks.planning.objnav_benchmark_runtime.methods.rpt_settings import RPTSettings

VALUES = OccupancyValues(free=0, occupied=100, unknown=-1)


def prior_grid(cells, resolution, origin):
    params = OccupancyGrid2DParams(resolution, origin[0], origin[1], "world")
    return OccupancyGrid2D(np.array(cells, dtype=np.int16), params, values=VALUES)


def fine_prior(size_m, resolution, origin=(-5.0, -5.0)):
    cells = int(size_m / resolution)
    grid = np.full((cells, cells), -1)
    grid[0, 0], grid[1, 1] = 0, 100
    return prior_grid(grid, resolution, origin)


def test_explicit_origin_is_used_before_any_pose():
    mapping = ObservedMap(10.0, 0.25, origin_xy=(-5.0, -5.0))
    assert (mapping.grid.origin_x, mapping.grid.origin_y) == (-5.0, -5.0)


def test_seed_prior_writes_free_and_occupied_cells_only():
    mapping = ObservedMap(10.0, 0.25, origin_xy=(-5.0, -5.0))
    mapping.seed_prior(fine_prior(10.0, 0.25))
    _, percent = mapping.grid.get_grid()
    assert percent[0, 0] <= 45 and percent[1, 1] >= 65
    assert np.count_nonzero(percent != -1) == 2


def test_seed_prior_does_not_count_as_observation():
    mapping = ObservedMap(10.0, 0.25, origin_xy=(-5.0, -5.0))
    mapping.seed_prior(fine_prior(10.0, 0.25))
    assert not mapping.grid.observed_mask.any()


def test_seed_prior_rejects_a_resolution_mismatch():
    mapping = ObservedMap(10.0, 0.25, origin_xy=(-5.0, -5.0))
    with pytest.raises(ValueError, match="resolution"):
        mapping.seed_prior(fine_prior(10.0, 0.5))


def test_seed_prior_rejects_an_origin_mismatch():
    mapping = ObservedMap(10.0, 0.25, origin_xy=(-5.0, -5.0))
    with pytest.raises(ValueError, match="origin"):
        mapping.seed_prior(fine_prior(10.0, 0.25, origin=(-4.0, -5.0)))


def test_seed_prior_rejects_a_shape_mismatch():
    mapping = ObservedMap(10.0, 0.25, origin_xy=(-5.0, -5.0))
    with pytest.raises(ValueError, match="shape"):
        mapping.seed_prior(fine_prior(8.0, 0.25))


def test_evidence_settings_default_to_the_established_values():
    mapping = ObservedMap(10.0, 0.25)
    assert mapping.grid.cfg.l_free == -0.40
    assert mapping.obstacle_min_height_m == 0.15


def test_free_evidence_and_obstacle_height_are_configurable():
    mapping = ObservedMap(10.0, 0.5, free_log_odds=-1.0, obstacle_min_height_m=0.4)
    assert mapping.grid.cfg.l_free == -1.0
    assert mapping.obstacle_min_height_m == 0.4


def test_settings_reject_a_non_negative_free_evidence():
    with pytest.raises(ValueError, match="free_log_odds"):
        RPTSettings(free_log_odds=0.2)


def test_settings_reject_an_obstacle_height_above_the_body():
    with pytest.raises(ValueError, match="obstacle_min_height_m"):
        RPTSettings(obstacle_min_height_m=2.0)
