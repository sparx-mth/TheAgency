"""Evaluator-only ground-truth coverage of an explored map (never a policy input)."""
import numpy as np
import pytest

from sparx_agency.core.planning.environment import OccupancyGrid2D, OccupancyGrid2DParams, OccupancyValues
from sparx_agency.tasks.planning.objnav_benchmark_runtime.experiments.multi_resolution.transfer_eval import (
    coverage_metrics, navigable_lattice_enu)

VALUES = OccupancyValues(free=0, occupied=100, unknown=-1)


class FakePathfinder:
    """Navigable on x_h in [0, 2), z_h in [0, 1); one island; height 0."""

    def get_bounds(self):
        return np.array([0.0, 0.0, 0.0]), np.array([2.0, 1.0, 1.0])

    def is_navigable(self, point, max_y_delta=0.5):
        return 0.0 <= point[0] < 2.0 and 0.0 <= point[2] < 1.0 and abs(point[1]) <= max_y_delta

    def get_island(self, point):
        return 0 if self.is_navigable(point) else -1


def world_grid(cells, resolution=0.5, origin=(0.0, 0.0)):
    params = OccupancyGrid2DParams(resolution, origin[0], origin[1], "world")
    return OccupancyGrid2D(np.array(cells, dtype=np.int16), params, values=VALUES)


def test_lattice_is_converted_to_enu_and_stays_on_the_floor():
    points = navigable_lattice_enu(FakePathfinder(), floor_height_m=0.0, start_habitat=(0.1, 0.0, 0.1), step_m=0.5)
    assert points.shape[1] == 2
    assert points[:, 0].max() <= 0.0 and points[:, 0].min() > -1.0
    assert points[:, 1].max() <= 0.0 and points[:, 1].min() > -2.0
    assert len(points) == 8


def test_lattice_excludes_other_islands():
    class TwoIslands(FakePathfinder):
        def get_island(self, point):
            return 0 if point[0] < 1.0 else 1

    points = navigable_lattice_enu(TwoIslands(), 0.0, (0.1, 0.0, 0.1), 0.5)
    assert len(points) == 4


def test_observed_coverage_counts_ground_truth_cells_the_sensor_saw():
    world = world_grid([[0, 0], [-1, -1]], origin=(0.0, 0.0))
    observed = np.array([[True, False], [False, False]])
    gt = np.array([[0.25, 0.25], [0.75, 0.25], [0.25, 0.75], [0.75, 0.75]])
    result = coverage_metrics(world, observed, gt)
    assert result["gt_points"] == 4
    assert result["observed_coverage"] == pytest.approx(0.25)
    assert result["known_coverage"] == pytest.approx(0.5)


def test_false_free_rate_counts_free_cells_away_from_navigable_space():
    world = world_grid([[0, 0, 0, 0]] * 4, resolution=1.0)
    gt = np.array([[0.5, 0.5]])
    result = coverage_metrics(world, np.ones((4, 4), bool), gt, support_radius_m=1.0)
    assert result["false_free_rate"] == pytest.approx(1 - 4 / 16)


def test_points_outside_the_map_are_ignored_not_crashing():
    world = world_grid([[0]], resolution=1.0)
    result = coverage_metrics(world, np.ones((1, 1), bool), np.array([[5.0, 5.0], [0.5, 0.5]]))
    assert result["gt_points"] == 1


def test_coverage_counts_points_so_resolutions_are_comparable():
    gt = np.array([[x + 0.125, y + 0.125] for x in np.arange(0, 1, 0.25) for y in np.arange(0, 1, 0.25)])
    coarse = world_grid([[0, 0], [0, 0]], resolution=0.5)
    fine = world_grid(np.zeros((4, 4)), resolution=0.25)
    observed_coarse = np.array([[True, False], [False, False]])
    observed_fine = np.zeros((4, 4), bool)
    observed_fine[:2, :2] = True
    assert coverage_metrics(coarse, observed_coarse, gt)["observed_coverage"] == pytest.approx(0.25)
    assert coverage_metrics(fine, observed_fine, gt)["observed_coverage"] == pytest.approx(0.25)
