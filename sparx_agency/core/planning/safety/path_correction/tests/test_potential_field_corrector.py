"""Tests for :class:`PotentialFieldPathCorrector` and the corrector factory.

These build a real ``OccupancyGrid2D`` corridor and run the full strategy
(``PotentialFieldLayer`` -> ``TrajectorySafetyCorrector`` -> unknown damping ->
collision clip), mirroring exactly what the ROS path-corrector node does, so the
extracted-from-the-node logic is exercised end to end.
"""
import numpy as np
import pytest

from sparx_agency.core.common.types import Path2D, Pose2D
from sparx_agency.core.planning.environment import (
    OccupancyGrid2D,
    OccupancyGrid2DParams,
    OccupancyValues,
)
from sparx_agency.core.planning.safety.path_correction import (
    InflatedGridCollisionChecker,
    PotentialFieldCorrectorConfig,
    PotentialFieldPathCorrector,
    make_path_corrector,
)

VALUES = OccupancyValues(free=0, occupied=100, unknown=-1)


def _corridor_grid(h=80, w=120, wall=10, res=0.1):
    """Horizontal corridor: occupied top/bottom bands, free in between.

    Free band rows ``[wall, h-wall)`` -> y in ``[wall*res, (h-wall)*res)``; for the
    defaults that is y in [1.0, 7.0) m, centre-line y = 4.0 m (a wide corridor, so
    an offset path can be recentred without its pinned start sitting inside the
    obstacle-inflation skirt).
    """
    data = np.full((h, w), VALUES.free, dtype=np.int16)
    data[:wall, :] = VALUES.occupied
    data[-wall:, :] = VALUES.occupied
    params = OccupancyGrid2DParams(resolution=res, origin_x=0.0, origin_y=0.0,
                                   frame_id="world")
    return OccupancyGrid2D(data, params, values=VALUES)


def _hugging_path(grid, off_y, res=0.1):
    """A straight path along ``off_y`` spanning the free corridor (>= 2 wps)."""
    pts = tuple(Pose2D(x * res, off_y) for x in range(15, 105, 10))
    return Path2D(points=pts, frame_id="world")


class TestPotentialFieldCorrector:
    def test_offset_path_recentres_toward_corridor_centre(self):
        grid = _corridor_grid()
        centre_y = (grid.height / 2) * grid.resolution        # 4.0 m
        path = _hugging_path(grid, off_y=centre_y - 1.0)       # 1 m below centre
        corrector = PotentialFieldPathCorrector(PotentialFieldCorrectorConfig(
            max_total_shift_m=1.5))

        result = corrector.correct(path, grid)
        out = result.path.points

        before = np.mean([abs(p.y - centre_y) for p in path.points[1:]])
        after = np.mean([abs(p.y - centre_y) for p in out[1:]])
        assert after < before                                  # pulled toward centre
        assert result.num_moved > 0
        assert corrector.field is not None                     # field exposed for viz

    def test_start_waypoint_is_pinned(self):
        grid = _corridor_grid()
        centre_y = (grid.height / 2) * grid.resolution
        path = _hugging_path(grid, off_y=centre_y - 0.5)
        out = PotentialFieldPathCorrector().correct(path, grid).path.points
        assert out[0].x == pytest.approx(path.points[0].x)
        assert out[0].y == pytest.approx(path.points[0].y)

    def test_corrected_path_stays_collision_free(self):
        grid = _corridor_grid()
        centre_y = (grid.height / 2) * grid.resolution
        path = _hugging_path(grid, off_y=centre_y - 1.0)
        corrector = PotentialFieldPathCorrector(PotentialFieldCorrectorConfig(
            inflate_radius_m=0.2, collision_recheck=True))
        out = corrector.correct(path, grid).path.points
        checker = InflatedGridCollisionChecker(grid, inflate_radius_m=0.2)
        assert not checker.path_collides(out)

    def test_open_map_leaves_path_essentially_unchanged(self):
        """With no walls in range the field is flat -> nothing to push off."""
        data = np.full((40, 120), VALUES.free, dtype=np.int16)
        grid = OccupancyGrid2D(
            data, OccupancyGrid2DParams(0.1, 0.0, 0.0, "world"), values=VALUES)
        path = _hugging_path(grid, off_y=2.0)
        result = PotentialFieldPathCorrector().correct(path, grid)
        assert result.num_moved == 0


class TestFactory:
    def test_builds_potential_field(self):
        c = make_path_corrector("potential_field", PotentialFieldCorrectorConfig())
        assert isinstance(c, PotentialFieldPathCorrector)

    def test_unknown_name_raises(self):
        with pytest.raises(ValueError):
            make_path_corrector("nope_not_a_corrector")

    def test_wrong_config_type_raises(self):
        with pytest.raises(TypeError):
            make_path_corrector("potential_field", object())
