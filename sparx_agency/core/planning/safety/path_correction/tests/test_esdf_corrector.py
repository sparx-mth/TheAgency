"""Tests for :class:`EsdfPathCorrector` and the corrector factory's esdf branch.

These build a real ``OccupancyGrid2D`` corridor and run the full ESDF strategy
(EsdfLayer distance transform -> gradient ascent -> unknown damping -> collision
clip), mirroring what the ROS path-corrector node does with ``~corrector=esdf``.
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
    EsdfCorrectorConfig,
    EsdfPathCorrector,
    InflatedGridCollisionChecker,
    make_path_corrector,
)

VALUES = OccupancyValues(free=0, occupied=100, unknown=-1)


def _corridor_grid(h=80, w=120, wall=10, res=0.1):
    """Horizontal corridor: occupied top/bottom bands, free in between.

    Free band rows ``[wall, h-wall)`` -> y in [1.0, 7.0) m, centre-line y = 4.0 m.
    """
    data = np.full((h, w), VALUES.free, dtype=np.int16)
    data[:wall, :] = VALUES.occupied
    data[-wall:, :] = VALUES.occupied
    params = OccupancyGrid2DParams(resolution=res, origin_x=0.0, origin_y=0.0,
                                   frame_id="world")
    return OccupancyGrid2D(data, params, values=VALUES)


def _hugging_path(grid, off_y, res=0.1):
    return Path2D(points=tuple(Pose2D(x * res, off_y) for x in range(15, 105, 10)),
                  frame_id="world")


class TestEsdfCorrector:
    def test_ascends_toward_corridor_centre(self):
        grid = _corridor_grid()
        centre_y = (grid.height / 2) * grid.resolution        # 4.0 m
        path = _hugging_path(grid, off_y=centre_y - 1.0)       # 1 m below centre
        corrector = EsdfPathCorrector(EsdfCorrectorConfig(max_total_shift_m=1.5))

        out = corrector.correct(path, grid).path.points
        before = np.mean([abs(p.y - centre_y) for p in path.points[1:]])
        after = np.mean([abs(p.y - centre_y) for p in out[1:]])
        assert after < before                                  # ascended toward centre
        assert corrector.field is not None

    def test_start_waypoint_pinned(self):
        grid = _corridor_grid()
        centre_y = (grid.height / 2) * grid.resolution
        path = _hugging_path(grid, off_y=centre_y - 1.0)
        out = EsdfPathCorrector().correct(path, grid).path.points
        assert out[0].x == pytest.approx(path.points[0].x)
        assert out[0].y == pytest.approx(path.points[0].y)

    def test_corrected_path_collision_free(self):
        grid = _corridor_grid()
        centre_y = (grid.height / 2) * grid.resolution
        path = _hugging_path(grid, off_y=centre_y - 1.0)
        out = EsdfPathCorrector(EsdfCorrectorConfig(
            inflate_radius_m=0.2, collision_recheck=True)).correct(path, grid).path.points
        assert not InflatedGridCollisionChecker(grid, 0.2).path_collides(out)

    def test_target_clearance_stops_when_safe(self):
        """With a finite target clearance, an already-safe waypoint is not moved."""
        grid = _corridor_grid()
        centre_y = (grid.height / 2) * grid.resolution
        path = _hugging_path(grid, off_y=centre_y)             # already centred (>2 m clear)
        out = EsdfPathCorrector(EsdfCorrectorConfig(
            target_clearance_m=0.5)).correct(path, grid).path.points
        assert all(p.y == pytest.approx(centre_y, abs=1e-6) for p in out)

    def test_open_map_no_movement(self):
        data = np.full((40, 120), VALUES.free, dtype=np.int16)
        grid = OccupancyGrid2D(
            data, OccupancyGrid2DParams(0.1, 0.0, 0.0, "world"), values=VALUES)
        path = _hugging_path(grid, off_y=2.0)
        assert EsdfPathCorrector().correct(path, grid).num_moved == 0


class TestFactoryEsdf:
    def test_builds_esdf(self):
        c = make_path_corrector("esdf", EsdfCorrectorConfig())
        assert isinstance(c, EsdfPathCorrector)

    def test_wrong_config_type_raises(self):
        with pytest.raises(TypeError):
            make_path_corrector("esdf", object())
