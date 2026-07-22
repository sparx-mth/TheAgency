"""Tests for the collision-aware post-correction smoothing (numpy, no torch)."""
import numpy as np

from sparx_agency.core.common.types import Path2D, Pose2D
from sparx_agency.core.planning.environment.occupancy_grid2d import (
    OccupancyGrid2D,
    OccupancyGrid2DParams,
)
from sparx_agency.core.planning.safety.path_correction.grid_collision import (
    InflatedGridCollisionChecker,
)
from sparx_agency.tasks.planning.finetune.common.frames import OCC_VALUES
from sparx_agency.tasks.planning.finetune.common.smoothing import smooth_path


def _grid(n_left=80, n_fwd=80, res=0.1, half=4.0):
    grid = np.full((n_left, n_fwd), OCC_VALUES.free, dtype=np.int16)
    params = OccupancyGrid2DParams(resolution=res, origin_x=0.0, origin_y=-half, frame_id="body")
    return OccupancyGrid2D(grid, params, values=OCC_VALUES)


def _bending(pts):
    """Sum of second-difference magnitudes -- a scalar 'kinkiness' measure."""
    p = np.array([(q.x, q.y) for q in pts], dtype=float)
    d2 = p[2:] - 2 * p[1:-1] + p[:-2]
    return float(np.sum(np.linalg.norm(d2, axis=1)))


def _zigzag_path():
    xs = np.linspace(0.0, 3.0, 13)
    ys = 0.4 * np.array([0, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 0, 0], dtype=float)
    return Path2D(points=tuple(Pose2D(float(x), float(y)) for x, y in zip(xs, ys)),
                  frame_id="body")


def test_smoothing_reduces_kinks_preserves_count_and_endpoints():
    path = _zigzag_path()
    smoothed, moved = smooth_path(path, _grid(), clearance_m=0.2,
                                  strength=0.5, passes=10, angle_deg=5.0)
    assert len(smoothed.points) == len(path.points)          # count preserved (label horizon)
    assert smoothed.points[0] == path.points[0]              # start pinned
    assert smoothed.points[-1] == path.points[-1]            # goal pinned
    assert moved > 0
    assert _bending(smoothed.points) < 0.5 * _bending(path.points)  # much smoother


def test_smoothing_never_enters_a_wall():
    occ = _grid()
    occ.grid[:, 40:44] = OCC_VALUES.occupied                 # a wall band at fwd ~4.0-4.4 m
    # a zigzag hugging just in front of the wall
    xs = np.linspace(0.0, 3.6, 13)
    ys = 0.3 * np.array([0, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 0, 0], dtype=float)
    path = Path2D(points=tuple(Pose2D(float(x), float(y)) for x, y in zip(xs, ys)),
                  frame_id="body")
    smoothed, _ = smooth_path(path, occ, clearance_m=0.2, strength=0.7, passes=15)
    checker = InflatedGridCollisionChecker(occ, 0.2)
    assert not checker.path_collides(list(smoothed.points))   # stayed clear of the inflated wall


def test_smoothing_noop_when_disabled_via_strength():
    path = _zigzag_path()
    smoothed, moved = smooth_path(path, _grid(), strength=0.0)
    assert moved == 0
    assert smoothed.points == path.points
