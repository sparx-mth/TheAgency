"""Tests for the per-frame PF/ESDF target generator (numpy, no torch)."""
import numpy as np
import pytest

from sparx_agency.core.common.types import Intrinsics
from sparx_agency.tasks.planning.finetune.common.esdf_target import (
    EsdfTargetConfig,
    generate_target,
    signed_sdf,
)
from sparx_agency.tasks.planning.finetune.common.frames import (
    LocalMapConfig,
    cloud_to_occupancy_grid,
)

INTR = Intrinsics(width=504, height=294, fx=322.635, fy=323.389, cx=242.065, cy=90.030)


def _obstacle_on_left_depth():
    """Depth frame with a near obstacle occupying the left half of the image."""
    depth = np.full((294, 504), 8.0, np.float32)
    depth[:, :252] = 2.5
    return depth


def _cfg():
    return EsdfTargetConfig(
        local_map=LocalMapConfig(resolution_m=0.1, forward_extent_m=6.0, half_width_m=3.0,
                                 z_band_m=(-0.5, 2.0), camera_height_m=1.0, pitch_deg=0.0, stride=4),
        corrector="potential_field",
        max_total_shift_m=1.5,
    )


def test_generate_target_shapes():
    res = generate_target(_obstacle_on_left_depth(), INTR, (4.0, 0.0), _cfg())
    assert res.sdf_m.shape == res.occupancy.grid.shape
    assert len(res.corrected_path) == 24  # n_seed_points
    assert len(res.seed_path) == 24
    assert res.sdf_m.dtype == np.float32


def test_signed_sdf_positive_free_negative_inside():
    res = generate_target(_obstacle_on_left_depth(), INTR, (4.0, 0.0), _cfg())
    # obstacle exists -> some cells occupied -> SDF has both signs
    assert res.sdf_m.min() < 0.0   # inside the wall
    assert res.sdf_m.max() > 0.0   # free space


def test_corrector_pushes_path_away_from_left_wall():
    res = generate_target(_obstacle_on_left_depth(), INTR, (4.0, 0.0), _cfg())
    seed = np.array([[p.x, p.y] for p in res.seed_path.points])
    corr = np.array([[p.x, p.y] for p in res.corrected_path.points])
    # obstacle is on +left; corrected path should move toward -left (right) vs seed
    assert corr[:, 1].mean() < seed[:, 1].mean() + 1e-6
    assert res.num_moved > 0


def test_all_free_sdf_is_uniform_positive():
    cfg = LocalMapConfig(resolution_m=0.1, forward_extent_m=5.0, half_width_m=2.0)
    grid = cloud_to_occupancy_grid(np.zeros((0, 3), np.float32), cfg)
    sdf = signed_sdf(grid, clamp_m=4.0)
    assert np.all(sdf == 4.0)


def test_esdf_corrector_variant_runs():
    cfg = _cfg()
    cfg = EsdfTargetConfig(local_map=cfg.local_map, corrector="esdf",
                           max_total_shift_m=1.0)
    res = generate_target(_obstacle_on_left_depth(), INTR, (4.0, 0.0), cfg)
    assert len(res.corrected_path) == 24
