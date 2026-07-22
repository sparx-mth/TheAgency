"""Tests for BevProjector.last_confidence (the per-cell OCCUPIED confidence)."""
import numpy as np

from sparx_agency.core.mapping.bev import BevConfig, BevProjector, OCCUPIED


def _cfg(temporal=True):
    """Tiny 8x8 BEV; a single occupied column at cell (4,4) is an OCC candidate."""
    return BevConfig(
        resolution_m=0.5, x_min=-2.0, x_max=2.0, y_min=-2.0, y_max=2.0,
        z_floor=0.3, z_ceil=2.3, z_peak=1.0, voxel_size_m=0.5,
        occ_weight_thresh=1.2, min_occ_voxels=2, occ_conf_full=3.0,
        confirm_3d=False, protect_openings=False, wall_fill_mode="off",
        temporal_filter=temporal)


# A full vertical column of occupied voxels at world (0, 0) -> cell (cx=4, cy=4).
_OCC = np.array([[0.0, 0.0, 0.55], [0.0, 0.0, 1.05],
                 [0.0, 0.0, 1.55], [0.0, 0.0, 2.05]], np.float32)
_FREE = np.empty((0, 3), np.float32)


def test_confidence_none_when_temporal_off():
    proj = BevProjector(_cfg(temporal=False))
    _, grid = proj.project(_OCC, _FREE)
    assert proj.last_confidence is None


def test_confidence_shape_and_range():
    proj = BevProjector(_cfg())
    _, grid = proj.project(_OCC, _FREE)
    c = proj.last_confidence
    assert c is not None
    assert c.shape == grid.shape
    assert c.dtype == np.float32
    assert c.min() >= 0.0 and c.max() <= 1.0
    assert c[0, 0] == 0.0                      # a never-occupied cell has no evidence


def test_confidence_rises_and_cell_latches_occupied():
    proj = BevProjector(_cfg())
    prev = -1.0
    latched_frame = None
    for i in range(10):
        _, grid = proj.project(_OCC, _FREE)
        conf = proj.last_confidence[4, 4]
        assert conf >= prev, "evidence must not decrease while re-observed occupied"
        prev = conf
        if latched_frame is None and grid[4, 4] == OCCUPIED:
            latched_frame = i
    assert latched_frame is not None, "a persistently-seen column must confirm OCC"
    assert prev > 0.5, "confidence saturates high after repeated observation"


def test_confidence_saturates_at_one():
    proj = BevProjector(_cfg())
    for _ in range(40):                        # well past t_max / (t_inc*conf)
        proj.project(_OCC, _FREE)
    assert proj.last_confidence[4, 4] == 1.0   # evidence clipped at t_max -> conf 1.0


def test_forced_obstacles_are_full_confidence():
    """A caller-forced CERTAIN obstacle (force_occ) is OCCUPIED but carries no
    temporal evidence; its confidence must read 1.0, not 0.0, so a downstream
    confidence gate never treats a known wall as a speckle to keep looking at."""
    proj = BevProjector(_cfg())
    force = np.zeros((proj.lattice.H, proj.lattice.W), bool)
    force[2, 2] = True                         # a config wall cell, never observed
    _, grid = proj.project(_FREE, _FREE, force_occ=force)
    assert grid[2, 2] == OCCUPIED
    assert proj.last_confidence[2, 2] == 1.0   # certain wall -> full confidence
    assert proj.last_confidence[0, 0] == 0.0   # an unobserved free cell stays 0


def test_forced_obstacle_confidence_none_when_temporal_off():
    """With the temporal filter off there is no confidence grid to stamp."""
    proj = BevProjector(_cfg(temporal=False))
    force = np.zeros((proj.lattice.H, proj.lattice.W), bool)
    force[2, 2] = True
    _, grid = proj.project(_FREE, _FREE, force_occ=force)
    assert grid[2, 2] == OCCUPIED
    assert proj.last_confidence is None
