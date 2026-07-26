"""Small-component (speck) removal: the morphology op and its wiring into the
BevProjector. Guards the failure the drone hit -- a lone phantom voxel in a turn
opening that blocks the planner and can never be re-observed free to clear.
"""
import numpy as np

from sparx_agency.core.mapping.bev import BevConfig, BevProjector
from sparx_agency.core.mapping.bev import morphology as morph


def test_remove_small_components_culls_speck_keeps_wall():
    m = np.zeros((10, 10), bool)
    m[2:8, 3] = True          # a 6-cell vertical wall
    m[5, 7] = True            # a lone 1-cell speck
    out, n = morph.remove_small_components(m, min_size=3)
    assert n == 1
    assert not out[5, 7]                 # speck gone
    assert out[2:8, 3].all()             # wall untouched


def test_run_filter_drops_compact_clump_keeps_straight_run():
    # The core distinction: a 2x2 clump is 4 cells (passes an area>=4 gate) but
    # is NOT a wall; a straight 3-run is a wall. min_wall_run must invert that.
    m = np.zeros((8, 8), bool)
    m[1:3, 1:3] = True        # 2x2 clump (4 cells, no 3-in-a-row)
    m[5, 2:5] = True          # horizontal 3-run (a wall segment)
    out, n = morph.remove_non_wall_components(m, min_run=3)
    assert n == 4                        # the whole clump removed
    assert not out[1:3, 1:3].any()
    assert out[5, 2:5].all()             # 3-run kept

    # Area gate does the opposite on the same input (keeps clump, drops run).
    a_out, _ = morph.remove_small_components(m, min_size=4)
    assert a_out[1:3, 1:3].all() and not a_out[5, 2:5].any()


def test_run_filter_keeps_diagonal_run_and_L_corner():
    m = np.zeros((8, 8), bool)
    for i in range(3):
        m[1 + i, 1 + i] = True           # 3-cell diagonal wall
    out, n = morph.remove_non_wall_components(m, min_run=3, connectivity=8)
    assert n == 0 and out.sum() == 3     # diagonal run survives

    # An L of two 3-runs sharing a corner: one 8-connected component, both arms
    # are 3-runs -> fully kept.
    L = np.zeros((8, 8), bool)
    L[2, 2:5] = True
    L[2:5, 2] = True
    out2, n2 = morph.remove_non_wall_components(L, min_run=3, connectivity=8)
    assert n2 == 0 and out2.sum() == int(L.sum())


def test_run_filter_drops_two_cell_and_L_tromino():
    m = np.zeros((6, 6), bool)
    m[1, 1:3] = True          # 2-cell segment
    m[4, 4] = True            # L-tromino
    m[4, 5] = True
    m[3, 5] = True
    out, n = morph.remove_non_wall_components(m, min_run=3)
    assert not out.any() and n == 5      # nothing is a 3-run -> all removed


def test_diagonal_corner_is_one_component_under_conn8():
    # An L-corner touching only diagonally must survive as one big component.
    m = np.zeros((6, 6), bool)
    for i in range(4):
        m[1 + i, 1 + i] = True           # diagonal staircase of 4 cells
    out8, n8 = morph.remove_small_components(m, min_size=4, connectivity=8)
    assert n8 == 0 and out8.sum() == 4   # 8-conn: one component of 4 -> kept
    out4, n4 = morph.remove_small_components(m, min_size=4, connectivity=4)
    assert n4 == 4 and out4.sum() == 0   # 4-conn: four singletons -> all culled


def test_min_size_le_1_is_noop():
    m = np.zeros((5, 5), bool)
    m[2, 2] = True
    out, n = morph.remove_small_components(m, min_size=1)
    assert n == 0 and out[2, 2]


def _wall_and_speck_clouds():
    """Occupied voxels: a solid wall column-run + one isolated phantom column,
    both filling the full height band so they pass the single-frame gates."""
    zs = np.linspace(0.45, 1.75, 14)
    occ = []
    for gx in range(-5, 5):                       # a 1 m wall along y=+1.0
        for z in zs:
            occ.append((gx * 0.1, 1.0, z))
    for z in zs:                                  # a lone phantom at (0.5, -1.0)
        occ.append((0.5, -1.0, z))
    return np.asarray(occ, np.float32), np.zeros((0, 3), np.float32)


def _cfg(min_wall_run):
    # wall_fill directional (as in production) so a one-cell gap in the wall is
    # bridged BEFORE speck removal -- the wall must survive as one component.
    return BevConfig(
        resolution_m=0.10, x_min=-2, x_max=2, y_min=-2, y_max=2,
        z_floor=0.40, z_ceil=1.80, z_peak=1.0, voxel_size_m=0.10,
        occ_weight_thresh=1.5, min_occ_voxels=3, occ_conf_full=3.0,
        confirm_3d=False, temporal_filter=False, wall_fill_mode="directional",
        protect_openings=False, min_wall_run=min_wall_run)


def test_projector_removes_isolated_phantom_keeps_wall():
    occ, free = _wall_and_speck_clouds()
    OCCUPIED = 100

    _, g_off = BevProjector(_cfg(0)).project(occ, free)
    _, g_on = BevProjector(_cfg(3)).project(occ, free)

    # The phantom column is a single BEV cell; the wall is a long run.
    gy_wall = int((1.0 - (-2)) / 0.10)
    gy_ph = int((-1.0 - (-2)) / 0.10)
    gx_ph = int((0.5 - (-2)) / 0.10)

    assert (g_off == OCCUPIED).sum() > (g_on == OCCUPIED).sum()   # something culled
    assert g_off[gy_ph, gx_ph] == OCCUPIED                        # phantom present w/o filter
    assert g_on[gy_ph, gx_ph] != OCCUPIED                         # phantom removed with filter
    assert (g_on[gy_wall] == OCCUPIED).sum() >= 8                 # wall preserved


def test_projector_min_wall_run_culls_2x2_clump():
    # A 2x2 phantom clump (4 cells) in open space: passes an area>=4 gate but is
    # not a wall, so min_wall_run=3 must remove it while keeping the wall.
    occ, free = _wall_and_speck_clouds()
    zs = np.linspace(0.45, 1.75, 14)
    # cell-centred coords (x_min=-2, res=0.1) so the four points land in a clean
    # 2x2 block of distinct cells (cols 24,25 x rows 9,10), not a collapsed pair.
    clump = np.array([[x, y, z]
                      for x in (0.45, 0.55) for y in (-1.05, -0.95) for z in zs],
                     np.float32)
    occ = np.vstack([occ, clump])
    OCCUPIED = 100

    p0 = BevProjector(_cfg(0)); _, g0 = p0.project(occ, free)
    p1 = BevProjector(_cfg(3)); _, g1 = p1.project(occ, free)

    gy_wall = int((1.0 - (-2)) / 0.10)
    assert p1.last_stats["speck"] == 4                     # the 2x2 clump culled
    assert (g0 == OCCUPIED).sum() - (g1 == OCCUPIED).sum() == 4
    assert (g1[gy_wall] == OCCUPIED).sum() >= 8            # wall preserved
