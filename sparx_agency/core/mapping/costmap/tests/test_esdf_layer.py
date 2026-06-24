"""Tests for :class:`EsdfLayer` (distance-to-obstacle field from a prob grid)."""
import numpy as np
import pytest

from sparx_agency.core.mapping.costmap.esdf_layer import EsdfLayer


def test_distance_increases_away_from_a_wall():
    """A single bottom wall row -> distance grows with height (in metres)."""
    res = 0.1
    p = np.zeros((50, 30), dtype=np.float32)   # all free
    p[0, :] = 1.0                              # occupied bottom row
    d = EsdfLayer(occ_thresh=0.5).compute_from_prob_grid(p, res)
    col = 15
    # Row r (free) is r cells above the wall -> ~r*res metres of clearance.
    assert d[1, col] < d[10, col] < d[30, col]
    assert d[10, col] == pytest.approx(10 * res, abs=2 * res)


def test_unknown_treated_as_free_by_default():
    """NaN (unknown) cells must not act as obstacles -> large clearance there."""
    res = 0.1
    p = np.full((40, 40), np.nan, dtype=np.float32)   # all unknown
    p[0, :] = 1.0                                     # one occupied row
    d_free = EsdfLayer(occ_thresh=0.5).compute_from_prob_grid(p, res)
    d_obst = EsdfLayer(occ_thresh=0.5, unknown_as_obstacle=True).compute_from_prob_grid(p, res)
    assert d_free[20, 20] > d_obst[20, 20]            # unknown-as-free is more open
    assert d_obst[20, 20] == pytest.approx(0.0, abs=1e-6)   # everything is an obstacle


def test_corridor_centre_is_a_local_maximum():
    """In a horizontal corridor the centre row has the greatest clearance."""
    res = 0.1
    p = np.zeros((41, 60), dtype=np.float32)
    p[:5, :] = 1.0
    p[-5:, :] = 1.0
    d = EsdfLayer(occ_thresh=0.5).compute_from_prob_grid(p, res)
    col = 30
    centre = 20
    assert d[centre, col] >= d[centre - 5, col]
    assert d[centre, col] >= d[centre + 5, col]
