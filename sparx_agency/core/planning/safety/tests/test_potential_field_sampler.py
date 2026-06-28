"""Unit tests for :class:`PotentialFieldSampler`.

The sampler is the frame + bilinear-sampling half of potential-field trajectory
correction; these tests pin down its coordinate contract and edge handling
independently of the descent algorithm.
"""
import numpy as np
import pytest

from sparx_agency.core.planning.safety import PotentialFieldSampler


def test_bilinear_matches_known_plane():
    """A field that is linear in x is sampled exactly by bilinear interpolation."""
    # U(x, y) = x  => U[row, col] = col * res (since x = col*res, origin 0).
    res = 0.25
    cols = np.arange(8)
    u = np.tile(cols * res, (8, 1)).astype(np.float64)   # value == world x
    s = PotentialFieldSampler(u, res, origin_x=0.0, origin_y=0.0)
    # Sample at a fractional world point well inside the grid.
    assert s.potential(0.6, 0.4) == pytest.approx(0.6, abs=1e-6)
    assert s.potential(1.1, 0.9) == pytest.approx(1.1, abs=1e-6)


def test_descent_sign_points_down_potential():
    """Descent of U=x is -x̂ everywhere (move toward lower potential)."""
    res = 0.25
    u = np.tile(np.arange(8) * res, (8, 1)).astype(np.float64)
    s = PotentialFieldSampler(u, res, 0.0, 0.0)
    g = s.descent(0.8, 0.8)
    assert g[0] < 0.0                 # x-component points toward smaller x
    assert abs(g[1]) < 1e-6           # no y dependence


def test_out_of_bounds_returns_none():
    s = PotentialFieldSampler(np.zeros((6, 6)), 0.1, 0.0, 0.0)
    assert s.potential(-1.0, 0.2) is None
    assert s.potential(100.0, 0.2) is None
    assert s.descent(0.2, -5.0) is None
    assert s.is_observed(100.0, 100.0) is False


def test_known_mask_gates_observation():
    known = np.ones((10, 10), dtype=bool)
    known[:, 5:] = False              # right half (high x) unobserved
    s = PotentialFieldSampler(np.zeros((10, 10)), 0.1, 0.0, 0.0, known_mask=known)
    assert s.is_observed(0.2, 0.2) is True     # col 2 observed
    assert s.is_observed(0.8, 0.2) is False    # col 8 unobserved


def test_clearance_optional():
    s_no = PotentialFieldSampler(np.zeros((6, 6)), 0.1, 0.0, 0.0)
    assert s_no.has_distance is False
    assert s_no.clearance(0.2, 0.2) is None
    s_d = PotentialFieldSampler(np.zeros((6, 6)), 0.1, 0.0, 0.0,
                                d_obs=np.full((6, 6), 0.7))
    assert s_d.has_distance is True
    assert s_d.clearance(0.2, 0.2) == pytest.approx(0.7, abs=1e-6)


def test_nonfinite_and_shape_guards():
    with pytest.raises(ValueError):
        PotentialFieldSampler(np.zeros((5,)), 0.1, 0.0, 0.0)
    bad = np.zeros((6, 6)); bad[0, 0] = np.nan
    with pytest.raises(ValueError):
        PotentialFieldSampler(bad, 0.1, 0.0, 0.0)
    with pytest.raises(ValueError):
        PotentialFieldSampler(np.zeros((6, 6)), 0.0, 0.0, 0.0)
    with pytest.raises(ValueError):
        PotentialFieldSampler(np.zeros((6, 6)), 0.1, 0.0, 0.0,
                              d_obs=np.zeros((4, 4)))
