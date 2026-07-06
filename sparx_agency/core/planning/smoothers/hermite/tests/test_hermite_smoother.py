"""Tests for the scipy-free cubic Hermite spline + smoother.

The numpy-only ``_CubicHermiteSpline`` replaced ``scipy.interpolate``'s so core
imports under ROS Noetic (no scipy). These tests pin its endpoint/derivative
behaviour, prove the end-to-end smoother works WITHOUT scipy, and — when scipy is
present (dev/CI) — assert bit-parity with it so the replacement cannot drift.
"""
import math

import numpy as np

from sparx_agency.core.common.types import KinematicLimits, Path2D, Pose2D
from sparx_agency.core.planning.interfaces.smoother import SmootherRequest
from sparx_agency.core.planning.smoothers.hermite import HermiteParams, HermiteSmoother
from sparx_agency.core.planning.smoothers.hermite.algorithm import _CubicHermiteSpline

_X = np.array([0.0, 1.0, 2.5, 4.0, 6.0])
_Y = np.array([0.0, 1.0, 0.5, 2.0, 1.5])
_D = np.array([1.0, 0.2, -0.5, 0.3, 0.0])


def test_cubic_hermite_endpoints_and_knot_derivatives():
    s = _CubicHermiteSpline(_X, _Y, _D)
    # Interpolates the knot values exactly...
    for xi, yi in zip(_X, _Y):
        assert abs(s(xi) - yi) < 1e-9, (xi, yi)
    # ...and reproduces the knot slopes (1st derivative).
    for xi, di in zip(_X, _D):
        assert abs(s(xi, 1) - di) < 1e-9, (xi, di)


def test_cubic_hermite_scalar_vs_array():
    s = _CubicHermiteSpline(_X, _Y, _D)
    xq = np.linspace(0.0, 6.0, 23)
    for nu in (0, 1, 2):
        arr = s(xq, nu)
        assert isinstance(arr, np.ndarray) and arr.shape == xq.shape
        for i, x in enumerate(xq):
            assert abs(float(s(x, nu)) - arr[i]) < 1e-12


def test_cubic_hermite_rejects_bad_knots():
    for bad in (np.array([0.0]), np.array([1.0, 1.0, 2.0]), np.array([2.0, 1.0])):
        raised = False
        try:
            _CubicHermiteSpline(bad, np.zeros_like(bad), np.zeros_like(bad))
        except ValueError:
            raised = True
        assert raised, bad


def test_parity_with_scipy_if_available():
    """Bit-parity with scipy across value/1st/2nd derivative (skipped if scipy
    is not installed, e.g. on the Noetic target)."""
    try:
        from scipy.interpolate import CubicHermiteSpline as _SP
    except ImportError:
        return  # scipy absent (the very environment this replacement is for)
    ours = _CubicHermiteSpline(_X, _Y, _D)
    ref = _SP(_X, _Y, _D)
    xq = np.linspace(_X[0], _X[-1], 101)
    for nu in (0, 1, 2):
        assert np.max(np.abs(ours(xq, nu) - ref(xq, nu))) < 1e-9


def test_smoother_produces_trajectory_without_scipy():
    """End-to-end: the smoother yields a dense trajectory through the endpoints
    with finite velocity/curvature -- no scipy involved."""
    path = Path2D(points=tuple(Pose2D(x, y) for x, y in
                               [(0, 0), (2, 0.5), (4, 1.0), (6, 0.0)]))
    traj = HermiteSmoother(HermiteParams()).smooth(
        SmootherRequest(path=path, limits=KinematicLimits(max_speed_xy=0.5,
                                                          max_yaw_rate=0.6)))
    samples = traj.sample_by_time(0.1)
    assert len(samples) > 10
    assert math.hypot(samples[0].x - 0.0, samples[0].y - 0.0) < 0.05
    assert math.hypot(samples[-1].x - 6.0, samples[-1].y - 0.0) < 0.05
    assert all(math.isfinite(s.vx) and math.isfinite(s.vy) for s in samples)
    assert all(s.curvature is not None and math.isfinite(s.curvature) for s in samples)


if __name__ == "__main__":
    import sys
    mod = sys.modules[__name__]
    fns = [getattr(mod, n) for n in dir(mod) if n.startswith("test_")]
    for fn in fns:
        fn()
        print("ok:", fn.__name__)
    print("all %d tests passed" % len(fns))
