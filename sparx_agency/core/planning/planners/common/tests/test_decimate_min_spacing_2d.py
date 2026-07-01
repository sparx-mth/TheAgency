"""Tests for :func:`decimate_min_spacing_2d` (dense-path thinning)."""
from math import hypot

from sparx_agency.core.common.types import Pose2D
from sparx_agency.core.planning.planners.common.utils_2d import decimate_min_spacing_2d


def _line(n, step):
    return [Pose2D(i * step, 0.0) for i in range(n)]


def test_thins_dense_path_to_min_spacing():
    pts = _line(24, 0.1)                       # 24 points, 0.1 m apart (2.3 m long)
    out = decimate_min_spacing_2d(pts, 0.5)
    # Endpoints preserved; every kept gap >= 0.5 m (last gap may shrink to land on end).
    assert out[0] is pts[0]
    assert out[-1] is pts[-1]
    gaps = [hypot(b.x - a.x, b.y - a.y) for a, b in zip(out[:-1], out[1:])]
    assert all(g >= 0.5 - 1e-9 for g in gaps[:-1])
    assert len(out) < len(pts)                 # actually thinned


def test_sparse_path_unchanged():
    pts = _line(5, 1.5)                         # already 1.5 m apart (A*-like)
    out = decimate_min_spacing_2d(pts, 0.5)
    assert [(p.x, p.y) for p in out] == [(p.x, p.y) for p in pts]


def test_disabled_or_too_short_returns_input():
    pts = _line(10, 0.1)
    assert len(decimate_min_spacing_2d(pts, 0.0)) == len(pts)     # disabled
    assert len(decimate_min_spacing_2d(pts[:2], 0.5)) == 2        # n < 3


def test_keeps_at_least_endpoints_when_all_crowded():
    pts = _line(6, 0.05)                        # 0.25 m total, all closer than 0.5
    out = decimate_min_spacing_2d(pts, 0.5)
    assert len(out) == 2 and out[0] is pts[0] and out[-1] is pts[-1]
