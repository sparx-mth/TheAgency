"""Tests for :func:`arclength_fraction_2d` (progress along a polyline)."""
import pytest

from sparx_agency.core.common.types import Pose2D
from sparx_agency.core.planning.planners.common.utils_2d import arclength_fraction_2d


def _line(n, step):
    return [Pose2D(i * step, 0.0) for i in range(n)]


def test_start_is_zero():
    pts = _line(5, 1.0)                          # 4 m straight line
    assert arclength_fraction_2d(pts, Pose2D(0.0, 0.0)) == pytest.approx(0.0)


def test_end_is_one():
    pts = _line(5, 1.0)
    assert arclength_fraction_2d(pts, Pose2D(4.0, 0.0)) == pytest.approx(1.0)


def test_midpoint_is_half():
    pts = _line(5, 1.0)                          # length 4 -> midpoint at x=2
    assert arclength_fraction_2d(pts, Pose2D(2.0, 0.0)) == pytest.approx(0.5)


def test_off_path_projects_perpendicularly():
    # A query 1 m to the side of x=2 still projects to the midpoint.
    pts = _line(5, 1.0)
    assert arclength_fraction_2d(pts, Pose2D(2.0, 1.0)) == pytest.approx(0.5)


def test_before_start_clamps_to_zero():
    pts = _line(5, 1.0)
    assert arclength_fraction_2d(pts, Pose2D(-5.0, 0.0)) == pytest.approx(0.0)


def test_past_end_clamps_to_one():
    pts = _line(5, 1.0)
    assert arclength_fraction_2d(pts, Pose2D(99.0, 0.0)) == pytest.approx(1.0)


def test_l_shaped_path_uses_arclength_not_euclidean():
    # An L: (0,0)->(2,0)->(2,2). Total length 4. The corner is at arclength 2 = 0.5.
    pts = [Pose2D(0.0, 0.0), Pose2D(2.0, 0.0), Pose2D(2.0, 2.0)]
    assert arclength_fraction_2d(pts, Pose2D(2.0, 0.0)) == pytest.approx(0.5)
    assert arclength_fraction_2d(pts, Pose2D(2.0, 1.0)) == pytest.approx(0.75)


def test_degenerate_paths_return_zero():
    assert arclength_fraction_2d([], Pose2D(0.0, 0.0)) == 0.0
    assert arclength_fraction_2d([Pose2D(1.0, 1.0)], Pose2D(0.0, 0.0)) == 0.0
    # Zero-length polyline (all coincident) -> 0.0, no division by zero.
    coincident = [Pose2D(1.0, 1.0), Pose2D(1.0, 1.0)]
    assert arclength_fraction_2d(coincident, Pose2D(1.0, 1.0)) == 0.0
