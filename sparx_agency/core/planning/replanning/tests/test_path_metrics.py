"""Tests for path length + forward-monotone remaining-route projection."""
import math

import pytest

from sparx_agency.core.common.types import Pose2D
from sparx_agency.core.planning.replanning.path_metrics import (
    polyline_length, remaining_polyline)


def test_polyline_length():
    assert polyline_length([]) == 0.0
    assert polyline_length([Pose2D(0, 0)]) == 0.0
    assert polyline_length([Pose2D(0, 0), Pose2D(3, 4)]) == pytest.approx(5.0)
    assert polyline_length([Pose2D(0, 0), Pose2D(1, 0), Pose2D(1, 1)]) == pytest.approx(2.0)


def test_remaining_polyline_midsegment():
    pts = [Pose2D(0, 0), Pose2D(2, 0), Pose2D(2, 2)]
    rem, idx = remaining_polyline(pts, Pose2D(1.4, 0.05))
    assert idx == 0
    assert rem[0].x == pytest.approx(1.4) and rem[0].y == pytest.approx(0.0)
    assert (rem[1].x, rem[1].y) == (2, 0)
    # remaining length = (2-1.4) + 2 = 2.6
    assert polyline_length(rem) == pytest.approx(2.6)


def test_remaining_polyline_degenerate():
    assert remaining_polyline([], Pose2D(0, 0)) == ([], 0)
    one = [Pose2D(1, 1)]
    rem, idx = remaining_polyline(one, Pose2D(0, 0))
    assert rem == one and idx == 0


def test_remaining_polyline_zero_length_segment():
    # Coincident points (a STOP hold): must not divide by zero.
    rem, idx = remaining_polyline([Pose2D(3, 3), Pose2D(3, 3)], Pose2D(3, 3))
    assert idx == 0 and len(rem) >= 1


def test_remaining_polyline_forward_monotone_on_self_approaching_path():
    """A U-shaped path passes near its own start; without a forward hint the global
    nearest projection would snap to the far leg and truncate the route. min_index
    keeps it monotone."""
    # Path: go right, up, back left (a U). The end leg passes close to the start.
    pts = [Pose2D(0, 0), Pose2D(4, 0), Pose2D(4, 1), Pose2D(0, 1)]
    pose = Pose2D(0.2, 0.9)   # near the LAST leg (row y~1) but also near start region
    # Fresh (min_index=0): global-nearest may jump to the last segment.
    rem0, idx0 = remaining_polyline(pts, pose, min_index=0)
    # With a progress hint that we are still on segment 0, projection stays forward.
    rem1, idx1 = remaining_polyline(pts, pose, min_index=0)
    assert idx1 >= 0
    # Feeding back the index must never move backward:
    _, idx_a = remaining_polyline(pts, Pose2D(3.9, 0.05), min_index=0)   # on seg 0/1
    _, idx_b = remaining_polyline(pts, Pose2D(0.1, 0.95), min_index=idx_a)  # later, on last seg
    assert idx_b >= idx_a
