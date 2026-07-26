"""Tests for route_obstacle_confidence -- confidence of an on-route obstacle."""
import numpy as np
import pytest

from sparx_agency.core.common.types import Pose2D
from sparx_agency.core.planning.environment import (
    OccupancyGrid2D, OccupancyGrid2DParams, OccupancyValues)
from sparx_agency.core.planning.replanning import route_obstacle_confidence

BEV = OccupancyValues(free=0, occupied=100, unknown=-1)
RES = 0.1
H = W = 20


def _grid(arr):
    params = OccupancyGrid2DParams(resolution=RES, origin_x=0.0, origin_y=0.0)
    return OccupancyGrid2D(arr.astype(np.int16), params, values=BEV)


def _straight_route():
    """A horizontal route along row y=1.0 (grid row 10) from x=0.2 to x=1.8."""
    return [Pose2D(0.2, 1.0), Pose2D(1.8, 1.0)]


def test_none_confidence_returns_none():
    g = _grid(np.zeros((H, W), np.int16))
    assert route_obstacle_confidence(g, None, _straight_route(), 1) is None


def test_shape_mismatch_returns_none():
    g = _grid(np.zeros((H, W), np.int16))
    bad = np.zeros((H + 1, W), np.float32)
    assert route_obstacle_confidence(g, bad, _straight_route(), 1) is None


def test_no_obstacle_on_route_returns_zero():
    g = _grid(np.zeros((H, W), np.int16))          # all FREE
    conf = np.full((H, W), 0.9, np.float32)        # high everywhere, but no OCC
    assert route_obstacle_confidence(g, conf, _straight_route(), 1) == 0.0


def test_off_route_obstacle_is_ignored():
    arr = np.zeros((H, W), np.int16)
    arr[2, 10] = 100                               # OCC far from the y=1.0 route
    g = _grid(arr)
    conf = np.zeros((H, W), np.float32)
    conf[2, 10] = 0.95
    # radius 1 cell corridor around row 10 does not reach row 2.
    assert route_obstacle_confidence(g, conf, _straight_route(), 1) == 0.0


def test_on_route_obstacle_returns_its_confidence():
    arr = np.zeros((H, W), np.int16)
    arr[10, 10] = 100                              # OCC on the route (row 10)
    g = _grid(arr)
    conf = np.zeros((H, W), np.float32)
    conf[10, 10] = 0.42
    got = route_obstacle_confidence(g, conf, _straight_route(), 1)
    assert got == pytest.approx(0.42, abs=1e-6)


def test_returns_peak_over_multiple_on_route_obstacles():
    arr = np.zeros((H, W), np.int16)
    arr[10, 8] = 100
    arr[10, 12] = 100
    g = _grid(arr)
    conf = np.zeros((H, W), np.float32)
    conf[10, 8] = 0.30
    conf[10, 12] = 0.80                            # the stronger one wins
    got = route_obstacle_confidence(g, conf, _straight_route(), 1)
    assert got == pytest.approx(0.80, abs=1e-6)


def test_radius_widens_the_inspected_band():
    arr = np.zeros((H, W), np.int16)
    arr[12, 10] = 100                              # 2 rows off the route line
    g = _grid(arr)
    conf = np.zeros((H, W), np.float32)
    conf[12, 10] = 0.7
    # radius 1 does not reach row 12; radius 2 does.
    assert route_obstacle_confidence(g, conf, _straight_route(), 1) == 0.0
    assert route_obstacle_confidence(g, conf, _straight_route(), 2) == pytest.approx(
        0.7, abs=1e-6)
