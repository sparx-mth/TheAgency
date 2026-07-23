"""Tests for NavDP / FlowNav label encoding (numpy, no torch)."""
import numpy as np
import pytest

from sparx_agency.core.common.types import Path2D, Pose2D
from sparx_agency.tasks.planning.vlas.common.finetune.common.label_format import (
    resample_arclength,
    to_flownav_label,
    to_navdp_label,
)


def _straight_path(length=6.0, n=13):
    xs = np.linspace(0, length, n)
    return np.stack([xs, np.zeros_like(xs)], axis=1).astype(np.float32)


def test_resample_preserves_endpoints():
    p = _straight_path(6.0, 13)
    r = resample_arclength(p, 25)
    assert r.shape == (25, 2)
    assert r[0] == pytest.approx(p[0], abs=1e-4)
    assert r[-1] == pytest.approx(p[-1], abs=1e-4)


def test_resample_degenerate():
    p = np.zeros((5, 2), np.float32)
    r = resample_arclength(p, 8)
    assert r.shape == (8, 2)
    assert np.all(r == 0)


def test_navdp_label_shape_and_reconstruction():
    # straight 6 m path over 24 steps -> 0.25 m/step -> action dx = 4*0.25 = 1.0
    label = to_navdp_label(_straight_path(6.0, 13), horizon=24)
    assert label.shape == (24, 3)
    assert label.min() >= -1.0 and label.max() <= 1.0
    # reconstruct: cumsum(action/4) should recover a monotonically forward path
    traj = np.cumsum(label[:, :2] / 4.0, axis=0)
    assert traj[-1, 0] == pytest.approx(6.0, abs=0.3)
    assert np.all(np.diff(traj[:, 0]) > 0)


def test_navdp_label_clamped():
    # a huge single step must clamp, not overflow
    p = np.array([[0, 0], [10, 0]], np.float32)
    label = to_navdp_label(p, horizon=24)
    assert label.max() <= 1.0 and label.min() >= -1.0


def test_flownav_label_shape_and_units():
    label = to_flownav_label(_straight_path(2.0, 9), horizon=8, metric_waypoint_spacing=0.25)
    assert label.shape == (8, 2)
    # last absolute waypoint ~ 2 m -> /0.25 = 8 waypoint-units
    assert label[-1, 0] == pytest.approx(8.0, abs=0.5)
    # monotonically increasing forward (absolute waypoints)
    assert np.all(np.diff(label[:, 0]) > 0)


def test_flownav_horizon_divisible_by_four():
    with pytest.raises(ValueError):
        to_flownav_label(_straight_path(2.0, 9), horizon=6)


def test_accepts_path2d():
    path = Path2D(points=tuple(Pose2D(float(x), 0.0) for x in np.linspace(0, 6, 13)))
    label = to_navdp_label(path)
    assert label.shape == (24, 3)
