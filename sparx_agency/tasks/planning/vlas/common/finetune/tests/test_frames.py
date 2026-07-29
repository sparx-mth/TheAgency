"""Tests for the per-frame depth -> body-FLU occupancy primitives (numpy, no torch)."""
import numpy as np
import pytest

from sparx_agency.core.common.types import Intrinsics
from sparx_agency.tasks.planning.vlas.common.finetune.common.frames import (
    LocalMapConfig,
    cloud_to_occupancy_grid,
    depth_to_body_cloud,
    occupancy_binary,
    occupancy_probability,
)

INTR = Intrinsics(width=100, height=100, fx=100.0, fy=100.0, cx=50.0, cy=50.0)


def test_forward_axis_maps_to_depth():
    # A point at the principal point with depth 3 should be straight ahead: fwd=3.
    depth = np.zeros((100, 100), np.float32)
    depth[50, 50] = 3.0
    cfg = LocalMapConfig(camera_height_m=0.0, pitch_deg=0.0, stride=1, depth_range_m=(0.1, 15.0))
    cloud = depth_to_body_cloud(depth, INTR, cfg)
    assert cloud.shape[1] == 3
    # exactly one valid pixel
    assert cloud.shape[0] == 1
    fwd, left, up = cloud[0]
    assert fwd == pytest.approx(3.0, abs=1e-4)
    assert left == pytest.approx(0.0, abs=1e-4)
    assert up == pytest.approx(0.0, abs=1e-4)


def test_left_of_image_is_positive_left():
    # A pixel left of the principal point (u < cx) must land on +left (body).
    depth = np.zeros((100, 100), np.float32)
    depth[50, 10] = 2.0
    cfg = LocalMapConfig(camera_height_m=0.0, pitch_deg=0.0, stride=1)
    cloud = depth_to_body_cloud(depth, INTR, cfg)
    assert cloud[0, 1] > 0.0  # left > 0


def test_camera_height_lifts_points():
    depth = np.zeros((100, 100), np.float32)
    depth[50, 50] = 3.0
    c0 = depth_to_body_cloud(depth, INTR, LocalMapConfig(camera_height_m=0.0, pitch_deg=0.0, stride=1))
    c1 = depth_to_body_cloud(depth, INTR, LocalMapConfig(camera_height_m=1.0, pitch_deg=0.0, stride=1))
    assert c1[0, 2] == pytest.approx(c0[0, 2] + 1.0, abs=1e-4)


def test_occupancy_grid_shape_and_origin():
    cfg = LocalMapConfig(resolution_m=0.1, forward_extent_m=5.0, half_width_m=2.0)
    grid = cloud_to_occupancy_grid(np.zeros((0, 3), np.float32), cfg)
    assert grid.grid.shape == (40, 50)  # (n_left, n_fwd)
    assert grid.origin_x == 0.0
    assert grid.origin_y == pytest.approx(-2.0)
    # empty cloud -> all free
    assert not occupancy_binary(grid).any()
    assert occupancy_probability(grid).max() == 0.0


def test_height_band_filters_floor_and_ceiling():
    cfg = LocalMapConfig(resolution_m=0.1, forward_extent_m=5.0, half_width_m=2.0,
                         z_band_m=(0.2, 1.5))
    # three points at fwd=2, left=0: one on the floor, one in-band, one on the ceiling
    cloud = np.array([[2.0, 0.0, 0.0],   # floor -> filtered
                      [2.0, 0.0, 1.0],   # in band -> kept
                      [2.0, 0.0, 3.0]],  # ceiling -> filtered
                     dtype=np.float32)
    grid = cloud_to_occupancy_grid(cloud, cfg)
    assert int(occupancy_binary(grid).sum()) == 1
