"""Unit tests for the ROS-free NavDP geometry (numpy-only, no ROS, no server)."""
import math

import numpy as np
import pytest

from sparx_agency.core.common.types import Intrinsics
from sparx_agency.core.planning.navdp.geometry import (
    NAVDP_MAX_FWD_M,
    NAVDP_MAX_LAT_M,
    anchor_trajectory_to_world,
    body_point_to_pixel,
    patch_median_depth,
    pixel_to_pointgoal,
    project_trajectory_to_pixels,
)

# A simple, axis-aligned pinhole: principal point at the image centre.
INTR = Intrinsics(width=504, height=392, fx=400.0, fy=400.0, cx=252.0, cy=196.0)


def _const_depth(value, h=392, w=504):
    return np.full((h, w), float(value), dtype=np.float32)


# ── patch_median_depth ───────────────────────────────────────────────
def test_patch_median_ignores_invalid():
    depth = _const_depth(2.5)
    depth[196, 252] = 0.0          # a hole at the centre
    depth[195, 251] = np.nan
    assert patch_median_depth(depth, 252, 196) == pytest.approx(2.5)


def test_patch_median_none_when_all_invalid():
    depth = np.zeros((50, 50), dtype=np.float32)
    assert patch_median_depth(depth, 25, 25) is None


# ── pixel_to_pointgoal ───────────────────────────────────────────────
def test_pointgoal_centre_is_straight_ahead():
    depth = _const_depth(3.0)
    gx, gy, d, bz = pixel_to_pointgoal(INTR.cx, INTR.cy, depth, INTR)
    assert gx == pytest.approx(3.0)
    assert gy == pytest.approx(0.0)
    assert d == pytest.approx(3.0)
    assert bz == pytest.approx(0.0)


def test_pointgoal_left_pixel_is_positive_left():
    # A pixel left of centre (u < cx) maps to +y (left) in the body frame.
    depth = _const_depth(4.0)
    u = INTR.cx - 100.0
    gx, gy, d, _ = pixel_to_pointgoal(u, INTR.cy, depth, INTR)
    assert gy > 0.0
    assert gy == pytest.approx(-(u - INTR.cx) * 4.0 / INTR.fx)
    assert gx == pytest.approx(4.0)


def test_pointgoal_scaling_preserves_bearing():
    # A far click (forward > 10 m) must be scaled as a whole, keeping its bearing.
    depth = _const_depth(40.0)         # 40 m forward, well past NAVDP_MAX_FWD_M
    u = INTR.cx - 200.0
    gx, gy, d, _ = pixel_to_pointgoal(u, INTR.cy, depth, INTR)
    bx_raw = 40.0
    by_raw = -(u - INTR.cx) * 40.0 / INTR.fx
    assert math.atan2(gy, gx) == pytest.approx(math.atan2(by_raw, bx_raw), abs=1e-6)
    assert gx <= NAVDP_MAX_FWD_M + 1e-6
    assert abs(gy) <= NAVDP_MAX_LAT_M + 1e-6
    assert d == pytest.approx(40.0)


def test_pointgoal_fallback_depth_when_no_valid():
    depth = np.zeros((50, 50), dtype=np.float32)
    gx, gy, d, _ = pixel_to_pointgoal(25, 25, depth, INTR, fallback_depth_m=3.0)
    assert d == pytest.approx(3.0)
    assert gx == pytest.approx(3.0)


# ── anchor_trajectory_to_world ───────────────────────────────────────
def test_anchor_identity_pose():
    traj = [(1.0, 0.0), (2.0, 0.5)]
    out = anchor_trajectory_to_world(traj, 0.0, 0.0, 0.0)
    assert out[0] == pytest.approx((1.0, 0.0))
    assert out[1] == pytest.approx((2.0, 0.5))


def test_anchor_yaw_90deg():
    # Facing +y (yaw=90deg): forward maps to +y, left maps to -x.
    out = anchor_trajectory_to_world([(2.0, 1.0)], 10.0, 5.0, math.pi / 2.0)
    # world_x = 10 + 2*cos90 - 1*sin90 = 10 - 1 = 9
    # world_y = 5  + 2*sin90 + 1*cos90 = 5 + 2 = 7
    assert out[0] == pytest.approx((9.0, 7.0))


def test_anchor_ignores_extra_columns():
    traj = np.array([[1.0, 0.0, 0.3], [2.0, 0.0, 0.1]], dtype=np.float32)  # fwd,left,yaw
    out = anchor_trajectory_to_world(traj, 1.0, 2.0, 0.0)
    assert out[0] == pytest.approx((2.0, 2.0))
    assert out[1] == pytest.approx((3.0, 2.0))


# ── body_point_to_pixel / project_trajectory_to_pixels ───────────────
def test_body_point_behind_camera_is_none():
    assert body_point_to_pixel(0.0, 0.0, INTR, cam_height_m=0.5) is None


def test_body_point_straight_ahead_on_ground():
    # Straight ahead (y_left=0) projects to u = cx; v below the centre (floor).
    u, v = body_point_to_pixel(5.0, 0.0, INTR, cam_height_m=0.5)
    assert u == pytest.approx(int(INTR.cx))
    assert v == int(INTR.fy * 0.5 / 5.0 + INTR.cy)
    assert v > INTR.cy            # ground is below the optical centre


def test_project_trajectory_length_matches():
    traj = [(0.01, 0.0), (1.0, 0.0), (2.0, -0.5)]
    px = project_trajectory_to_pixels(traj, INTR, cam_height_m=0.5)
    assert len(px) == 3
    assert px[0] is None          # 0.01 m < min_fwd_m -> dropped
    assert px[1] is not None and px[2] is not None
