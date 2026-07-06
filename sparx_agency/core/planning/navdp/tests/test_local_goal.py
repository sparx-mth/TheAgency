"""Tests for the A* -> NavDP local-goal selection (visibility + contiguity).

The drone is placed at the world origin facing ``+x`` (``ref = (0, 0, 0)``), so a
world waypoint ``(x, y)`` equals its body ``(forward, left)`` -- the projection
and occlusion logic can then be read straight off the waypoint coordinates.
"""
import numpy as np
import pytest

from sparx_agency.core.common.types import Intrinsics
from sparx_agency.core.planning.navdp.local_goal import (
    LocalGoal,
    point_visible,
    select_farthest_visible_waypoint,
)

# Axis-aligned pinhole, principal point centred. With cam_height 0.5 m a ground
# point is in-frame vertically once forward > fy*0.5/(H-cy) = 400*0.5/196 ~= 1.02 m.
INTR = Intrinsics(width=504, height=392, fx=400.0, fy=400.0, cx=252.0, cy=196.0)
CAM_H = 0.5
# Straight-ahead route at 0,1,2,3,4,5 m (world == body here).
ROUTE = [(0.0, 0.0), (1.0, 0.0), (2.0, 0.0), (3.0, 0.0), (4.0, 0.0), (5.0, 0.0)]


def _const_depth(value, h=392, w=504):
    return np.full((h, w), float(value), dtype=np.float32)


def _select(depth, **kw):
    return select_farthest_visible_waypoint(
        ROUTE, 0.0, 0.0, 0.0, depth, INTR, cam_height_m=CAM_H, **kw)


# ── point_visible ────────────────────────────────────────────────────
def test_visible_in_frame_and_unoccluded():
    assert point_visible(3.0, 0.0, _const_depth(10.0), INTR, CAM_H) is True


def test_too_close_is_not_visible():
    # forward below min_fwd_m -> at/under the camera, never visible.
    assert point_visible(0.1, 0.0, _const_depth(10.0), INTR, CAM_H) is False


def test_below_frame_is_not_visible():
    # forward 1.0 m projects below the image bottom (v ~= 396 > 392).
    assert point_visible(1.0, 0.0, _const_depth(10.0), INTR, CAM_H) is False


def test_occluded_by_nearer_surface():
    # A wall reads 2.0 m; a 3.0 m waypoint is behind it (3 > 2 + 0.5 tol).
    assert point_visible(3.0, 0.0, _const_depth(2.0), INTR, CAM_H) is False


def test_occlusion_ignored_when_not_required():
    assert point_visible(3.0, 0.0, _const_depth(2.0), INTR, CAM_H,
                         require_unoccluded=False) is True


# ── select_farthest_visible_waypoint ─────────────────────────────────
def test_picks_farthest_when_corridor_open():
    goal = _select(_const_depth(10.0))
    assert isinstance(goal, LocalGoal)
    assert goal.index == 5                      # the 5 m waypoint
    assert goal.world == pytest.approx((5.0, 0.0))
    assert goal.body == pytest.approx((5.0, 0.0))
    assert goal.goal == pytest.approx((5.0, 0.0))   # within NavDP range -> identity


def test_stops_at_wall():
    # Wall at 2.0 m: visible iff forward <= 2.5, so 2 m passes and 3 m breaks.
    goal = _select(_const_depth(2.0))
    assert goal is not None and goal.index == 2


def test_contiguity_rejects_waypoint_peeking_past_occluder():
    # Far everywhere, but a near patch occludes ONLY the 3 m waypoint's pixel.
    # Naive "farthest visible" would jump to 5 m; contiguity keeps the 2 m one.
    depth = _const_depth(10.0)
    depth[256:269, 246:259] = 1.0               # box around (u=252, v~263) = wp@3 m
    goal = _select(depth)
    assert goal is not None and goal.index == 2


def test_ignoring_occlusion_picks_farthest_in_frame():
    depth = _const_depth(2.0)
    goal = _select(depth, require_unoccluded=False)
    assert goal is not None and goal.index == 5


def test_none_when_nothing_visible():
    # Both waypoints are too close / below the frame -> no local goal.
    route = [(0.0, 0.0), (0.5, 0.0)]
    goal = select_farthest_visible_waypoint(
        route, 0.0, 0.0, 0.0, _const_depth(10.0), INTR, cam_height_m=CAM_H)
    assert goal is None


def test_no_valid_depth_fails_closed():
    # An all-invalid depth (no returns) is treated as NOT clear -> not visible,
    # so a goal is never placed past an unmeasured near occluder.
    invalid = np.zeros((392, 504), dtype=np.float32)   # all 0 -> no valid reading
    assert point_visible(3.0, 0.0, invalid, INTR, CAM_H) is False
    # ...but with the occlusion test off, it is in-frame -> visible.
    assert point_visible(3.0, 0.0, invalid, INTR, CAM_H,
                         require_unoccluded=False) is True


def test_route_turning_out_of_view_returns_none_not_leap():
    # The route turns hard left at 2 m (out of the horizontal FOV) before any
    # waypoint is visible, then a far waypoint at 4 m is straight ahead again.
    # Selection must NOT leap across the turn -> None (caller falls back to A*).
    route = [(0.0, 0.0), (2.0, 5.0), (4.0, 0.0)]
    goal = select_farthest_visible_waypoint(
        route, 0.0, 0.0, 0.0, _const_depth(10.0), INTR, cam_height_m=CAM_H)
    assert goal is None


def test_goal_is_scaled_into_navdp_range():
    # A far lateral waypoint (well outside NavDP's range) is scaled, bearing kept.
    route = [(0.0, 0.0), (3.0, 0.0), (40.0, 20.0)]
    # Far valid depth + big tolerance so both forward waypoints read unoccluded;
    # the 40 m lateral goal projects near the horizon and must be scaled into range.
    goal = select_farthest_visible_waypoint(
        route, 0.0, 0.0, 0.0, _const_depth(45.0), INTR, cam_height_m=CAM_H,
        depth_tol_m=100.0)
    assert goal is not None and goal.index == 2
    gx, gy = goal.goal
    assert gx <= 10.0 + 1e-6 and abs(gy) <= 10.0 + 1e-6
    # bearing preserved
    assert np.arctan2(gy, gx) == pytest.approx(np.arctan2(20.0, 40.0), abs=1e-6)
