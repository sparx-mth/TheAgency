"""Tests for detection->world geometry: rescale, robust depth, back-projection.

Synthetic pinholes throughout; the back-projection cases pin the frame chain
(optical -> body FLU -> world ENU) and act as a regression net for the old
SJTU semantic-mapper bug that rotated the optical ray by the body pose
directly.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from sparx_agency.core.common.math.se3 import quaternion_matrix
from sparx_agency.core.mapping.objects.geometry import (
    backproject_bbox_to_world,
    rescale_bbox_between_intrinsics,
    robust_bbox_depth,
)

DEPTH_K = (320.0, 320.0, 320.0, 240.0)   # fx, fy, cx, cy
RGB_K = (186.0, 186.0, 320.0, 180.0)


def _bbox_at(u, v, w=20.0, h=20.0):
    """Axis-aligned box of size (w, h) centered on pixel (u, v)."""
    return (u - 0.5 * w, v - 0.5 * h, u + 0.5 * w, v + 0.5 * h)


def _yaw_rotation(yaw):
    """World-from-body rotation for a level pose at the given yaw (radians)."""
    q = (0.0, 0.0, math.sin(0.5 * yaw), math.cos(0.5 * yaw))  # [x, y, z, w]
    return quaternion_matrix(q)[:3, :3]


class TestBackprojectBboxToWorld:
    def test_center_pixel_identity_pose_lands_ahead_on_body_x(self):
        fx, fy, cx, cy = DEPTH_K
        d = 2.5
        p = backproject_bbox_to_world(_bbox_at(cx, cy), d, DEPTH_K,
                                      np.eye(3), (0.0, 0.0, 0.0))
        assert p == pytest.approx((d, 0.0, 0.0), abs=1e-12)

    def test_old_bug_regression_not_straight_up(self):
        """The old stack rotated the optical ray by the body pose directly,
        so a dead-ahead object at depth d landed at (0, 0, d) — d meters
        ABOVE the drone — for an identity pose. Pin that it no longer does."""
        fx, fy, cx, cy = DEPTH_K
        d = 3.0
        p = np.array(backproject_bbox_to_world(_bbox_at(cx, cy), d, DEPTH_K,
                                               np.eye(3), (0.0, 0.0, 0.0)))
        assert not np.allclose(p, [0.0, 0.0, d])   # the buggy answer
        assert np.allclose(p, [d, 0.0, 0.0])       # the correct one

    def test_yaw_rotated_pose_rotates_and_translates(self):
        fx, fy, cx, cy = DEPTH_K
        d = 2.0
        t = (1.0, 2.0, 0.5)
        p = backproject_bbox_to_world(_bbox_at(cx, cy), d, DEPTH_K,
                                      _yaw_rotation(math.pi / 2), t)
        # Facing +y (east->north turn): the object is d meters north.
        assert p == pytest.approx((1.0, 2.0 + d, 0.5), abs=1e-12)

    def test_pixel_right_of_center_is_world_minus_y(self):
        fx, fy, cx, cy = DEPTH_K
        d = 2.0
        # One normalized unit right of the optical axis: optical x = d.
        p = backproject_bbox_to_world(_bbox_at(cx + fx, cy), d, DEPTH_K,
                                      np.eye(3), (0.0, 0.0, 0.0))
        # Right of the camera = body -y = world -y at identity pose.
        assert p == pytest.approx((d, -d, 0.0), abs=1e-12)

    def test_pixel_below_center_is_world_minus_z(self):
        fx, fy, cx, cy = DEPTH_K
        d = 2.0
        p = backproject_bbox_to_world(_bbox_at(cx, cy + fy), d, DEPTH_K,
                                      np.eye(3), (0.0, 0.0, 0.0))
        # Below the optical axis = body -z = below the drone.
        assert p == pytest.approx((d, 0.0, -d), abs=1e-12)

    def test_camera_offset_is_applied_in_the_body_frame(self):
        fx, fy, cx, cy = DEPTH_K
        d = 2.0
        offset = (0.1, 0.2, -0.05)
        p = backproject_bbox_to_world(_bbox_at(cx, cy), d, DEPTH_K,
                                      _yaw_rotation(math.pi), (0.0, 0.0, 0.0),
                                      camera_offset_body=offset)
        # v_body = (d + 0.1, 0.2, -0.05); yaw pi flips x and y in world.
        assert p == pytest.approx((-(d + 0.1), -0.2, -0.05), abs=1e-12)

    @pytest.mark.parametrize("bad_depth", [0.0, -1.0, float("nan"),
                                           float("inf")])
    def test_bad_depth_raises(self, bad_depth):
        with pytest.raises(ValueError):
            backproject_bbox_to_world(_bbox_at(320.0, 240.0), bad_depth,
                                      DEPTH_K, np.eye(3), (0.0, 0.0, 0.0))

    def test_bad_rotation_shape_raises(self):
        with pytest.raises(ValueError):
            backproject_bbox_to_world(_bbox_at(320.0, 240.0), 1.0, DEPTH_K,
                                      np.eye(4), (0.0, 0.0, 0.0))


class TestRescaleBboxBetweenIntrinsics:
    def test_principal_center_maps_to_principal_center(self):
        fx_r, fy_r, cx_r, cy_r = RGB_K
        fx_d, fy_d, cx_d, cy_d = DEPTH_K
        out = rescale_bbox_between_intrinsics(_bbox_at(cx_r, cy_r, 40.0, 30.0),
                                              RGB_K, DEPTH_K)
        ox = 0.5 * (out[0] + out[2])
        oy = 0.5 * (out[1] + out[3])
        assert (ox, oy) == pytest.approx((cx_d, cy_d))
        # Sizes scale by the focal ratio.
        assert out[2] - out[0] == pytest.approx(40.0 * fx_d / fx_r)
        assert out[3] - out[1] == pytest.approx(30.0 * fy_d / fy_r)

    def test_matches_the_old_node_center_math(self):
        """Same numbers as ObjectMapper._det_cb: kx = fx_d/fx_r etc."""
        fx_r, fy_r, cx_r, cy_r = RGB_K
        fx_d, fy_d, cx_d, cy_d = DEPTH_K
        u_r, v_r, w, h = 400.0, 200.0, 40.0, 30.0
        kx, ky = fx_d / fx_r, fy_d / fy_r
        out = rescale_bbox_between_intrinsics(_bbox_at(u_r, v_r, w, h),
                                              RGB_K, DEPTH_K)
        assert 0.5 * (out[0] + out[2]) == pytest.approx((u_r - cx_r) * kx + cx_d)
        assert 0.5 * (out[1] + out[3]) == pytest.approx((v_r - cy_r) * ky + cy_d)
        assert out[2] - out[0] == pytest.approx(w * kx)
        assert out[3] - out[1] == pytest.approx(h * ky)

    def test_round_trip_recovers_the_box(self):
        bbox = (100.0, 50.0, 220.0, 130.0)
        there = rescale_bbox_between_intrinsics(bbox, RGB_K, DEPTH_K)
        back = rescale_bbox_between_intrinsics(there, DEPTH_K, RGB_K)
        assert back == pytest.approx(bbox)

    def test_identity_when_intrinsics_match(self):
        bbox = (10.0, 20.0, 30.0, 40.0)
        assert rescale_bbox_between_intrinsics(bbox, DEPTH_K, DEPTH_K) \
            == pytest.approx(bbox)

    def test_nonpositive_focal_raises(self):
        with pytest.raises(ValueError):
            rescale_bbox_between_intrinsics((0, 0, 1, 1),
                                            (0.0, 186.0, 320.0, 180.0),
                                            DEPTH_K)


class TestRobustBboxDepth:
    # bbox (200, 200, 280, 280): center (240, 240), half-size 40; with the
    # default shrink 0.5 the sampled patch is exactly [220:260, 220:260).
    BBOX = (200.0, 200.0, 280.0, 280.0)

    @staticmethod
    def _blank():
        return np.full((480, 640), np.nan, dtype=np.float32)

    def test_percentile_with_nan_inf_and_range_holes(self):
        depth = self._blank()
        rng = np.random.default_rng(7)
        depth[220:260, 220:260] = rng.uniform(0.5, 4.5, size=(40, 40))
        depth[225, 225] = np.nan    # hole
        depth[230, 230] = np.inf    # hole
        depth[235, 235] = 0.1       # below min_depth_m
        depth[240, 240] = 6.0       # above max_depth_m
        patch = depth[220:260, 220:260]
        valid = np.isfinite(patch) & (patch >= 0.30) & (patch <= 5.0)
        expected = float(np.percentile(patch[valid], 30.0))
        assert robust_bbox_depth(depth, self.BBOX) == pytest.approx(expected)

    def test_min_valid_pixel_gate(self):
        depth = self._blank()
        flat = np.arange(40 * 40).reshape(40, 40)
        depth[220:260, 220:260] = np.where(flat < 19, 1.0, np.nan)
        assert robust_bbox_depth(depth, self.BBOX) is None      # 19 valid
        depth[220:260, 220:260] = np.where(flat < 20, 1.0, np.nan)
        assert robust_bbox_depth(depth, self.BBOX) == pytest.approx(1.0)

    def test_only_out_of_range_values_is_no_measurement(self):
        depth = self._blank()
        depth[220:260, 220:240] = 0.1   # nearer than min_depth_m
        depth[220:260, 240:260] = 6.0   # farther than max_depth_m
        assert robust_bbox_depth(depth, self.BBOX) is None

    def test_shrink_excludes_the_bbox_rim(self):
        """Background on the rim of the box must not pollute the depth."""
        depth = self._blank()
        depth[200:280, 200:280] = 4.9           # background over the full box
        depth[220:260, 220:260] = 2.0           # object in the shrunken core
        assert robust_bbox_depth(depth, self.BBOX) == pytest.approx(2.0)

    def test_bbox_outside_image_returns_none(self):
        depth = np.ones((480, 640), dtype=np.float32)
        assert robust_bbox_depth(depth, (700.0, 500.0, 800.0, 600.0)) is None

    def test_degenerate_bbox_returns_none(self):
        depth = np.ones((480, 640), dtype=np.float32)
        assert robust_bbox_depth(depth, (100.0, 100.0, 100.0, 100.0)) is None

    def test_non_2d_depth_raises(self):
        with pytest.raises(ValueError):
            robust_bbox_depth(np.ones((4, 4, 3), dtype=np.float32), self.BBOX)
