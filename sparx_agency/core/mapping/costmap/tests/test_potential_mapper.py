"""
test_potential_mapper.py
=========================
Unit tests for PotentialMapper.

No GPU required. Uses synthetic depth maps and intrinsics.
"""
from __future__ import annotations

import numpy as np
import pytest

from sparx_agency.core.common.types import Intrinsics
from sparx_agency.core.mapping.costmap.potential_mapper import (
    PotentialMapper,
    PotentialMapperConfig,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_intrinsics(width: int = 320, height: int = 240) -> Intrinsics:
    """Synthetic pinhole intrinsics with a 60° HFOV."""
    fx = width / (2.0 * np.tan(np.deg2rad(30.0)))
    fy = fx
    cx = width / 2.0
    cy = height / 2.0
    return Intrinsics(width=width, height=height, fx=fx, fy=fy, cx=cx, cy=cy)


def _flat_depth(height: int = 240, width: int = 320, value: float = 3.0) -> np.ndarray:
    """Uniform depth map: every pixel at the same distance."""
    return np.full((height, width), value, dtype=np.float32)


def _wall_depth(height: int = 240, width: int = 320) -> np.ndarray:
    """Simulates a wall close by in the image centre."""
    d = np.full((height, width), 10.0, dtype=np.float32)
    d[height // 4: 3 * height // 4, width // 4: 3 * width // 4] = 1.5
    return d


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

class TestPotentialMapperConfig:
    def test_invalid_alpha_zero(self):
        with pytest.raises(ValueError, match="alpha must be in"):
            PotentialMapperConfig(alpha=0.0)

    def test_invalid_alpha_gt_1(self):
        with pytest.raises(ValueError, match="alpha must be in"):
            PotentialMapperConfig(alpha=1.5)

    def test_invalid_sigma_zero(self):
        with pytest.raises(ValueError, match="sigma_m must be"):
            PotentialMapperConfig(sigma_m=0.0)

    def test_invalid_z_band(self):
        with pytest.raises(ValueError, match="z_band"):
            PotentialMapperConfig(z_band=(2.0, 1.0))

    def test_valid_config(self):
        cfg = PotentialMapperConfig(alpha=0.5)
        assert cfg.alpha == 0.5

    def test_prob_map_not_empty(self):
        """Enforce Rule: Occupancy map MUST NOT be empty when valid points are provided."""
        cfg = PotentialMapperConfig(size_m=10.0, resolution_m=0.5, occ_thresh=0.1)
        mapper = PotentialMapper(cfg)
        intr = _make_intrinsics()
        
        # Point at 3.0m in front (Forward axis)
        depth = _flat_depth(value=3.0)
        mapper.step(depth, intr)
        
        prob = mapper.get_prob_map()
        assert np.any(prob > 0.0), "Occupancy map is empty despite valid input!"
        assert np.max(prob) > 0.0


# ---------------------------------------------------------------------------
# Grid properties
# ---------------------------------------------------------------------------

class TestPotentialMapperGridShape:
    def test_grid_shape_matches_config(self):
        cfg = PotentialMapperConfig(size_m=10.0, resolution_m=0.1)
        mapper = PotentialMapper(cfg)
        expected_n = int(round(10.0 / 0.1))
        assert mapper.grid_shape == (expected_n, expected_n)

    def test_gradient_field_shape(self):
        cfg = PotentialMapperConfig(size_m=10.0, resolution_m=0.5)
        mapper = PotentialMapper(cfg)
        intr = _make_intrinsics()
        mapper.step(_flat_depth(value=3.0), intr)
        grad = mapper.get_gradient_field()
        n = int(round(10.0 / 0.5))
        assert grad.shape == (n, n, 2), f"Unexpected grad shape {grad.shape}"

    def test_prob_map_shape(self):
        cfg = PotentialMapperConfig(size_m=10.0, resolution_m=0.5)
        mapper = PotentialMapper(cfg)
        intr = _make_intrinsics()
        mapper.step(_flat_depth(value=3.0), intr)
        prob = mapper.get_prob_map()
        n = int(round(10.0 / 0.5))
        assert prob.shape == (n, n)


# ---------------------------------------------------------------------------
# Extrinsics
# ---------------------------------------------------------------------------

class TestExtrinsics:
    def test_height_projection(self):
        """Points at depth D with pitch=0, height=H should be at Up=H in base frame."""
        cfg = PotentialMapperConfig(height_m=2.0, pitch_deg=0.0)
        mapper = PotentialMapper(cfg)
        intr = _make_intrinsics()
        
        depth = _flat_depth(value=5.0)
        pts = mapper._backproject(depth, intr)
        
        # Base frame: [Left, Up, Forward]
        # index 1 (Up) should be 2.0 (height)
        mid_idx = pts.shape[0] // 2
        up_val = pts[mid_idx, 1]
        np.testing.assert_allclose(up_val, 2.0, atol=1e-3)

    def test_pitch_projection(self):
        """Points with pitch should have different Forward/Up based on Z-optical."""
        cfg = PotentialMapperConfig(height_m=1.0, pitch_deg=45.0)
        mapper = PotentialMapper(cfg)
        intr = _make_intrinsics()
        
        depth = _flat_depth(value=1.414) # sqrt(2)
        pts = mapper._backproject(depth, intr)
        
        # With 45 deg pitch down:
        # Forward ray (Z_opt=1.414, Y_opt=0) -> xl=0, yu=0, zf=1.414
        # z_final (Forward) = zf*cos(45) + yu*sin(45) = 1.414*0.707 + 0 = 1.0
        # y_final (Up)      = -zf*sin(45) + yu*cos(45) + height = -1.0 + 1.0 = 0.0
        mid_idx = pts.shape[0] // 2
        up_val = pts[mid_idx, 1]
        fwd_val = pts[mid_idx, 2]
        
        np.testing.assert_allclose(fwd_val, 1.0, atol=1e-3)
        np.testing.assert_allclose(up_val, 0.0, atol=1e-3)




# ---------------------------------------------------------------------------
# Accumulation / decay
# ---------------------------------------------------------------------------

class TestAccumulationDecay:
    def test_map_grows_then_decays_to_zero(self):
        """After adding obstacle, then running many empty-depth steps, M_acc → 0."""
        cfg = PotentialMapperConfig(
            alpha=0.5, size_m=20.0, resolution_m=0.5,
            range_min_m=0.1, range_max_m=20.0,
            z_band=(-3.0, 3.0),
        )
        mapper = PotentialMapper(cfg)
        intr = _make_intrinsics(320, 240)

        # Load with wall depth (creates obstacles)
        mapper.step(_wall_depth(), intr)
        prob_after_obs = mapper.get_prob_map().max()
        assert prob_after_obs > 0.0, "No obstacle registered"

        # Feed zeros / very-far depth so no points hit the grid
        blank = np.full((240, 320), 50.0, dtype=np.float32)  # out of range_max_m
        for _ in range(40):
            mapper.step(blank, intr)

        prob_decayed = mapper.get_prob_map().max()
        assert prob_decayed < prob_after_obs, "Map should decay"
        assert prob_decayed < 0.01, f"Map did not fully decay: {prob_decayed}"

    def test_alpha_1_replaces_each_frame(self):
        """With alpha=1.0, M_acc == M_temp exactly after each step."""
        cfg = PotentialMapperConfig(
            alpha=1.0, size_m=20.0, resolution_m=0.5,
            range_min_m=0.1, range_max_m=20.0,
            z_band=(-3.0, 3.0),
        )
        mapper = PotentialMapper(cfg)
        intr = _make_intrinsics(320, 240)

        mapper.step(_wall_depth(), intr)
        np.testing.assert_array_equal(
            mapper.get_prob_map(), mapper.get_temp_map(),
            err_msg="With alpha=1.0, M_acc must equal M_temp",
        )


# ---------------------------------------------------------------------------
# Reset
# ---------------------------------------------------------------------------

class TestReset:
    def test_reset_clears_all_maps(self):
        cfg = PotentialMapperConfig(
            size_m=20.0, resolution_m=0.5,
            range_min_m=0.1, range_max_m=30.0,
            z_band=(-3.0, 3.0),
        )
        mapper = PotentialMapper(cfg)
        intr = _make_intrinsics()
        mapper.step(_wall_depth(), intr)
        assert mapper.get_prob_map().max() > 0.0, "Setup: map should not be empty"

        mapper.reset()
        assert mapper.get_prob_map().max() == 0.0, "After reset: prob map should be zero"
        assert mapper.get_potential_map().max() == 0.0, "After reset: potential should be zero"
        np.testing.assert_array_equal(
            mapper.get_gradient_field(),
            np.zeros_like(mapper.get_gradient_field()),
        )


# ---------------------------------------------------------------------------
# Obstacle repulsion direction
# ---------------------------------------------------------------------------

class TestObstacleRepulsionDirection:
    def test_gradient_points_away_from_obstacle(self):
        """Gradient at a cell adjacent to an obstacle should point away from it."""
        cfg = PotentialMapperConfig(
            alpha=1.0,
            size_m=20.0,
            resolution_m=0.5,
            occ_thresh=0.1,   # low threshold so synthetic prob triggers occupancy
            sigma_m=1.0,
            inflation_radius_m=0.0,
            range_min_m=0.1,
            range_max_m=30.0,
            z_band=(-3.0, 3.0),
        )
        mapper = PotentialMapper(cfg)
        n = mapper._n

        # Manually inject a high-probability obstacle at the map centre
        mid = n // 2
        mapper._M_acc[mid, mid] = 1.0

        # Recompute potential directly (without step)
        from sparx_agency.core.mapping.costmap.potential_field_layer import PotentialFieldLayer
        pfl = PotentialFieldLayer(
            occ_thresh=cfg.occ_thresh,
            sigma_m=cfg.sigma_m,
            k_rep=1.0,
            inflation_radius_m=0.0,
            u_max=1.0,
        )
        U, _ = pfl.compute_from_prob_grid(mapper._M_acc, cfg.resolution_m)
        mapper._U_rep = U
        gy, gx = np.gradient(U, cfg.resolution_m)
        # Match PotentialMapper's negated gradient convention
        mapper._grad = np.stack([-gx, -gy], axis=-1).astype(np.float32)

        grad = mapper.get_gradient_field()

        # Cell to the RIGHT of centre: grad_x should be positive (pointing right = away)
        gx_right = grad[mid, mid + 2, 0]
        assert gx_right > 0.0, (
            f"Gradient at cell right of obstacle should point right (+x), got {gx_right}"
        )

        # Cell to the LEFT of centre: grad_x should be negative (pointing left = away)
        gx_left = grad[mid, mid - 2, 0]
        assert gx_left < 0.0, (
            f"Gradient at cell left of obstacle should point left (-x), got {gx_left}"
        )


# ---------------------------------------------------------------------------
# Navigation
# ---------------------------------------------------------------------------

class TestNavigation:
    def test_target_navigation(self):
        """Test that get_total_gradient includes a pull toward the goal."""
        cfg = PotentialMapperConfig(size_m=10.0, resolution_m=1.0, zeta=1.0)
        mapper = PotentialMapper(cfg)
        
        # Set goal at (5, 0)
        mapper.set_goal(5.0, 0.0)
        
        # At (0,0), repulsive grad is 0 (no obstacles).
        # Attractive should point toward (5, 0).
        # zeta * (Goal - Pos) = 1.0 * (5 - 0) = 5.0 in Forward dir.
        total_grad = mapper.get_total_gradient()
        
        # find index of (0,0) in grid
        # fwd_idx = (0 - (-5)) / 1.0 = 5
        # left_idx = (5 - 0) / 1.0 = 5
        mid_row, mid_col = 5, 5
        center_grad = total_grad[mid_row, mid_col]
        
        # Forward component index 0 should be positive (toward 5)
        np.testing.assert_allclose(center_grad[0], 5.0, atol=1e-3)
        np.testing.assert_allclose(center_grad[1], 0.0, atol=1e-3)


# ---------------------------------------------------------------------------
# Wall Segmentation (Revision 5)
# ---------------------------------------------------------------------------

class TestWallSegmentation:
    def test_detect_straight_wall(self):
        """A straight line of cells should be detected as a wall segment."""
        cfg = PotentialMapperConfig(size_m=20.0, resolution_m=1.0, min_wall_length_m=2.0)
        mapper = PotentialMapper(cfg)
        
        # 8m Vertical Wall at Col 5
        # Cells (5,5) to (12,5)
        for r in range(5, 13):
            mapper._M_acc[r, 5] = 1.0
            
        m_walls = mapper._detect_walls_and_clean()
        
        assert mapper._wall_segments.size > 0, "No wall segments detected"
        # Wall is at column 5, so x coordinates should be exactly 5
        line = mapper._wall_segments[0]
        # line is [x1, y1, x2, y2]
        assert line[0] == 5 and line[2] == 5
        
        # Check that m_walls (cleaned map) has the line
        assert m_walls[5, 5] > 0
        assert m_walls[12, 5] > 0



