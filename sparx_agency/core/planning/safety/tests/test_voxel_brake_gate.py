"""Tests for VoxelBrakeGate: accumulation semantics and braking geometry."""
import math

import numpy as np
import pytest

from sparx_agency.core.planning.safety.voxel_brake_gate import (
    VoxelBrakeGate,
    VoxelBrakeGateConfig,
)


def wall(x, y0, y1, z=1.2, step=0.1):
    ys = np.arange(y0, y1 + 1e-9, step)
    return np.stack([np.full_like(ys, x), ys, np.full_like(ys, z)], axis=1)


@pytest.fixture
def gate():
    return VoxelBrakeGate(VoxelBrakeGateConfig())


class TestIngest:
    def test_empty_map_never_blocks(self, gate):
        assert gate.blocked_distance((0, 0, 1.2), (1, 0), 5.0) is None
        assert gate.command_scale((0, 0, 1.2), (0.6, 0.0)) == (1.0, None)

    def test_occupied_then_free_removes(self, gate):
        pts = wall(2.0, -1, 1)
        gate.update_occupied(pts)
        assert gate.occupied_count() > 0
        gate.update_free(pts)
        assert gate.occupied_count() == 0
        assert gate.blocked_distance((0, 0, 1.2), (1, 0), 5.0) is None

    def test_out_of_band_ignored(self, gate):
        gate.update_occupied(wall(2.0, -1, 1, z=3.5))   # ceiling fixture
        assert gate.occupied_count() == 0
        gate.update_occupied(wall(2.0, -1, 1, z=0.05))  # floor clutter below band
        assert gate.occupied_count() == 0

    def test_duplicate_updates_idempotent(self, gate):
        pts = wall(2.0, -1, 1)
        gate.update_occupied(pts)
        n = gate.occupied_count()
        gate.update_occupied(pts)          # full sweep repeats the same voxels
        assert gate.occupied_count() == n


class TestBlockedDistance:
    def test_head_on_wall_distance(self, gate):
        gate.update_occupied(wall(2.0, -1, 1))
        d = gate.blocked_distance((0, 0, 1.2), (1, 0), 5.0)
        assert d is not None
        assert d == pytest.approx(2.0, abs=0.35)  # radius sweep hits early

    def test_wall_behind_is_ignored(self, gate):
        gate.update_occupied(wall(-2.0, -1, 1))
        assert gate.blocked_distance((0, 0, 1.2), (1, 0), 5.0) is None

    def test_lateral_radius_catches_offset_post(self, gate):
        # a post 0.2 m to the side of the ray: inside the 0.3 m radius sweep
        gate.update_occupied(np.array([[1.5, 0.2, 1.2]]))
        assert gate.blocked_distance((0, 0, 1.2), (1, 0), 5.0) is not None

    def test_clear_beyond_horizon(self, gate):
        gate.update_occupied(wall(4.0, -1, 1))
        assert gate.blocked_distance((0, 0, 1.2), (1, 0), 2.0) is None

    def test_zero_direction_is_clear(self, gate):
        gate.update_occupied(wall(0.2, -1, 1))
        assert gate.blocked_distance((0, 0, 1.2), (0, 0), 2.0) is None

    def test_diagonal_direction(self, gate):
        gate.update_occupied(wall(2.0, 1, 3))
        d = gate.blocked_distance((0, 0, 1.2), (1, 1), 5.0)
        assert d is not None
        assert d == pytest.approx(2.0 * math.sqrt(2.0), abs=0.5)


class TestCommandScale:
    def test_far_wall_full_speed(self, gate):
        gate.update_occupied(wall(4.0, -1, 1))
        scale, _ = gate.command_scale((0, 0, 1.2), (0.6, 0.0))
        assert scale == 1.0

    def test_close_wall_hard_stop(self, gate):
        gate.update_occupied(wall(0.25, -1, 1))
        scale, blocked = gate.command_scale((0, 0, 1.2), (0.6, 0.0))
        assert scale == 0.0
        cfg = VoxelBrakeGateConfig()
        assert blocked is not None
        assert blocked - cfg.drone_radius_m <= cfg.hard_stop_m + 1e-6

    def test_mid_wall_scales_down(self, gate):
        gate.update_occupied(wall(0.5, -1, 1))
        scale, blocked = gate.command_scale((0, 0, 1.2), (0.6, 0.0))
        assert 0.0 < scale < 1.0
        # the scaled speed must genuinely stop before the NOSE reaches the wall
        cfg = VoxelBrakeGateConfig()
        v = 0.6 * scale
        stop = v * cfg.react_s + v * v / (2 * cfg.brake_decel)
        assert stop <= blocked - cfg.drone_radius_m - cfg.margin_m + 1e-6

    def test_slow_creep_allowed_at_moderate_range(self, gate):
        gate.update_occupied(wall(1.2, -1, 1))
        scale, _ = gate.command_scale((0, 0, 1.2), (0.1, 0.0))
        assert scale == 1.0    # 0.1 m/s stops in ~5 cm; 1.2 m away is fine

    def test_hovering_never_blocked(self, gate):
        gate.update_occupied(wall(0.3, -1, 1))
        assert gate.command_scale((0, 0, 1.2), (0.0, 0.0)) == (1.0, None)

    def test_sideways_command_checked_sideways(self, gate):
        gate.update_occupied(wall(2.0, -1, 1))          # wall ahead in +x
        scale, _ = gate.command_scale((0, 0, 1.2), (0.0, 0.6))  # flying +y
        assert scale == 1.0

    def test_respawn_wipe_keeps_conservative_ghosts(self, gate):
        pts = wall(0.5, -1, 1)
        gate.update_occupied(pts)
        # planner respawn: voxels go UNKNOWN, appear in NO stream -- the ghost
        # wall must still brake the aircraft until re-observed free
        scale, _ = gate.command_scale((0, 0, 1.2), (0.6, 0.0))
        assert scale < 1.0
        gate.update_free(pts)
        scale, _ = gate.command_scale((0, 0, 1.2), (0.6, 0.0))
        assert scale == 1.0


class TestZLayers:
    def test_flying_over_a_desk_is_clear(self, gate):
        # desk top voxels at z 0.7-0.9; aircraft cruising at 1.6 clears it
        gate.update_occupied(wall(1.0, -1, 1, z=0.8))
        assert gate.blocked_distance((0, 0, 1.6), (1, 0), 3.0) is None
        scale, _ = gate.command_scale((0, 0, 1.6), (0.6, 0.0))
        assert scale == 1.0

    def test_flying_at_desk_height_is_blocked(self, gate):
        gate.update_occupied(wall(1.0, -1, 1, z=0.8))
        assert gate.blocked_distance((0, 0, 0.8), (1, 0), 3.0) is not None

    def test_band_edges_still_protect(self, gate):
        # obstacle 0.3 m above the flight plane: inside the 0.35 m half-height
        gate.update_occupied(wall(1.0, -1, 1, z=1.5))
        assert gate.blocked_distance((0, 0, 1.2), (1, 0), 3.0) is not None


class TestReplace:
    def test_replace_drops_ghosts(self, gate):
        gate.update_occupied(wall(0.6, -1, 1))          # a phantom
        gate.replace_occupied(wall(3.0, -1, 1))         # the real sweep
        scale, _ = gate.command_scale((0, 0, 1.2), (0.6, 0.0))
        assert scale == 1.0                              # phantom gone
        assert gate.blocked_distance((0, 0, 1.2), (1, 0), 5.0) is not None

    def test_replace_with_empty_clears(self, gate):
        gate.update_occupied(wall(0.6, -1, 1))
        gate.replace_occupied(np.empty((0, 3)))
        assert gate.occupied_count() == 0
