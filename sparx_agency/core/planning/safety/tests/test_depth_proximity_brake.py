"""Tests for DepthProximityBrake: corridor geometry and speed law."""
import math

import numpy as np
import pytest

from sparx_agency.core.planning.safety.depth_proximity_brake import (
    DepthProximityBrake,
    DepthProximityBrakeConfig,
)

H = W = 600


def frame(fill=10.0):
    return np.full((H, W), fill, dtype=np.float32)


def put_blob(img, cfg, x_lat, y_vert, depth, half_px=20):
    """Paint a square blob at the pixel where (x_lat, y_vert, depth) projects."""
    u = int(cfg.cx + x_lat * cfg.fx / depth)
    v = int(cfg.cy + y_vert * cfg.fy / depth)
    img[max(0, v - half_px):v + half_px, max(0, u - half_px):u + half_px] = depth
    return img


@pytest.fixture
def cfg():
    return DepthProximityBrakeConfig()


@pytest.fixture
def brake(cfg):
    return DepthProximityBrake(cfg)


class TestCorridor:
    def test_clear_frame_unrestricted(self, brake):
        v, d = brake.allowed_forward_speed(frame())
        assert d == pytest.approx(10.0)
        assert v > 3.0     # 10 m of room: no meaningful restriction

    def test_center_obstacle_is_seen(self, brake, cfg):
        img = put_blob(frame(), cfg, 0.0, 0.0, 2.5)
        assert brake.corridor_min_depth(img) == pytest.approx(2.5, abs=0.01)

    def test_off_corridor_obstacle_ignored(self, brake, cfg):
        # 1.2 m to the side at 2.5 m range: outside the 0.35 m corridor
        img = put_blob(frame(), cfg, 1.2, 0.0, 2.5)
        assert brake.corridor_min_depth(img) == pytest.approx(10.0)

    def test_floor_and_ceiling_ignored(self, brake, cfg):
        img = put_blob(frame(), cfg, 0.0, 0.9, 2.0)    # floor return
        img = put_blob(img, cfg, 0.0, -0.9, 2.0)       # ceiling fixture
        assert brake.corridor_min_depth(img) == pytest.approx(10.0)

    def test_near_noise_ignored(self, brake, cfg):
        img = frame()
        img[300:320, 300:320] = 0.05       # sub-clip garbage
        assert brake.corridor_min_depth(img) == pytest.approx(10.0)

    def test_nan_inf_handled(self, brake, cfg):
        img = frame()
        img[::7, ::7] = np.nan
        img[::11, ::11] = np.inf
        v, d = brake.allowed_forward_speed(img)
        assert d == pytest.approx(10.0)

    def test_thin_person_width_detected(self, brake, cfg):
        # 0.4 m wide torso at 2.8 m: the run-009 killer, ~28 px half-width
        img = put_blob(frame(), cfg, 0.0, 0.0, 2.8, half_px=28)
        assert brake.corridor_min_depth(img) == pytest.approx(2.8, abs=0.01)


class TestSpeedLaw:
    def test_at_nose_full_stop(self, brake, cfg):
        img = put_blob(frame(), cfg, 0.0, 0.0, 0.2)
        v, _ = brake.allowed_forward_speed(img)
        assert v == 0.0

    def test_monotone_in_distance(self, brake, cfg):
        speeds = []
        for d in (0.6, 1.2, 2.4, 4.8):
            img = put_blob(frame(), cfg, 0.0, 0.0, d)
            v, _ = brake.allowed_forward_speed(img)
            speeds.append(v)
        assert speeds == sorted(speeds)

    def test_allowed_speed_actually_stops_short(self, brake, cfg):
        img = put_blob(frame(), cfg, 0.0, 0.0, 1.5)
        v, d_min = brake.allowed_forward_speed(img)
        stop = v * cfg.react_s + v * v / (2 * cfg.brake_decel)
        assert stop <= d_min - cfg.nose_offset_m - cfg.margin_m + 1e-6

    def test_cruise_unimpeded_at_3m(self, brake, cfg):
        img = put_blob(frame(), cfg, 0.0, 0.0, 3.0)
        v, _ = brake.allowed_forward_speed(img)
        assert v >= 0.6    # campaign cruise speed passes untouched
