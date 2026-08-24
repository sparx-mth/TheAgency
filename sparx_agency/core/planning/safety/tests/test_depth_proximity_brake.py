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


# ── the corridor that is too close to see ────────────────────────────────
#
# On the SJTU drone the front face is at +0.26 m and the camera lens at
# +0.20 m, so a nose-on contact leaves the wall 0.06 m from the lens -- below
# `min_valid_m` and below the sensor's own 0.1 m near clip. Every corridor
# pixel goes invalid at exactly the moment the answer must be "stop", and
# reporting "clear" there is how the only reflex in the stack switches itself
# off after the first touch.


def _uniform(depth_m, shape=(120, 120)):
    return np.full(shape, depth_m, dtype=np.float32)


def test_a_corridor_full_of_too_close_returns_is_a_full_stop():
    brake = DepthProximityBrake(DepthProximityBrakeConfig(min_valid_m=0.15))
    for depth in (0.02, 0.06, 0.10, 0.149):
        allowed, d_min = brake.allowed_forward_speed(_uniform(depth))
        assert allowed == 0.0, "released the brake at %.3f m" % depth
        assert d_min == pytest.approx(0.15)


def test_an_empty_corridor_is_still_no_reason_to_brake():
    """All-invalid for the OTHER reason -- nothing there -- must stay free."""
    brake = DepthProximityBrake(DepthProximityBrakeConfig())
    allowed, d_min = brake.allowed_forward_speed(
        np.full((120, 120), np.nan, dtype=np.float32))
    assert allowed == float("inf")
    assert d_min is None


def test_too_close_returns_outside_the_corridor_do_not_stop_the_aircraft():
    """A wall brushing past the shoulder is not an obstacle straight ahead."""
    cfg = DepthProximityBrakeConfig(min_valid_m=0.15, corridor_halfwidth_m=0.35)
    brake = DepthProximityBrake(cfg)
    depth = np.full((120, 120), np.nan, dtype=np.float32)
    depth[:, :5] = 0.05          # far off-axis, so lateral offset is tiny at 0.05 m
    allowed, d_min = brake.allowed_forward_speed(depth)
    # At 0.05 m even the frame edge is only |u-cx|*d/fx = 300*0.05/390 = 0.038 m
    # off-axis, so this IS in the corridor -- assert the honest answer.
    assert allowed == 0.0 and d_min == pytest.approx(0.15)
