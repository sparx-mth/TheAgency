"""Tests for viewpoint/pitch augmentation (numpy + cv2, no torch)."""
import numpy as np
import pytest

from sparx_agency.core.common.types import Intrinsics
from sparx_agency.tasks.planning.finetune.common.augment import (
    ViewpointAugmentConfig,
    apply_viewpoint_augment,
    pitch_homography,
)

INTR = Intrinsics(width=64, height=64, fx=64.0, fy=64.0, cx=32.0, cy=32.0)


def test_zero_pitch_is_identity():
    h = pitch_homography(INTR, 0.0)
    assert np.allclose(h / h[2, 2], np.eye(3), atol=1e-6)


def test_pitch_homography_shifts_horizon():
    # a downward pitch should move image content vertically (non-identity, finite)
    h = pitch_homography(INTR, 8.0)
    assert h.shape == (3, 3)
    assert np.isfinite(h).all()
    assert not np.allclose(h / h[2, 2], np.eye(3), atol=1e-3)


def test_disabled_is_identity_passthrough():
    rng = np.random.default_rng(0)
    rgb = np.full((64, 64, 3), 100, np.uint8)
    depth = np.full((64, 64), 3.0, np.float32)
    out = apply_viewpoint_augment(INTR, ViewpointAugmentConfig(enabled=False), rng, rgb, depth)
    assert out.pitch_deg == 0.0 and out.depth_scale == 1.0
    assert np.array_equal(out.rgb, rgb)
    assert np.array_equal(out.depth_m, depth)


def test_augment_is_reproducible_and_registered():
    rgb = (np.random.RandomState(1).rand(64, 64, 3) * 255).astype(np.uint8)
    depth = np.full((64, 64), 3.0, np.float32)
    cfg = ViewpointAugmentConfig()
    a = apply_viewpoint_augment(INTR, cfg, np.random.default_rng(42), rgb.copy(), depth.copy())
    b = apply_viewpoint_augment(INTR, cfg, np.random.default_rng(42), rgb.copy(), depth.copy())
    # same seed -> same pitch and same warped output
    assert a.pitch_deg == b.pitch_deg
    assert np.array_equal(a.rgb, b.rgb)
    assert a.rgb.shape == rgb.shape
    assert a.depth_m.shape == depth.shape


def test_depth_scale_applied_within_range():
    depth = np.full((64, 64), 4.0, np.float32)
    cfg = ViewpointAugmentConfig(pitch_deg_range=(0.0, 0.0), depth_scale_range=(1.2, 1.2),
                                 depth_offset_m=0.0, brightness_range=(1.0, 1.0))
    out = apply_viewpoint_augment(INTR, cfg, np.random.default_rng(0), None, depth)
    valid = out.depth_m > 0
    assert np.allclose(out.depth_m[valid], 4.0 * 1.2, atol=1e-4)
