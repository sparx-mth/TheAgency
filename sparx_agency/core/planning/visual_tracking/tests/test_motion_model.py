"""Unit tests for the constant-velocity box motion model (alpha-beta filter).

Covers :class:`ConstantVelocityBoxModel` state lifecycle, velocity estimation
from constant image-plane motion, dead-reckoning via :meth:`predict`, re-anchoring
after a long gap, the speed clamp, and :class:`MotionModelConfig` validation.
"""
from __future__ import annotations

import numpy as np
import pytest

from sparx_agency.core.planning.visual_tracking.motion_model import (
    ConstantVelocityBoxModel,
    MotionModelConfig,
)


def test_no_state_before_first_update() -> None:
    """A fresh model has no estimate and a zero velocity until first measurement."""
    model = ConstantVelocityBoxModel()
    assert model.has_state is False
    assert model.state_cxcywh() is None
    assert model.velocity_px == (0.0, 0.0)


def test_first_update_anchors_with_zero_velocity() -> None:
    """The first measurement is passed through verbatim with velocity anchored to 0."""
    model = ConstantVelocityBoxModel()
    box = (100.0, 50.0, 30.0, 20.0)
    out = model.update(box, dt=0.1)

    assert model.has_state is True
    assert out == box
    assert model.state_cxcywh() == box
    assert model.velocity_px == (0.0, 0.0)


def test_constant_motion_velocity_sign_and_magnitude() -> None:
    """A box moving at a constant px/frame yields a matching velocity estimate.

    With a fixed ``dt`` the alpha-beta filter tracks a constant-velocity target
    with (near) zero steady-state error, so ``vx`` converges to ``step/dt`` and
    ``vy`` stays ~0.
    """
    model = ConstantVelocityBoxModel()
    dt = 0.1
    step_px = 10.0  # px per frame in +x
    expected_vx = step_px / dt  # px/s

    cx, cy = 100.0, 50.0
    for _ in range(40):
        cx += step_px
        model.update((cx, cy, 30.0, 20.0), dt=dt)

    vx, vy = model.velocity_px
    # Sign matches the +x motion, magnitude is in the right ballpark.
    assert vx > 0.0
    assert vx == pytest.approx(expected_vx, rel=0.2)
    # No vertical motion was injected.
    assert vy == pytest.approx(0.0, abs=1e-6)


def test_negative_motion_flips_velocity_sign() -> None:
    """Reversing the motion direction reverses the estimated velocity sign."""
    model = ConstantVelocityBoxModel()
    dt = 0.1
    cy = 50.0
    cx = 500.0
    for _ in range(40):
        cx -= 12.0  # moving in -x
        cy += 6.0   # moving in +y
        model.update((cx, cy, 30.0, 20.0), dt=dt)

    vx, vy = model.velocity_px
    assert vx < 0.0
    assert vy > 0.0
    assert vx == pytest.approx(-120.0, rel=0.2)
    assert vy == pytest.approx(60.0, rel=0.2)


def test_predict_advances_centre_and_holds_size() -> None:
    """`predict` dead-reckons the centre along velocity and keeps the box size."""
    model = ConstantVelocityBoxModel()
    dt = 0.1
    cx, cy = 100.0, 50.0
    w, h = 30.0, 20.0
    for _ in range(40):
        cx += 10.0
        model.update((cx, cy, w, h), dt=dt)

    cx0, cy0, w0, h0 = model.state_cxcywh()
    vx, vy = model.velocity_px

    pred_dt = 0.2
    pcx, pcy, pw, ph = model.predict(pred_dt)

    # Centre advanced exactly along the estimated velocity.
    assert pcx == pytest.approx(cx0 + vx * pred_dt)
    assert pcy == pytest.approx(cy0 + vy * pred_dt)
    # Size is held through dead reckoning.
    assert pw == w0
    assert ph == h0
    # Since motion is +x, the predicted centre moved forward.
    assert pcx > cx0


def test_predict_without_state_returns_none() -> None:
    """`predict` before any measurement has no state to advance -> None."""
    model = ConstantVelocityBoxModel()
    assert model.predict(0.1) is None


def test_long_gap_reanchors_and_resets_velocity() -> None:
    """A `dt` above `max_dt` re-anchors to the measurement with zero velocity."""
    cfg = MotionModelConfig()  # max_dt = 0.5
    model = ConstantVelocityBoxModel(cfg)
    dt = 0.1
    cx, cy = 100.0, 50.0
    for _ in range(20):
        cx += 15.0
        model.update((cx, cy, 30.0, 20.0), dt=dt)
    assert model.velocity_px[0] > 0.0  # velocity built up

    # A long stall: dt exceeds max_dt, so this is treated as a fresh anchor.
    gap_box = (900.0, 400.0, 35.0, 25.0)
    out = model.update(gap_box, dt=cfg.max_dt + 0.5)

    assert out == gap_box
    assert model.state_cxcywh() == gap_box
    assert model.velocity_px == (0.0, 0.0)  # reset, no explosion


def test_duplicate_timestamp_preserves_velocity() -> None:
    """A dt<=0 (duplicate/equal stamp) on an established track must NOT wipe the
    velocity estimate; it absorbs the measured position but holds velocity."""
    model = ConstantVelocityBoxModel(MotionModelConfig())
    cx, cy = 100.0, 50.0
    for _ in range(20):
        cx += 12.0
        model.update((cx, cy, 30.0, 20.0), dt=0.1)
    vx_before, vy_before = model.velocity_px
    assert vx_before > 0.0

    # Duplicate frame (dt == 0): velocity is preserved, not reset to zero.
    model.update((cx, cy, 30.0, 20.0), dt=0.0)
    assert model.velocity_px == (vx_before, vy_before)
    # A negative dt is treated the same way.
    model.update((cx, cy, 30.0, 20.0), dt=-0.05)
    assert model.velocity_px == (vx_before, vy_before)


def test_velocity_is_clamped_to_max_speed() -> None:
    """A large per-frame jump is clamped so the estimated speed stays bounded."""
    cfg = MotionModelConfig(max_speed_px=50.0)
    model = ConstantVelocityBoxModel(cfg)
    dt = 0.1
    cx, cy = 0.0, 0.0
    for _ in range(40):
        cx += 200.0  # would imply 2000 px/s, far above the clamp
        cy += 200.0
        model.update((cx, cy, 30.0, 20.0), dt=dt)

    vx, vy = model.velocity_px
    speed = (vx * vx + vy * vy) ** 0.5
    assert speed <= cfg.max_speed_px + 1e-6
    assert speed == pytest.approx(cfg.max_speed_px, rel=1e-3)


def test_velocity_estimate_is_robust_to_measurement_noise() -> None:
    """Under bounded noise around a ramp, the velocity sign/scale still hold."""
    rng = np.random.default_rng(1234)
    model = ConstantVelocityBoxModel()
    dt = 0.1
    step_px = 8.0
    cx, cy = 100.0, 200.0
    for _ in range(80):
        cx += step_px
        noisy = (cx + rng.normal(0.0, 1.0), cy + rng.normal(0.0, 1.0), 30.0, 20.0)
        model.update(noisy, dt=dt)

    vx, _ = model.velocity_px
    assert vx > 0.0
    assert vx == pytest.approx(step_px / dt, rel=0.3)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"alpha": 0.0},
        {"alpha": 1.5},
        {"alpha": -0.1},
        {"beta": 0.0},
        {"beta": 1.0001},
        {"size_alpha": 0.0},
        {"size_alpha": 2.0},
        {"max_speed_px": 0.0},
        {"max_speed_px": -10.0},
    ],
)
def test_config_rejects_invalid_params(kwargs) -> None:
    """Out-of-range blend factors and non-positive max speed raise ValueError."""
    with pytest.raises(ValueError):
        MotionModelConfig(**kwargs)


def test_config_accepts_boundary_values() -> None:
    """Blend factors of exactly 1.0 are valid (half-open interval upper bound)."""
    cfg = MotionModelConfig(alpha=1.0, beta=1.0, size_alpha=1.0, max_speed_px=1.0)
    assert cfg.alpha == 1.0
    assert cfg.beta == 1.0
    assert cfg.size_alpha == 1.0
