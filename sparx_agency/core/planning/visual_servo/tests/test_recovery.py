"""Unit tests for the re-search / recovery policy.

Covers :func:`infer_exit_side` (exit-side inference from the last track's box and
image-plane velocity) and :class:`ReSearchPolicy` (hold -> active yaw sweep ->
give-up), plus :class:`ReSearchConfig` validation.
"""
from __future__ import annotations

import numpy as np
import pytest

from sparx_agency.core.common.types import ControlCommand
from sparx_agency.core.common.types.perception import Track2D
from sparx_agency.core.planning.visual_servo.recovery import (
    ReSearchConfig,
    ReSearchDecision,
    ReSearchPolicy,
    infer_exit_side,
)

# Deterministic: fix the RNG even though these tests build fixed inputs.
np.random.seed(0)

FRAME_W = 640
FRAME_H = 480


def _track(bbox, velocity_px=(0.0, 0.0)):
    """Build a Track2D with a chosen box / velocity in the test frame."""
    return Track2D(
        label="target",
        bbox_xyxy=bbox,
        frame_w=FRAME_W,
        frame_h=FRAME_H,
        velocity_px=velocity_px,
    )


# A box centred horizontally in the frame (ox == 0), so only velocity decides.
_CENTER_BOX = (300.0, 220.0, 340.0, 260.0)   # cx == 320 == frame_w / 2
_RIGHT_BOX = (500.0, 220.0, 600.0, 260.0)    # cx == 550 -> right of centre
_LEFT_BOX = (40.0, 220.0, 140.0, 260.0)      # cx == 90  -> left of centre


# --------------------------------------------------------------------------- #
# infer_exit_side
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("default", [1.0, -1.0])
def test_infer_exit_side_none_track_returns_default(default):
    assert infer_exit_side(None, FRAME_W, FRAME_H, 0.5, default) == default


def test_infer_exit_side_box_on_right_returns_minus_one():
    # Box near the right edge, no velocity -> exited right -> -1.
    side = infer_exit_side(_track(_RIGHT_BOX), FRAME_W, FRAME_H, 0.5, 1.0)
    assert side == -1.0


def test_infer_exit_side_box_on_left_returns_plus_one():
    side = infer_exit_side(_track(_LEFT_BOX), FRAME_W, FRAME_H, 0.5, 1.0)
    assert side == 1.0


def test_infer_exit_side_velocity_right_returns_minus_one():
    # Centred box, but moving right (vx > 0) -> exited right -> -1.
    side = infer_exit_side(
        _track(_CENTER_BOX, velocity_px=(50.0, 0.0)), FRAME_W, FRAME_H, 0.5, 1.0
    )
    assert side == -1.0


def test_infer_exit_side_velocity_left_returns_plus_one():
    side = infer_exit_side(
        _track(_CENTER_BOX, velocity_px=(-50.0, 0.0)), FRAME_W, FRAME_H, 0.5, 1.0
    )
    assert side == 1.0


def test_infer_exit_side_centered_zero_velocity_returns_default():
    # score == 0 (centred, no motion) -> falls back to default_direction.
    assert infer_exit_side(
        _track(_CENTER_BOX), FRAME_W, FRAME_H, 0.5, 1.0
    ) == 1.0
    assert infer_exit_side(
        _track(_CENTER_BOX), FRAME_W, FRAME_H, 0.5, -1.0
    ) == -1.0


# --------------------------------------------------------------------------- #
# ReSearchPolicy.command -- hold window
# --------------------------------------------------------------------------- #
def test_command_hold_phase_zero_command():
    policy = ReSearchPolicy(ReSearchConfig(hold_before_search_s=0.3))
    dec = policy.command(_track(_RIGHT_BOX), lost_for_s=0.1,
                         frame_w=FRAME_W, frame_h=FRAME_H)
    assert isinstance(dec, ReSearchDecision)
    assert dec.phase == "hold"
    assert dec.give_up is False
    assert dec.exit_side == 0.0
    cmd = dec.command
    assert isinstance(cmd, ControlCommand)
    assert cmd.x == 0.0 and cmd.y == 0.0 and cmd.z == 0.0
    assert cmd.yaw_rate == 0.0


# --------------------------------------------------------------------------- #
# ReSearchPolicy.command -- search window (yaw sign vs exit side)
# --------------------------------------------------------------------------- #
def test_command_search_right_yaws_cw():
    # exit_side == -1 (right) -> yaw_rate < 0 (CW).
    policy = ReSearchPolicy(ReSearchConfig(search_yaw_rate=0.5,
                                           hold_before_search_s=0.3,
                                           max_search_s=8.0))
    dec = policy.command(_track(_RIGHT_BOX), lost_for_s=1.0,
                         frame_w=FRAME_W, frame_h=FRAME_H)
    assert dec.phase == "search"
    assert dec.exit_side == -1.0
    assert dec.command.yaw_rate < 0.0
    assert dec.command.yaw_rate == pytest.approx(-0.5)
    assert dec.give_up is False


def test_command_search_left_yaws_ccw():
    # exit_side == +1 (left) -> yaw_rate > 0 (CCW).
    policy = ReSearchPolicy(ReSearchConfig(search_yaw_rate=0.5,
                                           hold_before_search_s=0.3,
                                           max_search_s=8.0))
    dec = policy.command(_track(_LEFT_BOX), lost_for_s=1.0,
                         frame_w=FRAME_W, frame_h=FRAME_H)
    assert dec.phase == "search"
    assert dec.exit_side == 1.0
    assert dec.command.yaw_rate > 0.0
    assert dec.command.yaw_rate == pytest.approx(0.5)
    assert dec.give_up is False


def test_command_search_none_track_uses_default_direction():
    # No prior track -> exit side is the configured default (-1 here) -> CW.
    policy = ReSearchPolicy(ReSearchConfig(default_direction=-1.0,
                                           hold_before_search_s=0.3))
    dec = policy.command(None, lost_for_s=1.0, frame_w=FRAME_W, frame_h=FRAME_H)
    assert dec.phase == "search"
    assert dec.exit_side == -1.0
    assert dec.command.yaw_rate < 0.0


# --------------------------------------------------------------------------- #
# ReSearchPolicy.command -- give up
# --------------------------------------------------------------------------- #
def test_command_gives_up_after_max_search():
    policy = ReSearchPolicy(ReSearchConfig(max_search_s=8.0,
                                           hold_before_search_s=0.3))
    dec = policy.command(_track(_RIGHT_BOX), lost_for_s=8.0,
                         frame_w=FRAME_W, frame_h=FRAME_H)
    assert dec.give_up is True
    assert dec.phase == "search"


def test_command_no_give_up_just_before_max_search():
    policy = ReSearchPolicy(ReSearchConfig(max_search_s=8.0,
                                           hold_before_search_s=0.3))
    dec = policy.command(_track(_RIGHT_BOX), lost_for_s=7.999,
                         frame_w=FRAME_W, frame_h=FRAME_H)
    assert dec.give_up is False


def test_command_default_config_hold_boundary():
    # Default hold_before_search_s == 0.3: at exactly the boundary we are no
    # longer holding (strict <), so the policy searches.
    policy = ReSearchPolicy()  # default config
    dec_hold = policy.command(_track(_RIGHT_BOX), lost_for_s=0.29,
                              frame_w=FRAME_W, frame_h=FRAME_H)
    dec_search = policy.command(_track(_RIGHT_BOX), lost_for_s=0.3,
                                frame_w=FRAME_W, frame_h=FRAME_H)
    assert dec_hold.phase == "hold"
    assert dec_search.phase == "search"


# --------------------------------------------------------------------------- #
# ReSearchConfig validation
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("bad_rate", [0.0, -0.1, -1.0])
def test_config_rejects_non_positive_yaw_rate(bad_rate):
    with pytest.raises(ValueError):
        ReSearchConfig(search_yaw_rate=bad_rate)


@pytest.mark.parametrize("bad_dir", [0.0, 2.0, -2.0, 0.5])
def test_config_rejects_invalid_default_direction(bad_dir):
    with pytest.raises(ValueError):
        ReSearchConfig(default_direction=bad_dir)


def test_config_accepts_valid_defaults():
    cfg = ReSearchConfig()
    assert cfg.search_yaw_rate > 0.0
    assert cfg.default_direction in (-1.0, 1.0)
