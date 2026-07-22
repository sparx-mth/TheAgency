"""Unit tests for the re-search / recovery policy.

Covers :func:`infer_exit_side` (exit-side inference from the last track's box and
image-plane velocity) and :class:`ReSearchPolicy` (hold -> directional chase OR
occluder peek -> give-up), plus :class:`ReSearchConfig` validation.
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
# ReSearchPolicy.command -- directional chase (target left a side)
# --------------------------------------------------------------------------- #
def test_command_right_exit_yaws_cw_and_crabs_right():
    # A box far to the right -> exited right (-1) -> yaw CW (< 0) and crab right (vy < 0).
    policy = ReSearchPolicy(ReSearchConfig(search_yaw_rate=0.5,
                                           hold_before_search_s=0.3,
                                           max_search_s=8.0))
    dec = policy.command(_track(_RIGHT_BOX), lost_for_s=1.0,
                         frame_w=FRAME_W, frame_h=FRAME_H)
    assert dec.phase == "directional"
    assert dec.exit_side == -1.0
    assert dec.command.yaw_rate == pytest.approx(-0.5)
    assert dec.command.y < 0.0          # crab toward the right (+vy is left)
    assert dec.give_up is False


def test_command_left_exit_yaws_ccw_and_crabs_left():
    policy = ReSearchPolicy(ReSearchConfig(search_yaw_rate=0.5,
                                           hold_before_search_s=0.3,
                                           max_search_s=8.0))
    dec = policy.command(_track(_LEFT_BOX), lost_for_s=1.0,
                         frame_w=FRAME_W, frame_h=FRAME_H)
    assert dec.phase == "directional"
    assert dec.exit_side == 1.0
    assert dec.command.yaw_rate == pytest.approx(0.5)
    assert dec.command.y > 0.0          # crab toward the left


def test_command_none_track_uses_default_direction():
    # No prior track -> directional sweep toward the configured default (-1 here) -> CW.
    policy = ReSearchPolicy(ReSearchConfig(default_direction=-1.0,
                                           hold_before_search_s=0.3))
    dec = policy.command(None, lost_for_s=1.0, frame_w=FRAME_W, frame_h=FRAME_H)
    assert dec.phase == "directional"
    assert dec.exit_side == -1.0
    assert dec.command.yaw_rate < 0.0


# --------------------------------------------------------------------------- #
# ReSearchPolicy.command -- occluder peek (target vanished from centre)
# --------------------------------------------------------------------------- #
def test_command_center_loss_triggers_peek():
    # Centred box, no velocity -> vanished from the centre -> peek, not directional.
    policy = ReSearchPolicy(ReSearchConfig(hold_before_search_s=0.3))
    dec = policy.command(_track(_CENTER_BOX), lost_for_s=1.0,
                         frame_w=FRAME_W, frame_h=FRAME_H)
    assert dec.phase == "peek"


def test_peek_holds_one_direction_no_flip_flop():
    cfg = ReSearchConfig(hold_before_search_s=0.0)
    policy = ReSearchPolicy(cfg)
    # Same last_track (frozen during RECOVER) at very different times -> the sidestep
    # and yaw keep the SAME sign the whole episode (no jarring reversal).
    a = policy.command(_track(_CENTER_BOX), 0.5, FRAME_W, FRAME_H)
    b = policy.command(_track(_CENTER_BOX), 3.5, FRAME_W, FRAME_H)
    assert a.phase == "peek" and b.phase == "peek"
    assert a.command.y != 0.0
    assert (a.command.y > 0.0) == (b.command.y > 0.0)            # roll: same side, held
    assert (a.command.yaw_rate > 0.0) == (b.command.yaw_rate > 0.0)  # yaw: same, held


def test_peek_forward_nudge_is_bounded():
    cfg = ReSearchConfig(hold_before_search_s=0.0, peek_forward_speed=0.06,
                         peek_forward_s=0.6)
    policy = ReSearchPolicy(cfg)
    early = policy.command(_track(_CENTER_BOX), 0.2, FRAME_W, FRAME_H)  # within nudge
    late = policy.command(_track(_CENTER_BOX), 1.5, FRAME_W, FRAME_H)   # after nudge
    assert early.command.x > 0.0        # a little forward to clear the occluder edge
    assert late.command.x == 0.0        # then no further advance (bounded travel)


def test_peek_orbit_yaws_opposite_the_sidestep():
    # Default peek_orbit=True: as it sidesteps one way it yaws the other, to keep
    # looking back around the occluder.
    policy = ReSearchPolicy(ReSearchConfig(hold_before_search_s=0.0))
    dec = policy.command(_track(_CENTER_BOX), 0.1, FRAME_W, FRAME_H)
    assert (dec.command.y > 0.0) != (dec.command.yaw_rate > 0.0)


# --------------------------------------------------------------------------- #
# ReSearchPolicy.command -- give up
# --------------------------------------------------------------------------- #
def test_command_gives_up_after_max_search():
    policy = ReSearchPolicy(ReSearchConfig(max_search_s=8.0,
                                           hold_before_search_s=0.3))
    dec = policy.command(_track(_RIGHT_BOX), lost_for_s=8.0,
                         frame_w=FRAME_W, frame_h=FRAME_H)
    assert dec.give_up is True
    assert dec.phase == "directional"


def test_command_no_give_up_just_before_max_search():
    policy = ReSearchPolicy(ReSearchConfig(max_search_s=8.0,
                                           hold_before_search_s=0.3))
    dec = policy.command(_track(_RIGHT_BOX), lost_for_s=7.999,
                         frame_w=FRAME_W, frame_h=FRAME_H)
    assert dec.give_up is False


def test_command_default_config_hold_boundary():
    # Default hold_before_search_s == 0.3: at exactly the boundary we are no
    # longer holding (strict <), so the policy manoeuvres.
    policy = ReSearchPolicy()  # default config
    dec_hold = policy.command(_track(_RIGHT_BOX), lost_for_s=0.29,
                              frame_w=FRAME_W, frame_h=FRAME_H)
    dec_move = policy.command(_track(_RIGHT_BOX), lost_for_s=0.3,
                              frame_w=FRAME_W, frame_h=FRAME_H)
    assert dec_hold.phase == "hold"
    assert dec_move.phase == "directional"


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


@pytest.mark.parametrize("bad", [-0.1, -1.0])
def test_config_rejects_negative_center_exit_frac(bad):
    with pytest.raises(ValueError):
        ReSearchConfig(center_exit_frac=bad)


def test_config_accepts_valid_defaults():
    cfg = ReSearchConfig()
    assert cfg.search_yaw_rate > 0.0
    assert cfg.default_direction in (-1.0, 1.0)
    assert cfg.peek_orbit is True
