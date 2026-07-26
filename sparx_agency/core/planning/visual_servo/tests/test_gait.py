"""Unit tests for the pulse-and-settle closure gait.

Drives :class:`ClosureGait` tick by tick with shaped-style commands and asserts on
the emitted duty cycle (move-a-little / stop-and-look), the turn<->forward
transition settle, natural-stop pass-through, and config validation. Pure and
deterministic — no clock, one command per tick.
"""
from __future__ import annotations

import pytest

from sparx_agency.core.common.types import ControlCommand
from sparx_agency.core.planning.visual_servo.gait import (
    MOVING,
    SETTLING,
    ClosureGait,
    ClosureGaitConfig,
)


# ── helpers ───────────────────────────────────────────────────────────
def _fwd(v: float = 0.3) -> ControlCommand:
    return ControlCommand.velocity(v, 0.0, 0.0, 0.0)


def _yaw(w: float = 0.7) -> ControlCommand:
    return ControlCommand.velocity(0.0, 0.0, 0.0, w)


def _stop() -> ControlCommand:
    return ControlCommand.velocity(0.0, 0.0, 0.0, 0.0)


def _moving(cmd: ControlCommand) -> bool:
    return abs(cmd.x) + abs(cmd.y) + abs(cmd.yaw_rate) > 0.0


# ── config validation ─────────────────────────────────────────────────
@pytest.mark.parametrize("bad", [0, -1])
def test_invalid_move_ticks_raises(bad):
    with pytest.raises(ValueError):
        ClosureGaitConfig(move_ticks=bad)


def test_invalid_settle_ticks_raises():
    with pytest.raises(ValueError):
        ClosureGaitConfig(settle_ticks=-1)


def test_active_flag():
    assert ClosureGaitConfig(settle_ticks=3, enabled=True).active is True
    assert ClosureGaitConfig(settle_ticks=0).active is False
    assert ClosureGaitConfig(settle_ticks=3, enabled=False).active is False


# ── disabled / inert pass-through ─────────────────────────────────────
def test_settle_zero_passes_through():
    gait = ClosureGait(ClosureGaitConfig(move_ticks=2, settle_ticks=0))
    for _ in range(10):
        assert _moving(gait.step(_fwd()))


def test_disabled_passes_through():
    gait = ClosureGait(ClosureGaitConfig(enabled=False))
    for _ in range(10):
        assert _moving(gait.step(_yaw()))


# ── duty cycle: move a little, stop, move a little ────────────────────
def test_forward_duty_cycle():
    gait = ClosureGait(ClosureGaitConfig(move_ticks=2, settle_ticks=3,
                                         settle_on_axis_change=True))
    pattern = [_moving(gait.step(_fwd())) for _ in range(10)]
    # 2 move, 3 stop, 2 move, 3 stop, ...
    assert pattern == [True, True, False, False, False,
                       True, True, False, False, False]


def test_settle_state_transitions():
    gait = ClosureGait(ClosureGaitConfig(move_ticks=2, settle_ticks=2))
    gait.step(_fwd())                      # move 1/2
    assert gait.state == MOVING
    gait.step(_fwd())                      # move 2/2 -> reaches move_ticks, settling
    assert gait.state == SETTLING
    gait.step(_fwd())                      # stop 1/2
    assert gait.state == SETTLING
    gait.step(_fwd())                      # stop 2/2 (still settling this tick)
    assert gait.state == SETTLING
    d = gait.step(_fwd())                  # settle done -> resumes moving this tick
    assert gait.state == MOVING
    assert _moving(d)


# ── transition between turn and forward inserts a stop ────────────────
def test_turn_to_forward_settles_first():
    gait = ClosureGait(ClosureGaitConfig(move_ticks=5, settle_ticks=2,
                                         settle_on_axis_change=True))
    # One yaw burst tick (well under move_ticks), then switch to forward.
    assert _moving(gait.step(_yaw()))
    # The category change forces a stop before forward is allowed through.
    out = gait.step(_fwd())
    assert not _moving(out)
    assert gait.state == SETTLING
    # After the settle, forward flows.
    gait.step(_fwd())                      # settle tick 2/2
    assert _moving(gait.step(_fwd()))


def test_transition_settle_can_be_disabled():
    gait = ClosureGait(ClosureGaitConfig(move_ticks=5, settle_ticks=2,
                                         settle_on_axis_change=False))
    gait.step(_yaw())
    # No transition settle: forward passes straight through (same burst).
    assert _moving(gait.step(_fwd()))


# ── natural stop pass-through does not consume a burst ────────────────
def test_natural_stop_passes_through_and_resets_burst():
    gait = ClosureGait(ClosureGaitConfig(move_ticks=2, settle_ticks=3))
    gait.step(_fwd())                      # 1 motion tick
    assert not _moving(gait.step(_stop()))  # servo idles -> passes, burst reset
    # Burst counter reset, so two fresh motion ticks are allowed before settling.
    assert _moving(gait.step(_fwd()))
    assert _moving(gait.step(_fwd()))
    assert not _moving(gait.step(_fwd()))   # now it settles


# ── reset ─────────────────────────────────────────────────────────────
def test_reset_clears_settle():
    gait = ClosureGait(ClosureGaitConfig(move_ticks=1, settle_ticks=5))
    gait.step(_fwd())                      # move_ticks=1 -> immediately settling
    assert gait.state == SETTLING
    gait.reset()
    assert gait.state == MOVING
    assert _moving(gait.step(_fwd()))      # fresh burst allowed right away


# ── metadata is preserved through a settle ────────────────────────────
def test_stop_preserves_metadata():
    gait = ClosureGait(ClosureGaitConfig(move_ticks=1, settle_ticks=2))
    gait.step(ControlCommand.velocity(0.3, 0.0, 0.0, 0.0, source="servo"))
    out = gait.step(ControlCommand.velocity(0.3, 0.0, 0.0, 0.0, source="servo"))
    assert not _moving(out)
    assert out.metadata.get("source") == "servo"
