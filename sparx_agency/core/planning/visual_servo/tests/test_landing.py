"""Unit tests for the terminal LAND trigger of the visual-approach state machine.

Covers the ``land_range_m`` / ``land_confirm_ticks`` behaviour added to
:class:`VisualApproachStateMachine`: entering LAND after a sustained in-range depth
reading during APPROACH or HOVER_LOCK, the confirm-tick debounce, the streak reset on
any out-of-range / no-range / state-change tick, LAND being terminal, and the
disabled-by-default (``land_range_m is None``) path that preserves hover-lock.

The FSM is pure: transitions depend only on the booleans plus the optional metric
``range_m``, so every test drives it with explicit values and asserts on the returned
:class:`ApproachDecision`.
"""
from __future__ import annotations

import pytest

from sparx_agency.core.planning.visual_servo.state_machine import (
    APPROACH,
    HOVER_LOCK,
    LAND,
    RECOVER,
    SEARCH,
    ApproachFSMConfig,
    VisualApproachStateMachine,
)


# ── fixtures / helpers ────────────────────────────────────────────────
def _land_fsm(land_range_m=1.0, land_confirm_ticks=3, recover_confirm_ticks=2,
              recover_timeout_s=10.0) -> VisualApproachStateMachine:
    """FSM with the LAND trigger enabled and acquire-stop skipped (confirm -> APPROACH)."""
    return VisualApproachStateMachine(ApproachFSMConfig(
        recover_timeout_s=recover_timeout_s,
        recover_confirm_ticks=recover_confirm_ticks,
        land_range_m=land_range_m,
        land_confirm_ticks=land_confirm_ticks))


def _to_approach(fsm: VisualApproachStateMachine):
    """Drive a fresh SEARCH machine straight into APPROACH (acquire_stop_s == 0)."""
    d = fsm.update(confirmed=True, track_valid=True, at_target=False, dt=0.1)
    assert d.mode == APPROACH
    return d


def _tick(fsm, range_m=None, track_valid=True, at_target=False):
    return fsm.update(confirmed=True, track_valid=track_valid, at_target=at_target,
                      dt=0.1, range_m=range_m)


# ── config validation ─────────────────────────────────────────────────
def test_land_disabled_by_default():
    assert ApproachFSMConfig().land_range_m is None
    assert ApproachFSMConfig().land_confirm_ticks == 3


@pytest.mark.parametrize("bad", [0.0, -0.1, -2.0])
def test_invalid_land_range_raises(bad):
    with pytest.raises(ValueError):
        ApproachFSMConfig(land_range_m=bad)


@pytest.mark.parametrize("bad", [0, -1])
def test_invalid_land_confirm_ticks_raises(bad):
    with pytest.raises(ValueError):
        ApproachFSMConfig(land_confirm_ticks=bad)


def test_positive_land_range_is_valid():
    cfg = ApproachFSMConfig(land_range_m=1.5, land_confirm_ticks=1)
    assert cfg.land_range_m == 1.5


# ── disabled path (None) keeps hover-lock ─────────────────────────────
def test_disabled_never_lands_even_when_in_range():
    # land_range_m None: a close range never triggers LAND, it stays APPROACH.
    fsm = VisualApproachStateMachine(ApproachFSMConfig(recover_timeout_s=10.0))
    _to_approach(fsm)
    for _ in range(10):
        d = _tick(fsm, range_m=0.05)
        assert d.mode == APPROACH
        assert d.land is False


# ── LAND from APPROACH ─────────────────────────────────────────────────
def test_approach_lands_after_confirm_ticks():
    fsm = _land_fsm(land_range_m=1.0, land_confirm_ticks=3)
    _to_approach(fsm)
    # First two in-range ticks accumulate but do not land (streak < 3).
    assert _tick(fsm, range_m=0.9).mode == APPROACH
    assert _tick(fsm, range_m=0.8).mode == APPROACH
    # Third consecutive in-range tick commits to LAND.
    d = _tick(fsm, range_m=0.7)
    assert d.mode == LAND
    assert d.land is True
    assert d.drive_cmd_vel is False


def test_land_needs_strictly_within_or_equal_range():
    # Exactly at land_range_m counts (<=); just above does not.
    fsm = _land_fsm(land_range_m=1.0, land_confirm_ticks=1)
    _to_approach(fsm)
    assert _tick(fsm, range_m=1.0001).mode == APPROACH
    assert _tick(fsm, range_m=1.0).mode == LAND


def test_out_of_range_tick_resets_the_streak():
    fsm = _land_fsm(land_range_m=1.0, land_confirm_ticks=3)
    _to_approach(fsm)
    _tick(fsm, range_m=0.8)          # streak 1
    _tick(fsm, range_m=0.8)          # streak 2
    assert _tick(fsm, range_m=1.5).mode == APPROACH   # out of range -> streak 0
    # Must take a fresh three consecutive in-range ticks now.
    assert _tick(fsm, range_m=0.8).mode == APPROACH   # 1
    assert _tick(fsm, range_m=0.8).mode == APPROACH   # 2
    assert _tick(fsm, range_m=0.8).mode == LAND       # 3


def test_missing_range_tick_resets_the_streak():
    # A None range (depth dropped this frame) is treated as out-of-range: reset.
    fsm = _land_fsm(land_range_m=1.0, land_confirm_ticks=3)
    _to_approach(fsm)
    _tick(fsm, range_m=0.8)          # streak 1
    _tick(fsm, range_m=0.8)          # streak 2
    assert _tick(fsm, range_m=None).mode == APPROACH  # no range -> streak 0
    assert _tick(fsm, range_m=0.8).mode == APPROACH   # 1
    assert _tick(fsm, range_m=0.8).mode == APPROACH   # 2
    assert _tick(fsm, range_m=0.8).mode == LAND       # 3


def test_single_tick_confirm_lands_immediately():
    fsm = _land_fsm(land_range_m=1.0, land_confirm_ticks=1)
    _to_approach(fsm)
    assert _tick(fsm, range_m=0.9).mode == LAND


# ── LAND is terminal ───────────────────────────────────────────────────
def test_land_is_terminal_ignores_lost_track():
    # recover_confirm_ticks=1 makes this a genuine mutation test: WITHOUT the terminal
    # guard in update(), a LAND state falls through to the RECOVER block and a single
    # valid tick (>= recover_confirm_ticks) would escape LAND -> APPROACH. With the
    # guard, every tick below must stay LAND.
    fsm = _land_fsm(land_range_m=1.0, land_confirm_ticks=1, recover_confirm_ticks=1)
    _to_approach(fsm)
    assert _tick(fsm, range_m=0.5).mode == LAND
    # A lost track would normally go to RECOVER; LAND never leaves LAND.
    d = _tick(fsm, range_m=None, track_valid=False)
    assert d.mode == LAND
    assert d.land is True
    # A re-acquired, in-frame target -- which without the guard would immediately
    # re-enter APPROACH via the recovery path -- must NOT restart the approach.
    for _ in range(3):
        d = _tick(fsm, range_m=5.0, track_valid=True, at_target=False)
        assert d.mode == LAND
        assert d.land is True
    # Even an at-target re-acquire stays terminal.
    assert _tick(fsm, range_m=0.5, track_valid=True, at_target=True).mode == LAND


# ── LAND from HOVER_LOCK (land closer than the hover standoff) ─────────
def test_hover_lock_lands_when_range_falls_within_land_range():
    # land closer (0.4 m) than the at-target hover; hover is entered first, then a
    # sustained closer reading lands.
    fsm = _land_fsm(land_range_m=0.4, land_confirm_ticks=3)
    _to_approach(fsm)
    # at_target with range still outside land range -> HOVER_LOCK, no land.
    assert _tick(fsm, range_m=0.6, at_target=True).mode == HOVER_LOCK
    assert _tick(fsm, range_m=0.3, at_target=True).mode == HOVER_LOCK   # streak 1
    assert _tick(fsm, range_m=0.3, at_target=True).mode == HOVER_LOCK   # streak 2
    assert _tick(fsm, range_m=0.3, at_target=True).mode == LAND         # streak 3


# ── streak does not survive a state change ────────────────────────────
def test_streak_resets_across_a_recover_excursion():
    fsm = _land_fsm(land_range_m=1.0, land_confirm_ticks=3, recover_confirm_ticks=1)
    _to_approach(fsm)
    _tick(fsm, range_m=0.8)                      # streak 1
    _tick(fsm, range_m=0.8)                      # streak 2
    assert _tick(fsm, range_m=None, track_valid=False).mode == RECOVER  # streak reset
    assert _tick(fsm, range_m=0.8, track_valid=True).mode == APPROACH   # regained
    # The pre-RECOVER progress is gone: two in-range ticks are not yet enough.
    assert _tick(fsm, range_m=0.8).mode == APPROACH   # 1
    assert _tick(fsm, range_m=0.8).mode == APPROACH   # 2
    assert _tick(fsm, range_m=0.8).mode == LAND       # 3


# ── reset() clears land progress ──────────────────────────────────────
def test_reset_clears_land_streak():
    fsm = _land_fsm(land_range_m=1.0, land_confirm_ticks=3)
    _to_approach(fsm)
    _tick(fsm, range_m=0.8)          # streak 1
    _tick(fsm, range_m=0.8)          # streak 2
    fsm.reset()
    assert fsm.state == SEARCH
    _to_approach(fsm)
    # Fresh episode: two in-range ticks are not enough (streak restarted at 0).
    assert _tick(fsm, range_m=0.8).mode == APPROACH
    assert _tick(fsm, range_m=0.8).mode == APPROACH
    assert _tick(fsm, range_m=0.8).mode == LAND
