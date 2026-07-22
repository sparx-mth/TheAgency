"""Unit tests for the coordinate-arrival LAND trigger of the visual-approach FSM.

Covers the ``land_at_goal`` / ``arrive_land_confirm_ticks`` behaviour: entering the
terminal LAND from SEARCH once the coordinate route has arrived at its goal still
unconfirmed for the required consecutive ticks ("reached the object by A* alone"),
the confirm-tick debounce, the streak reset when the drone leaves the arrival radius,
confirm+lock taking precedence over arrival-land, the disabled-by-default path (arrival
falls back to the legacy SCAN sweep), independence from depth (no ``range_m`` needed),
and co-existence with the depth ``land_range_m`` trigger.

The FSM is pure: transitions depend only on the booleans, so every test drives it with
explicit values and asserts on the returned :class:`ApproachDecision`.
"""
from __future__ import annotations

import pytest

from sparx_agency.core.planning.visual_servo.state_machine import (
    APPROACH,
    LAND,
    SCAN,
    SEARCH,
    ApproachFSMConfig,
    VisualApproachStateMachine,
)


# ── fixtures / helpers ────────────────────────────────────────────────
def _goal_land_fsm(arrive_land_confirm_ticks=3, land_range_m=None):
    """FSM with the coordinate-arrival LAND trigger enabled."""
    return VisualApproachStateMachine(ApproachFSMConfig(
        recover_timeout_s=10.0,
        land_at_goal=True,
        arrive_land_confirm_ticks=arrive_land_confirm_ticks,
        land_range_m=land_range_m))


def _arrived(fsm, arrived_at_goal=True, confirmed=False, track_valid=False):
    """One SEARCH tick with the given arrival / confirmation state (no depth)."""
    return fsm.update(confirmed=confirmed, track_valid=track_valid, at_target=False,
                      dt=0.1, arrived_at_goal=arrived_at_goal)


# ── config validation ─────────────────────────────────────────────────
def test_land_at_goal_disabled_by_default():
    cfg = ApproachFSMConfig()
    assert cfg.land_at_goal is False
    assert cfg.arrive_land_confirm_ticks == 5


@pytest.mark.parametrize("bad", [0, -1])
def test_invalid_arrive_land_confirm_ticks_raises(bad):
    with pytest.raises(ValueError):
        ApproachFSMConfig(arrive_land_confirm_ticks=bad)


# ── disabled path: arrival -> SCAN, never LAND ────────────────────────
def test_disabled_arrival_scans_never_lands():
    fsm = VisualApproachStateMachine(ApproachFSMConfig(recover_timeout_s=10.0))
    for _ in range(10):
        d = _arrived(fsm)
        assert d.mode == SCAN
        assert d.land is False


# ── LAND from SEARCH on sustained arrival ─────────────────────────────
def test_arrival_lands_after_confirm_ticks():
    fsm = _goal_land_fsm(arrive_land_confirm_ticks=3)
    # First two arrived ticks accumulate but do not land (streak < 3); the machine
    # stays in passive SEARCH (the follower keeps the drone at the goal).
    d = _arrived(fsm)
    assert d.mode == SEARCH and d.land is False
    assert _arrived(fsm).mode == SEARCH
    # Third consecutive arrived tick commits to LAND.
    d = _arrived(fsm)
    assert d.mode == LAND
    assert d.land is True
    assert d.drive_cmd_vel is False


def test_single_tick_confirm_lands_immediately():
    fsm = _goal_land_fsm(arrive_land_confirm_ticks=1)
    assert _arrived(fsm).mode == LAND


def test_leaving_the_radius_resets_the_streak():
    fsm = _goal_land_fsm(arrive_land_confirm_ticks=3)
    _arrived(fsm)                                   # streak 1
    _arrived(fsm)                                   # streak 2
    assert _arrived(fsm, arrived_at_goal=False).mode == SEARCH   # left radius -> reset
    # A fresh three consecutive arrived ticks are now required.
    assert _arrived(fsm).mode == SEARCH             # 1
    assert _arrived(fsm).mode == SEARCH             # 2
    assert _arrived(fsm).mode == LAND               # 3


def test_arrival_land_needs_no_depth():
    # The whole sequence runs with range_m defaulting to None (no depth at all).
    fsm = _goal_land_fsm(arrive_land_confirm_ticks=2)
    assert _arrived(fsm).mode == SEARCH
    assert _arrived(fsm).mode == LAND


# ── confirm+lock beats arrival-land ───────────────────────────────────
def test_confirmed_lock_takes_precedence_over_arrival_land():
    # An arrived tick that is ALSO confirmed+locked goes to the visual approach,
    # not arrival-LAND -- we prefer to close on the object we can actually see.
    fsm = _goal_land_fsm(arrive_land_confirm_ticks=3)
    _arrived(fsm)                                   # streak 1
    d = _arrived(fsm, confirmed=True, track_valid=True)
    assert d.mode == APPROACH                       # acquire wins
    assert d.land is False


def test_confirmed_without_lock_does_not_advance_or_land():
    # confirmed but track_valid False: the object WAS seen but the tracker cannot hold a
    # box this tick. It must NOT count as arrival-land (that is only for an object the
    # detector never saw) -- the streak breaks and the machine holds in passive SEARCH
    # for the tracker to re-acquire and visually approach, never blind-landing on it.
    fsm = _goal_land_fsm(arrive_land_confirm_ticks=2)
    assert _arrived(fsm).mode == SEARCH             # unconfirmed arrival -> streak 1
    # A confirmed (but untracked) arrived tick resets the streak and does NOT land.
    d = fsm.update(confirmed=True, track_valid=False, at_target=False, dt=0.1,
                   arrived_at_goal=True)
    assert d.mode == SEARCH
    assert d.land is False
    # Even sustained: a persistently confirmed-but-untracked object never arrival-lands.
    for _ in range(5):
        d = fsm.update(confirmed=True, track_valid=False, at_target=False, dt=0.1,
                       arrived_at_goal=True)
        assert d.mode == SEARCH
        assert d.land is False
    # A fresh UNCONFIRMED streak must start from zero (the confirmed ticks reset it).
    assert _arrived(fsm).mode == SEARCH             # 1
    assert _arrived(fsm).mode == LAND               # 2


# ── active APPROACH never lands via the arrival path ──────────────────
def test_active_approach_ignores_arrived_flag_for_landing():
    # Once visually approaching, arrived_at_goal must not trigger a coordinate LAND
    # (that trigger lives only in SEARCH). With no depth range, APPROACH just holds.
    fsm = _goal_land_fsm(arrive_land_confirm_ticks=1, land_range_m=None)
    d = fsm.update(confirmed=True, track_valid=True, at_target=False, dt=0.1)
    assert d.mode == APPROACH
    for _ in range(5):
        d = fsm.update(confirmed=True, track_valid=True, at_target=False, dt=0.1,
                       arrived_at_goal=True, range_m=None)
        assert d.mode == APPROACH
        assert d.land is False


# ── terminal + reset ──────────────────────────────────────────────────
def test_arrival_land_is_terminal():
    fsm = _goal_land_fsm(arrive_land_confirm_ticks=1)
    assert _arrived(fsm).mode == LAND
    # No input restarts the mission from the terminal LAND.
    for _ in range(3):
        d = fsm.update(confirmed=True, track_valid=True, at_target=True, dt=0.1,
                       arrived_at_goal=False, range_m=0.2)
        assert d.mode == LAND
        assert d.land is True


def test_reset_clears_arrival_streak():
    fsm = _goal_land_fsm(arrive_land_confirm_ticks=3)
    _arrived(fsm)                                   # streak 1
    _arrived(fsm)                                   # streak 2
    fsm.reset()
    assert fsm.state == SEARCH
    # Fresh episode: two arrived ticks are not enough (streak restarted at 0).
    assert _arrived(fsm).mode == SEARCH
    assert _arrived(fsm).mode == SEARCH
    assert _arrived(fsm).mode == LAND


# ── co-existence with the depth land trigger ──────────────────────────
def test_arrival_and_depth_land_coexist():
    # Both enabled: with no confirmation the arrival trigger still lands from SEARCH,
    # independent of the depth land_range_m (which only acts in APPROACH/HOVER_LOCK).
    fsm = _goal_land_fsm(arrive_land_confirm_ticks=2, land_range_m=1.0)
    assert _arrived(fsm).mode == SEARCH
    assert _arrived(fsm).mode == LAND
