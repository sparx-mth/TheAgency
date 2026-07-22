"""Unit tests for the visual-approach mission state machine.

Covers :meth:`VisualApproachStateMachine.update` across every documented
SEARCH / APPROACH / HOVER_LOCK / RECOVER transition, plus the
:class:`ApproachFSMConfig` validation and :meth:`reset`.

The FSM is pure: transitions depend only on the ``confirmed`` /
``track_valid`` / ``at_target`` booleans and the ``dt`` recovery timer, so
every test drives it with explicit ``dt`` values and asserts on the returned
:class:`ApproachDecision`.
"""
from __future__ import annotations

import pytest

from sparx_agency.core.planning.visual_servo.state_machine import (
    ACQUIRE_STOP,
    APPROACH,
    HOVER_LOCK,
    RECOVER,
    SCAN,
    SEARCH,
    ApproachDecision,
    ApproachFSMConfig,
    VisualApproachStateMachine,
)


# ── fixtures / helpers ────────────────────────────────────────────────
def _make(recover_timeout_s: float = 3.0,
          hover_release_ticks: int = 3,
          recover_confirm_ticks: int = 2) -> VisualApproachStateMachine:
    """FSM with small, explicit thresholds for readable tests."""
    return VisualApproachStateMachine(
        ApproachFSMConfig(recover_timeout_s=recover_timeout_s,
                          hover_release_ticks=hover_release_ticks,
                          recover_confirm_ticks=recover_confirm_ticks)
    )


def _to_approach(fsm: VisualApproachStateMachine) -> ApproachDecision:
    """Drive a fresh SEARCH machine into APPROACH and return the decision."""
    return fsm.update(confirmed=True, track_valid=True, at_target=False, dt=0.1)


# ── config validation ─────────────────────────────────────────────────
def test_default_config_is_valid():
    fsm = VisualApproachStateMachine()
    assert fsm.state == SEARCH
    assert fsm.cfg.recover_timeout_s == 6.0
    assert fsm.cfg.hover_release_ticks == 5


@pytest.mark.parametrize("bad", [0.0, -1.0])
def test_invalid_recover_timeout_raises(bad):
    with pytest.raises(ValueError):
        ApproachFSMConfig(recover_timeout_s=bad)


@pytest.mark.parametrize("bad", [0, -1])
def test_invalid_hover_release_ticks_raises(bad):
    with pytest.raises(ValueError):
        ApproachFSMConfig(hover_release_ticks=bad)


# ── initial state ─────────────────────────────────────────────────────
def test_starts_in_search_passive():
    fsm = _make()
    assert fsm.state == SEARCH
    d = fsm.update(confirmed=False, track_valid=False, at_target=False, dt=0.1)
    assert d.mode == SEARCH
    assert d.drive_cmd_vel is False
    assert d.reset_acquisition is False
    assert d.lost_for_s == 0.0


# ── SEARCH transitions ────────────────────────────────────────────────
def test_search_confirmed_but_track_invalid_stays_search():
    fsm = _make()
    d = fsm.update(confirmed=True, track_valid=False, at_target=False, dt=0.1)
    assert d.mode == SEARCH
    assert d.drive_cmd_vel is False


def test_search_track_valid_but_unconfirmed_stays_search():
    fsm = _make()
    d = fsm.update(confirmed=False, track_valid=True, at_target=False, dt=0.1)
    assert d.mode == SEARCH
    assert d.drive_cmd_vel is False


def test_search_confirmed_and_track_valid_enters_approach():
    fsm = _make()
    d = _to_approach(fsm)
    assert d.mode == APPROACH
    assert d.drive_cmd_vel is True
    assert d.reset_acquisition is False
    assert d.lost_for_s == 0.0


# ── SCAN transitions (arrived at goal, still looking) ─────────────────
def test_search_arrived_at_goal_enters_scan():
    fsm = _make()
    d = fsm.update(confirmed=False, track_valid=False, at_target=False, dt=0.1,
                   arrived_at_goal=True)
    assert d.mode == SCAN
    assert d.drive_cmd_vel is True          # the node now drives the sweep
    assert d.reset_acquisition is False
    assert d.lost_for_s == 0.0


def test_search_confirm_beats_arrived():
    # Confirmed+locked on the same tick we arrive -> go straight to APPROACH.
    fsm = _make()
    d = fsm.update(confirmed=True, track_valid=True, at_target=False, dt=0.1,
                   arrived_at_goal=True)
    assert d.mode == APPROACH


def test_scan_confirmed_and_locked_enters_approach():
    fsm = _make()
    fsm.update(confirmed=False, track_valid=False, at_target=False, dt=0.1,
               arrived_at_goal=True)
    assert fsm.state == SCAN
    d = fsm.update(confirmed=True, track_valid=True, at_target=False, dt=0.1,
                   arrived_at_goal=True)
    assert d.mode == APPROACH
    assert d.drive_cmd_vel is True


def test_scan_stays_while_arrived_and_unconfirmed():
    fsm = _make()
    fsm.update(confirmed=False, track_valid=False, at_target=False, dt=0.1,
               arrived_at_goal=True)
    d = fsm.update(confirmed=False, track_valid=False, at_target=False, dt=0.1,
                   arrived_at_goal=True)
    assert d.mode == SCAN


def test_scan_goal_cleared_falls_back_to_search():
    # Goal changed out from under us (no longer at the goal) -> hand back to planner.
    fsm = _make()
    fsm.update(confirmed=False, track_valid=False, at_target=False, dt=0.1,
               arrived_at_goal=True)
    d = fsm.update(confirmed=False, track_valid=False, at_target=False, dt=0.1,
                   arrived_at_goal=False)
    assert d.mode == SEARCH
    assert d.drive_cmd_vel is False


def test_recover_giveup_then_rescans_when_still_at_goal():
    # A closure that was reached from a scan, then lost the object: on the recover
    # give-up it returns to SEARCH (reset), and re-enters SCAN the next tick because
    # the drone is still standing at the goal.
    fsm = _make(recover_timeout_s=1.0)
    fsm.update(confirmed=True, track_valid=True, at_target=False, dt=0.1,
               arrived_at_goal=True)                      # APPROACH
    fsm.update(confirmed=True, track_valid=False, at_target=False, dt=0.1,
               arrived_at_goal=True)                      # RECOVER
    d = fsm.update(confirmed=True, track_valid=False, at_target=False, dt=2.0,
                   arrived_at_goal=True)                  # timeout -> SEARCH + reset
    assert d.mode == SEARCH
    assert d.reset_acquisition is True
    d = fsm.update(confirmed=False, track_valid=False, at_target=False, dt=0.1,
                   arrived_at_goal=True)                  # back to SCAN
    assert d.mode == SCAN


# ── ACQUIRE_STOP settle (confirm+lock -> brief stop -> approach) ───────
def test_default_config_skips_settle():
    # acquire_stop_s defaults to 0: confirm+lock goes straight to APPROACH (legacy).
    assert ApproachFSMConfig().acquire_stop_s == 0.0
    fsm = _make()
    d = _to_approach(fsm)
    assert d.mode == APPROACH


@pytest.mark.parametrize("bad", [-0.1, -1.0])
def test_invalid_acquire_stop_raises(bad):
    with pytest.raises(ValueError):
        ApproachFSMConfig(acquire_stop_s=bad)


def _make_settle(acquire_stop_s: float = 1.0) -> VisualApproachStateMachine:
    return VisualApproachStateMachine(
        ApproachFSMConfig(recover_timeout_s=3.0, acquire_stop_s=acquire_stop_s))


def test_search_confirmed_enters_acquire_stop_when_settle_enabled():
    fsm = _make_settle(1.0)
    d = fsm.update(confirmed=True, track_valid=True, at_target=False, dt=0.1)
    assert d.mode == ACQUIRE_STOP
    assert d.drive_cmd_vel is True          # node owns cmd_vel to publish the stop
    assert d.lost_for_s == 0.0


def test_scan_confirmed_enters_acquire_stop_when_settle_enabled():
    fsm = _make_settle(1.0)
    fsm.update(confirmed=False, track_valid=False, at_target=False, dt=0.1,
               arrived_at_goal=True)        # -> SCAN
    assert fsm.state == SCAN
    d = fsm.update(confirmed=True, track_valid=True, at_target=False, dt=0.1,
                   arrived_at_goal=True)
    assert d.mode == ACQUIRE_STOP


def test_acquire_stop_holds_then_advances_to_approach():
    # dt=0.5 is exactly representable, so the 1.0s threshold is hit cleanly.
    fsm = _make_settle(1.0)
    fsm.update(confirmed=True, track_valid=True, at_target=False, dt=0.5)  # ACQUIRE_STOP
    # First accumulating tick: 0.5s < 1.0s -> still stopped in place.
    d = fsm.update(confirmed=True, track_valid=True, at_target=False, dt=0.5)
    assert d.mode == ACQUIRE_STOP
    # Second tick reaches acquire_stop_s (1.0s) -> start the approach.
    d = fsm.update(confirmed=True, track_valid=True, at_target=False, dt=0.5)
    assert d.mode == APPROACH


def test_acquire_stop_completes_even_if_track_flickers():
    # The settle is time-boxed only: a mid-settle track drop does not abort it (the
    # drone still brakes to a stop); APPROACH handles any real loss afterwards.
    fsm = _make_settle(0.3)
    fsm.update(confirmed=True, track_valid=True, at_target=False, dt=0.1)  # ACQUIRE_STOP
    d = fsm.update(confirmed=True, track_valid=False, at_target=False, dt=0.1)
    assert d.mode == ACQUIRE_STOP
    d = fsm.update(confirmed=True, track_valid=False, at_target=False, dt=0.2)
    assert d.mode == APPROACH


def test_acquire_stop_settle_timer_resets_between_episodes():
    # A give-up back to SEARCH then a fresh confirm must restart the settle from 0.
    fsm = _make_settle(0.3)
    fsm.update(confirmed=True, track_valid=True, at_target=False, dt=0.1)  # ACQUIRE_STOP
    fsm.reset()
    assert fsm.state == SEARCH
    d = fsm.update(confirmed=True, track_valid=True, at_target=False, dt=0.1)
    assert d.mode == ACQUIRE_STOP          # fresh settle, not instantly elapsed


# ── APPROACH transitions ──────────────────────────────────────────────
def test_approach_track_lost_enters_recover():
    fsm = _make()
    _to_approach(fsm)
    d = fsm.update(confirmed=True, track_valid=False, at_target=False, dt=0.1)
    assert d.mode == RECOVER
    assert d.drive_cmd_vel is True


def test_approach_at_target_enters_hover_lock():
    fsm = _make()
    _to_approach(fsm)
    d = fsm.update(confirmed=True, track_valid=True, at_target=True, dt=0.1)
    assert d.mode == HOVER_LOCK
    assert d.drive_cmd_vel is True


def test_approach_stays_when_tracking_and_not_at_target():
    fsm = _make()
    _to_approach(fsm)
    d = fsm.update(confirmed=True, track_valid=True, at_target=False, dt=0.1)
    assert d.mode == APPROACH


# ── HOVER_LOCK transitions ────────────────────────────────────────────
def test_hover_lock_holds_below_release_ticks():
    fsm = _make(hover_release_ticks=3)
    _to_approach(fsm)
    fsm.update(confirmed=True, track_valid=True, at_target=True, dt=0.1)
    assert fsm.state == HOVER_LOCK
    # Two consecutive not-at-target ticks (< 3) must stay in HOVER_LOCK.
    for _ in range(2):
        d = fsm.update(confirmed=True, track_valid=True, at_target=False, dt=0.1)
        assert d.mode == HOVER_LOCK
        assert d.drive_cmd_vel is True


def test_hover_lock_releases_at_release_ticks():
    fsm = _make(hover_release_ticks=3)
    _to_approach(fsm)
    fsm.update(confirmed=True, track_valid=True, at_target=True, dt=0.1)
    # First two not-at-target ticks stay; the third crosses the threshold.
    fsm.update(confirmed=True, track_valid=True, at_target=False, dt=0.1)
    fsm.update(confirmed=True, track_valid=True, at_target=False, dt=0.1)
    d = fsm.update(confirmed=True, track_valid=True, at_target=False, dt=0.1)
    assert d.mode == APPROACH


def test_hover_lock_release_counter_resets_on_recenter():
    fsm = _make(hover_release_ticks=3)
    _to_approach(fsm)
    fsm.update(confirmed=True, track_valid=True, at_target=True, dt=0.1)
    # Two not-at-target ticks, then re-centre resets the release counter...
    fsm.update(confirmed=True, track_valid=True, at_target=False, dt=0.1)
    fsm.update(confirmed=True, track_valid=True, at_target=False, dt=0.1)
    d = fsm.update(confirmed=True, track_valid=True, at_target=True, dt=0.1)
    assert d.mode == HOVER_LOCK
    # ...so two more slips still hold instead of releasing.
    fsm.update(confirmed=True, track_valid=True, at_target=False, dt=0.1)
    d = fsm.update(confirmed=True, track_valid=True, at_target=False, dt=0.1)
    assert d.mode == HOVER_LOCK


def test_hover_lock_track_lost_enters_recover():
    fsm = _make()
    _to_approach(fsm)
    fsm.update(confirmed=True, track_valid=True, at_target=True, dt=0.1)
    assert fsm.state == HOVER_LOCK
    d = fsm.update(confirmed=True, track_valid=False, at_target=True, dt=0.1)
    assert d.mode == RECOVER


# ── RECOVER transitions ───────────────────────────────────────────────
def _into_recover(fsm: VisualApproachStateMachine) -> None:
    _to_approach(fsm)
    fsm.update(confirmed=True, track_valid=False, at_target=False, dt=0.1)
    assert fsm.state == RECOVER


def test_recover_track_regained_enters_approach_after_confirm_ticks():
    # A true re-acquisition needs recover_confirm_ticks consecutive valid ticks;
    # a single valid tick is not enough (guards against a spurious re-detection).
    fsm = _make(recover_timeout_s=10.0, recover_confirm_ticks=2)
    _into_recover(fsm)
    d = fsm.update(confirmed=True, track_valid=False, at_target=False, dt=1.0)
    assert d.mode == RECOVER and d.lost_for_s == pytest.approx(1.0)
    # First valid tick: still RECOVER (not yet confirmed), lost timer NOT reset.
    d = fsm.update(confirmed=True, track_valid=True, at_target=False, dt=1.0)
    assert d.mode == RECOVER
    assert d.lost_for_s == pytest.approx(2.0)
    # Second consecutive valid tick: confirmed -> APPROACH, timer cleared.
    d = fsm.update(confirmed=True, track_valid=True, at_target=False, dt=1.0)
    assert d.mode == APPROACH
    assert d.lost_for_s == 0.0
    assert d.reset_acquisition is False


def test_recover_flicker_cannot_starve_the_timeout():
    # Intermittent single-frame re-detections (valid, invalid, valid, ...) must NOT
    # keep resetting the episode: the lost timer accumulates across the whole
    # recovery and eventually gives up to SEARCH with a re-acquisition reset.
    fsm = _make(recover_timeout_s=3.0, recover_confirm_ticks=2)
    _into_recover(fsm)
    last = None
    for i in range(10):
        last = fsm.update(confirmed=True, track_valid=(i % 2 == 0),
                          at_target=False, dt=1.0)
        if last.mode == SEARCH:
            break
    assert last.mode == SEARCH
    assert last.reset_acquisition is True


def test_recover_accumulates_lost_time():
    fsm = _make(recover_timeout_s=10.0)
    _into_recover(fsm)
    d1 = fsm.update(confirmed=True, track_valid=False, at_target=False, dt=1.5)
    d2 = fsm.update(confirmed=True, track_valid=False, at_target=False, dt=2.0)
    assert d1.mode == RECOVER
    assert d1.lost_for_s == pytest.approx(1.5)
    assert d2.lost_for_s == pytest.approx(3.5)


def test_recover_timeout_gives_up_to_search_with_reset():
    fsm = _make(recover_timeout_s=3.0)
    _into_recover(fsm)
    # Two 1.0s ticks: still lost (2.0 < 3.0).
    fsm.update(confirmed=True, track_valid=False, at_target=False, dt=1.0)
    d = fsm.update(confirmed=True, track_valid=False, at_target=False, dt=1.0)
    assert d.mode == RECOVER
    assert d.lost_for_s == pytest.approx(2.0)
    # Third tick reaches the timeout -> give up to SEARCH, flag re-acquisition.
    d = fsm.update(confirmed=True, track_valid=False, at_target=False, dt=1.0)
    assert d.mode == SEARCH
    assert d.reset_acquisition is True
    assert d.drive_cmd_vel is False
    assert d.lost_for_s == 0.0


def test_recover_reset_flag_only_on_giveup_edge():
    fsm = _make(recover_timeout_s=2.0)
    _into_recover(fsm)
    d = fsm.update(confirmed=True, track_valid=False, at_target=False, dt=1.0)
    assert d.reset_acquisition is False  # still recovering, no give-up yet


# ── reset() ───────────────────────────────────────────────────────────
def test_reset_returns_to_search():
    fsm = _make()
    _into_recover(fsm)
    fsm.update(confirmed=True, track_valid=False, at_target=False, dt=1.0)
    assert fsm.state == RECOVER
    fsm.reset()
    assert fsm.state == SEARCH
    # Timers cleared: a fresh RECOVER entry starts its lost timer from zero.
    _into_recover(fsm)
    d = fsm.update(confirmed=True, track_valid=False, at_target=False, dt=1.0)
    assert d.lost_for_s == pytest.approx(1.0)


# ── dt hygiene ────────────────────────────────────────────────────────
def test_negative_dt_is_clamped_to_zero():
    fsm = _make(recover_timeout_s=3.0)
    _into_recover(fsm)
    d = fsm.update(confirmed=True, track_valid=False, at_target=False, dt=-5.0)
    assert d.mode == RECOVER
    assert d.lost_for_s == pytest.approx(0.0)
