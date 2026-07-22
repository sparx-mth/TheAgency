"""Unit tests for the bounded room sweep and the pre-arm gate in the visual FSM.

Three mission behaviours layered on top of the staged approach:

  * ``aim_fail_to_scan`` -- a failed aim SWEEPS the room in place for the object rather
    than escalating the goal onto the object's imprecise catalogued coordinate;
  * ``scan_land_revolutions`` -- that sweep is bounded: after N full in-place turns
    without a confirmation, the mission gives up and lands;
  * ``arm_ready`` -- the BLIND visual take-over is forbidden (the machine is pinned in
    passive SEARCH) until the drone has flown the hard part of the route into the room.

The FSM is pure, so each test drives it with explicit booleans / swept angles and
asserts on the returned :class:`ApproachDecision`.
"""
from __future__ import annotations

import math

import pytest

from sparx_agency.core.planning.visual_servo.state_machine import (
    APPROACH,
    AIM,
    LAND,
    SCAN,
    SEARCH,
    ApproachFSMConfig,
    VisualApproachStateMachine,
)

TAU = 2.0 * math.pi


# ── aim_fail_to_scan ──────────────────────────────────────────────────
def test_aim_done_sweeps_instead_of_escalating():
    """With aim_fail_to_scan on, a failed aim rotates to look, never flies at the object."""
    fsm = VisualApproachStateMachine(ApproachFSMConfig(aim_fail_to_scan=True))
    fsm.update(confirmed=False, track_valid=False, at_target=False, dt=0.1,
               arrived_at_goal=True, aim_ready=True)                # -> AIM
    assert fsm.state == AIM
    dec = fsm.update(confirmed=False, track_valid=False, at_target=False, dt=0.1,
                     arrived_at_goal=True, aim_ready=True, aim_done=True)
    assert dec.mode == SCAN
    assert dec.escalate_goal is False        # never re-targets the object's coordinate


def test_legacy_aim_done_still_escalates_when_off():
    """Default (off) keeps the older staged fallback: escalate the goal to the object."""
    fsm = VisualApproachStateMachine(ApproachFSMConfig(aim_fail_to_scan=False))
    fsm.update(confirmed=False, track_valid=False, at_target=False, dt=0.1,
               arrived_at_goal=True, aim_ready=True)
    dec = fsm.update(confirmed=False, track_valid=False, at_target=False, dt=0.1,
                     arrived_at_goal=True, aim_ready=True, aim_done=True)
    assert dec.mode == SEARCH
    assert dec.escalate_goal is True


# ── scan_land_revolutions ─────────────────────────────────────────────
def _into_scan(fsm):
    """Drive an unstaged FSM into SCAN (arrived, unconfirmed, land_at_goal off)."""
    return fsm.update(confirmed=False, track_valid=False, at_target=False, dt=0.1,
                      arrived_at_goal=True)


def test_scan_lands_after_configured_revolutions():
    fsm = VisualApproachStateMachine(ApproachFSMConfig(scan_land_revolutions=3.0))
    assert _into_scan(fsm).mode == SCAN
    # Just under three full turns: still sweeping.
    dec = fsm.update(confirmed=False, track_valid=False, at_target=False, dt=1.0,
                     arrived_at_goal=True, scan_swept_delta=3 * TAU - 0.01)
    assert dec.mode == SCAN and not dec.land
    # One more nudge tips it over three turns -> give-up LAND.
    dec = fsm.update(confirmed=False, track_valid=False, at_target=False, dt=1.0,
                     arrived_at_goal=True, scan_swept_delta=0.02)
    assert dec.mode == LAND
    assert dec.land is True


def test_scan_revolutions_accumulate_across_ticks():
    fsm = VisualApproachStateMachine(ApproachFSMConfig(scan_land_revolutions=1.0))
    _into_scan(fsm)
    swept = 0.0
    for _ in range(1000):
        dec = fsm.update(confirmed=False, track_valid=False, at_target=False, dt=0.1,
                         arrived_at_goal=True, scan_swept_delta=0.05)
        swept += 0.05
        if dec.mode == LAND:
            break
    assert dec.mode == LAND
    assert swept >= TAU                       # took at least one full turn to give up


def test_scan_cap_disabled_sweeps_forever():
    fsm = VisualApproachStateMachine(ApproachFSMConfig(scan_land_revolutions=0.0))
    _into_scan(fsm)
    dec = None
    for _ in range(100):
        dec = fsm.update(confirmed=False, track_valid=False, at_target=False, dt=1.0,
                         arrived_at_goal=True, scan_swept_delta=TAU)
    assert dec.mode == SCAN                    # never lands with the cap off


def test_reentering_scan_resets_the_revolution_count():
    """A sweep interrupted (goal cleared -> SEARCH) and re-reached counts from zero, so
    a stale angle cannot land the drone early."""
    fsm = VisualApproachStateMachine(ApproachFSMConfig(scan_land_revolutions=2.0))
    _into_scan(fsm)
    fsm.update(confirmed=False, track_valid=False, at_target=False, dt=1.0,
               arrived_at_goal=True, scan_swept_delta=1.9 * TAU)    # almost there
    # Goal cleared -> SEARCH, then re-arrive -> a fresh SCAN.
    assert fsm.update(confirmed=False, track_valid=False, at_target=False, dt=0.1,
                      arrived_at_goal=False).mode == SEARCH
    assert _into_scan(fsm).mode == SCAN
    dec = fsm.update(confirmed=False, track_valid=False, at_target=False, dt=1.0,
                     arrived_at_goal=True, scan_swept_delta=0.5 * TAU)
    assert dec.mode == SCAN                    # 0.5 turns, not 2.4 -> still sweeping


def test_invalid_scan_land_revolutions_raises():
    with pytest.raises(ValueError):
        ApproachFSMConfig(scan_land_revolutions=-1.0)


# ── arm_ready gate ────────────────────────────────────────────────────
def test_not_armed_stays_passive_despite_confirmation():
    """The blind take-over must not engage before the drone is in the room."""
    fsm = VisualApproachStateMachine(ApproachFSMConfig())
    dec = fsm.update(confirmed=True, track_valid=True, at_target=False, dt=0.1,
                     arm_ready=False)
    assert dec.mode == SEARCH
    assert dec.drive_cmd_vel is False


def test_not_armed_ignores_arrival_and_aim():
    fsm = VisualApproachStateMachine(ApproachFSMConfig(aim_fail_to_scan=True))
    dec = fsm.update(confirmed=False, track_valid=False, at_target=False, dt=0.1,
                     arrived_at_goal=True, aim_ready=True, arm_ready=False)
    assert dec.mode == SEARCH


def test_arming_then_confirmation_acquires():
    """Held passive while not armed; the same confirmation acquires once armed."""
    fsm = VisualApproachStateMachine(ApproachFSMConfig(acquire_stop_s=0.0))
    assert fsm.update(confirmed=True, track_valid=True, at_target=False, dt=0.1,
                      arm_ready=False).mode == SEARCH
    assert fsm.update(confirmed=True, track_valid=True, at_target=False, dt=0.1,
                      arm_ready=True).mode == APPROACH


def test_arm_ready_defaults_to_armed():
    """A caller that never gates (the default) is unaffected."""
    fsm = VisualApproachStateMachine(ApproachFSMConfig(acquire_stop_s=0.0))
    assert fsm.update(confirmed=True, track_valid=True, at_target=False,
                      dt=0.1).mode == APPROACH
