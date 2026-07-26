"""Unit tests for the staged approach's AIM state in the visual-approach FSM.

The mission flies to a *staging* vantage point rather than onto the object's
catalogued (imprecise) coordinate. Arriving there must therefore NOT be treated as
arriving at the object: instead of landing or sweeping, the machine enters AIM, and
the node turns the nose onto the object's bearing and looks. Covered here:

  * arrival with ``aim_ready`` enters AIM, and AIM outranks BOTH the arrival-land
    and the room sweep (the two things that must not fire at a staging point);
  * a confirmation during AIM acquires the target — the outcome aiming exists for;
  * ``aim_done`` (looked / timed out, still unseen) returns to SEARCH and raises
    ``escalate_goal`` exactly once, so the node re-targets the object's coordinate;
  * after escalation (``aim_ready`` False, the goal now IS the object) arrival goes
    back to the ordinary land/scan behaviour;
  * a retarget mid-aim (``arrived_at_goal`` cleared) hands back to the planner
    WITHOUT escalating;
  * AIM owns ``/cmd_vel`` (the node drives the turn);
  * with ``aim_ready`` False nothing changes at all (default-off compatibility).

The FSM is pure, so each test drives it with explicit booleans and asserts on the
returned :class:`ApproachDecision`.
"""
from __future__ import annotations

from sparx_agency.core.planning.visual_servo.state_machine import (
    ACQUIRE_STOP,
    AIM,
    APPROACH,
    LAND,
    SCAN,
    SEARCH,
    ApproachFSMConfig,
    VisualApproachStateMachine,
)


# ── helpers ───────────────────────────────────────────────────────────
def _fsm(land_at_goal=True, arrive_land_confirm_ticks=2, acquire_stop_s=0.0):
    """A mission-realistic FSM: arrival-land on, so AIM's precedence is meaningful."""
    return VisualApproachStateMachine(ApproachFSMConfig(
        land_at_goal=land_at_goal,
        arrive_land_confirm_ticks=arrive_land_confirm_ticks,
        acquire_stop_s=acquire_stop_s))


def _tick(fsm, confirmed=False, track_valid=False, at_target=False, dt=0.1,
          arrived_at_goal=True, aim_ready=True, aim_done=False):
    return fsm.update(confirmed=confirmed, track_valid=track_valid,
                      at_target=at_target, dt=dt, arrived_at_goal=arrived_at_goal,
                      aim_ready=aim_ready, aim_done=aim_done)


# ── entering AIM ──────────────────────────────────────────────────────
def test_arrival_at_a_staging_goal_aims_instead_of_landing():
    """The whole point: land_at_goal is ON, yet arriving at a STAGING point aims."""
    fsm = _fsm(land_at_goal=True)
    dec = _tick(fsm)
    assert dec.mode == AIM
    assert not dec.land


def test_aim_outranks_the_room_sweep_too():
    fsm = _fsm(land_at_goal=False)          # arrival would otherwise SCAN
    assert _tick(fsm).mode == AIM


def test_sustained_arrival_never_accumulates_a_land_streak_while_aiming():
    """Arrival-land needs consecutive SEARCH ticks; aiming must starve it entirely,
    however long the drone stands at the staging point."""
    fsm = _fsm(arrive_land_confirm_ticks=2)
    for _ in range(20):
        dec = _tick(fsm)
    assert dec.mode == AIM
    assert not dec.land


def test_aim_owns_cmd_vel():
    """The node drives the turn, so the follower must be handed off (not passive)."""
    assert _tick(_fsm()).drive_cmd_vel is True


# ── leaving AIM ───────────────────────────────────────────────────────
def test_confirmation_during_aim_acquires_the_target():
    """Aiming worked: the detector saw the object down the bearing."""
    fsm = _fsm(acquire_stop_s=1.5)
    _tick(fsm)                                          # -> AIM
    dec = _tick(fsm, confirmed=True, track_valid=True)
    assert dec.mode == ACQUIRE_STOP
    assert not dec.escalate_goal


def test_confirmation_during_aim_can_go_straight_to_approach():
    fsm = _fsm(acquire_stop_s=0.0)                      # no settle configured
    _tick(fsm)
    assert _tick(fsm, confirmed=True, track_valid=True).mode == APPROACH


def test_aim_done_returns_to_search_and_escalates_the_goal():
    fsm = _fsm()
    _tick(fsm)                                          # -> AIM
    dec = _tick(fsm, aim_done=True)
    assert dec.mode == SEARCH
    assert dec.escalate_goal is True


def test_escalation_is_raised_once_not_every_tick():
    """The node acts on the edge (it publishes a goal), so a repeat would re-publish
    and, worse, re-enter AIM forever."""
    fsm = _fsm()
    _tick(fsm)
    assert _tick(fsm, aim_done=True).escalate_goal is True
    # The node clears aim_ready when it escalates; the FSM must not re-raise anyway.
    assert _tick(fsm, aim_ready=False, aim_done=True).escalate_goal is False


def test_retarget_mid_aim_hands_back_without_escalating():
    """A new goal under the drone (a live retarget) is not an aim failure."""
    fsm = _fsm()
    _tick(fsm)                                          # -> AIM
    dec = _tick(fsm, arrived_at_goal=False)
    assert dec.mode == SEARCH
    assert dec.escalate_goal is False


# ── after escalation: the ordinary behaviour resumes ──────────────────
def test_after_escalation_arrival_lands_as_before():
    """Once the goal IS the object (aim_ready False), arriving there is the legacy
    reached-by-A*-alone land."""
    fsm = _fsm(arrive_land_confirm_ticks=2)
    _tick(fsm)                                          # -> AIM
    _tick(fsm, aim_done=True)                           # -> SEARCH + escalate
    _tick(fsm, aim_ready=False)                         # streak 1
    dec = _tick(fsm, aim_ready=False)                   # streak 2 -> LAND
    assert dec.mode == LAND
    assert dec.land is True


def test_after_escalation_arrival_scans_when_land_at_goal_is_off():
    fsm = _fsm(land_at_goal=False)
    _tick(fsm)
    _tick(fsm, aim_done=True)
    assert _tick(fsm, aim_ready=False).mode == SCAN


# ── default-off compatibility ─────────────────────────────────────────
def test_without_aim_ready_the_machine_is_unchanged():
    fsm = _fsm(arrive_land_confirm_ticks=2)
    assert _tick(fsm, aim_ready=False).mode == SEARCH    # streak 1, still searching
    assert _tick(fsm, aim_ready=False).mode == LAND      # streak 2 -> the old path


def test_aim_is_never_entered_before_arrival():
    """aim_ready alone means nothing: the drone must actually be AT the staging
    point, or it would stop and turn in the middle of the route."""
    fsm = _fsm()
    assert _tick(fsm, arrived_at_goal=False).mode == SEARCH


def test_a_confirmed_target_on_arrival_skips_aiming_entirely():
    """Already seeing it: acquire straight away rather than turn away from it."""
    fsm = _fsm(acquire_stop_s=0.0)
    assert _tick(fsm, confirmed=True, track_valid=True).mode == APPROACH
