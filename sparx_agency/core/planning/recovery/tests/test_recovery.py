"""Unit tests for the controller-agnostic recovery loop.

Covers the three primitives (:class:`StuckDetector`, :class:`EscapeManeuver`,
:class:`RecoverySupervisor`) and one end-to-end episode: a drone driven into a
wall it cannot see must be noticed, backed out, and reported once for a replan
from the recovered pose.

Run from the repo root by path:
    pytest sparx_agency/core/planning/recovery
"""
import math

import pytest

from sparx_agency.core.common.types import Pose2D, normalize_angle
from sparx_agency.core.planning.recovery import (
    AXIS_FORWARD,
    AXIS_YAW,
    EscapeManeuver,
    EscapeParams,
    EscapeState,
    RecoveryParams,
    RecoverySupervisor,
    StuckDetector,
    StuckParams,
)

DT = 0.05


# ─────────────────────────── helpers ────────────────────────────
def _advance(pose, vx, vy, wz, response, dt=DT):
    """Move ``pose`` by ``response`` * the body-frame command (REP-103)."""
    r = response(vx, vy, wz)
    nx = pose.x + (vx * math.cos(pose.yaw) - vy * math.sin(pose.yaw)) * r * dt
    ny = pose.y + (vx * math.sin(pose.yaw) + vy * math.cos(pose.yaw)) * r * dt
    nyaw = normalize_angle(pose.yaw + wz * r * dt)
    return Pose2D(nx, ny, nyaw)


def _run_detector(det, cmd_vx, cmd_wz, response, ticks, dt=DT, trustworthy=True):
    """Drive ``det`` with a fixed command and a world ``response`` factor."""
    pose = Pose2D(0.0, 0.0, 0.0)
    prev_vx, prev_wz = 0.0, 0.0
    out = []
    for _ in range(ticks):
        out.append(det.update(pose, prev_vx, prev_wz, dt, trustworthy))
        pose = _advance(pose, cmd_vx, 0.0, cmd_wz, response, dt)
        prev_vx, prev_wz = cmd_vx, cmd_wz
    return out


def _run_supervisor(sup, ctrl_vx, ctrl_wz, response, ticks, dt=DT,
                    frozen=False, trustworthy=True):
    """Drive ``sup`` wrapping a controller that always asks for ``(ctrl_vx, ctrl_wz)``.

    Returns ``(decisions, final_pose)``. The supervisor's escape overrides the
    controller command when it owns the tick.
    """
    pose = Pose2D(0.0, 0.0, 0.0)
    prev_vx, prev_wz = 0.0, 0.0
    decisions = []
    for _ in range(ticks):
        d = sup.update(pose, prev_vx, prev_wz, dt, pose_trustworthy=trustworthy,
                       frozen=frozen)
        decisions.append(d)
        if d.override:
            vx, vy, wz = d.vx, d.vy, d.wz
        else:
            vx, vy, wz = ctrl_vx, 0.0, ctrl_wz
        pose = _advance(pose, vx, vy, wz, response, dt)
        prev_vx, prev_wz = vx, wz
    return decisions, pose


PINNED = lambda vx, vy, wz: 0.0                       # nothing the drone does moves it
OBEYS = lambda vx, vy, wz: 1.0                        # the world honours every command
# A wall dead ahead: forward is blocked, reverse and sideways are free.
WALL_AHEAD = lambda vx, vy, wz: 0.0 if vx > 0.01 else 1.0


# ─────────────────────────── StuckDetector ──────────────────────
def test_detector_confirms_forward_block_when_pinned():
    det = StuckDetector()
    out = _run_detector(det, 0.3, 0.0, PINNED, ticks=30)
    assert any(v.stuck for v in out)
    final = out[-1]
    assert final.axis == AXIS_FORWARD
    assert final.sign == 1


def test_detector_clears_once_the_drone_moves():
    det = StuckDetector()
    _run_detector(det, 0.3, 0.0, PINNED, ticks=30)
    assert det.verdict.stuck
    # Now the world starts honouring the command: progress resumes, block clears.
    out = _run_detector(det, 0.3, 0.0, OBEYS, ticks=15)
    assert not out[-1].stuck


def test_detector_silent_when_not_commanding():
    det = StuckDetector()
    # Below the min-command floors -> never counted as "trying", never confirms.
    out = _run_detector(det, 0.01, 0.0, PINNED, ticks=40)
    assert not any(v.stuck for v in out)


def test_detector_ignores_untrustworthy_pose():
    det = StuckDetector()
    out = _run_detector(det, 0.3, 0.0, PINNED, ticks=40, trustworthy=False)
    assert not any(v.stuck for v in out)


def test_detector_confirms_yaw_block():
    det = StuckDetector()
    out = _run_detector(det, 0.0, 0.5, PINNED, ticks=30)
    assert out[-1].axis == AXIS_YAW


def test_detector_stale_clears_a_standing_block():
    det = StuckDetector(StuckParams(stale_clear_s=0.5))
    _run_detector(det, 0.3, 0.0, PINNED, ticks=30)
    assert det.verdict.stuck
    # Stop pushing the axis entirely; after stale_clear_s the claim is dropped.
    out = _run_detector(det, 0.0, 0.0, PINNED, ticks=20)
    assert not out[-1].stuck


def test_detector_disabled_never_fires():
    det = StuckDetector(StuckParams(enabled=False))
    out = _run_detector(det, 0.3, 0.0, PINNED, ticks=40)
    assert not any(v.stuck for v in out)


def test_detector_rejects_non_positive_dt():
    det = StuckDetector()
    with pytest.raises(ValueError):
        det.update(Pose2D(0, 0, 0), 0.2, 0.0, 0.0)


def test_stuck_params_validation():
    with pytest.raises(ValueError):
        StuckParams(progress_frac=1.5)
    with pytest.raises(ValueError):
        StuckParams(window_s=0.0)
    with pytest.raises(ValueError):
        StuckParams(confirm_ticks=0)


# ─────────────────────────── EscapeManeuver ─────────────────────
def _drive_escape(esc, ticks, dt=DT):
    cmds = []
    for _ in range(ticks):
        cmds.append(esc.step(dt))
    return cmds


def test_escape_runs_brake_back_probe_settle():
    esc = EscapeManeuver(EscapeParams(brake_s=0.1, back_s=0.1, probe_s=0.1,
                                      settle_s=0.1))
    started = esc.trigger(_verdict_forward())
    assert started
    states = {c.state for c in _drive_escape(esc, 20)}
    assert EscapeState.BRAKE in states
    assert EscapeState.BACK in states
    assert EscapeState.PROBE in states
    assert EscapeState.SETTLE in states
    assert not esc.active           # finished within 20 ticks


def test_escape_back_phase_reverses():
    esc = EscapeManeuver(EscapeParams(brake_s=0.05, back_s=0.3, back_speed=0.12,
                                      probe_s=0.05, settle_s=0.05))
    esc.trigger(_verdict_forward())
    cmds = _drive_escape(esc, 20)
    backs = [c for c in cmds if c.state == EscapeState.BACK]
    assert backs and all(c.vx < 0.0 for c in backs)
    assert all(c.wz == 0.0 for c in cmds)          # an escape never rotates


def test_escape_skips_probe_without_lateral():
    esc = EscapeManeuver(EscapeParams(brake_s=0.05, back_s=0.05, probe_s=0.2,
                                      settle_s=0.05, allow_lateral=False))
    esc.trigger(_verdict_forward())
    states = {c.state for c in _drive_escape(esc, 20)}
    assert EscapeState.PROBE not in states
    assert EscapeState.BACK in states


def test_escape_exhausts_after_max_attempts():
    esc = EscapeManeuver(EscapeParams(brake_s=0.02, back_s=0.02, probe_s=0.02,
                                      settle_s=0.02, max_attempts=2))
    assert esc.trigger(_verdict_forward())
    _drive_escape(esc, 10)
    assert esc.trigger(_verdict_forward())          # second attempt allowed
    _drive_escape(esc, 10)
    assert esc.exhausted
    assert not esc.trigger(_verdict_forward())       # third refused


def test_escape_flips_probe_side_on_second_attempt():
    esc = EscapeManeuver(EscapeParams(brake_s=0.02, back_s=0.02, probe_s=0.1,
                                      settle_s=0.02, max_attempts=2))
    esc.trigger(_verdict_forward(), prefer_left=True)
    first = [c.vy for c in _drive_escape(esc, 10) if c.state == EscapeState.PROBE]
    esc.trigger(_verdict_forward(), prefer_left=True)
    second = [c.vy for c in _drive_escape(esc, 10) if c.state == EscapeState.PROBE]
    assert first and second
    assert first[0] * second[0] < 0.0               # opposite directions


def test_escape_abort_stops_immediately():
    esc = EscapeManeuver()
    esc.trigger(_verdict_forward())
    esc.step(DT)
    esc.abort()
    assert not esc.active
    assert not esc.step(DT).active


def _verdict_forward():
    from sparx_agency.core.planning.recovery import StuckVerdict
    return StuckVerdict(AXIS_FORWARD, 1, 0.0)


# ─────────────────────────── RecoverySupervisor ─────────────────
def test_supervisor_transparent_when_disabled():
    sup = RecoverySupervisor(RecoveryParams(enabled=False))
    decisions, _ = _run_supervisor(sup, 0.3, 0.0, PINNED, ticks=80)
    assert not any(d.override for d in decisions)
    assert not any(d.request_replan for d in decisions)


def test_supervisor_nominal_when_moving():
    sup = RecoverySupervisor()
    decisions, _ = _run_supervisor(sup, 0.3, 0.0, OBEYS, ticks=80)
    assert not any(d.override for d in decisions)
    assert not any(d.request_replan for d in decisions)


def test_supervisor_pinned_escapes_then_reports_once():
    sup = RecoverySupervisor()
    decisions, _ = _run_supervisor(sup, 0.3, 0.0, PINNED, ticks=120)
    assert any(d.override for d in decisions)                 # ran a back-out
    replans = [d for d in decisions if d.request_replan]
    assert len(replans) == 1                                  # reported exactly once
    assert replans[0].stuck_axis == AXIS_FORWARD
    assert not replans[0].override                            # never both at once


def test_supervisor_backs_the_drone_out_of_the_wall():
    sup = RecoverySupervisor()
    decisions, final = _run_supervisor(sup, 0.3, 0.0, WALL_AHEAD, ticks=120)
    assert any(d.override for d in decisions)
    assert any(d.request_replan for d in decisions)
    # The back-out phase reversed the drone: it ends up behind where it stuck.
    assert final.x < 0.05


def test_supervisor_frozen_never_escapes_or_reports():
    sup = RecoverySupervisor()
    decisions, _ = _run_supervisor(sup, 0.3, 0.0, PINNED, ticks=120, frozen=True)
    assert not any(d.override for d in decisions)
    assert not any(d.request_replan for d in decisions)


def test_supervisor_freeze_mid_escape_aborts_without_reporting():
    sup = RecoverySupervisor()
    pose = Pose2D(0.0, 0.0, 0.0)
    prev_vx = prev_wz = 0.0
    saw_escape = False
    saw_replan = False
    for i in range(120):
        # Freeze the moment the escape takes over, to prove an aborted manoeuvre
        # is not mistaken for a completed recovery.
        frozen = saw_escape
        d = sup.update(pose, prev_vx, prev_wz, DT, frozen=frozen)
        if d.state == "ESCAPE":
            saw_escape = True
        if d.request_replan:
            saw_replan = True
        vx, vy, wz = (d.vx, d.vy, d.wz) if d.override else (0.3, 0.0, 0.0)
        pose = _advance(pose, vx, vy, wz, PINNED)
        prev_vx, prev_wz = vx, wz
    assert saw_escape
    assert not saw_replan


def test_supervisor_reports_repeated_sticks_as_separate_episodes():
    """A drone that keeps clipping the same spot escalates once per episode."""
    sup = RecoverySupervisor()
    # Pin, let it escape+report, then let it move (clears), then pin again.
    d1, _ = _run_supervisor(sup, 0.3, 0.0, PINNED, ticks=120)
    d2, _ = _run_supervisor(sup, 0.3, 0.0, OBEYS, ticks=20)      # gets free
    d3, _ = _run_supervisor(sup, 0.3, 0.0, PINNED, ticks=120)    # sticks again
    assert sum(d.request_replan for d in d1) == 1
    assert sum(d.request_replan for d in d3) == 1

