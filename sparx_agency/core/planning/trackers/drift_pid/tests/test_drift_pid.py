"""Tests for the drift-PID controller.

The interesting ones are at the bottom: a closed-loop simulation of a drone
being pushed sideways, which is the whole reason this controller exists. The
rest pin the pieces it is built from.
"""
from __future__ import annotations

from math import cos, radians, sin

import pytest

from sparx_agency.core.common.types import Pose2D
from sparx_agency.core.planning.trackers.drift_pid import (
    AXIS_FORWARD,
    AXIS_YAW,
    AxisPid,
    BlockageMonitor,
    BlockageParams,
    ConfidenceParams,
    ConfidenceScheduler,
    DriftPidFollower,
    DriftPidParams,
    DriftPidState,
    EnvelopeParams,
    EscapeManeuver,
    EscapeParams,
    EscapeState,
    ForceEnvelope,
    LocalizationQuality,
    PidGains,
)

DT = 0.1  # 10 Hz control loop, matching ctrl_rate_hz on the drone


def _good(conf=0.5, eff=1.0):
    """A healthy localization snapshot."""
    return LocalizationQuality(confidence=conf, pos_std_m=0.02, age_s=0.05,
                               coasting=False, cmd_effectiveness=eff, valid=True)


# ── PID ──────────────────────────────────────────────────────────
def test_integral_learns_a_standing_bias():
    """The I term converges on the output that cancels a constant error."""
    pid = AxisPid(PidGains(kp=0.0, ki=0.5, kd=0.0, i_limit=1.0, out_limit=1.0))
    for _ in range(200):
        out = pid.update(0.2, DT)
    assert out == pytest.approx(pid.drift, abs=1e-9)
    assert pid.drift > 0.05


def test_frozen_integrator_holds_rather_than_decays():
    """A frozen integrator keeps what it learned -- that is the point of freezing."""
    pid = AxisPid(PidGains(ki=0.5, i_limit=1.0, out_limit=1.0))
    for _ in range(50):
        pid.update(0.2, DT)
    learned = pid.drift
    for _ in range(50):
        pid.update(0.9, DT, integrate=False)
    assert pid.drift == pytest.approx(learned, abs=1e-9)


def test_deadband_stops_the_integrator_learning_from_jitter():
    """Error inside the deadband must not accumulate into a phantom drift."""
    pid = AxisPid(PidGains(ki=1.0, i_limit=1.0, deadband=0.05, out_limit=1.0))
    for i in range(200):
        pid.update(0.04 if i % 2 else -0.04, DT)
    assert pid.drift == pytest.approx(0.0, abs=1e-9)


def test_integral_does_not_wind_up_past_its_limit():
    """A long saturation must not hide an integral that unwinds later."""
    pid = AxisPid(PidGains(kp=0.1, ki=1.0, kd=0.0, i_limit=0.2, out_limit=0.5))
    for _ in range(500):
        pid.update(5.0, DT)
    assert pid.drift <= 0.2 + 1e-9
    # Reverse the error: it must respond within a couple of ticks, not after
    # unwinding a huge hidden accumulator.
    for _ in range(20):
        out = pid.update(-5.0, DT)
    assert out < 0.0


def test_gain_scale_spares_the_integral():
    """Backing off P/D on a poor pose must not throw away the learned drift."""
    pid = AxisPid(PidGains(kp=1.0, ki=0.5, i_limit=1.0, out_limit=2.0))
    for _ in range(100):
        pid.update(0.2, DT)
    assert pid.drift > 0.0
    out = pid.update(0.2, DT, gain_scale=0.0)
    # P and D are scaled away, so what is left IS the learned drift.
    assert out == pytest.approx(pid.drift, abs=1e-9)


# ── Force envelope ───────────────────────────────────────────────
def test_combined_budget_scales_every_axis_together():
    """Multi-axis demand is cut proportionally, so the command keeps its shape."""
    env = ForceEnvelope(EnvelopeParams(
        max_vx=0.3, max_vy=0.3, max_wz=0.3, max_translation=1.0,
        combined_effort=1.0, accel_xy=100.0, decel_xy=100.0,
        accel_wz=100.0, decel_wz=100.0,
        min_vx=0.01, min_vy=0.01, min_wz=0.01))
    vx, vy, wz = env.apply(0.3, 0.3, 0.3, DT)
    assert env.effort(vx, vy, wz) == pytest.approx(1.0, abs=1e-6)
    assert vx == pytest.approx(vy, abs=1e-9)
    assert vy == pytest.approx(wz, abs=1e-9)


def test_slew_limits_how_fast_a_command_may_change():
    """A step demand becomes a ramp -- what makes corrections prolonged."""
    env = ForceEnvelope(EnvelopeParams(accel_xy=0.35, min_vx=0.001))
    vx, _, _ = env.apply(0.25, 0.0, 0.0, DT)
    assert vx == pytest.approx(0.035, abs=1e-9)


def test_minimum_force_snaps_up_or_drops_to_zero():
    """Below the motor floor a command is either committed or abandoned."""
    env = ForceEnvelope(EnvelopeParams(min_vx=0.06, release_frac=0.5,
                                       accel_xy=100.0, decel_xy=100.0))
    assert env.apply(0.04, 0.0, 0.0, DT)[0] == pytest.approx(0.06)
    env.reset()
    assert env.apply(0.01, 0.0, 0.0, DT)[0] == 0.0


def test_speed_scale_shrinks_the_caps_not_the_floors():
    """A low-confidence pose flies the same command, slower -- but still flyably."""
    env = ForceEnvelope(EnvelopeParams(max_vx=0.3, max_translation=0.3,
                                       min_vx=0.06, accel_xy=100.0,
                                       decel_xy=100.0))
    vx, _, _ = env.apply(0.3, 0.0, 0.0, DT, speed_scale=0.5)
    assert vx == pytest.approx(0.15, abs=1e-9)


def test_unreachable_per_axis_cap_is_rejected():
    """A max_translation below max_vx would make the forward dial a no-op."""
    with pytest.raises(ValueError, match="max_translation"):
        EnvelopeParams(max_vx=0.4, max_translation=0.2)


def test_reverse_is_capped_harder_than_forward():
    """Backward flight is blind; it must never be the fast direction."""
    env = ForceEnvelope(EnvelopeParams(max_vx=0.30, max_vx_back=0.10,
                                       max_translation=0.30,
                                       accel_xy=100.0, decel_xy=100.0))
    assert env.apply(0.5, 0.0, 0.0, DT)[0] == pytest.approx(0.30)
    env.reset()
    assert env.apply(-0.5, 0.0, 0.0, DT)[0] == pytest.approx(-0.10)
    with pytest.raises(ValueError, match="max_vx_back"):
        EnvelopeParams(max_vx=0.2, max_vx_back=0.3)


def test_braking_is_faster_than_accelerating():
    """Taking thrust off must not be rate-limited by the ramp-up's gentleness."""
    env = ForceEnvelope(EnvelopeParams(accel_xy=0.35, decel_xy=0.70,
                                       min_vx=0.001))
    up = env.apply(0.25, 0.0, 0.0, DT)[0]
    assert up == pytest.approx(0.035, abs=1e-9)         # one accel step
    for _ in range(20):
        env.apply(0.25, 0.0, 0.0, DT)                    # reach cruise
    down = env.brake(DT)[0]
    assert down == pytest.approx(0.25 - 0.070, abs=1e-9)  # one decel step
    with pytest.raises(ValueError, match="decel"):
        EnvelopeParams(accel_xy=0.5, decel_xy=0.3)


def test_stale_ticks_do_not_double_the_derivative():
    """A held pose repeated for a tick must not turn into a derivative spike.

    The error steps 0 -> 0.1 across TWO ticks (one stale, one fresh). Honest
    rate: 0.1 / (2*DT). A naive implementation differentiates over one DT and
    reports double.
    """
    honest = AxisPid(PidGains(kd=1.0, d_tau_s=0.0, out_limit=10.0))
    honest.update(0.0, DT, fresh=True)
    honest.update(0.0, DT, fresh=False)          # stale: pose stream gapped
    out = honest.update(0.1, DT, fresh=True)
    assert out == pytest.approx(0.1 / (2 * DT), abs=1e-9)

    naive = AxisPid(PidGains(kd=1.0, d_tau_s=0.0, out_limit=10.0))
    naive.update(0.0, DT, fresh=True)
    naive.update(0.0, DT, fresh=True)            # mislabeled as fresh
    assert naive.update(0.1, DT, fresh=True) == pytest.approx(0.1 / DT, abs=1e-9)


def test_wider_deadband_for_this_tick_suppresses_the_error():
    """deadband_extra widens the band: a within-noise error is not corrected."""
    pid = AxisPid(PidGains(kp=1.0, ki=1.0, i_limit=1.0, deadband=0.03,
                           out_limit=1.0))
    assert pid.update(0.08, DT, deadband_extra=0.10) == pytest.approx(0.0)
    assert pid.drift == pytest.approx(0.0)       # and nothing was learned
    assert pid.update(0.08, DT, deadband_extra=0.0) > 0.0


# ── Confidence scheduling ────────────────────────────────────────
def test_stale_pose_holds_the_drone():
    sched = ConfidenceScheduler(ConfidenceParams(max_age_s=0.6))
    auth = sched.evaluate(LocalizationQuality(confidence=0.5, age_s=1.0,
                                              valid=True))
    assert auth.hold and auth.speed_scale == 0.0


def test_coasting_slows_down_and_freezes_learning():
    """A dead-reckoned pose may still be flown on briefly, but never learned from."""
    sched = ConfidenceScheduler(ConfidenceParams())
    auth = sched.evaluate(LocalizationQuality(confidence=0.2, age_s=0.05,
                                              coasting=True, valid=True))
    assert not auth.hold
    assert not auth.integrate
    assert auth.speed_scale < 0.6


def test_lone_tag_confidence_still_flies():
    """A one-tag fix caps near 0.21; the schedule must not treat that as failure."""
    auth = ConfidenceScheduler(ConfidenceParams()).evaluate(_good(conf=0.21))
    assert not auth.hold and auth.speed_scale > 0.5


def test_a_vague_pose_widens_the_deadbands():
    """pos_std above the crisp reference opens the tracking deadband honestly."""
    sched = ConfidenceScheduler(ConfidenceParams(std_ref_m=0.05,
                                                 std_deadband_gain=0.6,
                                                 deadband_extra_max_m=0.15))
    crisp = sched.evaluate(LocalizationQuality(confidence=0.5, pos_std_m=0.02,
                                               age_s=0.05, valid=True))
    assert crisp.deadband_extra_m == pytest.approx(0.0)
    vague = sched.evaluate(LocalizationQuality(confidence=0.5, pos_std_m=0.25,
                                               age_s=0.05, valid=True))
    assert vague.deadband_extra_m == pytest.approx(0.6 * 0.20, abs=1e-9)
    coasting = sched.evaluate(LocalizationQuality(confidence=0.15, pos_std_m=0.5,
                                                  age_s=0.05, coasting=True,
                                                  valid=True))
    assert coasting.deadband_extra_m == pytest.approx(0.15)   # capped
    # Low confidence widens the yaw deadband via the provider's yaw-std law
    # (0.02 + 0.20*(1-conf)^2): barely at mid confidence, decisively when low.
    assert vague.yaw_deadband_extra_rad == pytest.approx(0.012, abs=1e-3)
    low = sched.evaluate(LocalizationQuality(confidence=0.12, pos_std_m=0.25,
                                             age_s=0.05, valid=True))
    assert low.yaw_deadband_extra_rad > 0.05


def test_unproven_commands_earn_their_speed():
    """A fresh flight starts near the earned-speed floor and speeds up as the
    effectiveness EMA learns that commands really move the drone."""
    sched = ConfidenceScheduler(ConfidenceParams(eff_speed_floor=0.5))
    unproven = sched.evaluate(_good(conf=0.5, eff=0.15))
    proven = sched.evaluate(_good(conf=0.5, eff=0.9))
    assert unproven.speed_scale == pytest.approx(0.5)
    assert proven.speed_scale == pytest.approx(1.0)
    assert not unproven.hold                       # throttled, never grounded


def test_latency_lead_is_earned_and_never_applied_while_coasting():
    sched = ConfidenceScheduler(ConfidenceParams(latency_s=0.12))
    assert sched.evaluate(_good(eff=1.0)).lead_s == pytest.approx(0.12)
    assert sched.evaluate(_good(eff=0.10)).lead_s == pytest.approx(0.0)
    coasting = LocalizationQuality(confidence=0.2, age_s=0.05, coasting=True,
                                   cmd_effectiveness=1.0, valid=True)
    # A coasted pose is ALREADY propagated by the commands: leading it would
    # count the same commands twice.
    assert sched.evaluate(coasting).lead_s == pytest.approx(0.0)


def test_yaw_scale_floor_keeps_the_turn_at_full_authority_on_a_vague_pose():
    """A vague pose slows the flying, never the turning: turning in place is
    what sweeps tags back into view, and it carries no position risk."""
    sched = ConfidenceScheduler(ConfidenceParams(yaw_scale_floor=1.0))
    vague = _good(conf=0.15)                   # between conf_min and conf_full
    auth = sched.evaluate(vague)
    assert auth.speed_scale < 1.0              # translation slowed
    assert auth.yaw_speed_scale == pytest.approx(1.0)   # yaw untouched
    held = sched.evaluate(LocalizationQuality(confidence=0.01, age_s=0.05,
                                              valid=True))
    assert held.hold and held.yaw_speed_scale == 0.0    # a hold still holds


def test_envelope_scales_yaw_separately_from_translation():
    env = ForceEnvelope(EnvelopeParams(max_vx=0.4, max_wz=0.6,
                                       max_translation=0.4,
                                       combined_effort=5.0,
                                       accel_xy=100.0, accel_wz=100.0,
                                       decel_xy=100.0, decel_wz=100.0))
    vx, _, wz = env.apply(1.0, 0.0, 1.0, DT, speed_scale=0.5,
                          yaw_speed_scale=1.0)
    assert vx == pytest.approx(0.2)            # halved by speed_scale
    assert wz == pytest.approx(0.6)            # yaw keeps its full cap


def test_turn_rides_a_pitch_bias_so_the_yaw_bites():
    """A pure in-place yaw coasts flat on this airframe; the bias converts the
    turn into the measured 3-6x better turning-while-translating regime."""
    p = DriftPidParams(turn_pitch_bias=0.08,
                       yaw_engage_rad=radians(22.0))
    f = DriftPidFollower(p)
    f.set_quality(_good())
    # Path 90 degrees to the left: heading error far above engage -> TURN.
    f.set_path([Pose2D(0.0, 0.0, 0.0), Pose2D(0.0, 3.0, 0.0)])
    cmd = None
    for _ in range(10):
        cmd = f.step(Pose2D(0.0, 0.0, 0.0), DT)
    assert cmd.state == DriftPidState.TURN
    assert cmd.wz > 0.0                        # turning left
    assert cmd.vx >= p.envelope.min_vx         # and riding the forward bias

    unbiased = DriftPidFollower(DriftPidParams(yaw_engage_rad=radians(22.0)))
    unbiased.set_quality(_good())
    unbiased.set_path([Pose2D(0.0, 0.0, 0.0), Pose2D(0.0, 3.0, 0.0)])
    for _ in range(10):
        cmd = unbiased.step(Pose2D(0.0, 0.0, 0.0), DT)
    assert cmd.state == DriftPidState.TURN
    assert cmd.vx == pytest.approx(0.0)        # default: the pure yaw of old


# ── Blockage detection ───────────────────────────────────────────
def test_wedged_drone_is_detected_on_the_forward_axis():
    mon = BlockageMonitor(BlockageParams(window_s=0.5, confirm_ticks=3))
    pose = Pose2D(0.0, 0.0, 0.0)
    verdict = None
    for _ in range(30):
        verdict = mon.update(pose, 0.2, 0.0, DT, _good())
    assert verdict.axis == AXIS_FORWARD and verdict.sign == 1


def test_a_moving_drone_is_not_reported_blocked():
    mon = BlockageMonitor(BlockageParams(window_s=0.5, confirm_ticks=3))
    x = 0.0
    for _ in range(30):
        x += 0.2 * DT
        verdict = mon.update(Pose2D(x, 0.0, 0.0), 0.2, 0.0, DT, _good())
    assert not verdict.blocked


def test_sideways_drift_is_not_forward_progress():
    """Drifting left while the forward axis is pinned must still read as blocked."""
    mon = BlockageMonitor(BlockageParams(window_s=0.5, confirm_ticks=3))
    y = 0.0
    for _ in range(30):
        y += 0.2 * DT
        verdict = mon.update(Pose2D(0.0, y, 0.0), 0.2, 0.0, DT, _good())
    assert verdict.axis == AXIS_FORWARD


def test_coasted_poses_are_never_used_as_evidence():
    """A coasted pose is propagated BY the command, so it always looks obedient."""
    mon = BlockageMonitor(BlockageParams(window_s=0.5, confirm_ticks=3))
    coasted = LocalizationQuality(confidence=0.15, age_s=0.05, coasting=True,
                                  cmd_effectiveness=0.0, valid=True)
    for _ in range(50):
        verdict = mon.update(Pose2D(0.0, 0.0, 0.0), 0.2, 0.0, DT, coasted)
    assert not verdict.blocked


def test_blocked_turn_is_detected_on_the_yaw_axis():
    mon = BlockageMonitor(BlockageParams(window_s=0.5, confirm_ticks=3))
    for _ in range(30):
        verdict = mon.update(Pose2D(0.0, 0.0, 0.0), 0.0, 0.4, DT, _good())
    assert verdict.axis == AXIS_YAW and verdict.sign == 1


def test_disabled_monitor_never_blocks_and_drops_a_standing_verdict():
    """The kill switch for a platform that under-delivers on every axis: honest
    weakness looks exactly like a wall, so the operator must be able to turn the
    detector off outright -- and flipping it off must also clear a verdict that
    latched before the switch."""
    mon = BlockageMonitor(BlockageParams(window_s=0.5, confirm_ticks=3))
    pose = Pose2D(0.0, 0.0, 0.0)
    for _ in range(30):                       # wedged: forward confirmed
        verdict = mon.update(pose, 0.2, 0.0, DT, _good())
    assert verdict.blocked
    mon.params = BlockageParams(window_s=0.5, confirm_ticks=3, enabled=False)
    verdict = mon.update(pose, 0.2, 0.0, DT, _good())
    assert not verdict.blocked                # standing verdict dropped
    for _ in range(30):                       # and it can never re-confirm
        verdict = mon.update(pose, 0.2, 0.0, DT, _good())
    assert not verdict.blocked


def test_stale_blockage_clears_once_the_drone_stops_pushing_into_it():
    """A phantom block must not latch the drone forever. Once it stops driving
    into the spot -- boxed in, rerouted away, or held -- the verdict goes stale."""
    mon = BlockageMonitor(BlockageParams(window_s=0.5, confirm_ticks=3,
                                         clear_ticks=3, stale_clear_s=1.0))
    pose = Pose2D(0.0, 0.0, 0.0)
    for _ in range(30):                       # wedged: forward confirmed
        verdict = mon.update(pose, 0.2, 0.0, DT, _good())
    assert verdict.axis == AXIS_FORWARD
    for _ in range(int(1.0 / DT) + 2):        # stop commanding forward for > 1 s
        verdict = mon.update(pose, 0.0, 0.0, DT, _good())
    assert not verdict.blocked


def test_a_live_escape_does_not_clear_its_own_blockage():
    """stale_clear_s must outlast an escape's idle phases (~2 s of brake/probe/
    settle), or the detector would drop the blockage the escape is reacting to and
    hand it a fresh, endless set of attempts."""
    mon = BlockageMonitor(BlockageParams(window_s=0.5, confirm_ticks=3,
                                         stale_clear_s=2.0))
    pose = Pose2D(0.0, 0.0, 0.0)
    for _ in range(30):
        verdict = mon.update(pose, 0.2, 0.0, DT, _good())
    assert verdict.axis == AXIS_FORWARD
    for _ in range(15):                       # 1.5 s of no forward command < 2 s
        verdict = mon.update(pose, 0.0, 0.0, DT, _good())
    assert verdict.blocked and verdict.axis == AXIS_FORWARD


# ── Escape reflexes ──────────────────────────────────────────────
def test_forward_escape_brakes_backs_off_then_probes():
    esc = EscapeManeuver(EscapeParams(brake_s=0.2, back_s=0.2, probe_s=0.2,
                                      settle_s=0.2))
    from sparx_agency.core.planning.trackers.drift_pid import Blockage
    assert esc.trigger(Blockage(AXIS_FORWARD, 1, 0.0), prefer_left=True)
    seen = []
    for _ in range(40):
        cmd = esc.step(DT)
        seen.append(cmd.state)
        if not cmd.active:
            break
    assert EscapeState.BRAKE in seen
    assert EscapeState.BACK in seen
    assert EscapeState.PROBE in seen
    assert not esc.active


def test_yaw_escape_rolls_toward_the_attempted_turn_and_never_backs_up():
    """A blocked left turn means the obstruction is on the right: roll left."""
    esc = EscapeManeuver(EscapeParams(brake_s=0.1, yaw_probe_s=0.3, settle_s=0.1))
    from sparx_agency.core.planning.trackers.drift_pid import Blockage
    esc.trigger(Blockage(AXIS_YAW, 1, 0.0))
    lateral = []
    for _ in range(40):
        cmd = esc.step(DT)
        if cmd.state == EscapeState.BACK:
            pytest.fail("a yaw escape must never reverse")
        lateral.append(cmd.vy)
        if not cmd.active:
            break
    assert max(lateral) > 0.0


def test_escape_alternates_sides_then_gives_up():
    esc = EscapeManeuver(EscapeParams(brake_s=0.1, back_s=0.1, probe_s=0.1,
                                      settle_s=0.1, max_attempts=2))
    from sparx_agency.core.planning.trackers.drift_pid import Blockage
    blockage = Blockage(AXIS_FORWARD, 1, 0.0)
    sides = []
    for _ in range(2):
        assert esc.trigger(blockage, prefer_left=True)
        while esc.active:
            cmd = esc.step(DT)
            if cmd.state == EscapeState.PROBE:
                sides.append(cmd.vy)
    assert sides[0] * sides[-1] < 0.0, "the second probe must try the other side"
    assert esc.exhausted
    assert not esc.trigger(blockage)


# ── Closed loop: the reason this controller exists ───────────────
class _Plant:
    """A first-order drone that is being pushed by a constant world-frame drift."""

    def __init__(self, drift_vx=0.0, drift_vy=0.0, drift_wz=0.0, lag=0.45):
        self.pose = Pose2D(0.0, 0.0, 0.0)
        self.drift = (drift_vx, drift_vy, drift_wz)
        self.lag = lag
        self._v = [0.0, 0.0, 0.0]

    def apply(self, vx, vy, wz, dt):
        for i, target in enumerate((vx, vy, wz)):
            self._v[i] += self.lag * (target - self._v[i])
        bx, by, bwz = self._v
        dvx, dvy, dwz = self.drift
        yaw = self.pose.yaw
        # Body velocity into the world, plus the drift (already world-frame).
        wx = bx * cos(yaw) - by * sin(yaw) + dvx
        wy = bx * sin(yaw) + by * cos(yaw) + dvy
        self.pose = Pose2D(self.pose.x + wx * dt, self.pose.y + wy * dt,
                           self.pose.yaw + (bwz + dwz) * dt)
        return self.pose


def _fly(follower, plant, ticks=1200, quality=None):
    """Run the loop, returning the worst cross-track error after settling."""
    worst_late = 0.0
    for i in range(ticks):
        follower.set_quality(quality or _good())
        cmd = follower.step(plant.pose, DT)
        plant.apply(cmd.vx, cmd.vy, cmd.wz, DT)
        if i > 200:
            worst_late = max(worst_late, abs(cmd.telemetry.cross_track_m))
        if cmd.done:
            break
    return worst_late


def test_drone_reaches_the_goal_with_no_drift():
    follower = DriftPidFollower(DriftPidParams())
    plant = _Plant()
    follower.set_path([Pose2D(0.0, 0.0), Pose2D(3.0, 0.0)], plant.pose)
    _fly(follower, plant)
    assert follower.done
    assert plant.pose.x > 2.6


def test_constant_sideways_drift_is_learned_and_cancelled():
    """The headline case: a steady lateral push must not leave a standing offset."""
    follower = DriftPidFollower(DriftPidParams())
    plant = _Plant(drift_vy=-0.035)      # pushed steadily to the drone's right
    follower.set_path([Pose2D(0.0, 0.0), Pose2D(6.0, 0.0)], plant.pose)
    worst = _fly(follower, plant, ticks=2000)
    assert follower._lat.drift > 0.02, "the controller never learned the drift"
    # The same run with ki=0 settles at ~0.085 m, so this threshold genuinely
    # distinguishes "learned the drift" from "P-only pushing back against it".
    # The residual floor is the cross-track deadband (0.03 m) by design.
    assert worst < 0.055, "settled cross-track error is too large: %.3f" % worst


def test_drift_is_still_cancelled_through_measurement_latency():
    """The vision pose arrives a frame late; the command lead absorbs it.

    Same drifting plant, but the follower only ever sees the pose from one
    control tick ago (~100 ms) — the realistic AprilTag pipeline delay. Without
    the latency lead this phase lag costs tracking margin; with it, settled
    error stays at the no-delay level.
    """
    follower = DriftPidFollower(DriftPidParams())
    plant = _Plant(drift_vy=-0.035)
    follower.set_path([Pose2D(0.0, 0.0), Pose2D(6.0, 0.0)], plant.pose)
    delayed = [plant.pose, plant.pose]
    worst = 0.0
    for i in range(2000):
        follower.set_quality(_good())
        cmd = follower.step(delayed[0], DT)      # a tick-old measurement
        plant.apply(cmd.vx, cmd.vy, cmd.wz, DT)
        delayed.append(plant.pose)
        delayed.pop(0)
        if i > 300:
            worst = max(worst, abs(cmd.telemetry.cross_track_m))
        if cmd.done:
            break
    assert follower.done
    assert worst < 0.06, "latency destroyed the tracking: %.3f" % worst


def test_controller_holds_when_localization_dies():
    follower = DriftPidFollower(DriftPidParams())
    plant = _Plant()
    follower.set_path([Pose2D(0.0, 0.0), Pose2D(3.0, 0.0)], plant.pose)
    dead = LocalizationQuality(confidence=0.0, age_s=5.0, valid=False)
    for _ in range(30):
        follower.set_quality(dead)
        cmd = follower.step(plant.pose, DT)
        plant.apply(cmd.vx, cmd.vy, cmd.wz, DT)
    assert (cmd.vx, cmd.vy, cmd.wz) == (0.0, 0.0, 0.0)
    assert cmd.state == DriftPidState.HOLD


def test_a_wedged_drone_escapes_then_reports_when_reflexes_are_spent():
    """Blocked -> reflexes -> still blocked -> tell the planner exactly once."""
    params = DriftPidParams(
        blockage=BlockageParams(window_s=0.5, confirm_ticks=3),
        escape=EscapeParams(brake_s=0.1, back_s=0.2, probe_s=0.2, settle_s=0.1,
                            max_attempts=2))
    follower = DriftPidFollower(params)
    follower.set_path([Pose2D(0.0, 0.0), Pose2D(3.0, 0.0)], Pose2D(0.0, 0.0, 0.0))
    pinned = Pose2D(0.0, 0.0, 0.0)       # never moves, whatever we command
    states, reports = set(), 0
    for _ in range(300):
        follower.set_quality(_good(eff=0.02))
        cmd = follower.step(pinned, DT)
        states.add(cmd.telemetry.escape_state)
        reports += int(cmd.report_blocked)
    assert EscapeState.PROBE in states, "the drone never tried to get free"
    assert reports == 1, "expected exactly one blockage report, got %d" % reports


def test_turn_engages_past_the_deadband_and_releases_with_hysteresis():
    follower = DriftPidFollower(DriftPidParams())
    plant = _Plant()
    plant.pose = Pose2D(0.0, 0.0, radians(90.0))   # facing 90 deg off the leg
    follower.set_path([Pose2D(0.0, 0.0), Pose2D(3.0, 0.0)], plant.pose)
    follower.set_quality(_good())
    cmd = follower.step(plant.pose, DT)
    assert cmd.state == DriftPidState.TURN
    assert cmd.wz < 0.0, "should rotate clockwise to face +x"
    _fly(follower, plant)
    assert follower.done
