"""Tests for the windowed pose+velocity estimator (ROS-free).

Run:
    .venv/bin/python -m pytest \
        sparx_agency/core/localization/tests/test_pose_estimator.py
"""
from __future__ import annotations

import math

from sparx_agency.core.common.types import circular_mean
from sparx_agency.core.localization.pose_estimator import (
    PoseEstimatorParams,
    WindowedPoseEstimator,
)

DT = 0.1   # 10 Hz localization


def _feed(est, samples, t0=0.0, dt=DT):
    """Feed (x, y, yaw) samples at dt spacing; return the timestamp of the last."""
    t = t0
    for (x, y, yaw) in samples:
        est.add_measurement(x, y, yaw, t)
        t += dt
    return t - dt


# --------------------------------------------------------------------------
# stopped: hover-drift rejection
# --------------------------------------------------------------------------
def test_stopped_rejects_hover_drift():
    est = WindowedPoseEstimator()
    est.set_command(0.0, 0.0)
    xs = [1.00, 1.04, 0.97, 1.03, 0.96, 1.02]      # ±4 cm wander
    t_last = _feed(est, [(x, -0.5 + 0.03 * ((i % 2) * 2 - 1), 0.01 * ((i % 2) * 2 - 1))
                         for i, x in enumerate(xs)])
    e = est.estimate(t_last)
    assert e.mode == "stopped" and e.stopped
    assert abs(e.x - sum(xs) / len(xs)) < 1e-6       # the centroid, not chased
    assert e.vx == 0.0 and e.wz == 0.0               # wander is not phantom motion


def test_stopped_single_jump_attenuated():
    est = WindowedPoseEstimator()
    est.set_command(0.0, 0.0)
    base = [(1.0, 0.0, 0.0)] * 5 + [(1.06, 0.0, 0.0)]   # one small (6 cm) outlier
    t_last = _feed(est, base)
    e = est.estimate(t_last)
    # Within the settle_vx tolerance so it stays STOPPED, and the jump is averaged
    # into the centroid: 5*1.0 + 1.06 = 6.06/6 = 1.01 -> moved by only ~1/N.
    assert e.stopped
    assert abs(e.x - 1.01) < 1e-6
    assert e.vx == 0.0 and e.wz == 0.0


# --------------------------------------------------------------------------
# turning: true rate recovery + feed-forward
# --------------------------------------------------------------------------
def test_turning_recovers_true_rate():
    est = WindowedPoseEstimator()
    est.set_command(0.0, 0.7)
    noise = [0.0, 0.06, -0.05, 0.04, -0.03, 0.05]
    t = 0.0
    samples = []
    for k in range(6):
        samples.append((0.0, 0.0, 0.7 * t + noise[k]))
        t += DT
    t_last = _feed(est, samples)
    e = est.estimate(t_last)
    assert e.mode == "turning"
    assert abs(e.wz - 0.7) < 0.15                    # true rate recovered
    assert abs(e.wz_deg_s() - 40.1) < 9.0            # ~40 deg/s


def test_coast_reports_measured_slope():
    # Command just went to zero but the heading is still rising (yaw coast). The
    # settle_wz guard keeps it OUT of 'stopped', and with FF=min it reports ~the
    # measured slope so the controller can see the real coasting rate.
    est = WindowedPoseEstimator()
    est.set_command(0.0, 0.0)
    samples = [(0.0, 0.0, 0.7 * (k * DT)) for k in range(6)]
    t_last = _feed(est, samples)
    e = est.estimate(t_last)
    assert e.mode == "coast" and not e.stopped
    assert e.wz > 0.4                                # reports the real coast, not 0


def test_forward_coast_not_frozen_as_stopped():
    # Stop commanded, but the drone still glides forward (linear inertia). The
    # symmetric vx settle guard must keep it OUT of 'stopped' so the position
    # estimate is not frozen behind the true position.
    est = WindowedPoseEstimator()
    est.set_command(0.0, 0.0)
    samples = [(0.3 * (k * DT), 0.0, 0.0) for k in range(6)]   # gliding +x at 0.3 m/s
    t_last = _feed(est, samples)
    e = est.estimate(t_last)
    assert not e.stopped and e.mode != "stopped"
    assert e.vx > 0.1                                  # residual forward speed reported


def test_blend_monotonic_in_command():
    # Same rising-yaw data, larger |wz_cmd| -> wz_est leans closer to the command.
    samples = [(0.0, 0.0, 0.5 * (k * DT)) for k in range(6)]
    est_lo = WindowedPoseEstimator(); est_lo.set_command(0.0, 0.2)
    est_hi = WindowedPoseEstimator(); est_hi.set_command(0.0, 0.7)
    t_lo = _feed(est_lo, samples); t_hi = _feed(est_hi, samples)
    e_lo = est_lo.estimate(t_lo); e_hi = est_hi.estimate(t_hi)
    # higher command pulls wz_est further above the ~0.5 measured slope
    assert e_hi.wz > e_lo.wz


# --------------------------------------------------------------------------
# forward: speed + lateral drift rejection
# --------------------------------------------------------------------------
def test_forward_speed_and_lateral_rejection():
    est = WindowedPoseEstimator()
    est.set_command(0.3, 0.0)
    lat = [0.0, 0.02, -0.02, 0.02, -0.02, 0.0]       # perpendicular wander
    samples = [(0.3 * (k * DT), lat[k], 0.0) for k in range(6)]   # heading +x
    t_last = _feed(est, samples)
    e = est.estimate(t_last)
    assert e.mode == "forward"
    assert abs(e.vx - 0.3) < 0.12                    # forward speed recovered
    assert abs(e.y) < 0.05                           # lateral wander not propagated


# --------------------------------------------------------------------------
# yaw wrap / dropout / degenerate
# --------------------------------------------------------------------------
def test_yaw_unwrap_across_pi():
    est = WindowedPoseEstimator()
    est.set_command(0.0, 0.0)
    near = math.pi - 0.05
    samples = [(0.0, 0.0, near + 0.02 * ((k % 2) * 2 - 1)) for k in range(6)]
    samples[2] = (0.0, 0.0, -math.pi + 0.03)         # straddles ±pi
    t_last = _feed(est, samples)
    e = est.estimate(t_last)
    assert abs(abs(e.yaw) - math.pi) < 0.2           # ~±pi, no 2pi blowup


def test_dropout_coast_dead_reckons():
    est = WindowedPoseEstimator()
    est.set_command(0.0, 0.7)
    samples = [(0.0, 0.0, 0.7 * (k * DT)) for k in range(6)]
    t_last = _feed(est, samples)
    e = est.estimate(t_last + 0.30)                  # 3-tick gap
    assert e.mode == "coast"
    assert e.yaw > est.estimate(t_last).yaw          # heading advanced on command
    assert e.confidence < est.estimate(t_last).confidence


def test_long_dropout_holds_stale():
    est = WindowedPoseEstimator()
    est.set_command(0.0, 0.0)
    t_last = _feed(est, [(2.0, 3.0, 1.0)] * 4)
    e = est.estimate(t_last + 1.2)                    # > max_coast_s
    assert e.mode == "hold"
    assert abs(e.x - 2.0) < 1e-6 and abs(e.y - 3.0) < 1e-6
    assert e.vx == 0.0 and e.wz == 0.0 and e.confidence < 0.05


def test_nonincreasing_timestamp_dropped():
    est = WindowedPoseEstimator()
    est.add_measurement(0.0, 0.0, 0.0, 1.0)
    est.add_measurement(9.0, 9.0, 9.0, 1.0)          # duplicate t -> dropped
    est.add_measurement(9.0, 9.0, 9.0, 0.5)          # older t -> dropped
    assert len(est._buf) == 1


def test_empty_buffer_invalid():
    est = WindowedPoseEstimator()
    e = est.estimate(5.0)
    assert e.mode == "invalid" and e.confidence == 0.0


def test_degenerate_single_sample():
    est = WindowedPoseEstimator()
    est.set_command(0.0, 0.0)
    est.add_measurement(1.0, 2.0, 0.3, 0.0)
    e = est.estimate(0.0)
    assert abs(e.x - 1.0) < 1e-9 and abs(e.yaw - 0.3) < 1e-9
    assert e.vx == 0.0 and e.wz == 0.0


def test_estimate_is_pure():
    a = WindowedPoseEstimator(); a.set_command(0.0, 0.7)
    b = WindowedPoseEstimator(); b.set_command(0.0, 0.7)
    samples = [(0.0, 0.0, 0.7 * (k * DT)) for k in range(6)]
    # a: estimate() interleaved with adds; b: all adds then one estimate.
    ta = 0.0
    for s in samples:
        a.add_measurement(s[0], s[1], s[2], ta); a.estimate(ta); ta += DT
    _feed(b, samples)
    ea = a.estimate(ta - DT); eb = b.estimate(ta - DT)
    assert abs(ea.yaw - eb.yaw) < 1e-12 and abs(ea.wz - eb.wz) < 1e-12


def test_settle_equivalence_with_circular_mean():
    # STOPPED-mode heading equals circular_mean over the same window -> proves it
    # generalises the follower's YAW_SETTLE averaging.
    est = WindowedPoseEstimator()
    est.set_command(0.0, 0.0)
    yaws = [0.10, 0.12, 0.08, 0.11, 0.09, 0.10]
    t_last = _feed(est, [(0.0, 0.0, y) for y in yaws])
    e = est.estimate(t_last)
    assert abs(e.yaw - circular_mean(yaws)) < 1e-9


# --------------------------------------------------------------------------
# holonomic crab: the optional commanded lateral (vy) is propagated, not dropped
# (multi-axis follower). vy defaults to 0, so all tests above are unaffected.
# --------------------------------------------------------------------------
def test_crab_lateral_preserved_with_commanded_vy():
    """A pure sideways crab (facing +x, moving +y) is tracked when vy is fed; the
    legacy two-argument call drops it and the estimate lags."""
    samples = [(0.0, 0.2 * (i * DT), 0.0) for i in range(7)]    # y = 0.2*t, yaw 0
    t_now = 6 * DT                                              # true y(now) = 0.12
    fix = WindowedPoseEstimator()
    fix.set_command(0.0, 0.0, vy=0.2)                          # forward 0, crab 0.2 left
    _feed(fix, samples)
    e_fix = fix.estimate(t_now)
    leg = WindowedPoseEstimator()
    leg.set_command(0.0, 0.0)                                   # legacy: no vy
    _feed(leg, samples)
    e_leg = leg.estimate(t_now)
    assert abs(e_fix.y - 0.12) < 0.015                          # tracks the crab
    assert abs(e_fix.y - 0.12) < abs(e_leg.y - 0.12)           # and beats the legacy lag


def test_slow_crab_not_frozen_as_stopped():
    """A slow crab (measured speed below settle_vx_eps) is NOT frozen as 'stopped'
    when vy is commanded; without vy the legacy call freezes it (drift rejection)."""
    samples = [(0.0, 0.05 * (i * DT), 0.0) for i in range(7)]   # 0.05 m/s < settle_vx_eps
    t_now = 6 * DT
    fix = WindowedPoseEstimator()
    fix.set_command(0.0, 0.0, vy=0.05)
    _feed(fix, samples)
    assert fix.estimate(t_now).mode != "stopped"
    leg = WindowedPoseEstimator()
    leg.set_command(0.0, 0.0)
    _feed(leg, samples)
    assert leg.estimate(t_now).mode == "stopped"
