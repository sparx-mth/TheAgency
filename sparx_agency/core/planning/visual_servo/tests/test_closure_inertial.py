"""Closed-loop check that the coarse-yaw actuation model does not overshoot.

Drives the real :class:`VisualServoController` output through a
:class:`PulseShaper` into a first-order-lag (inertial, coasting) yaw plant, and
compares the centring trajectory with vs without the actuation model. Without it
(a small yaw deadband + analog command) the inertial coast carries the target past
centre and the loop oscillates — the exact failure the model is for. With it (a
coast-sized yaw deadband + minimum burst + a brake tick) the target glides to
centre and stops with no overshoot.
"""
from __future__ import annotations

from sparx_agency.core.common.types import Intrinsics
from sparx_agency.core.common.types.perception import Track2D
from sparx_agency.core.planning.visual_servo import (
    AxisForceProfile,
    PulseShaper,
    VisualServoController,
    VisualServoParams,
    VisualServoRequest,
)

W, H = 504, 294
HALF_FOV = 0.66          # rad; ~38 deg half field of view
DT = 0.1                 # 10 Hz
TAU = 0.37               # yaw inertia/coast time constant (~15 deg coast at 0.7 rad/s)
INTR = Intrinsics(width=W, height=H, fx=float(W), fy=float(W), cx=W / 2, cy=H / 2)
START = 0.35             # target starts 0.35 rad off to the right


def _track_at(theta: float) -> Track2D:
    ox = max(-1.0, min(1.0, theta / HALF_FOV))
    cx = W / 2.0 * (1.0 + ox)
    return Track2D("t", (cx - 50, H / 2 - 50, cx + 50, H / 2 + 50), W, H, valid=True)


def _run(yaw_deadband: float, use_shaper: bool):
    p = VisualServoParams(use_depth=False, target_area_frac=0.9,   # never "close": pure yaw
                          yaw_deadband=yaw_deadband, max_yaw_rate=0.7, kp_yaw=1.5)
    servo = VisualServoController(p)
    shaper = None
    if use_shaper:
        prof = AxisForceProfile(min_magnitude=0.7, max_magnitude=0.7, mode="fixed")
        shaper = PulseShaper(prof, prof, prof, min_burst_ticks=2, brake_ticks=1)
    theta, wz_act, hist = START, 0.0, [START]
    for _ in range(80):
        res = servo.step(VisualServoRequest(track=_track_at(theta), intrinsics=INTR, dt=DT))
        wz_cmd = shaper.shape(res.command).yaw_rate if shaper else res.command.yaw_rate
        wz_act += (wz_cmd - wz_act) * (DT / (TAU + DT))   # first-order lag = inertia + coast
        theta += wz_act * DT                              # ox>0 -> wz<0 -> theta decreases
        hist.append(theta)
    return hist


def test_analog_servo_overshoots_on_an_inertial_plant():
    hist = _run(yaw_deadband=0.03, use_shaper=False)
    assert min(hist) < -0.05        # coast carries the target well past centre


def test_actuation_model_converges_without_overshoot():
    hist = _run(yaw_deadband=0.35, use_shaper=True)
    assert min(hist) >= -0.02       # essentially no overshoot past centre
    assert abs(hist[-1]) < START    # and it did close the offset (glided in)
    assert abs(hist[-1]) < 0.2      # settled near centre (crab would finish the rest)
