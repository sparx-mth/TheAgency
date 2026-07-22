"""Stop-distance and acquisition-angle tests for the visual-servo controller.

The stop distance (``target_range_m``) is the forward standoff the drone holds; the
acquisition angle (``yaw_deadband`` / ``lateral_deadband`` / ``center_tol``) is the
allowed centring deviation — on a pulsed platform we cannot correct to a single
degree, so a target already within the acquisition angle must NOT be yawed at (a
correction burst would overshoot to the other side).
"""
from __future__ import annotations

from sparx_agency.core.common.types import Intrinsics
from sparx_agency.core.common.types.perception import Track2D
from sparx_agency.core.planning.visual_servo import (
    VisualServoController,
    VisualServoParams,
    VisualServoRequest,
)

W, H = 504, 294
INTR = Intrinsics(W, H, fx=float(W), fy=float(W), cx=W / 2, cy=H / 2)


def _box_at(ox: float):
    cx = W / 2.0 * (1.0 + ox)
    return (cx - 40, H / 2 - 40, cx + 40, H / 2 + 40)


def _step(servo, ox=0.0, rng=None):
    servo.reset()
    tr = Track2D("t", _box_at(ox), W, H, valid=True)
    return servo.step(VisualServoRequest(track=tr, intrinsics=INTR, range_m=rng, dt=0.1))


# ── stop distance ────────────────────────────────────────────────────────
def test_default_stop_distance_is_half_a_metre():
    assert VisualServoParams().target_range_m == 0.5


def test_forward_stops_at_the_target_range():
    servo = VisualServoController(VisualServoParams())   # target_range_m = 0.5
    assert _step(servo, ox=0.0, rng=2.0).command.x > 0.0        # far -> advance
    at = _step(servo, ox=0.0, rng=0.5)                          # at the standoff
    assert at.command.x == 0.0 and at.at_target is True
    closer = _step(servo, ox=0.0, rng=0.4)                      # closer -> still stopped
    assert closer.command.x == 0.0 and closer.at_target is True


# ── acquisition angle (yaw) ──────────────────────────────────────────────
def test_yaw_suppressed_within_the_acquisition_angle():
    p = VisualServoParams(use_depth=False, target_area_frac=0.9, yaw_deadband=0.35)
    servo = VisualServoController(p)
    assert _step(servo, ox=0.10).command.yaw_rate == 0.0      # within angle -> no yaw
    assert _step(servo, ox=0.30).command.yaw_rate == 0.0      # still within
    assert _step(servo, ox=0.50).command.yaw_rate != 0.0      # outside -> correct


def test_lateral_deadband_suppresses_tiny_crab():
    p = VisualServoParams(use_depth=False, target_area_frac=0.9,
                          use_lateral=True, lateral_deadband=0.12)
    servo = VisualServoController(p)
    assert _step(servo, ox=0.05).command.y == 0.0            # within crab deadband
    assert _step(servo, ox=0.30).command.y != 0.0            # outside -> crab
