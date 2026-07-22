"""Unit tests for :class:`VisualServoController`.

Drives ``VisualServoController.step`` with hand-built :class:`Track2D` +
:class:`Intrinsics` (640x360) and asserts on the returned
:class:`VisualServoResult` / :class:`ControlCommand`.

Sign conventions (REP-103 body frame, image origin top-left):
    * A target to the RIGHT of image centre (``ox > 0``) is centred by yawing CW
      (``yaw_rate < 0``) and/or crabbing right (``vy = command.y < 0``).
    * ``command.x`` is forward speed (``vx``), gated by how centred the target is.

Each independent case uses a fresh controller (or ``reset()``) so the output EMA
smoothing and the xor sub-mode state do not leak across cases.
"""
from __future__ import annotations

import numpy as np

from sparx_agency.core.common.types import Intrinsics
from sparx_agency.core.common.types.perception import Track2D
from sparx_agency.core.planning.visual_servo.controller import VisualServoController
from sparx_agency.core.planning.visual_servo.interface import VisualServoRequest
from sparx_agency.core.planning.visual_servo.params import VisualServoParams

# Deterministic (no RNG is used by the controller, but pin a seed regardless).
np.random.seed(0)

W, H = 640, 360  # image size for every case


# ── helpers ───────────────────────────────────────────────────────────
def _intr(w: int = W, h: int = H) -> Intrinsics:
    """Pinhole intrinsics; only width/height are read by the controller."""
    return Intrinsics(width=w, height=h, fx=500.0, fy=500.0, cx=w / 2.0, cy=h / 2.0)


def _track(cx: float, cy: float, bw: float, bh: float,
           valid: bool = True, label: str = "target") -> Track2D:
    """Build a Track2D whose box is centred at ``(cx, cy)`` with size ``bw x bh``."""
    return Track2D(
        label=label,
        bbox_xyxy=(cx - bw / 2.0, cy - bh / 2.0, cx + bw / 2.0, cy + bh / 2.0),
        frame_w=W, frame_h=H, valid=valid,
    )


def _req(track: Track2D, range_m=None) -> VisualServoRequest:
    return VisualServoRequest(track=track, intrinsics=_intr(), range_m=range_m)


# ── centring: yaw + crab ──────────────────────────────────────────────
def test_target_right_yaws_cw_and_crabs_right():
    """Box centre right of image centre => yaw_rate < 0 and command.y <= 0."""
    ctrl = VisualServoController()
    tr = _track(cx=480.0, cy=180.0, bw=40.0, bh=40.0)  # ox = +0.5
    res = ctrl.step(_req(tr, range_m=3.0))

    assert res.x_offset > 0.0
    assert res.command.yaw_rate < 0.0   # yaw clockwise toward the target
    assert res.command.y <= 0.0         # crab right (+vy is left)


# ── forward centring gate ─────────────────────────────────────────────
def test_forward_reduced_when_off_center():
    """Centred & far advances; the same range off-centre advances less."""
    ctrl = VisualServoController()
    res_centred = ctrl.step(_req(_track(320.0, 180.0, 40.0, 40.0), range_m=3.0))
    vx_centred = res_centred.command.x

    ctrl.reset()  # clear EMA smoothing before the independent second case
    res_off = ctrl.step(_req(_track(396.0, 180.0, 40.0, 40.0), range_m=3.0))
    vx_off = res_off.command.x

    assert vx_centred > 0.0            # advancing when centred
    assert vx_off > 0.0               # still advancing off-centre
    assert vx_off < vx_centred        # but gated down by the centring gain


# ── depth-driven approach / terminal ──────────────────────────────────
def test_depth_far_advances_not_at_target():
    """use_depth=True, range far => advancing and not yet at target."""
    ctrl = VisualServoController()
    res = ctrl.step(_req(_track(320.0, 180.0, 40.0, 40.0), range_m=3.0))

    assert res.range_m == 3.0
    assert res.command.x > 0.0
    assert res.at_target is False


def test_depth_close_centered_at_target_hard_stop():
    """range <= target_range_m and centred => at_target and hard-stopped forward."""
    ctrl = VisualServoController()
    res = ctrl.step(_req(_track(320.0, 180.0, 40.0, 40.0), range_m=0.5))

    assert res.at_target is True
    assert res.command.x == 0.0       # hard stop on success


# ── area-based terminal (no depth) ────────────────────────────────────
def test_area_based_at_target_without_depth():
    """use_depth=False, area >= target_area_frac and centred => at_target."""
    params = VisualServoParams(use_depth=False)
    ctrl = VisualServoController(params)
    # 200x160 box -> area_frac = 32000 / 230400 = 0.1389 >= 0.12.
    res = ctrl.step(_req(_track(320.0, 180.0, 200.0, 160.0), range_m=0.5))

    assert res.range_m is None                       # depth ignored
    assert res.area_frac >= params.target_area_frac
    assert res.at_target is True
    assert res.command.x == 0.0


# ── invalid track ─────────────────────────────────────────────────────
def test_invalid_track_holds_zero_command():
    """Lost track => zero command on every axis, at_target False, mode 'hold'."""
    ctrl = VisualServoController()
    res = ctrl.step(_req(_track(480.0, 180.0, 40.0, 40.0, valid=False), range_m=3.0))

    assert res.at_target is False
    assert res.mode == "hold"
    assert res.command.x == 0.0
    assert res.command.y == 0.0
    assert res.command.z == 0.0
    assert res.command.yaw_rate == 0.0


# ── xor mode: YAW <-> ADVANCE with a brake tick on switch ─────────────
def test_xor_mode_alternates_with_brake_tick():
    """Driving |x_off| across the hysteresis band toggles YAW/ADVANCE.

    Each switch emits a one-tick brake (mode label suffixed with ``*``).
    Thresholds: yaw_deadband_exit=0.08 (YAW->ADVANCE), yaw_deadband_enter=0.20
    (ADVANCE->YAW). No depth and a tiny box keep the target off-target so the
    at-target hard-stop never masks the returned sub-mode label.
    """
    ctrl = VisualServoController(VisualServoParams(mode="yaw_forward_xor"))
    intr = _intr()

    def mode_for(ox: float) -> str:
        cx = 320.0 + ox * 320.0
        tr = _track(cx=cx, cy=180.0, bw=40.0, bh=40.0)
        return ctrl.step(VisualServoRequest(track=tr, intrinsics=intr)).mode

    assert mode_for(0.50) == "YAW"         # start in YAW, large offset holds it
    assert mode_for(0.02) == "ADVANCE*"    # inside exit band -> switch (brake)
    assert mode_for(0.02) == "ADVANCE"     # settled in ADVANCE
    assert mode_for(0.50) == "YAW*"        # past enter band -> switch (brake)
    assert mode_for(0.50) == "YAW"         # settled back in YAW
