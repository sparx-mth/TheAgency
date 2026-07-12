"""Visual-servo controller: a tracked box -> a body-frame velocity command.

Turns the target's image-plane offset and proximity into a REP-103 velocity that
centres the object and closes in until it is "in front of and very close" (the
mission's success condition). Two modes:

  * ``holonomic`` (default): yaw to centre, forward gated by how centred we are,
    plus optional lateral crab and vertical centring — all commanded at once.
    Smooth and fast on a holonomic platform like the XTEND.
  * ``yaw_forward_xor``: the reference behaviour — pure-yaw XOR pure-forward with
    hysteresis and a brake tick on each switch — for platforms that reject Twists
    mixing forward and yaw.

Depth-aware: given a metric range it ramps forward on true distance and declares
success at ``target_range_m``; without depth it falls back to the box area
fraction. Stateless per-axis math lives in
:mod:`sparx_agency.core.planning.visual_servo.algorithm`; this class owns only the
mode/hysteresis state and the output EMA smoothing.
"""
from __future__ import annotations

from typing import Optional

from sparx_agency.core.common.types import ControlCommand, KinematicLimits
from sparx_agency.core.common.math.bbox import area_frac as bbox_area_frac
from sparx_agency.core.common.math.bbox import center_offset_norm
from sparx_agency.core.planning.visual_servo.params import VisualServoParams
from sparx_agency.core.planning.visual_servo.interface import (
    VisualServoRequest,
    VisualServoResult,
)
from sparx_agency.core.planning.visual_servo import algorithm as alg


class VisualServoController:
    """Reactive image-based visual servo (see module docstring)."""

    name = "visual_servo"

    def __init__(self, params: Optional[VisualServoParams] = None,
                 default_limits: Optional[KinematicLimits] = None) -> None:
        self.p = params or VisualServoParams()
        self.default_limits = default_limits
        self.reset()

    def reset(self) -> None:
        """Clear smoothing + xor sub-mode state."""
        self._sub_mode = "YAW"     # xor mode only
        self._vx_s = 0.0           # smoothed forward speed
        self._wz_s = 0.0           # smoothed yaw rate

    def step(self, request: VisualServoRequest) -> VisualServoResult:
        """Compute one command from the current tracked box."""
        p = self.p
        track = request.track
        intr = request.intrinsics
        W, H = int(intr.width), int(intr.height)
        limits = request.limits or self.default_limits

        ox, oy = center_offset_norm(track.bbox_xyxy, W, H)
        area = bbox_area_frac(track.bbox_xyxy, W, H)
        use_depth = bool(p.use_depth) and request.range_m is not None
        rng = float(request.range_m) if use_depth else None

        # Effective saturations (params capped by kinematic limits when supplied).
        max_yaw = self._cap(p.max_yaw_rate, getattr(limits, "max_yaw_rate", None))
        max_fwd = self._cap(p.vx_max, getattr(limits, "max_speed_xy", None))
        max_lat = self._cap(p.max_lateral_speed, getattr(limits, "max_speed_xy", None))
        max_vert = self._cap(p.max_vertical_speed, getattr(limits, "max_speed_z", None))

        # Lost track -> hold (the FSM normally routes recovery, this is defensive).
        if not track.valid:
            return self._hold(ox, oy, area, rng, reason="track_invalid")

        # Degenerate (collapsed) box: area_frac would read ~0 and drive full-speed
        # forward. Treat it as an unusable measurement and hold instead.
        if track.w < 1.0 or track.h < 1.0:
            return self._hold(ox, oy, area, rng, reason="degenerate_box")

        close = (rng <= p.target_range_m) if use_depth else (area >= p.target_area_frac)
        centered = abs(ox) <= p.center_tol
        at_target = bool(close and centered)

        # Coarse-yaw platform: the yaw deadband is larger than the crab/climb one
        # (min-burst + coast) and GROWS as we close, so a small yaw does not sweep
        # the (now large) target out of frame -- crab does the fine centring instead.
        closeness = self._closeness(area, rng, use_depth)
        yaw_db = (p.yaw_deadband if p.yaw_deadband is not None else p.center_deadband) \
            + closeness * p.yaw_close_deadband

        vx_raw = self._forward_raw(area, rng, use_depth, max_fwd)

        if p.mode == "yaw_forward_xor":
            vx, vy, vz, wz, mode = self._xor_step(ox, vx_raw, max_yaw, yaw_db)
        else:
            vx, vy, vz, wz, mode = self._holonomic_step(
                ox, oy, vx_raw, at_target, max_yaw, max_fwd, max_lat, max_vert, yaw_db
            )

        # Output smoothing on the two dominant axes.
        self._vx_s = alg.ema(self._vx_s, vx, p.speed_smoothing)
        self._wz_s = alg.ema(self._wz_s, wz, p.yaw_smoothing)
        vx, wz = self._vx_s, self._wz_s

        # Success: hard-stop forward so we hold "close and in front" without the
        # smoothed speed creeping the drone into the object. Centring (yaw/crab)
        # stays live so a moving target is still followed.
        if at_target:
            self._vx_s = 0.0
            vx = 0.0

        cmd = ControlCommand.velocity(
            vx, vy, vz, wz, source=self.name, mode=mode, at_target=at_target
        )
        return VisualServoResult(
            command=cmd, at_target=at_target, centered=centered,
            x_offset=ox, y_offset=oy, area_frac=area, range_m=rng, mode=mode,
            metadata={"vx_raw": vx_raw, "n_matches": track.n_matches,
                      "predicted": track.predicted, "closeness": closeness,
                      "yaw_deadband": yaw_db},
        )

    # ── modes ────────────────────────────────────────────────────────
    def _holonomic_step(self, ox, oy, vx_raw, at_target, max_yaw, max_fwd,
                        max_lat, max_vert, yaw_db):
        p = self.p
        wz = alg.yaw_command(ox, p.kp_yaw, max_yaw, yaw_db)
        gate = alg.centering_gain(ox, p.advance_offset_max)
        vx = 0.0 if at_target else vx_raw * gate
        # Stiction floor, scaled by the centring gate so it only lifts a genuine
        # advance and still collapses to 0 as the target goes off-axis (rather
        # than re-injecting forward motion the gate deliberately ramped out).
        floor = p.min_forward_speed * gate
        if 0.0 < vx < floor:
            vx = floor
        vx = alg.saturate(vx, max_fwd)
        vy = alg.lateral_command(ox, p.kp_lateral, max_lat, p.center_deadband) \
            if p.use_lateral else 0.0
        vz = alg.vertical_command(oy, p.kp_vertical, max_vert, p.center_deadband) \
            if p.use_vertical else 0.0
        return vx, vy, vz, wz, "holonomic"

    def _xor_step(self, ox, vx_raw, max_yaw, yaw_db):
        # yaw_db is unused here: the yaw XOR forward mode uses its own
        # enter/exit submode hysteresis as the yaw deadband (a second large
        # deadband would fight it). The coast/range-aware yaw deadband applies
        # to the holonomic closure (the FALCON default).
        p = self.p
        prev = self._sub_mode
        ax = abs(ox)
        if prev == "YAW" and ax < p.yaw_deadband_exit:
            self._sub_mode = "ADVANCE"
        elif prev == "ADVANCE" and ax > p.yaw_deadband_enter:
            self._sub_mode = "YAW"
        if self._sub_mode != prev:
            # Brake tick on switch: let the platform settle one axis before the next.
            self._vx_s = 0.0
            self._wz_s = 0.0
            return 0.0, 0.0, 0.0, 0.0, self._sub_mode + "*"
        if self._sub_mode == "YAW":
            wz = alg.saturate(-p.kp_yaw * ox, max_yaw)
            return 0.0, 0.0, 0.0, wz, "YAW"
        return vx_raw, 0.0, 0.0, 0.0, "ADVANCE"

    # ── helpers ──────────────────────────────────────────────────────
    def _closeness(self, area, rng, use_depth):
        """Proximity in [0, 1]: 0 far, 1 at the target range/area (for yaw scaling)."""
        p = self.p
        if use_depth and rng is not None:
            if p.slowdown_range_m <= p.target_range_m:
                return 1.0 if rng <= p.target_range_m else 0.0
            frac = (p.slowdown_range_m - rng) / (p.slowdown_range_m - p.target_range_m)
        else:
            frac = area / p.target_area_frac if p.target_area_frac > 0.0 else 0.0
        return 0.0 if frac < 0.0 else (1.0 if frac > 1.0 else frac)

    def _forward_raw(self, area, rng, use_depth, max_fwd):
        p = self.p
        if use_depth:
            return alg.forward_from_range(
                rng, p.target_range_m, p.slowdown_range_m, max_fwd, p.kp_forward)
        return alg.forward_from_area(
            area, p.target_area_frac, p.slowdown_area_frac, max_fwd)

    def _hold(self, ox, oy, area, rng, reason):
        self._vx_s = 0.0
        self._wz_s = 0.0
        cmd = ControlCommand.velocity(0.0, 0.0, 0.0, 0.0, source=self.name,
                                      hold=reason)
        return VisualServoResult(
            command=cmd, at_target=False, centered=abs(ox) <= self.p.center_tol,
            x_offset=ox, y_offset=oy, area_frac=area, range_m=rng, mode="hold",
            metadata={"hold": reason})

    @staticmethod
    def _cap(value: float, limit: Optional[float]) -> float:
        if limit is None:
            return float(value)
        return float(min(value, limit))
