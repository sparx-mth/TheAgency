"""Parameters for the visual-servo control law.

Body-frame convention (REP-103, what the FALCON follower/adapter consume):
``+vx`` forward, ``+vy`` left, ``+vz`` up, ``+yaw_rate`` CCW. Image convention:
origin top-left, ``+x`` right, ``+y`` down. A target to the **right** of the image
centre (``ox > 0``) is centred by yawing CW (``yaw_rate < 0``) and/or crabbing
right (``vy < 0``); a target **above** centre (``oy < 0``) by climbing (``vz > 0``).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class VisualServoParams:
    """Tuning for :class:`~...visual_servo.controller.VisualServoController`.

    Attributes:
        mode: ``"holonomic"`` (default: yaw + gated forward, optional crab/climb,
            all at once) or ``"yaw_forward_xor"`` (reference behaviour: pure-yaw
            XOR pure-forward with hysteresis, for platforms that reject mixed
            forward+yaw Twists).

        kp_yaw: P-gain from normalised x-offset to yaw rate (rad/s per unit offset).
        max_yaw_rate: Saturation on commanded yaw rate (rad/s).
        center_deadband: ``|ox|``/``|oy|`` below which no yaw/crab/climb is commanded
            (anti-jitter).
        yaw_deadband: ``|ox|`` below which no YAW is commanded, separate from
            ``center_deadband`` so a *coarse-yaw, high-inertia* platform can leave
            fine centring to the (smoother) lateral crab. On such a platform the
            smallest reliable yaw correction is ~half of one minimum burst plus the
            coast (e.g. ~0.25 of the image half-width at 10 Hz), so aligning tighter
            than that just oscillates. ``None`` reuses ``center_deadband`` (no change).
        yaw_close_deadband: Extra yaw deadband added *in proportion to closeness*
            (0 far → this value at the target range/area). As the drone closes in, a
            given yaw sweeps the (now large) target out of frame fast, so yaw is
            suppressed near the object and crab does the fine centring. 0 disables.

        use_lateral: Also crab sideways to help centre (holonomic mode only).
        kp_lateral: P-gain from x-offset to lateral speed (m/s per unit offset).
        max_lateral_speed: Saturation on crab speed (m/s).

        use_vertical: Climb/descend to centre the target vertically. Off by
            default — the platform holds altitude and the mission is a horizontal
            approach.
        kp_vertical: P-gain from y-offset to vertical speed (m/s per unit offset).
        max_vertical_speed: Saturation on vertical speed (m/s).

        vx_max: Maximum forward approach speed (m/s).
        kp_forward: P-gain from range error (m) to forward speed, when depth is used.
        advance_offset_max: ``|ox|`` at/above which forward speed is gated to ~0 —
            don't fly forward while the target is far off-centre. Below it, forward
            speed scales up smoothly as the target centres.
        min_forward_speed: Floor on forward speed while still advancing (m/s), to
            beat platform stiction; snapped to 0 once at target.

        use_depth: Prefer a metric range (from depth) over the area-fraction proxy
            for the approach/terminal logic. Falls back to area if range is absent.
        target_range_m: **Stop distance** — the forward ramp reaches zero here and
            the mission declares "close and in front" (m). The drone holds this
            standoff (default 0.5 m).
        slowdown_range_m: Range at which the forward ramp begins (m). Must be
            greater than ``target_range_m``.
        target_area_frac: Success box-area fraction when no depth is available.
        slowdown_area_frac: Area fraction at which the forward ramp begins.

        center_tol: The **acquisition angle** — ``|ox|`` below which the target
            counts as "centred" for the at-target (hover-lock) decision. It is the
            angular analogue of a waypoint's acquisition *radius*: on a pulsed,
            coarse-yaw platform we cannot centre to a single degree (a correction
            burst would overshoot to the other side), so "centred" is deliberately a
            small allowed deviation, not exact zero. Keep it >= what the fine axis
            (crab) can actually settle to.
        lateral_deadband: ``|ox|`` below which no lateral crab is commanded, separate
            from ``center_deadband`` so the crab that does the fine centring is also
            pulse/coast-aware and does not oscillate about centre. ``None`` reuses
            ``center_deadband`` (no change).

        speed_smoothing: EMA blend on the forward speed output (0..1; 1 = no smoothing).
        yaw_smoothing: EMA blend on the yaw-rate output (0..1; 1 = no smoothing).

        yaw_deadband_enter: (xor mode) ``|ox|`` at which ADVANCE switches back to YAW.
        yaw_deadband_exit: (xor mode) ``|ox|`` at which YAW switches to ADVANCE
            (must be < ``yaw_deadband_enter`` for hysteresis).
    """

    mode: str = "holonomic"

    # Horizontal centring (yaw)
    kp_yaw: float = 1.2
    max_yaw_rate: float = 0.6
    center_deadband: float = 0.03
    yaw_deadband: Optional[float] = None      # None -> use center_deadband
    yaw_close_deadband: float = 0.0           # extra yaw deadband scaled by closeness

    # Horizontal centring (lateral crab)
    use_lateral: bool = True
    kp_lateral: float = 0.25
    max_lateral_speed: float = 0.25
    lateral_deadband: Optional[float] = None  # None -> use center_deadband

    # Vertical centring
    use_vertical: bool = False
    kp_vertical: float = 0.25
    max_vertical_speed: float = 0.2

    # Forward approach
    vx_max: float = 0.35
    kp_forward: float = 0.6
    advance_offset_max: float = 0.35
    min_forward_speed: float = 0.0

    # Terminal / range logic
    use_depth: bool = True
    target_range_m: float = 0.5          # stop distance: hold ~0.5 m from the object
    slowdown_range_m: float = 2.0
    target_area_frac: float = 0.12
    slowdown_area_frac: float = 0.03

    center_tol: float = 0.15

    # Output smoothing
    speed_smoothing: float = 0.5
    yaw_smoothing: float = 0.5

    # xor-mode hysteresis
    yaw_deadband_enter: float = 0.20
    yaw_deadband_exit: float = 0.08

    def __post_init__(self) -> None:
        if self.mode not in ("holonomic", "yaw_forward_xor"):
            raise ValueError(
                "mode must be 'holonomic' or 'yaw_forward_xor', got %r" % self.mode
            )
        if self.advance_offset_max <= 0.0:
            raise ValueError("advance_offset_max must be > 0.")
        if self.yaw_deadband is not None and self.yaw_deadband < 0.0:
            raise ValueError("yaw_deadband must be >= 0 when given.")
        if self.yaw_close_deadband < 0.0:
            raise ValueError("yaw_close_deadband must be >= 0.")
        if self.lateral_deadband is not None and self.lateral_deadband < 0.0:
            raise ValueError("lateral_deadband must be >= 0 when given.")
        if self.slowdown_range_m <= self.target_range_m:
            raise ValueError("slowdown_range_m must be > target_range_m.")
        if self.target_area_frac <= self.slowdown_area_frac:
            raise ValueError("target_area_frac must be > slowdown_area_frac.")
        for name in ("speed_smoothing", "yaw_smoothing"):
            v = getattr(self, name)
            if not (0.0 < v <= 1.0):
                raise ValueError("%s must be in (0, 1], got %r" % (name, v))
        if self.yaw_deadband_exit >= self.yaw_deadband_enter:
            raise ValueError("yaw_deadband_exit must be < yaw_deadband_enter.")
