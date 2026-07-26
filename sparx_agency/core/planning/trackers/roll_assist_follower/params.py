"""Parameters for the cross-track ROLL corrector layered on the waypoint follower.

The corrector never decides *where* to go — the one-axis
:class:`~sparx_agency.core.planning.trackers.waypoint_follower.WaypointFollower`
still owns navigation (look at the next point, align, then advance). The
corrector adds one thing: a lateral ("ROLL", ``+vy`` = left) velocity whose only
job is to pull the drone back onto its trajectory when it drifts sideways, plus,
while turning or holding, a small forward/back nudge for along-track drift.

The correction is scaled by *what the base follower is doing this tick*, because
ROLL is safe in different amounts in each regime:

  * **Advancing** (flying forward): full lateral gain — stick to the track. No
    forward correction (the base is already driving forward).
  * **Turning** (in-place yaw): a weak lateral gain. ROLL during a turn perturbs
    the rotation itself, so only a small correction is allowed; a small
    forward/back nudge may also be added for along-track drift.
  * **Holding** (stopped, no command): a small gain — the platform is already
    fairly stable at rest, so only large drifts are worth chasing.

Every axis has a **minimum effective command** (``min_vy`` / ``min_vx``): below
it the motors do not move the drone, so a nonzero command is snapped up to that
floor (or, if below ``release_frac`` of it, dropped to zero).
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CrossTrackRollParams:
    """Tuning for :class:`CrossTrackRollCorrector` (SI units, body frame REP-103).

    Attributes:
        kp_lat: Proportional gain mapping cross-track error (m) to lateral speed
            (m/s), before the per-mode scaling below.
        lateral_speed_max: Hard cap on the commanded lateral (ROLL) speed (m/s).
        deadband_m: Cross-track error below this is left uncorrected (m) — ride
            out localization noise instead of chattering the ROLL axis.
        advance_frac: Fraction of ``kp_lat`` applied while advancing (0..1).
            Full (1.0) by default: forward flight is where sticking to the track
            matters most and is safest.
        turn_frac: Fraction of ``kp_lat`` applied while turning in place. Small
            by default — a large ROLL during a turn would spoil the rotation.
        hold_frac: Fraction of ``kp_lat`` applied while stopped/holding. Small by
            default (the platform is fairly stable at rest).
        kp_fwd: Proportional gain mapping along-track error (m) to a forward/back
            speed (m/s), used only while turning or holding (advancing already
            drives forward).
        forward_speed_max: Hard cap on the along-track correction speed (m/s).
        forward_deadband_m: Along-track error below this is left uncorrected (m).
        turn_fwd_frac: Fraction of ``kp_fwd`` applied while turning (0..1).
        hold_fwd_frac: Fraction of ``kp_fwd`` applied while holding (0..1).
        min_vy: Smallest lateral speed that actually moves the platform (m/s).
        min_vx: Smallest forward speed that actually moves the platform (m/s).
        release_frac: A per-axis correction below ``release_frac * min_*`` is
            dropped to zero; between that and ``min_*`` it is snapped up. 0..1.
        cmd_zero_eps: Magnitude treated as exactly zero (numerical dust guard).
        accel_limit: Acceleration limit for slew-shaping the corrections (m/s^2).
        yaw_active_eps: Base yaw rate above which the tick counts as "turning"
            (rad/s); used to pick the turn vs hold regime.
    """

    # Lateral (cross-track / ROLL) correction
    kp_lat: float = 0.8
    lateral_speed_max: float = 0.25
    deadband_m: float = 0.05
    advance_frac: float = 1.0
    turn_frac: float = 0.35
    hold_frac: float = 0.25

    # Along-track (forward/back) correction — turning / holding only
    kp_fwd: float = 0.6
    forward_speed_max: float = 0.15
    forward_deadband_m: float = 0.08
    turn_fwd_frac: float = 0.35
    hold_fwd_frac: float = 0.25

    # Minimum effective command per axis ("minimum force")
    min_vy: float = 0.06
    min_vx: float = 0.06
    release_frac: float = 0.5
    cmd_zero_eps: float = 1e-3

    # Slew shaping + regime threshold
    accel_limit: float = 1.0
    yaw_active_eps: float = 0.05

    def __post_init__(self) -> None:
        """Validate the invariants the correction law relies on."""
        for name in ("kp_lat", "kp_fwd", "deadband_m", "forward_deadband_m",
                     "advance_frac", "turn_frac", "hold_frac",
                     "turn_fwd_frac", "hold_fwd_frac", "yaw_active_eps"):
            if getattr(self, name) < 0.0:
                raise ValueError(name + " must be >= 0")
        for name in ("lateral_speed_max", "forward_speed_max", "min_vy",
                     "min_vx", "accel_limit"):
            if getattr(self, name) <= 0.0:
                raise ValueError(name + " must be > 0")
        if not 0.0 <= self.release_frac <= 1.0:
            raise ValueError("release_frac must be in [0, 1]")
