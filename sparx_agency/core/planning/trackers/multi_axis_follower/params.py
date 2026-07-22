"""Parameters for the multi-axis (vx + vy + yaw) waypoint follower.

Unlike the one-axis :class:`~sparx_agency.core.planning.trackers.waypoint_follower`,
this tracker may drive forward, sideways (lateral / "crab") and yaw at the same
time. The platform's localization and depth are noisiest *while yawing* and while
standing still, and cleanest while flying forward, so the controller's whole
strategy is to reach the waypoint with the least yaw possible:

  * small heading errors are absorbed by lateral motion (ROLL, not YAW),
  * yaw is engaged only past a deadband (with hysteresis) and is then commanded
    continuously *while still translating* — never a stop-and-spin,
  * altitude is never commanded (``vz`` is always 0; the platform holds height).

Every axis has a **minimum effective command** (``min_vx`` / ``min_vy`` /
``min_wz``): below it the motors do not actually move the drone, so a nonzero
command is snapped up to that floor (or, if it is below ``release_frac`` of the
floor, dropped to zero). This is the "minimum force per axis" the platform needs.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import radians


@dataclass(frozen=True)
class MultiAxisFollowerParams:
    """Tuning for :class:`MultiAxisFollower` (SI units, body frame REP-103).

    Body frame: ``+vx`` forward, ``+vy`` left, ``+wz`` counter-clockwise. The
    follower outputs a velocity command; another layer maps it to the platform.

    Attributes:
        cruise_speed: Nominal translation speed while pursuing a waypoint (m/s).
        lateral_speed_max: Cap on the sideways (crab) speed (m/s). Usually lower
            than ``cruise_speed`` because lateral authority is weaker; a large
            lateral demand is what pushes the heading error past the yaw deadband,
            so the drone turns instead of crabbing forever.
        yaw_rate: Saturation on the commanded yaw rate during a turn (rad/s).
        pos_radius: Waypoint capture ("attack") radius (m). Inside it the waypoint
            counts as reached.
        slow_radius: Distance to the FINAL goal at which the translation speed
            starts ramping down from ``cruise_speed`` toward ``arrive_speed_min``
            for a gentle arrival (m). Intermediate waypoints are glided through at
            cruise (no ramp).
        arrive_speed_min: Floor on the translation speed while still outside
            ``pos_radius`` (m/s). Keeps the drone moving instead of crawling to a
            noisy near-stop short of the point.
        yaw_engage_rad: Heading error above which yaw is engaged (rad). Below it
            the drone crabs without rotating (the low-noise regime).
        yaw_release_rad: Heading error below which yaw is released once engaged
            (rad). ``< yaw_engage_rad`` — the hysteresis band that stops the yaw
            command chattering on noise around the threshold.
        yaw_kp: Proportional gain mapping heading error to yaw rate (1/s); the
            result is saturated to ``yaw_rate``.
        travel_cone_rad: Maximum body-frame angle off straight-ahead that the
            translation is allowed to point (rad). A steeper target is approached
            at the cone edge while yaw brings it forward — so the drone never flies
            fast into a direction its forward camera cannot see. Set to ``pi`` for
            fully holonomic crabbing.
        translate_suppress_rad: Heading error at/above which translation is
            throttled to ``translate_suppress_floor`` so a grossly mis-pointed
            drone mostly yaws first (rad). Must exceed ``travel_cone_rad``.
        translate_suppress_floor: Fraction of speed kept at/beyond
            ``translate_suppress_rad`` (0..1); keeps creeping, never fully stops.
        min_vx: Smallest forward speed that actually moves the platform (m/s).
        min_vy: Smallest lateral speed that actually moves the platform (m/s).
        min_wz: Smallest yaw rate that actually rotates the platform (rad/s).
        release_frac: A per-axis command below ``release_frac * min_*`` is dropped
            to zero (deadband); between that and ``min_*`` it is snapped up to
            ``min_*`` (committed). 0..1.
        cmd_zero_eps: Magnitude below which a command is treated as exactly zero,
            so numerical dust never triggers a spurious minimum-force pulse.
        hold_deadband: In station-keeping, positional error below this is left
            uncorrected (ride out the localization noise instead of chasing it) (m).
        hold_kp: Proportional gain for the station-keeping correction (1/s).
        hold_speed_max: Cap on the station-keeping correction speed (m/s).
        hold_reacquire_margin: If the drone drifts more than
            ``pos_radius + hold_reacquire_margin`` from the goal while holding, it
            re-enters full pursuit instead of a gentle nudge (m).
        passed_bearing_rad: If a NON-final waypoint's bearing exceeds this it is
            treated as passed (now behind) and skipped (rad). The final goal is
            never skipped — the drone turns back to it.
        vel_xy_sat: Saturation on the translation-speed vector magnitude (m/s).
        yaw_rate_sat: Saturation on the published yaw rate (rad/s).
        accel_limit: Translation acceleration limit for slew shaping (m/s^2).
        yaw_accel_limit: Yaw acceleration limit for slew shaping (rad/s^2).
    """

    # Speeds / limits
    cruise_speed: float = 0.3
    lateral_speed_max: float = 0.25
    yaw_rate: float = 0.6

    # Capture / arrival
    pos_radius: float = 0.35
    slow_radius: float = 0.8
    arrive_speed_min: float = 0.08

    # Roll-vs-yaw allocation
    yaw_engage_rad: float = radians(25.0)
    yaw_release_rad: float = radians(10.0)
    yaw_kp: float = 1.2
    travel_cone_rad: float = radians(80.0)
    translate_suppress_rad: float = radians(120.0)
    translate_suppress_floor: float = 0.2

    # Minimum effective command per axis ("minimum force")
    min_vx: float = 0.06
    min_vy: float = 0.06
    min_wz: float = radians(8.0)
    release_frac: float = 0.5
    cmd_zero_eps: float = 1e-3

    # Station-keeping at the final goal
    hold_deadband: float = 0.18
    hold_kp: float = 0.8
    hold_speed_max: float = 0.2
    hold_reacquire_margin: float = 0.15

    # Sequencing
    passed_bearing_rad: float = radians(110.0)

    # Slew + saturation
    vel_xy_sat: float = 1.0
    yaw_rate_sat: float = 1.5
    accel_limit: float = 1.0
    yaw_accel_limit: float = 2.5

    def __post_init__(self) -> None:
        """Validate the invariants the allocation math relies on."""
        if self.yaw_release_rad >= self.yaw_engage_rad:
            raise ValueError("yaw_release_rad must be < yaw_engage_rad (hysteresis)")
        if self.slow_radius < self.pos_radius:
            raise ValueError("slow_radius must be >= pos_radius")
        if not 0.0 <= self.release_frac <= 1.0:
            raise ValueError("release_frac must be in [0, 1]")
        if not 0.0 <= self.translate_suppress_floor <= 1.0:
            raise ValueError("translate_suppress_floor must be in [0, 1]")
        # The ramp tail at pos_radius is arrive_speed_min; if it is below the
        # minimum-force release threshold the deadband snaps it to zero and the
        # drone stalls just outside the capture radius. Forbid that.
        if self.arrive_speed_min < self.release_frac * self.min_vx:
            raise ValueError("arrive_speed_min must be >= release_frac*min_vx "
                             "(else the drone stalls just outside pos_radius)")
        for name in ("cruise_speed", "yaw_rate", "pos_radius", "vel_xy_sat",
                     "yaw_rate_sat", "accel_limit", "yaw_accel_limit"):
            if getattr(self, name) <= 0.0:
                raise ValueError(name + " must be > 0")
