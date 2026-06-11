"""Parameters for the one-axis-at-a-time waypoint follower."""
from __future__ import annotations

from dataclasses import dataclass
from math import radians


@dataclass(frozen=True)
class WaypointFollowerParams:
    """Tuning for :class:`WaypointFollower`.

    The follower is deliberately "stupid": it never moves on two axes at
    once. Every command is either a pure forward advance (``vx``, ``wz=0``)
    or a pure in-place rotation (``wz``, ``vx=0``). This mirrors a real
    platform that cannot translate and rotate simultaneously.

    Yaw alignment is a discrete "pulse -> settle -> re-measure" loop, not a
    continuous controller. The platform yaws slowly (~0.7 rad/s), has strong
    inertia (commanding 0 coasts on; too short a pulse does not overcome the
    deadband) and its yaw localization is unreliable *while rotating* but
    accurate when still. So YAW_ALIGN commits to a short open-loop burst sized
    from the last *settled* heading, then YAW_SETTLE coasts to a stop and dwells
    so localization re-converges, then the heading is re-measured and the loop
    repeats only if still off. The ADVANCE gate is deliberately gentle (see
    ``yaw_capture_tol_m`` / ``yaw_acquire_max``): the drone starts moving as soon
    as going straight would still capture the waypoint, rather than nailing an
    exact heading.

    Attributes:
        vel_x: Forward cruise speed in the ADVANCE state (m/s).
        yaw_rate: Nominal rotation speed during a YAW_ALIGN burst (rad/s).
        pos_radius: Waypoint acquisition radius (m). Closer than this counts
            as "reached".
        yaw_settle: Legacy heading tolerance (rad). Retained for compatibility;
            the live ADVANCE decision uses the predictive gate below.
        yaw_drift_thresh: Heading drift from the ADVANCE-entry heading that
            triggers a brake-and-realign (rad).
        skip_yaw_thresh: If the next waypoint's bearing is within this of the
            current heading, glide straight to it instead of re-aligning (rad).
        vx_brake_thresh: Forward speed below which BRAKE is considered settled
            (m/s).
        brake_timeout_s: Hard cap on how long BRAKE waits for the slew to
            reach zero (s).
        passed_bearing_rad: If the bearing to the current target exceeds this,
            the waypoint is treated as passed (it is now mostly behind) and
            the follower advances rather than chasing it (rad).
        yaw_settle_dwell_s: After a burst coasts to a stop, dwell this long with
            zero command (sensors unfrozen) so localization re-converges before
            re-measuring the heading (s).
        yaw_settle_eps: Yaw rate below which the post-burst coast is considered
            finished and the (unfrozen) dwell begins (rad/s).
        yaw_burst_min_ticks: Minimum burst length in control ticks. Sets the
            deadband floor so a burst always actually rotates the platform.
        yaw_burst_max_ticks: Hard cap on a single burst (runaway guard).
        yaw_burst_max_rad: Hard cap on a single open-loop burst angle (rad);
            larger turns re-measure between bursts.
        yaw_coast_rad: Physical yaw the platform keeps sweeping after the burst
            command stops (inertia). The burst aims this much short to land on
            the target instead of overshooting.
        yaw_capture_tol_m: Cross-track tolerance for the predictive ADVANCE gate
            (m): advance once going straight on the current heading would pass
            within this distance of the waypoint. Wired from the launch's
            ``yaw_acquisition_radius``.
        yaw_acquire_max: Hard cap on the heading error the predictive gate will
            ever accept (rad), regardless of how close the waypoint is.
        yaw_lead_pct: Deprecated (unused by the burst loop); kept so existing
            configs/tests that pass it still construct.
        vel_xy_sat: Saturation on the published forward speed (m/s).
        yaw_rate_sat: Saturation on the published yaw rate (rad/s).
        accel_limit: Forward acceleration limit used for slew shaping (m/s^2).
        yaw_accel_limit: Yaw acceleration limit used for slew shaping (rad/s^2).
        forward_only: Skip YAW_ALIGN/YAW_SETTLE entirely; treat every alignment
            as an immediate ADVANCE. Useful when already pointed down a corridor.
    """

    # Speeds
    vel_x: float = 0.3
    yaw_rate: float = 0.7

    # Acquisition / settle thresholds
    pos_radius: float = 0.35
    yaw_settle: float = 0.05

    # Strict-separation thresholds
    yaw_drift_thresh: float = 0.40
    skip_yaw_thresh: float = 0.25
    vx_brake_thresh: float = 0.05
    brake_timeout_s: float = 2.0
    passed_bearing_rad: float = radians(100.0)

    # Pulse -> settle -> re-measure (yaw inertia + jumpy localization).
    yaw_settle_dwell_s: float = 0.8
    yaw_settle_eps: float = 0.05
    yaw_burst_min_ticks: int = 2
    yaw_burst_max_ticks: int = 30
    yaw_burst_max_rad: float = radians(135.0)
    yaw_coast_rad: float = radians(15.0)

    # Gentle predictive ADVANCE gate.
    yaw_capture_tol_m: float = 0.20
    yaw_acquire_max: float = radians(35.0)

    # Deprecated (kept for config/test compatibility).
    yaw_lead_pct: float = 10.0

    # Slew + saturations
    vel_xy_sat: float = 1.25
    yaw_rate_sat: float = 2.4
    accel_limit: float = 1.5
    yaw_accel_limit: float = 3.5

    # Behaviour
    forward_only: bool = False
