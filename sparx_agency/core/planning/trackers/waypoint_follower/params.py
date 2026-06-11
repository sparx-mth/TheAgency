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

    Attributes:
        vel_x: Forward cruise speed in the ADVANCE state (m/s).
        yaw_rate: Nominal rotation speed in the YAW_ALIGN state (rad/s).
        pos_radius: Waypoint acquisition radius (m). Closer than this counts
            as "reached".
        yaw_settle: Heading error below which YAW_ALIGN is satisfied (rad).
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
        yaw_lead_pct: Stop the rotation this many percent short of the desired
            heading, leaving the slew ramp-down to consume the rest. Clamped to
            [0, 40].
        vel_xy_sat: Saturation on the published forward speed (m/s).
        yaw_rate_sat: Saturation on the published yaw rate (rad/s).
        accel_limit: Forward acceleration limit used for slew shaping (m/s^2).
        yaw_accel_limit: Yaw acceleration limit used for slew shaping (rad/s^2).
        forward_only: Skip YAW_ALIGN entirely; treat every alignment as an
            immediate ADVANCE. Useful when already pointed down a corridor.
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

    # Yaw lead (inertia compensation), percent of the initial sweep.
    yaw_lead_pct: float = 10.0

    # Slew + saturations
    vel_xy_sat: float = 1.25
    yaw_rate_sat: float = 2.4
    accel_limit: float = 1.5
    yaw_accel_limit: float = 3.5

    # Behaviour
    forward_only: bool = False
