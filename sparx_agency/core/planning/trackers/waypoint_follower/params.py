"""Parameters for the one-axis-at-a-time waypoint follower."""
from __future__ import annotations

from dataclasses import dataclass
from math import degrees, radians


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
        min_motion_ticks: Minimum number of consecutive control ticks for any
            motion command, forward OR yaw. A single 5 Hz pulse cannot overcome
            the platform's deadband/inertia (it neither turns nor moves), so the
            follower never emits a lone motion tick: every yaw burst and every
            forward advance lasts at least this many ticks (default 2). Set 1 to
            disable.
        yaw_burst_max_ticks: Hard cap on a single burst (runaway guard).
        yaw_burst_max_rad: Maximum a single burst will rotate (rad) before
            stopping to re-measure and let the map update. A large turn is split
            into several bursts of at most this size -- each followed by a stop +
            voxel update + re-measure -- instead of one big open-loop sweep. So
            e.g. an 80 deg turn becomes a few ~25 deg chunks, not one 80 deg burst.
        yaw_coast_rad: Physical yaw the platform keeps sweeping after the burst
            command stops (inertia). The burst aims this much short to land on
            the target instead of overshooting.
        freeze_on_rotation: Master on/off for the whole freeze-during-rotation
            feature. False keeps the map LIVE through every turn (no freeze, no
            forced re-observation) — use it if depth is trusted during rotation.
            Navigation (the burst/settle loop) is unchanged either way.
        freeze_yaw_thresh_rad: Heading change above which an alignment *episode*
            is treated as a real rotation and FREEZES the map: while depth and
            localization are unreliable during a turn, no voxel is fused. A small
            heading correction (error at or below this) is instead executed with
            the sensors LIVE -- the map keeps updating through the gentle nudge --
            because freezing (and the post-turn re-observation it forces) is not
            worth it for a few degrees. Measured once, from the settled heading at
            the start of the episode, and LATCHED for the whole episode (so the
            small residual burst that ends an 80 deg turn stays frozen, not just
            the first chunk); it clears on reaching ADVANCE. Set ``0.0`` to freeze
            every turn (legacy behaviour); set very large to never freeze on yaw.
        settle_map_updates: Fresh map/voxel updates a *frozen* turn's YAW_SETTLE
            must see (stopped, sensors live) before the follower will move on --
            i.e. the drone re-observes the scene at least this many times from the
            new, converged heading before advancing or turning again, so it never
            acts on a map built before the turn. Only enforced for frozen episodes
            (a small live correction needs none); surfaced to the adapter via
            :attr:`WaypointFollower.settle_map_updates_required` and gated together
            with the ``map_ready`` input to :meth:`WaypointFollower.step`.
        yaw_capture_tol_m: Cross-track tolerance for the predictive ADVANCE gate
            (m): advance once going straight on the current heading would pass
            within this distance of the waypoint. Wired from the launch's
            ``yaw_acquisition_radius``. MUST be < ``pos_radius`` (validated): the
            gate promises only to pass within this distance, so a tolerance wider
            than the radius that counts as *reaching* a waypoint means none are
            ever acquired.
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
    min_motion_ticks: int = 2          # min consecutive ticks for forward OR yaw
    yaw_burst_max_ticks: int = 30
    yaw_burst_max_rad: float = radians(25.0)   # per-burst increment; split big turns
    yaw_coast_rad: float = radians(15.0)

    # Freeze-vs-live decision for the turn (see docstring). A turn larger than
    # freeze_yaw_thresh_rad freezes the map and then re-observes settle_map_updates
    # times while stopped; a smaller correction stays live and needs no re-observe.
    # freeze_on_rotation is the master switch: False => never freeze / never wait
    # to re-observe (the map stays live through every turn), for when depth is
    # trusted during rotation. Navigation (bursts/settle) is unaffected.
    freeze_on_rotation: bool = True
    freeze_yaw_thresh_rad: float = radians(20.0)
    settle_map_updates: int = 2

    # Graded-pulse / mid-burst-feedback / anti-deadlock yaw upgrades. ALL default
    # to today's behaviour (flags off / 0) so the change is inert until enabled;
    # the FALCON launch turns them on (see waypoint_follower_node + nav_stack.launch).
    yaw_graded_pulses: bool = False     # size each burst by tick count {2,4,6}, cap 6
    yaw_burst_grade_max_ticks: int = 6  # hard cap on a graded burst (req: 6 ticks)
    yaw_settle_dwell_per_tick: float = 0.0  # extra settle dwell per burst tick (inertia)
    yaw_burst_live_feedback: bool = False   # cut a burst short on confirmed live overshoot
    yaw_fb_reach_rad: float = 0.0       # remaining-in-burst-dir <= this counts as reached
    yaw_fb_confirm_ticks: int = 2       # consecutive reach ticks before cutting (noise guard)
    yaw_max_reversals: int = 0          # force ADVANCE after this many sign-flips (0 = off)
    yaw_accept_growth_rad: float = 0.0  # widen the accept band per reversal (anti-deadlock)

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

    def __post_init__(self) -> None:
        # The predictive gate only promises to pass within yaw_capture_tol_m of the
        # waypoint. If that is wider than the radius which COUNTS as reaching one,
        # the drone flies a route it never acquires: every waypoint is missed and
        # then retired late by passed_bearing_rad (100 deg, i.e. well behind it).
        # This flies visibly worse and the cause is invisible from the outside --
        # both numbers look individually reasonable, only the pair is wrong.
        if self.yaw_capture_tol_m >= self.pos_radius:
            raise ValueError(
                "yaw_capture_tol_m (%.2f m) must be < pos_radius (%.2f m): the "
                "ADVANCE gate would let the drone pass further from a waypoint "
                "than the distance that counts as reaching it, so waypoints would "
                "never be acquired -- only retired once %.0f deg behind."
                % (self.yaw_capture_tol_m, self.pos_radius,
                   degrees(self.passed_bearing_rad)))
