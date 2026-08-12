"""Tuning for the outer loop.

Four terms, tuned separately because they answer different questions:

* **Feedforward** -- the trajectory's own acceleration, sent straight out. It is
  a *lookup*, not a calculation: the value is already in the curve, exact and
  with no lag at all, which is why it can act before an error exists. On a plan
  the airframe can fly, this term does most of the work and the other three are
  small.
* **Position feedback** -- pulls back onto the curve, on a *clamped* error, so
  being far from the plan never becomes flying straight at it.
* **Velocity feedback** -- matches the plan's speed. This is what damps the
  loop, and it is deliberately not derived from the position error: measured
  velocity is a clean signal the estimator hands over, while differentiating
  position is mostly noise and the filtering it needs costs phase.
* **Integral** -- learns the standing biases, and only near the curve.

The speed limit here is **relative to the plan**, not absolute, and that
distinction is the whole point. An absolute clamp below the planner's own limit
leaves the aircraft permanently behind — a real failure mode, and the reason
this shape avoided one at first. But "no ceiling at all" is also wrong: the
damping term balances the position term only once the aircraft is
``kp * clamp / kd`` faster than the plan, which measured out at 42% of a flight
above 1.1 m/s on a 0.6 m/s plan, and FALCON's clearance is computed for the
speed FALCON planned. So the ceiling is ``planned speed + max_overspeed``, which
moves with the plan and can never hold the aircraft back.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from math import radians

from sparx_agency.core.control.flatness.limits import AccelerationLimits
from sparx_agency.core.control.reference.params import ReferenceParams
from sparx_agency.core.planning.trackers.drift_pid.pid import PidGains
from sparx_agency.core.planning.trajectories.bspline.projection import ProjectionParams


def _default_horizontal_pid():
    # type: () -> PidGains
    """Horizontal position loop, one instance per world axis.

    ``kd`` is zero on purpose: damping comes from the measured velocity error,
    not from differentiating position. Gains are in acceleration units -- a
    metre of error asks for 2 m/s^2 -- and with ``velocity_damping_xy`` at 2.2
    the pair is a slightly under-damped second-order system, which settles
    faster than critical without overshooting into a wall.
    """
    return PidGains(kp=2.0, ki=0.4, kd=0.0, i_limit=1.5, d_tau_s=0.3,
                    deadband=0.01, out_limit=6.0)


def _default_vertical_pid():
    # type: () -> PidGains
    """Vertical position loop.

    Stiffer than horizontal, with a larger integral allowance. A multirotor
    always has a standing thrust bias to learn, and altitude is the axis that
    ends a flight against a floor or a ceiling.
    """
    return PidGains(kp=3.0, ki=0.8, kd=0.0, i_limit=2.5, d_tau_s=0.3,
                    deadband=0.01, out_limit=6.0)


@dataclass(frozen=True)
class TrajectoryTrackerParams:
    """Gains, limits and freshness rules for the outer loop.

    Attributes:
        horizontal_pid: Position-loop gains for world x and y. Both axes share
            one tuning: the airframe is symmetric in the horizontal plane and
            the world frame has no preferred direction.
        vertical_pid: Position-loop gains for world z.
        limits: Acceleration ceilings. Applied here as well as in the flatness
            stage -- not redundantly, but so the integrator can be told when the
            command it is contributing to has saturated.
        projection: Nearest-point search settings.
        use_projection: Track the nearest point on the curve rather than the
            point at the current time.

            On by default, but not for the reason usually given: measured, it
            does *not* beat time indexing through a bend on a trajectory
            FALCON's optimiser has already made feasible. It wins when the
            aircraft has been displaced in time from the plan -- a hold while
            FALCON replans, a stall -- where a time-indexed reference has moved
            seconds down the route and pulls the aircraft across everything
            between. Off reproduces the time-indexed behaviour, and is worth
            keeping to measure against rather than as a fallback.
        velocity_damping_xy: Gain on the horizontal velocity error
            (``v_ref - v_measured``), in m/s^2 per m/s. With the position gain
            above this sets the damping ratio.
        velocity_damping_z: The same for the vertical axis.
        schedule_gain_per_s: Along-track catch-up, in m/s of extra target speed
            per second of schedule lag.

            Projection deliberately throws the plan's *timing* away -- it asks
            where the aircraft is on the curve, not where it should be by now --
            and something has to put the timing back, because FALCON replans
            from its own previous curve at ``now + planning_time``. An aircraft
            perfectly on the path but two seconds behind schedule gets a new
            trajectory starting two seconds' worth of flying ahead of it, and
            that gap is a step.

            The catch-up is added to the *target velocity along the tangent*,
            never to the position error. That is the whole trick: an along-track
            term cannot cut a corner, because it only ever pushes the aircraft
            in the direction the path already goes. It is what lets projection
            remove corner cutting without giving up schedule.
        max_catchup_speed: Ceiling on that catch-up, m/s. Bounded well under
            cruise so a large lag -- which usually means something went wrong
            rather than that the aircraft is merely late -- cannot turn into a
            dash along the route.

            **Small on purpose, and it was not always.** At 0.5 it saturated for
            most of every simulated flight, because on Isaac Sim the lag is
            largely not a schedule deficit at all: the simulator runs at about
            0.7x real time while FALCON plans on the wall clock, so the aircraft
            is structurally behind a deadline it cannot meet (see
            ``falcon_pegasus/link/sim_clock.py``). Chasing that fictitious lag
            drove the aircraft at 1.1-1.3 m/s along a route FALCON had cleared
            for 0.6 -- measured, 44% of one flight above the governor's own
            ceiling -- and it is how two soak rounds ended against a wall.

            Being late is benign; being fast in a corridor sized for a slower
            aircraft is not. The same reasoning as ``_diagnose``: along-track
            error is the cheap one, cross-track is what hits things.

            A further cut to 0.10 (with ``max_overspeed`` 0.12) was tried and
            reverted. It did reduce the overspeed -- the aircraft had been
            sitting AT its ceiling for 54% of the ticks where the plan had a
            speed, running 0.74 m/s against a 0.52 m/s plan -- but it cost 24%
            of the coverage RATE on the stub (1077 -> 822 m3 in 180 s), and the
            bar has to be met inside a fixed flight budget. The stub cannot
            price the other side of that trade because it has no collisions, so
            there was no evidence the safety gain paid for the coverage loss.
            Do not tune this pair again on stub numbers alone.
        max_overspeed: How much faster than the plan the aircraft may be flown,
            m/s. This is a **clearance** setting, not a comfort one.

            FALCON's trajectory is checked against the map at the speed FALCON
            planned, with `bspline_opt/safe_distance` of clearance around it.
            Fly the same curve faster and that margin is spent on stopping
            distance: the airframe is 0.7 m across, the clutter in this building
            sits at cruise height, and the difference between rounding an
            obstacle and hitting it is a few tenths of a metre.

            It needs a ceiling because the position loop has no natural one. A
            metre of error asks for ``kp * clamp`` of acceleration, which the
            damping term balances only once the aircraft is
            ``kp * clamp / kd`` **faster than the plan** -- about 0.9 m/s with
            these gains. Measured on a 0.6 m/s plan before this existed: 42% of
            the flight above 1.1 m/s, 29% above 1.5, peaking at 2.85, and the
            flight ended embedded in a desk at cruise height.

            The only sanctioned reason to exceed the plan's speed is recovering
            schedule, so this must be at least ``max_catchup_speed`` or the two
            fight each other.
        position_error_clamp_m: Position error is clamped to this, per axis,
            **before** the position loop sees it.

            This is the difference between rounding an obstacle and flying
            through it. The position term points straight at the reference, and
            a trajectory routing around a wall puts its reference on the far
            side of that wall for a second or two; without a clamp, an aircraft
            that has fallen behind turns that displacement into a beeline.
            Clamping the *error* rather than the output keeps the correction
            pointed the right way and bounds only how hard it pulls.

            Less load-bearing than it was in the velocity-cut controller,
            because projection means the reference is no longer on the far side
            of anything -- but it costs nothing and the two failure modes it
            guards are not the same.
        integral_band_m: Per-axis error within which the integrator may learn.
            Outside it the axis is not holding a bias, it is *travelling*, and
            integrating that charges a correction which arrives after the error
            is gone.
        max_position_error_m: Error beyond which ``diverged`` is set. Advisory.
        max_trajectory_age_s: How long past a trajectory's end the aircraft will
            keep flying to its final point before giving up and holding station
            where it is. Covers the normal gap between replans; a longer silence
            means the planner died, and flying to a stale endpoint then is a
            guess.
        max_yaw_rate: Ceiling on how fast the *commanded* heading may slew.
        yaw_rate_margin: Multiplier on that ceiling, above 1.0 so a planner
            respecting its own yaw-rate limit passes through untouched. The slew
            then bites only when the reference heading jumps, which happens when
            the aircraft has fallen far enough behind that FALCON replanned from
            somewhere else.
    """

    horizontal_pid: PidGains = field(default_factory=_default_horizontal_pid)
    vertical_pid: PidGains = field(default_factory=_default_vertical_pid)
    limits: AccelerationLimits = field(default_factory=AccelerationLimits)
    projection: ProjectionParams = field(default_factory=ProjectionParams)
    use_projection: bool = True
    velocity_damping_xy: float = 2.2
    velocity_damping_z: float = 3.0
    schedule_gain_per_s: float = 0.6
    max_catchup_speed: float = 0.15
    max_overspeed: float = 0.25
    position_error_clamp_m: float = 1.0
    integral_band_m: float = 0.5
    max_position_error_m: float = 2.0
    max_trajectory_age_s: float = 2.0
    max_yaw_rate: float = radians(90.0)
    yaw_rate_margin: float = 1.5

    # ── standing-force feedforward ───────────────────────────────────────
    # Drag opposes velocity, so holding a cruise speed costs a permanent
    # acceleration nothing else in this law supplies. The damping term can only
    # shrink the resulting velocity deficit (deficit = drag / kd, about
    # 0.14 m/s at 1 m/s on the measured airframe), and the integrator that
    # could remove it is gated off outside integral_band_m -- precisely when
    # the aircraft is furthest from the plan. Feeding the measured drag curve
    # forward removes the bias structurally instead of asking feedback to
    # fight a force we can predict.
    #
    # Computed from the PLANNED velocity, deliberately: basing it on measured
    # velocity would make it positive feedback on speed. Zero by default
    # because the numbers are a property of one airframe -- the Pegasus Iris
    # measures 0.176*v + 0.121 m/s^2 (fitted from the residual between
    # specific force and thrust axis over 501 samples of steady cruise) -- and
    # core/ carries no platform's constants.
    drag_per_mps: float = 0.0
    drag_offset_mps2: float = 0.0

    # ── attitude-lag lead ────────────────────────────────────────────────
    # The chain commands an attitude and the airframe reaches it a time
    # constant later (PX4's attitude+rate response, ~0.18 s here). That is
    # transport delay: at 1 m/s it is ~18 cm of tracking error no gain can
    # remove, because reacting faster to the past does not shorten the future.
    # Sampling the FEEDFORWARD acceleration and jerk this far ahead of the
    # reference commands, now, the attitude the plan wants when the airframe
    # will actually have it. Only the feedforward leads -- position and
    # velocity feedback stay at the reference, correcting errors that exist
    # rather than errors that are predicted.
    attitude_lead_s: float = 0.0

    def reference_params(self):
        # type: () -> ReferenceParams
        """The subset of this tuning that describes the *plan* rather than the airframe.

        Kept as three flat fields here rather than as a nested dataclass so the
        parameter surface every caller already constructs does not change; this
        method is the seam that hands them to the shared
        :class:`~sparx_agency.core.control.reference.feed.TrajectoryFeed`, which
        the velocity backend uses too. The two backends disagree about what to
        command and must never disagree about what the plan says.
        """
        return ReferenceParams(projection=self.projection,
                               use_projection=self.use_projection,
                               max_trajectory_age_s=self.max_trajectory_age_s)

    def __post_init__(self):
        # type: () -> None
        """Validate the invariants the control law relies on."""
        for name in ("velocity_damping_xy", "velocity_damping_z", "schedule_gain_per_s"):
            if getattr(self, name) < 0.0:
                raise ValueError("%s must be >= 0, got %r" % (name, getattr(self, name)))
        for name in ("drag_per_mps", "drag_offset_mps2", "attitude_lead_s"):
            if getattr(self, name) < 0.0:
                raise ValueError("%s must be >= 0, got %r" % (name, getattr(self, name)))
        if self.max_overspeed < self.max_catchup_speed:
            raise ValueError(
                "max_overspeed (%r) must be at least max_catchup_speed (%r), or the "
                "speed ceiling cancels the catch-up it is supposed to allow"
                % (self.max_overspeed, self.max_catchup_speed))
        for name in ("max_overspeed", "max_catchup_speed", "position_error_clamp_m",
                     "integral_band_m", "max_position_error_m",
                     "max_trajectory_age_s", "max_yaw_rate", "yaw_rate_margin"):
            if getattr(self, name) <= 0.0:
                raise ValueError("%s must be > 0, got %r" % (name, getattr(self, name)))
