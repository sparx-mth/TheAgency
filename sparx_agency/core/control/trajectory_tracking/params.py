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

There is no speed limit here, and none is needed. The plan's own velocity sets
the speed; the damping term opposes any excess over it, so the closed loop
settles at the planned speed rather than at some ceiling. A tracker with a speed
clamp *below* the planner's limit falls permanently behind, which is a failure
mode this shape simply does not have.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from math import radians

from sparx_agency.core.control.flatness.limits import AccelerationLimits
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
    max_catchup_speed: float = 0.5
    position_error_clamp_m: float = 1.0
    integral_band_m: float = 0.5
    max_position_error_m: float = 2.0
    max_trajectory_age_s: float = 2.0
    max_yaw_rate: float = radians(90.0)
    yaw_rate_margin: float = 1.5

    def __post_init__(self):
        # type: () -> None
        """Validate the invariants the control law relies on."""
        for name in ("velocity_damping_xy", "velocity_damping_z", "schedule_gain_per_s"):
            if getattr(self, name) < 0.0:
                raise ValueError("%s must be >= 0, got %r" % (name, getattr(self, name)))
        for name in ("max_catchup_speed", "position_error_clamp_m",
                     "integral_band_m", "max_position_error_m",
                     "max_trajectory_age_s", "max_yaw_rate", "yaw_rate_margin"):
            if getattr(self, name) <= 0.0:
                raise ValueError("%s must be > 0, got %r" % (name, getattr(self, name)))
