"""Tuning for the 3D reference tracker.

The controller has two halves and they are tuned separately, because they answer
different questions:

  * **Feed-forward** -- replay what the planner asked for. The reference already
    carries a velocity and an acceleration; sending them straight out is what
    makes the aircraft *fly the shape* of the trajectory rather than chase a
    point along it. :attr:`ReferenceTrackerParams.accel_lead_s` is how much of
    the planner's own intent survives.
  * **Feedback** -- cancel the difference between where the planner thinks the
    aircraft is and where it actually is. This half exists only because there is
    physics: a planner validated in a geometry-only simulator never needed it,
    because there the commanded state *was* the state.

The feedback limits are deliberately smaller than the flight speeds. A
correction that can out-run the trajectory is a correction that flies the
aircraft, and then the planner's dynamic-feasibility guarantees no longer say
anything about what the airframe does.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from math import radians

from sparx_agency.core.common.types import KinematicLimits
from sparx_agency.core.planning.trackers.drift_pid.pid import PidGains


def _default_horizontal_pid():
    # type: () -> PidGains
    """Horizontal station-keeping loop, one instance per world axis.

    ``kp`` is the dominant term: a metre of lag asks for about a metre per second
    of catch-up. The integral is small and tightly capped -- its job is the
    standing bias (a draught, a trim error, a heavy battery), not the transient
    lag, which is the P term's.

    ``out_limit`` deserves its own paragraph, because it is the one number here
    that was tuned against a real airframe rather than reasoned about, and the
    obvious reasoning is wrong.

    The correction points straight at the reference, so a *large* cap makes a
    displaced aircraft turn hard toward the plan -- and when the plan is rounding
    an obstacle, "toward the plan" can be through it. That argues for a small
    cap. But a small cap was measured on Isaac Sim and made things worse, not
    better: dropping it from 1.0 to 0.35 m/s against a 0.45 m/s plan took the mean
    tracking error from 0.4-1.4 m to 2.0-2.7 m, and an aircraft that is
    permanently two metres off the plan is in more danger than one that
    occasionally turns sharply back onto it. Convergence *is* the safety property.

    So it stays at 1.0. If an aircraft is cutting corners into obstacles, the dial
    to turn is the planner's speed (see
    ``tasks/planning/falcon_pegasus/adapter/launch/``), not this one.
    """
    return PidGains(kp=1.0, ki=0.10, kd=0.05, i_limit=0.20, d_tau_s=0.3,
                    deadband=0.01, out_limit=1.0)


def _default_vertical_pid():
    # type: () -> PidGains
    """Vertical loop.

    Tighter than horizontal and with a larger integral allowance: altitude error
    is the one that ends a flight against a ceiling or a floor, and a multirotor
    always has a standing thrust bias to learn.
    """
    return PidGains(kp=1.2, ki=0.15, kd=0.05, i_limit=0.30, d_tau_s=0.3,
                    deadband=0.01, out_limit=0.8)


def _default_limits():
    # type: () -> KinematicLimits
    """Command ceilings for an indoor multirotor tracking a planned trajectory.

    ``max_speed_xy`` must sit *above* the planner's own velocity limit, or the
    tracker clips trajectories the planner considered feasible and the aircraft
    falls permanently behind.
    """
    return KinematicLimits(max_speed_xy=1.5, max_speed_z=0.8,
                           max_yaw_rate=radians(90.0),
                           max_accel_xy=2.0, max_accel_z=1.5)


@dataclass(frozen=True)
class ReferenceTrackerParams:
    """Gains, limits and freshness rules for :class:`~.tracker.ReferenceTracker3D`.

    Attributes:
        horizontal_pid: Feedback gains for the world x and y axes. Both axes share
            one tuning -- the airframe is symmetric in the horizontal plane and the
            world frame has no preferred direction.
        vertical_pid: Feedback gains for the world z axis.
        limits: Ceilings on the *commanded* velocity and yaw rate. They bound the
            sum of feed-forward and feedback, so ``max_speed_xy`` must be at least
            the planner's own velocity limit.
        accel_lead_s: How far ahead, in seconds, the reference acceleration is
            projected onto the velocity command (``v += a * accel_lead_s``). This
            is the lead that lets the aircraft start turning *before* a position
            error appears, which is the whole benefit of the planner emitting an
            acceleration at all. Roughly one inner-loop time constant; 0 disables
            the term.
        integral_band_m: Per-axis error magnitude within which the integrator is
            allowed to learn (integral separation). Outside it the axis is not
            holding a standing bias, it is *travelling* -- a large error is the
            aircraft catching up, and integrating it charges a correction that
            arrives after the error is gone and overshoots by exactly that much.
            Measured: a 1 m step overshot 9 cm without this and 1 cm with it.
        max_position_error_m: Distance between the reference and the aircraft
            beyond which :attr:`~.types.TrackedSetpoint.diverged` is set. Purely
            advisory -- the tracker keeps trying -- but it is how a mission
            notices the aircraft is no longer flying the plan.
        reference_timeout_s: Age beyond which a reference is stale and the tracker
            holds the last commanded position instead of following it. The planner
            republishes continuously; silence means it died, and flying on the last
            velocity would be the worst possible response.
        yaw_rate_margin: Multiplier applied to ``limits.max_yaw_rate`` when slewing
            the commanded heading. Above 1.0 so that in normal operation the
            planner's own (already rate-limited) yaw trajectory passes through
            untouched, and the slew engages only when the aircraft has fallen far
            enough behind that the reference heading jumped.
    """

    horizontal_pid: PidGains = field(default_factory=_default_horizontal_pid)
    vertical_pid: PidGains = field(default_factory=_default_vertical_pid)
    limits: KinematicLimits = field(default_factory=_default_limits)
    accel_lead_s: float = 0.25
    integral_band_m: float = 0.5
    max_position_error_m: float = 2.0
    reference_timeout_s: float = 1.0
    yaw_rate_margin: float = 1.5

    def __post_init__(self):
        # type: () -> None
        """Validate the invariants the control law relies on."""
        if self.accel_lead_s < 0.0:
            raise ValueError("accel_lead_s must be >= 0, got %r" % (self.accel_lead_s,))
        if self.integral_band_m <= 0.0:
            raise ValueError("integral_band_m must be > 0, got %r"
                             % (self.integral_band_m,))
        if self.max_position_error_m <= 0.0:
            raise ValueError("max_position_error_m must be > 0, got %r"
                             % (self.max_position_error_m,))
        if self.reference_timeout_s <= 0.0:
            raise ValueError("reference_timeout_s must be > 0, got %r"
                             % (self.reference_timeout_s,))
        if self.yaw_rate_margin <= 0.0:
            raise ValueError("yaw_rate_margin must be > 0, got %r"
                             % (self.yaw_rate_margin,))
