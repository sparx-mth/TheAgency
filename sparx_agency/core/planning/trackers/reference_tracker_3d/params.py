"""Tuning for the 3D reference tracker.

The controller has three parts, and they are tuned separately because they
answer different questions:

  * **Feed-forward** -- replay what the planner asked for. The reference already
    carries a velocity and an acceleration; sending them straight out is what
    makes the aircraft *fly the shape* of the trajectory rather than chase a
    point along it. :attr:`ReferenceTrackerParams.accel_lead_s` is how much of
    the planner's own intent survives.
  * **Velocity feedback** -- damp the difference between the planned velocity and
    the measured one. This is what an inner loop with lag needs, and it does most
    of the tracking work.
  * **Position feedback** -- cancel accumulated displacement, on a *clamped*
    error, so that being far from the plan never becomes flying straight at it.
    See :attr:`ReferenceTrackerParams.position_error_clamp_m`; that clamp is a
    collision property, not a tuning nicety.

The split, and the numbers, come from a controller that flew this planner's
trajectories well on another simulator: ``sjtu_drone``'s ``minsnap_tracker``,
``v_cmd = v_ff + Kp * clamp(e_p) + Kd * e_v``. The lesson worth carrying over is
that the damping term should act on the **measured velocity error** rather than
on the derivative of the position error -- the first is a clean signal the
simulator hands over exactly, the second is mostly noise and needs filtering
that costs phase.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from math import radians

from sparx_agency.core.common.types import KinematicLimits
from sparx_agency.core.planning.trackers.drift_pid.pid import PidGains


def _default_horizontal_pid():
    # type: () -> PidGains
    """Horizontal position loop, one instance per world axis.

    ``kd`` is zero on purpose: damping comes from the velocity-error term, which
    is measured rather than differenced. ``out_limit`` therefore bounds only the
    proportional-plus-integral contribution, and the clamp that actually keeps
    the aircraft off walls is
    :attr:`ReferenceTrackerParams.position_error_clamp_m`, applied to the error
    before this ever sees it.
    """
    return PidGains(kp=1.0, ki=0.10, kd=0.0, i_limit=0.20, d_tau_s=0.3,
                    deadband=0.01, out_limit=1.0)


def _default_vertical_pid():
    # type: () -> PidGains
    """Vertical position loop.

    A larger integral allowance than horizontal: a multirotor always has a
    standing thrust bias to learn, and altitude error is the one that ends a
    flight against a ceiling or a floor.
    """
    return PidGains(kp=1.2, ki=0.15, kd=0.0, i_limit=0.30, d_tau_s=0.3,
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
        horizontal_pid: Position-loop gains for the world x and y axes. Both axes
            share one tuning -- the airframe is symmetric in the horizontal plane
            and the world frame has no preferred direction.
        vertical_pid: Position-loop gains for the world z axis.
        limits: Ceilings on the *commanded* velocity and yaw rate. They bound the
            sum of every term, so ``max_speed_xy`` must be at least the planner's
            own velocity limit.
        accel_lead_s: How far ahead, in seconds, the reference acceleration is
            projected onto the velocity command (``v += a * accel_lead_s``). This
            is the lead that lets the aircraft start turning *before* a position
            error appears, which is the whole benefit of the planner emitting an
            acceleration at all. Roughly one inner-loop time constant; 0 disables
            the term.
        velocity_damping_xy: Gain on the horizontal velocity error
            (``v_ref - v_measured``). This is the term that closes the lag of an
            inner loop reaching a commanded velocity through tilt.

            Lower than the source controller's 0.4 because of what sits
            underneath: PX4's own velocity controller is already a damped loop,
            so this one is cascaded on top of it and a high gain buys oscillation
            rather than tracking. A drone commanded directly, as in the Gazebo
            original, has no such inner loop to fight.
        velocity_damping_z: The same for the vertical axis.
        position_error_clamp_m: Position error is clamped to this, per axis,
            **before** the position loop sees it.

            This is the difference between rounding an obstacle and flying
            through it. The position term points straight at the reference, and a
            trajectory routing around a wall puts its reference on the far side
            of that wall for a second or two; without a clamp, an aircraft that
            has fallen behind turns that displacement into a beeline. Clamping
            the *error* rather than the output keeps the correction pointed the
            right way and bounds only how hard it pulls, while the velocity
            damping -- unclamped, and what actually tracks -- carries on working.

            Set to a metre, not to the half-metre the source controller used,
            and the difference is the airframe. That controller commanded a
            Gazebo drone directly, so its inner loop barely lagged and the
            aircraft was never far enough behind for the clamp to bind. Here
            PX4's velocity controller reaches a commanded velocity through tilt,
            the aircraft genuinely does fall a metre or two behind, and a
            half-metre clamp starves the recovery: measured, it took the mean
            tracking error to 3 m and ended the flight in twelve seconds. A
            metre leaves the normal regime untouched and still bounds the far
            field, which is all it was ever for.
        command_smoothing_alpha: Weight of the new command in the exponential
            smoothing of the output; 1.0 disables it. A multirotor answers a step
            in commanded velocity with a step in tilt, so smoothing costs a
            little phase and buys a camera that is not being whipped around --
            which, for a planner that aims its sensor at frontiers, is part of
            the job rather than a cosmetic.
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
    velocity_damping_xy: float = 0.25
    velocity_damping_z: float = 0.2
    position_error_clamp_m: float = 1.0
    command_smoothing_alpha: float = 0.5
    integral_band_m: float = 0.5
    max_position_error_m: float = 2.0
    reference_timeout_s: float = 1.0
    yaw_rate_margin: float = 1.5

    def __post_init__(self):
        # type: () -> None
        """Validate the invariants the control law relies on."""
        if self.accel_lead_s < 0.0:
            raise ValueError("accel_lead_s must be >= 0, got %r" % (self.accel_lead_s,))
        for name in ("velocity_damping_xy", "velocity_damping_z"):
            if getattr(self, name) < 0.0:
                raise ValueError("%s must be >= 0, got %r" % (name, getattr(self, name)))
        if self.position_error_clamp_m <= 0.0:
            raise ValueError("position_error_clamp_m must be > 0, got %r"
                             % (self.position_error_clamp_m,))
        if not 0.0 < self.command_smoothing_alpha <= 1.0:
            raise ValueError("command_smoothing_alpha must be in (0, 1], got %r"
                             % (self.command_smoothing_alpha,))
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
