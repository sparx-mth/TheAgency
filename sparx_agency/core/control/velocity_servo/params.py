"""Tuning for the velocity backend.

The law has four terms and only one of them is a gain anybody has to guess:

* **Velocity feedforward** -- the plan's own velocity, sent straight out. On an
  autopilot whose velocity loop has unit DC gain this term alone flies the
  trajectory in steady state, and the rest is there for the transients.
* **Inverse-plant lead** -- ``tau * a_plan``. Not a tuning parameter: ``tau`` is
  *measured* off the airframe and ``a_plan`` is read exactly from the B-spline.
  It is what cancels the autopilot's own lag, and it is the single term that
  distinguishes this from the P + feedforward controller it replaces.
* **Position feedback** -- closes the loop on where the aircraft actually is.
  The only genuinely tuned gain, and it is bounded from above by the plant's
  transport delay rather than by preference.
* **Integral** -- learns the standing biases (wind, a trim error, a battery
  sagging), and only near the curve.

There is deliberately **no velocity damping term**. That is the difference
between this backend and the acceleration one, and getting it wrong is the
classic failure of bolting an outer loop onto an autopilot: the airframe already
contains a velocity loop with unit DC gain, so a second one closed around it
double-counts the same feedback, drops the phase margin and rings. Damping here
comes from the plant, which is exactly what the plant model is for.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from sparx_agency.core.control.reference.params import ReferenceParams
from sparx_agency.core.control.velocity_servo.limits import VelocityLimits
from sparx_agency.core.control.velocity_servo.plant import VelocityPlant
from sparx_agency.core.planning.trackers.drift_pid.pid import PidGains


def _default_horizontal_pid():
    # type: () -> PidGains
    """Horizontal position loop, one instance per world axis.

    Gains are in **velocity** units -- a metre of error asks for 1.2 m/s -- which
    is an order of magnitude smaller than the acceleration backend's, and that
    is the point rather than an oversight. This loop closes around an autopilot
    that already takes half a second to reach a commanded velocity and a fifth
    of a second to notice the request; 1.2 sits just under the
    ``1 / (3 * delay)`` bound that keeps 20 degrees of phase margin at
    crossover.

    ``kd`` is zero because there is no clean derivative available: differencing
    the position error at the state-estimate rate is mostly noise, and the
    damping this loop needs is already in the plant.
    """
    return PidGains(kp=1.2, ki=0.25, kd=0.0, i_limit=0.4, d_tau_s=0.3,
                    deadband=0.02, out_limit=1.5)


def _default_vertical_pid():
    # type: () -> PidGains
    """Vertical position loop.

    Stiffer than horizontal because the axis is faster -- thrust changes without
    waiting for the airframe to rotate -- and because altitude is the axis that
    ends a flight against a floor or a ceiling.
    """
    return PidGains(kp=1.8, ki=0.4, kd=0.0, i_limit=0.5, d_tau_s=0.3,
                    deadband=0.02, out_limit=1.5)


@dataclass(frozen=True)
class VelocityServoParams:
    """Gains, limits and plant model for the velocity backend.

    Attributes:
        horizontal_pid: Position-loop gains for world x and y. Both axes share
            one tuning: the airframe is symmetric in the horizontal plane and
            the world frame has no preferred direction.
        vertical_pid: Position-loop gains for world z.
        plant: The autopilot's measured velocity response. **Measure this**; the
            defaults are a representative airframe and are wrong for any
            specific one. The lead term is only as good as ``time_constant_s``
            and the achievable bandwidth is set by ``delay_s``.
        limits: Speed and slew ceilings.
        reference: How the reference point is chosen from the trajectory.
        yaw_gain: Proportional gain on heading error, rad/s per rad.
        yaw_deadband_rad: Heading error below which the proportional term is not
            applied, so the aircraft does not dither about a heading it has
            already reached.
        use_feedforward_lead: Apply the ``tau * a_plan`` inverse-plant term.
            On by default. Off reproduces a plain P + velocity-feedforward
            controller, which is what the previous stack flew and what the
            improvement is measured against -- keep it as a comparison, not as
            a fallback.
        predict_reference: Read the feedforward off the plan at
            ``plant.feedforward_lead_s`` in the future, so the aircraft arrives
            on schedule rather than one transport delay late.

            Applied to the **feedforward only**, never to the feedback. Leaning
            the whole reference forward is the tempting version and it is wrong:
            it settles at a constant position error of ``lead * speed``, which
            the position gain turns into a standing forward push. Predicting
            only the feedforward has no such equilibrium error, because the
            feedback term still measures against where the plan says the
            aircraft should be *now*.
        schedule_gain_per_s: Along-track catch-up, in m/s of extra commanded
            speed per second of schedule lag. Added along the tangent only, so
            unlike a lookahead it can never pull the aircraft across a corner.
        max_catchup_speed: Ceiling on that catch-up, m/s. Bounded well under
            cruise so a large lag -- which usually means something went wrong
            rather than that the aircraft is merely late -- cannot turn into a
            dash along the route.
        max_overspeed: How much faster than the plan the aircraft may be
            commanded, m/s. A **clearance** setting, not a comfort one: FALCON
            checks its trajectory against the map at the speed it planned, with
            a fixed margin around it, and flying the same curve faster spends
            that margin on stopping distance. Must be at least
            ``max_catchup_speed`` or the two fight each other.
        integral_band_m: Per-axis error within which the integrator may learn.
            Outside it the axis is not holding a bias, it is *travelling*, and
            integrating that charges a correction which arrives after the error
            is gone.
        position_error_clamp_m: Position error is clamped to this before the
            position loop sees it, as a horizontal pair plus an independent
            vertical. Clamping the *error* rather than the output keeps the
            correction pointed at the reference and bounds only how hard it
            pulls -- a trajectory routing around a wall puts its reference on
            the far side of that wall for a second or two, and an aircraft that
            has fallen behind must not turn that into a beeline.
        max_position_error_m: Error beyond which ``diverged`` is set. Advisory.
        hold_speed_xy: Horizontal ceiling while holding station, m/s. A hold has
            no plan, so it has no planned speed to be relative to, and an
            aircraft returning to a latched point has no business doing so at
            cruise -- it is the case where it is most likely to be near whatever
            pushed it off.
    """

    horizontal_pid: PidGains = field(default_factory=_default_horizontal_pid)
    vertical_pid: PidGains = field(default_factory=_default_vertical_pid)
    plant: VelocityPlant = field(default_factory=VelocityPlant)
    limits: VelocityLimits = field(default_factory=VelocityLimits)
    reference: ReferenceParams = field(default_factory=ReferenceParams)
    yaw_gain: float = 1.5
    yaw_deadband_rad: float = 0.01
    use_feedforward_lead: bool = True
    predict_reference: bool = True
    schedule_gain_per_s: float = 0.6
    max_catchup_speed: float = 0.15
    max_overspeed: float = 0.25
    integral_band_m: float = 0.5
    position_error_clamp_m: float = 1.0
    max_position_error_m: float = 2.0
    hold_speed_xy: float = 0.5

    def __post_init__(self):
        # type: () -> None
        """Validate the invariants the control law relies on."""
        if self.yaw_gain < 0.0:
            raise ValueError("yaw_gain must be >= 0, got %r" % (self.yaw_gain,))
        if self.yaw_deadband_rad < 0.0:
            raise ValueError("yaw_deadband_rad must be >= 0, got %r"
                             % (self.yaw_deadband_rad,))
        if self.schedule_gain_per_s < 0.0:
            raise ValueError("schedule_gain_per_s must be >= 0, got %r"
                             % (self.schedule_gain_per_s,))
        if self.max_overspeed < self.max_catchup_speed:
            raise ValueError(
                "max_overspeed (%r) must be at least max_catchup_speed (%r), or the "
                "speed ceiling cancels the catch-up it is supposed to allow"
                % (self.max_overspeed, self.max_catchup_speed))
        for name in ("max_catchup_speed", "max_overspeed", "integral_band_m",
                     "position_error_clamp_m", "max_position_error_m", "hold_speed_xy"):
            if getattr(self, name) <= 0.0:
                raise ValueError("%s must be > 0, got %r" % (name, getattr(self, name)))

    def yaw_limits(self):
        # type: () -> tuple
        """``(max_rate, max_accel)`` for the heading servo, from ``limits``."""
        return self.limits.max_yaw_rate, self.limits.max_yaw_accel
