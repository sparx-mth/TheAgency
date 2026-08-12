"""Tuning for the thrust-to-throttle model.

The single number this package exists to know is **how much acceleration one
unit of throttle buys**, and the reason it is a whole package rather than a
constant is that the number moves. It falls as the battery sags, it changes with
air density, and it is different for every airframe. A controller that assumes
it stays put spends the flight fighting a bias on the vertical axis -- and
because that bias enters below the position loop, the integrator dutifully
learns it and every gain above is then tuned against a lie.
"""
from __future__ import annotations

from dataclasses import dataclass

from sparx_agency.core.control.constants import GRAVITY_MPS2


@dataclass(frozen=True)
class ThrustModelParams:
    """Bounds and learning rate for :class:`~.model.ThrustModel`.

    Attributes:
        hover_throttle: Starting guess for the throttle that holds a hover. Only
            a seed -- the estimator converges away from it within seconds of
            flight -- but a bad seed is a bad first few seconds, which on a
            takeoff is when it is least welcome.
        min_hover_throttle: Floor on the learned hover throttle. Below this the
            estimate implies a thrust-to-weight the airframe does not have, and
            is far more likely to be a bad measurement.
        max_hover_throttle: Ceiling on the same. Above it the aircraft has
            almost no climb authority left, which is a reason to distrust the
            estimate rather than act on it.
        min_throttle: Floor on the emitted throttle. Never zero: rotors that
            have spun down cannot produce a torque either, so the attitude
            controller underneath loses authority exactly when it is needed.
        max_throttle: Ceiling on the emitted throttle, kept below 1.0 so the
            mixer retains headroom to add the differential thrust that holds
            attitude. Commanding full collective leaves nothing to steer with.
        learn_tau_s: Time constant of the estimator, seconds. Long: battery sag
            happens over minutes, and a fast estimator tracks measurement noise
            into the one number the whole vertical axis depends on.
        min_observation_throttle: Observations below this throttle are ignored.
            The estimate divides by the commanded throttle, so a near-zero
            command turns a small acceleration error into an enormous scale
            error.
        max_observation_ratio: Reject an observation implying a scale more than
            this factor away from the current estimate. A propeller strike, a
            landing, or a rotor hitting a table all produce accelerations that
            have nothing to do with the thrust curve.
    """

    hover_throttle: float = 0.5
    min_hover_throttle: float = 0.2
    max_hover_throttle: float = 0.8
    min_throttle: float = 0.06
    max_throttle: float = 0.9
    learn_tau_s: float = 8.0
    min_observation_throttle: float = 0.15
    max_observation_ratio: float = 2.0

    def __post_init__(self):
        # type: () -> None
        """Validate the invariants the estimator relies on."""
        if not 0.0 < self.min_hover_throttle < self.max_hover_throttle < 1.0:
            raise ValueError("hover throttle bounds must satisfy 0 < min < max < 1")
        if not self.min_hover_throttle <= self.hover_throttle <= self.max_hover_throttle:
            raise ValueError("hover_throttle %r is outside its own bounds"
                             % (self.hover_throttle,))
        if not 0.0 < self.min_throttle < self.max_throttle <= 1.0:
            raise ValueError("throttle output bounds must satisfy 0 < min < max <= 1")
        if self.learn_tau_s <= 0.0:
            raise ValueError("learn_tau_s must be > 0, got %r" % (self.learn_tau_s,))
        if not 0.0 < self.min_observation_throttle < 1.0:
            raise ValueError("min_observation_throttle must be in (0, 1)")
        if self.max_observation_ratio <= 1.0:
            raise ValueError("max_observation_ratio must be > 1")

    @property
    def initial_full_scale_mps2(self):
        # type: () -> float
        """Specific thrust at full throttle implied by the seed hover throttle."""
        return GRAVITY_MPS2 / self.hover_throttle
