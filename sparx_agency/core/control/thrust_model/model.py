"""How much throttle a wanted acceleration costs, learned in flight.

Everything above this module reasons in metres per second squared. The autopilot
underneath wants a number between 0 and 1. The conversion between them is one
scalar -- the specific thrust available at full throttle -- and getting it wrong
is not a small error: a 10% mistake is a persistent 1 m/s^2 bias on the vertical
axis, which the position integrator absorbs and then hides, leaving every gain
above tuned against a lie.

So it is measured rather than assumed. The aircraft is continuously running the
experiment anyway: it commands a throttle and something accelerates. Comparing
the two gives the scale, and filtering that comparison over seconds tracks the
battery sagging without chasing vibration.

The measurement is the **specific force along the thrust axis**, not the
vertical acceleration -- a tilted aircraft producing 12 m/s^2 of thrust shows
only part of it as climb, and using the vertical component would make the
estimate a function of how hard the aircraft happens to be cornering.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from sparx_agency.core.control.constants import GRAVITY_MPS2
from sparx_agency.core.control.thrust_model.params import ThrustModelParams

_MAX_WARMUP_WEIGHT = 0.5
"""Ceiling on the running-mean weight, so no one early sample defines the scale."""


def specific_force_along(acceleration_world, body_z_world):
    # type: (object, object) -> float
    """Thrust per unit mass, inferred from a measured acceleration.

    An accelerometer measures specific force -- thrust minus gravity, in body
    axes. Going the other way, an aircraft whose measured world acceleration is
    ``a`` must be producing ``a + g`` worth of thrust, and the part of that
    along its own thrust axis is what its rotors actually delivered.

    Args:
        acceleration_world: Measured world ``(ax, ay, az)``, m/s^2, +z up.
        body_z_world: The aircraft's thrust axis in world coordinates, unit
            length.

    Returns:
        Specific thrust, m/s^2. Negative results are physically impossible and
        mean the inputs disagree; the caller should reject them.
    """
    acceleration = np.asarray(acceleration_world, dtype=float).reshape(3)
    axis = np.asarray(body_z_world, dtype=float).reshape(3)
    total = acceleration + np.array([0.0, 0.0, GRAVITY_MPS2], dtype=float)
    return float(np.dot(total, axis))


class ThrustModel:
    """Converts specific thrust to throttle, learning the scale as it flies.

    Stateful: the learned scale is the whole point. :meth:`reset` when the
    aircraft stops flying, so a scale learned in ground effect or during a
    landing does not open the next flight.

    Args:
        params: Bounds and learning rate.
    """

    def __init__(self, params=None):
        # type: (Optional[ThrustModelParams]) -> None
        self.params = params or ThrustModelParams()
        self._full_scale = self.params.initial_full_scale_mps2
        self._observations = 0

    def reset(self):
        # type: () -> None
        """Forget what was learned and return to the seed hover throttle."""
        self._full_scale = self.params.initial_full_scale_mps2
        self._observations = 0

    @property
    def full_scale_mps2(self):
        # type: () -> float
        """Specific thrust the airframe produces at full throttle, m/s^2."""
        return self._full_scale

    @property
    def hover_throttle(self):
        # type: () -> float
        """Throttle currently believed to hold a hover.

        The single most useful number to log from this package: it should sit
        somewhere near the seed, drift slowly upward as the battery sags, and
        never jump. A jump means an observation got through that should not
        have.
        """
        return GRAVITY_MPS2 / self._full_scale

    @property
    def observations(self):
        # type: () -> int
        """How many measurements the estimate has accepted."""
        return self._observations

    def normalized(self, specific_thrust_mps2):
        # type: (float) -> float
        """Throttle to command for a wanted specific thrust.

        Args:
            specific_thrust_mps2: Thrust over mass, m/s^2, from the flatness
                conversion. Always positive.

        Returns:
            Throttle in ``[min_throttle, max_throttle]``.
        """
        throttle = float(specific_thrust_mps2) / self._full_scale
        return max(self.params.min_throttle, min(self.params.max_throttle, throttle))

    def observe(self, commanded_throttle, acceleration_world, body_z_world, dt):
        # type: (float, object, object, float) -> bool
        """Fold one throttle-versus-acceleration measurement into the estimate.

        Args:
            commanded_throttle: The throttle that was actually sent, in [0, 1].
            acceleration_world: The measured world acceleration that resulted.
            body_z_world: The aircraft's measured thrust axis, unit length.
            dt: Seconds since the previous observation. Must be > 0.

        Returns:
            True if the observation was accepted. False means it was rejected as
            implausible, which is normal and not an error -- a landing, a
            propeller strike, or simply too little throttle to divide by.

        Raises:
            ValueError: If ``dt`` is not positive.
        """
        if dt <= 0.0:
            raise ValueError("ThrustModel.observe: dt must be > 0, got %r" % (dt,))
        throttle = float(commanded_throttle)
        if throttle < self.params.min_observation_throttle:
            return False
        measured = specific_force_along(acceleration_world, body_z_world)
        if measured <= 0.0:
            return False

        observed_scale = measured / throttle
        ratio = observed_scale / self._full_scale
        if ratio > self.params.max_observation_ratio or ratio < 1.0 / self.params.max_observation_ratio:
            return False

        updated = self._full_scale + self._weight(dt) * (observed_scale - self._full_scale)
        # Clamp through the hover throttle rather than the scale, because that is
        # the quantity the bounds are expressed in and the one an operator reads.
        low = GRAVITY_MPS2 / self.params.max_hover_throttle
        high = GRAVITY_MPS2 / self.params.min_hover_throttle
        self._full_scale = max(low, min(high, updated))
        self._observations += 1
        return True

    def _weight(self, dt):
        # type: (float) -> float
        """How much this observation moves the estimate.

        Two regimes, and the handover between them is automatic. While the
        estimate is young the weight is ``1 / (n + 1)``, which makes the estimate
        the plain **running mean** of everything seen so far -- the right thing
        for a quantity that is roughly constant and completely unknown, and it
        converges in a second or two rather than waiting out a time constant.
        Once enough samples have accumulated that the running mean would move
        more slowly than the exponential filter, the filter takes over and the
        estimate starts *tracking* instead of averaging, which is what a sagging
        battery needs.

        Without this, the estimate is only as good as its seed for the first
        time constant of the flight. Measured: seeded at 0.50 against an
        airframe that hovers at 0.62, the aircraft sank 0.6 m before the plain
        exponential filter caught up.

        The running-mean weight is capped so that no single early measurement
        can define the estimate on its own.
        """
        exponential = dt / (self.params.learn_tau_s + dt)
        running_mean = min(1.0 / float(self._observations + 1), _MAX_WARMUP_WEIGHT)
        return max(exponential, running_mean)
