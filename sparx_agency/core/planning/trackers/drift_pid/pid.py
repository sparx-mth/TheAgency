"""One-axis PID with the anti-windup this platform actually needs.

There is no other PID in ``core`` — every existing tracker is P-only, and that is
a deliberate choice documented in
:mod:`sparx_agency.core.planning.trackers.multi_axis_follower.params`: the
localization is noisy at standstill and while yawing, so an integrator that
learns from that noise would fly the drone into the noise.

This controller needs the I term anyway, because the thing it exists to cancel
*is* a bias: the drone drifts sideways (and, while turning or standing, fore/aft)
at a rate that no proportional term can null — P only ever pushes back in
proportion to how far the drift has already carried you, so it settles at a
standing offset. The integral is the drift estimate, and reading
:attr:`AxisPid.drift` is how an operator sees what the controller has learned.

The noise problem is answered not by dropping the I term but by refusing to feed
it bad data:

  * ``integrate=False`` freezes the integrator outright — the caller passes this
    whenever localization is coasting or its confidence is below the trust floor.
    A frozen integrator holds its learned drift; it does not decay to zero and it
    does not learn from a pose that is a guess.
  * Conditional integration stops winding up when the output is already saturated
    in the direction the error is pushing (the classic clamp).
  * ``leak_tau_s`` bleeds the estimate back toward zero over minutes, so a drift
    learned on one battery/air-current regime does not outlive it.
  * The derivative is low-pass filtered, because a raw difference of a ~10 Hz
    noisy pose is almost entirely noise.

Sign convention: the output pushes in the direction that *reduces* ``error``, so
``error`` must be "setpoint minus measurement" (e.g. cross-track error measured
as the offset from the drone to the trajectory: positive = trajectory is to the
left = command positive ``vy`` = left).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


def _clamp(value, limit):
    # type: (float, float) -> float
    """Clamp ``value`` to the symmetric interval ``[-limit, limit]``."""
    if value > limit:
        return limit
    if value < -limit:
        return -limit
    return value


def _deadband(value, width):
    # type: (float, float) -> float
    """Continuous deadband: 0 within +-``width``, else the excess past it.

    Subtracting the width rather than hard-cutting keeps the response continuous
    across the threshold, so a correction eases in from zero instead of snapping
    on. ``width <= 0`` passes the value through unchanged.
    """
    if width <= 0.0:
        return value
    if value > width:
        return value - width
    if value < -width:
        return value + width
    return 0.0


@dataclass(frozen=True)
class PidGains:
    """Tuning for one :class:`AxisPid` (SI units for the axis it drives).

    Attributes:
        kp: Proportional gain, error units -> output units. The fast term; keep it
            low on this platform so a noisy pose does not become a twitchy command.
        ki: Integral gain, error units -> output units per second. This is the
            drift-learning rate: how fast the controller decides "there is a
            standing bias here" and starts feeding it forward. Deliberately small
            — the drift is slow, so learning it should be slow too.
        kd: Derivative gain, error-rate units -> output units. Damping. Small,
            and always used with ``d_tau_s``.
        i_limit: Hard cap on the integral's contribution to the output, in output
            units. This IS the "how much drift may I assume?" dial — the largest
            standing correction the controller may hold on its own.
        d_tau_s: Low-pass time constant on the derivative (s). 0 disables the
            filter (raw difference). At a ~10 Hz noisy pose, anything below ~0.2
            mostly differentiates noise.
        deadband: Error magnitude left uncorrected (error units). Rides out
            localization jitter instead of chattering the axis. Applied before
            every term, so inside the deadband the integrator does not learn
            either — jitter must never look like drift.
        out_limit: Hard cap on the total output magnitude (output units).
    """

    kp: float = 0.0
    ki: float = 0.0
    kd: float = 0.0
    i_limit: float = 0.0
    d_tau_s: float = 0.3
    deadband: float = 0.0
    out_limit: float = 1.0

    def __post_init__(self):
        # type: () -> None
        """Validate the invariants the control law relies on."""
        for name in ("kp", "ki", "kd", "i_limit", "d_tau_s", "deadband"):
            if getattr(self, name) < 0.0:
                raise ValueError("PidGains." + name + " must be >= 0")
        if self.out_limit <= 0.0:
            raise ValueError("PidGains.out_limit must be > 0")
        if self.i_limit > self.out_limit:
            raise ValueError("PidGains.i_limit must not exceed out_limit "
                             "(the integral alone would saturate the axis)")


class AxisPid:
    """Stateful PID for a single control axis.

    The integral term doubles as the axis's **drift estimate**: it is the standing
    output the controller has learned it must hold to stay on the trajectory. Read
    it through :attr:`drift` to see, in the axis's own units, how hard this drone
    is being pushed sideways / fore-aft / around.
    """

    def __init__(self, gains=None, leak_tau_s=120.0):
        # type: (Optional[PidGains], float) -> None
        """Create a PID.

        Args:
            gains: Tuning. Defaults to all-zero gains (an inert controller).
            leak_tau_s: Time constant over which a learned drift bleeds back
                toward zero while the controller is running (s). Long by design:
                the drift is a property of the airframe and the room, not of this
                second. Set <= 0 to never forget.
        """
        self.gains = gains or PidGains()
        self.leak_tau_s = float(leak_tau_s)
        self.reset()

    def reset(self):
        # type: () -> None
        """Clear the integral and derivative memory (call on a fresh path)."""
        self._i = 0.0
        self._prev_error = None      # type: Optional[float]
        self._d_filt = 0.0
        self._stale_dt = 0.0

    def reset_derivative(self):
        # type: () -> None
        """Drop only the derivative memory, keeping the learned drift.

        Call this when the *setpoint* jumps — a new path, a new segment, a switch
        from tracking to station-keeping. Without it the step change in error
        becomes a one-tick derivative spike that kicks the axis. The integral is
        deliberately kept: the drift did not change just because the goal did.
        """
        self._prev_error = None
        self._d_filt = 0.0
        self._stale_dt = 0.0

    @property
    def drift(self):
        # type: () -> float
        """The learned standing bias, as its contribution to the output.

        Positive means the controller is holding a positive command on this axis
        just to stay on track — i.e. the world is pushing the drone in the
        negative direction. This is the number to publish for an operator.
        """
        return _clamp(self.gains.ki * self._i, self.gains.i_limit)

    def update(self, error, dt, integrate=True, gain_scale=1.0,
               deadband_extra=0.0, fresh=True):
        # type: (float, float, bool, float, float, bool) -> float
        """Advance the controller one tick and return the axis command.

        Args:
            error: Setpoint minus measurement, in the axis's error units.
            dt: Seconds since the previous call. Must be > 0.
            integrate: False freezes the integrator for this tick (the pose is
                coasted, or its confidence is below the trust floor, or the axis
                is being overridden). The learned drift is held, not cleared.
            gain_scale: Multiplier on the P and D terms only (0..1), used to back
                off the fast terms when localization is poor. The integral is
                deliberately NOT scaled — a drift learned while the pose was good
                stays valid when it degrades, and that feed-forward is exactly
                what carries the drone through a bad patch.
            deadband_extra: Widens the deadband for this tick only (error units,
                >= 0). Fed from the localization's own error estimate: when the
                provider says this pose is only good to +-20 cm, a 3 cm error is
                not a fact to correct, and definitely not one to learn drift from.
            fresh: False when NO new measurement arrived since the previous call
                (the ~10 Hz vision pose is event-driven; the control loop is not).
                A held pose repeats the previous error exactly, so differentiating
                it yields 0 and then a double-height spike when the next frame
                lands. On a stale tick the derivative is held and the elapsed time
                is banked; the next fresh tick differentiates over the TRUE
                interval. The integral keeps running — a zero-order-held error is
                still the best available estimate of the continuous error.

        Returns:
            The commanded output for this axis, clamped to ``out_limit``.
        """
        if dt <= 0.0:
            raise ValueError("AxisPid.update: dt must be > 0")
        g = self.gains
        band = g.deadband + (deadband_extra if deadband_extra > 0.0 else 0.0)
        err = _deadband(float(error), band)

        # Derivative on error, low-pass filtered, only across FRESH measurements.
        # Inside the deadband err is 0, so a drone sitting quietly on track
        # produces no derivative at all.
        if not fresh:
            self._stale_dt += dt
        else:
            span = dt + self._stale_dt
            self._stale_dt = 0.0
            if self._prev_error is None:
                raw_d = 0.0
            else:
                raw_d = (err - self._prev_error) / span
            self._prev_error = err
            if g.d_tau_s > 0.0:
                alpha = span / (g.d_tau_s + span)
                self._d_filt += alpha * (raw_d - self._d_filt)
            else:
                self._d_filt = raw_d

        scale = max(0.0, float(gain_scale))
        p_term = g.kp * err * scale
        d_term = g.kd * self._d_filt * scale

        # Provisional output WITHOUT this tick's integration, so the anti-windup
        # test below asks the right question: "would adding to the integral push
        # an already-saturated axis further into its stop?"
        i_term = _clamp(g.ki * self._i, g.i_limit)
        provisional = p_term + d_term + i_term

        if integrate and g.ki > 0.0:
            saturated_same_way = (
                (provisional >= g.out_limit and err > 0.0)
                or (provisional <= -g.out_limit and err < 0.0))
            at_i_limit_same_way = (
                (i_term >= g.i_limit and err > 0.0)
                or (i_term <= -g.i_limit and err < 0.0))
            if not saturated_same_way and not at_i_limit_same_way:
                self._i += err * dt
            # Leak toward zero so a stale drift estimate expires. Applied only
            # while integrating: a frozen integrator must hold, not fade.
            if self.leak_tau_s > 0.0:
                self._i -= self._i * (dt / self.leak_tau_s)
            # Keep the raw accumulator bounded too, so a long freeze after a long
            # saturation cannot hide an enormous number that unwinds later.
            if g.ki > 0.0:
                self._i = _clamp(self._i, g.i_limit / g.ki)

        i_term = _clamp(g.ki * self._i, g.i_limit)
        return _clamp(p_term + d_term + i_term, g.out_limit)
