"""Closed-loop velocity control for a platform driven only by a remote-control axis.

Some airframes expose no velocity, acceleration or attitude interface at all --
only the stick a human would hold. The ROBOTICAN "Rooster" is one: the single
actuation path is ``ManualControl`` with axes in ``[-1000, 1000]``, which is
neither a velocity lock nor a clean thrust command but something in between,
with a large dead band and a platform-dependent gain.

Feeding such an axis from a measured inverse curve alone is open loop, and it
under-delivers: measured against Sphera ground truth 2026-08-18, a 0.30 m/s
request produced 0.11 m/s (ratio 0.31) even though the inverse curve was
calibrated. The curve captures the *steady-state* response of an isolated axis
step; it cannot capture drag, residual tilt, wind-up of the platform's own
attitude loop, or the fact that the aircraft spends most of its time
accelerating rather than settled.

So the curve becomes a feed-forward term and the measured velocity closes the
loop around it. The integrator is what actually finds the extra deflection the
curve is missing, and it does so per axis and per flight condition, without
anyone having to re-measure a gain.

This module is deliberately platform-agnostic and ROS-free: it knows about a
signed axis range, a dead band, and a measured velocity. Which topic the
velocity came from and where the axis goes is the caller's business.

Not to be confused with ``core/control/velocity_servo/``, which inverts an
autopilot that genuinely accepts a velocity setpoint. That plant model has no
dead band and no stick; this one is the opposite case, where no velocity
interface exists at all.
"""

from __future__ import annotations


def clamp(value, low, high):
    # type: (float, float, float) -> float
    """Clamp ``value`` into ``[low, high]``."""
    return low if value < low else (high if value > high else value)


def feedforward_axis(v_mps, deadzone, v_full, axis_limit=1000.0):
    # type: (float, float, float, float) -> float
    """Measured-curve inverse: the axis value that holds ``v_mps`` in steady state.

    The horizontal axes are dead below ``deadzone`` counts and then ramp
    roughly linearly to ``v_full`` at ``axis_limit``. Sign is carried through;
    a zero request returns zero rather than a dead-band-edge command.

    Args:
        v_mps: Requested velocity along this axis, m/s (signed).
        deadzone: Axis counts below which the platform does not move at all.
        v_full: Velocity produced at ``axis_limit``, m/s. Must be > 0.
        axis_limit: Magnitude of full deflection.

    Returns:
        Signed axis value in ``[-axis_limit, axis_limit]``.

    Raises:
        ValueError: If ``v_full`` is not positive.
    """
    if v_full <= 0.0:
        raise ValueError("v_full must be positive, got %r" % (v_full,))
    if v_mps == 0.0:
        return 0.0
    span = max(1.0, axis_limit - deadzone)
    magnitude = deadzone + min(1.0, abs(v_mps) / v_full) * span
    magnitude = clamp(magnitude, 0.0, axis_limit)
    return magnitude if v_mps > 0.0 else -magnitude


class AxisVelocityServo(object):
    """PI-corrected velocity servo for one remote-control axis.

    ``axis = feedforward(v_cmd) + kp * e + ki * integral(e)``, with ``e`` the
    velocity error in m/s.

    The integrator is the important half and the dangerous half. Three guards,
    all of which exist because of how this platform actually behaves:

    - **Dead-band-aware anti-windup.** Below the dead band the platform cannot
      move at all, so an integrator would happily wind up to full deflection
      while the aircraft sits still and then slam the stick. The integral is
      frozen whenever the *output* is already saturated in the direction the
      error is pushing.
    - **A bounded correction.** The correction is capped independently of the
      total, so a stale or noisy velocity estimate can bias the command but
      never take it over.
    - **Reset on stop.** A zero request clears the integrator instead of
      leaving it to discharge through the next command.

    Args:
        deadzone: Axis counts below which the platform does not move.
        v_full: Velocity produced at full deflection, m/s.
        kp: Proportional gain, axis counts per (m/s) of error.
        ki: Integral gain, axis counts per (m/s * s) of error.
        max_correction: Largest magnitude the PI term may reach, in counts.
        axis_limit: Magnitude of full deflection.
        min_command_mps: Requests smaller than this are treated as "stop"
            (the platform cannot hold speeds below its first dead-band step).
    """

    def __init__(self, deadzone, v_full, kp=0.0, ki=0.0, max_correction=250.0,
                 axis_limit=1000.0, min_command_mps=0.0):
        # type: (float, float, float, float, float, float, float) -> None
        self.deadzone = float(deadzone)
        self.v_full = float(v_full)
        self.kp = float(kp)
        self.ki = float(ki)
        self.max_correction = float(max_correction)
        self.axis_limit = float(axis_limit)
        self.min_command_mps = float(min_command_mps)
        self._integral = 0.0
        self.last_error = 0.0
        self.last_correction = 0.0

    def reset(self):
        # type: () -> None
        """Clear the integrator. Call on stop, mode change, or loss of feedback."""
        self._integral = 0.0
        self.last_error = 0.0
        self.last_correction = 0.0

    @property
    def integral(self):
        # type: () -> float
        """Accumulated integral term, in axis counts."""
        return self._integral

    def update(self, v_cmd, v_meas, dt):
        # type: (float, float, float) -> float
        """Advance the servo one tick and return the axis command.

        Args:
            v_cmd: Requested velocity along this axis, m/s (signed).
            v_meas: Measured velocity along this axis, m/s (signed). Must come
                from a source independent of the platform's own estimator --
                PX4's estimate was measured drifting convincingly while the
                aircraft sat still (see LESSONS.md).
            dt: Seconds since the previous update. Non-positive dt integrates
                nothing, so a duplicated tick cannot double-count.

        Returns:
            Signed axis value in ``[-axis_limit, axis_limit]``.
        """
        if abs(v_cmd) < self.min_command_mps:
            self.reset()
            return 0.0

        ff = feedforward_axis(v_cmd, self.deadzone, self.v_full, self.axis_limit)
        error = v_cmd - v_meas
        self.last_error = error

        # Provisional integral, then commit only if the output is not saturated
        # against this error's direction -- the dead band makes windup here a
        # real failure, not a textbook one.
        candidate = self._integral + (self.ki * error * dt if dt > 0.0 else 0.0)
        candidate = clamp(candidate, -self.max_correction, self.max_correction)
        correction = clamp(self.kp * error + candidate,
                           -self.max_correction, self.max_correction)
        axis = clamp(ff + correction, -self.axis_limit, self.axis_limit)

        saturated_high = axis >= self.axis_limit and error > 0.0
        saturated_low = axis <= -self.axis_limit and error < 0.0
        if not (saturated_high or saturated_low):
            self._integral = candidate

        self.last_correction = correction
        return axis
