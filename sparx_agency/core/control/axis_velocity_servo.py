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
        output_limit: Real ceiling on the returned axis value, if the caller
            clamps further than ``axis_limit``. Bounds the output AND the
            anti-windup test; <=0 means ``axis_limit``.
    """

    def __init__(self, deadzone, v_full, kp=0.0, ki=0.0, max_correction=250.0,
                 axis_limit=1000.0, min_command_mps=0.0,
                 v_full_moving=0.0, deadzone_moving=0.0, move_eps_mps=0.10,
                 brake_release_margin_mps=0.15, integral_hold_s=0.6,
                 output_limit=0.0, curve=None):
        # type: (float, float, float, float, float, float, float, float, float, float, float, float, float, object) -> None
        #: Measured response curve (an ``AxisResponseCurve``). When given, the
        #: feedforward is its inverse and the dead-band machinery below --
        #: ``deadzone``/``v_full``, the standing/moving regime pair, and the
        #: never-mute floor -- is bypassed entirely: the 2026-08-31 manual
        #: calibration measured the horizontal axes as an expo curve with NO
        #: dead band, so there is no edge to hold the stick at. Callers should
        #: also pass ``output_limit=curve.max_counts`` so the anti-windup and
        #: the ceiling agree with the curve's own last measured point.
        self.curve = curve
        self.deadzone = float(deadzone)
        self.v_full = float(v_full)
        #: Velocity at full deflection once the aircraft is ALREADY MOVING.
        #: The platform's effective gain is regime-dependent: a hover-start step
        #: measured 0.26 m/s at axis 700, while in flight the same axis produced
        #: ~0.9-1.0 m/s. One steady-state curve therefore cannot fit both, and
        #: fitting it to the standing case is what made the aircraft fly 2-4x
        #: faster than commanded once moving. <=0 falls back to ``v_full``.
        self.v_full_moving = float(v_full_moving)
        #: Dead band once ALREADY MOVING. <=0 keeps ``deadzone`` in both regimes.
        #:
        #: A regime is a CURVE -- an offset and a slope together -- and the two
        #: halves cannot be taken from different regimes. Switching only the
        #: full scale, while the offset stayed at the standing value, was flown
        #: and over-commanded badly: peak speed rose from ~1.5-2.1 to 3.1-3.5
        #: m/s and a third of the flight ended up stopped. Set this alongside
        #: ``v_full_moving`` or set neither.
        self.deadzone_moving = float(deadzone_moving)
        self.move_eps_mps = float(move_eps_mps)
        #: Only release the stick below the dead band when the aircraft is
        #: genuinely this much over the demand -- i.e. when braking is what is
        #: wanted. Otherwise a negative correction merely mutes the axis, the
        #: aircraft coasts, and the loop re-commands: a limit cycle.
        self.brake_release_margin_mps = float(brake_release_margin_mps)
        #: Seconds a below-minimum demand keeps the learned integral before it
        #: is dropped. A follower that emits brief zeros would otherwise destroy
        #: the standing bias several times a second.
        self.integral_hold_s = float(integral_hold_s)
        self.kp = float(kp)
        self.ki = float(ki)
        self.max_correction = float(max_correction)
        self.axis_limit = float(axis_limit)
        #: Largest axis value the CALLER will actually send, if it clamps the
        #: output further down. Anti-windup is only anti-windup if it knows the
        #: real ceiling: with the servo free to 1000 while the caller clipped at
        #: 900, the integral kept accumulating across a 100-count band the
        #: actuator never saw -- and 900 is exactly where this airframe stops
        #: behaving (roll excursions past 30 deg follow a full-stick command).
        #: Kept separate from ``axis_limit`` because that one is the measured
        #: curve's full-scale reference: moving it would silently re-scale the
        #: calibration instead of bounding the output.
        self.output_limit = (float(output_limit) if float(output_limit) > 0.0
                             else float(axis_limit))
        self.min_command_mps = float(min_command_mps)
        self._integral = 0.0
        self._idle_s = 0.0
        self.last_error = 0.0
        self.last_correction = 0.0

    def reset(self):
        # type: () -> None
        """Clear the integrator. Call on stop, mode change, or loss of feedback."""
        self._integral = 0.0
        self._idle_s = 0.0
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
            # Hold the learned bias briefly rather than dropping it: a follower
            # that dips through zero for a tick or two must not cost the loop
            # everything it learned about the standing shortfall.
            self._idle_s += max(0.0, float(dt))
            if self._idle_s >= self.integral_hold_s:
                self.reset()
            return 0.0
        self._idle_s = 0.0

        if self.curve is not None:
            # Measured expo curve: one law for every speed and regime, no dead
            # band -- so the floor below never engages (deadzone 0).
            ff = self.curve.axis_for(v_cmd)
            deadzone = 0.0
        else:
            moving = abs(v_meas) >= self.move_eps_mps
            v_full = self.v_full
            deadzone = self.deadzone
            if moving and self.v_full_moving > 0.0:
                v_full = self.v_full_moving
                # Both halves of the curve move together, or neither does.
                if self.deadzone_moving > 0.0:
                    deadzone = self.deadzone_moving
            ff = feedforward_axis(v_cmd, deadzone, v_full, self.axis_limit)
        error = v_cmd - v_meas
        self.last_error = error

        # Provisional integral, then commit only if the output is not saturated
        # against this error's direction -- the dead band makes windup here a
        # real failure, not a textbook one.
        candidate = self._integral + (self.ki * error * dt if dt > 0.0 else 0.0)
        candidate = clamp(candidate, -self.max_correction, self.max_correction)
        correction = clamp(self.kp * error + candidate,
                           -self.max_correction, self.max_correction)
        axis = clamp(ff + correction, -self.output_limit, self.output_limit)

        # Never mute the stick while still asking for motion. Below the dead
        # band the platform does not merely go slower -- it stops, and on this
        # airframe a released stick is an active brake (PX4 Position mode), so
        # the aircraft then has to climb back through the dead band to restart.
        # The one case where releasing IS what the loop wants is a genuine
        # overspeed, so that stays allowed.
        # The floor uses the SAME regime's dead band as the feedforward above.
        # Flooring a moving-regime command back up to the standing dead band
        # would quietly reintroduce the mixed pair this class exists to avoid.
        braking = abs(v_meas) > abs(v_cmd) + self.brake_release_margin_mps
        if not braking and abs(axis) < deadzone:
            axis = deadzone if v_cmd > 0.0 else -deadzone

        saturated_high = axis >= self.output_limit and error > 0.0
        saturated_low = axis <= -self.output_limit and error < 0.0
        if not (saturated_high or saturated_low):
            self._integral = candidate

        self.last_correction = correction
        return axis
