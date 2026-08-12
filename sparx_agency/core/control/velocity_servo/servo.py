"""Fly a planned trajectory on an autopilot that takes a velocity setpoint.

The sibling of ``trajectory_tracking``, for the other kind of aircraft. Where
that one owns the attitude loop and commands an acceleration, this one sits on
top of an autopilot that already closes its own velocity loop and will only
accept a twist -- a Gazebo model plugin, a hobby flight controller in a
velocity mode, an indoor platform whose manufacturer exposes nothing lower.

The temptation with such an airframe is to command the plan's velocity and add a
proportional pull toward the plan's position. That is what the previous stack
did, and it cannot work well, for a reason that is measurable rather than
arguable: **the autopilot underneath is slow.** Measured on the reference
airframe, a horizontal velocity command takes 0.18 s to have any effect and a
further 0.51 s to arrive. Commanding ``v_plan`` therefore produces an aircraft
whose velocity matches the plan roughly 0.7 s late, which at a 0.6 m/s cruise is
0.4 m of standing position error -- before any disturbance, any noise or any
replan. Raising the position gain to chase it is exactly the wrong move, because
that same delay is what limits how high the gain can go.

The fix is to stop treating the autopilot as a black box and invert it. A
first-order lag is inverted by a lead:

.. code-block:: text

    plant:      tau * dv/dt + v = v_command
    so:         v_command = v_wanted + tau * (dv/dt)_wanted

and ``(dv/dt)_wanted`` is the plan's **acceleration**, which the B-spline
carries analytically and exactly. That single term is the difference between
this backend and the one it replaces, and it costs nothing: no gain to tune, no
derivative to filter, no extra state. It is the reason the trajectory is carried
as a curve rather than as a stream of sampled points.

The whole law:

.. code-block:: text

    v_cmd = v_plan(t+L)  +  tau * a_plan(t+L)     <- feedforward, read ahead by
            \\__________/     \\______________/        the transport delay
             the plan          invert the lag
          +  Kp * clamp(p_plan(t) - p)             <- position feedback, at t
          +  Ki * integral(...)                    <- standing bias, near the curve
          +  catchup * tangent                     <- schedule, along-track only

then clamped to the plan's own speed plus a margin, slew limited, and rotated
into the body **once**, at the very end.

Frames: world is any right-handed frame with +z up -- world ENU here (REP-103).
The body is FLU, so the emitted twist is x forward, y left, z up.
"""
from __future__ import annotations

import math
from typing import Optional

import numpy as np

from sparx_agency.core.common.types import normalize_angle
from sparx_agency.core.control.reference.feed import TrajectoryFeed
from sparx_agency.core.control.velocity_servo.limits import limit_velocity, slew_velocity
from sparx_agency.core.control.velocity_servo.params import VelocityServoParams
from sparx_agency.core.control.velocity_servo.types import BodyTwistCommand
from sparx_agency.core.control.velocity_servo.yaw import YawServo
from sparx_agency.core.planning.trackers.drift_pid.pid import AxisPid


class VelocityServo:
    """Turns a B-spline trajectory and a measured state into a body twist.

    One instance per aircraft. Stateful -- the integrators hold the learned
    standing bias, the feed holds where on the curve the aircraft was last tick,
    the yaw servo holds the rate it last commanded, and the slew limit is
    measured against the previous command -- so it must be stepped at a steady
    rate and :meth:`reset` whenever the aircraft stops flying the plan.

    Args:
        params: Gains, limits and the measured plant. The defaults are for an
            indoor multirotor on an autopilot with a unit-DC-gain velocity loop;
            **measure the plant** rather than trusting them.
    """

    def __init__(self, params=None):
        # type: (Optional[VelocityServoParams]) -> None
        self.params = params or VelocityServoParams()
        self._pid = (
            AxisPid(self.params.horizontal_pid),
            AxisPid(self.params.horizontal_pid),
            AxisPid(self.params.vertical_pid),
        )
        self._feed = TrajectoryFeed(self.params.reference)
        max_rate, max_accel = self.params.yaw_limits()
        self._yaw = YawServo(gain=self.params.yaw_gain, max_rate=max_rate,
                             max_accel=max_accel,
                             deadband_rad=self.params.yaw_deadband_rad)
        self._previous = None       # type: Optional[np.ndarray]
        self._hold = None           # type: Optional[np.ndarray]
        # Per axis, not one flag for the whole command. A horizontal clamp says
        # nothing about whether the vertical axis may keep learning its bias,
        # and freezing all three on any saturation costs the altitude
        # integrator the trim it exists to hold.
        self._saturated = [False, False, False]

    def reset(self, hold_position=None):
        # type: (Optional[object]) -> None
        """Forget every accumulated correction before a new phase of flight.

        Call whenever the aircraft stops flying the plan -- before handing over
        from a takeoff climb, after a landing, on a planner restart. The
        integrators hold a bias learned in a different flight regime, and
        carrying it across a handover puts a step into the first command of the
        new one.

        Args:
            hold_position: World ``(x, y, z)`` to hold while nothing is being
                followed. None latches wherever the aircraft is when that first
                happens.
        """
        for axis in self._pid:
            axis.reset()
        self._feed.reset()
        self._yaw.reset()
        self._previous = None
        self._hold = None if hold_position is None else np.asarray(hold_position, dtype=float)
        self._saturated = [False, False, False]

    def set_trajectory(self, trajectory):
        # type: (object) -> bool
        """Queue a newly planned trajectory. See :class:`TrajectoryFeed`."""
        return self._feed.set_trajectory(trajectory)

    @property
    def trajectory_id(self):
        # type: () -> int
        """Id of the trajectory being flown, or -1 when none is."""
        return self._feed.trajectory_id

    def update(self, position, velocity, yaw, dt, now_s, follow=True):
        # type: (object, object, float, float, float, bool) -> BodyTwistCommand
        """Advance one control tick.

        Args:
            position: Measured world ``(x, y, z)``, metres.
            velocity: Measured world ``(vx, vy, vz)``, m/s. Used only for the
                diagnostics and the hold -- deliberately *not* fed back as a
                damping term, because the autopilot underneath already closes
                that loop and a second one around it rings.
            yaw: Measured heading, radians CCW from world +x.
            dt: Seconds since the previous call. Must be > 0.
            now_s: Current time, on the clock the trajectories' start times are
                stamped on.
            follow: False holds station instead of tracking.

        Returns:
            The body twist to command, with the diagnostics that say whether the
            aircraft is actually flying the plan.

        Raises:
            ValueError: If ``dt`` is not positive.
        """
        if dt <= 0.0:
            raise ValueError("VelocityServo.update: dt must be > 0, got %r" % (dt,))
        measured = np.asarray(position, dtype=float).reshape(3)
        measured_velocity = np.asarray(velocity, dtype=float).reshape(3)
        yaw = normalize_angle(float(yaw))

        if self._feed.promote(now_s):
            # A new curve is a different parameterisation, so the derivative
            # memory built on the old one is meaningless on it. The learned
            # bias is kept: the wind did not change because the plan did.
            for axis in self._pid:
                axis.reset_derivative()

        reference = self._feed.resolve(measured, now_s) if follow else None
        if reference is None:
            return self._hold_station(measured, measured_velocity, yaw, dt)

        self._hold = None
        world, planned_speed = self._world_command(reference, measured, dt)
        command, saturated, rate_limited = self._shape(world, planned_speed, dt)

        reference_yaw = reference.yaw
        yaw_rate = self._yaw.update(reference_yaw, reference.yaw_rate, yaw, dt)
        body = self._to_body(command, yaw, yaw_rate, dt)
        return BodyTwistCommand(
            vx=float(body[0]), vy=float(body[1]), vz=float(body[2]),
            yaw_rate=float(yaw_rate),
            world_vx=float(command[0]), world_vy=float(command[1]),
            world_vz=float(command[2]),
            commanded_yaw=float(yaw if reference_yaw is None else reference_yaw),
            position_error_m=reference.gap_m,
            along_track_lag_m=reference.along_track_lag_m,
            cross_track_error_m=reference.cross_track_error_m,
            yaw_error_rad=self._yaw.error_rad,
            reference_time_s=reference.reference_time_s,
            trajectory_id=reference.trajectory_id,
            diverged=reference.gap_m > self.params.max_position_error_m,
            past_end=reference.past_end,
            saturated=saturated, rate_limited=rate_limited)

    def _world_command(self, reference, measured, dt):
        # type: (object, np.ndarray, float) -> tuple
        """Assemble the wanted world velocity, before any clamp.

        Returns:
            ``(world_velocity, planned_speed)``. The planned speed is carried
            out because the speed ceiling is relative to it.
        """
        lead = self._feedforward(reference)
        planned = np.array([lead.vx, lead.vy, lead.vz], dtype=float)
        feed_forward = planned.copy()
        if self.params.use_feedforward_lead:
            feed_forward = feed_forward + self._lead_term(lead)

        error = np.array(reference.position, dtype=float) - measured
        clamped = self._clamp_error(error)
        correction = np.array([self._correct(i, float(error[i]), float(clamped[i]), dt)
                               for i in range(3)])
        catchup = self._catchup(planned, reference.along_track_lag_m)
        return feed_forward + correction + catchup, float(np.linalg.norm(planned))

    def _feedforward(self, reference):
        # type: (object) -> object
        """The plan's state to feed forward, read ahead by the transport delay.

        The aircraft responds to this tick's command one delay from now, so the
        feedforward must describe the plan *then*, not now. Applied to the
        feedforward only -- see ``predict_reference`` in the params for why
        leaning the feedback forward too is a trap with a standing error.
        """
        trajectory = self._feed.trajectory
        if not self.params.predict_reference or trajectory is None:
            return reference.point
        lead = self.params.plant.feedforward_lead_s
        if lead <= 0.0:
            return reference.point
        return trajectory.sample(reference.reference_time_s + lead)

    def _lead_term(self, point):
        # type: (object) -> np.ndarray
        """``tau * a_plan`` -- the term that cancels the autopilot's own lag.

        Per axis, because the horizontal and vertical loops of a multirotor have
        genuinely different time constants: horizontal has to rotate the whole
        airframe to produce a force, vertical only has to change the thrust.
        """
        plant = self.params.plant
        return np.array([plant.horizontal.time_constant_s * point.ax,
                         plant.horizontal.time_constant_s * point.ay,
                         plant.vertical.time_constant_s * point.az], dtype=float)

    def _catchup(self, planned_velocity, lag_m):
        # type: (np.ndarray, float) -> np.ndarray
        """Extra commanded speed along the tangent, to recover schedule.

        Projection asks *where* the aircraft is on the curve and deliberately
        forgets *when* it should have been there. This puts the timing back, and
        it does so along the direction of travel only -- so unlike a lookahead
        it cannot pull the aircraft across a corner. Cross-track and along-track
        are corrected by different terms because they are different mistakes.

        The gain is per *second* of lateness rather than per metre, so the
        response means the same thing at any planned speed: two seconds late is
        two seconds late whether the plan is crawling through a doorway or
        running down a corridor.
        """
        speed = float(np.linalg.norm(planned_velocity))
        if speed <= 1e-6 or self.params.schedule_gain_per_s <= 0.0:
            return np.zeros(3)
        wanted = self.params.schedule_gain_per_s * (float(lag_m) / speed)
        bounded = max(-self.params.max_catchup_speed,
                      min(self.params.max_catchup_speed, wanted))
        return (planned_velocity / speed) * bounded

    def _clamp_error(self, error):
        # type: (np.ndarray) -> np.ndarray
        """Bound how hard the position loop pulls, without changing where it pulls.

        The horizontal pair is scaled **together**; only the vertical axis is
        clipped on its own. Clamping each axis independently silently rotates
        the correction: an error of (5.0, 1.0) m clamps to (1.0, 1.0) and points
        33.7 degrees away from the reference, worst case a full 45 -- the
        opposite of what the clamp is for, and most wrong exactly when the
        aircraft is furthest off the plan.
        """
        clamp = self.params.position_error_clamp_m
        clamped = np.array(error, dtype=float)
        horizontal = float(np.hypot(clamped[0], clamped[1]))
        if horizontal > clamp:
            clamped[0] *= clamp / horizontal
            clamped[1] *= clamp / horizontal
        clamped[2] = max(-clamp, min(clamp, clamped[2]))
        return clamped

    def _correct(self, axis, error, clamped, dt):
        # type: (int, float, float, float) -> float
        """One axis of position feedback.

        The PID sees the *clamped* error; the integral gate tests the *true*
        one, because "am I near the curve" is a question about where the
        aircraft actually is. Outside ``integral_band_m`` the axis is not
        holding a standing bias, it is travelling, and integrating that charges
        a correction which arrives after the error is gone. The integrator is
        also frozen while **this axis** is saturated -- not while any axis is,
        which would let a hard turn cost the altitude loop its trim.
        """
        near = abs(error) <= self.params.integral_band_m
        return self._pid[axis].update(clamped, dt,
                                      integrate=near and not self._saturated[axis])

    def _shape(self, wanted, planned_speed, dt):
        # type: (np.ndarray, float, float) -> tuple
        """Clamp the magnitude, then the rate of change. Both in world.

        The speed ceiling moves with the plan rather than being absolute, so it
        can never hold the aircraft permanently behind a faster trajectory --
        but it is a ceiling nonetheless, because FALCON checks its route against
        the map at the speed it planned and flying faster spends the clearance
        on stopping distance.
        """
        ceiling = planned_speed + self.params.max_overspeed
        command, saturated = limit_velocity(wanted, self.params.limits, max_speed_xy=ceiling)
        command, rate_limited = slew_velocity(self._previous, command,
                                              self.params.limits, dt)
        self._previous = command.copy()
        self._note_saturation(wanted, command)
        return command, saturated, rate_limited

    def _note_saturation(self, wanted, delivered, tolerance=1e-9):
        # type: (np.ndarray, np.ndarray, float) -> None
        """Record, per axis, whether the command asked for survived to the wire.

        Per axis rather than one flag for the whole command, because the
        integrator gate reads it: a horizontal clamp during a hard turn says
        nothing about whether the altitude loop may keep learning its trim, and
        freezing all three on any saturation is how a vertical integrator
        quietly loses the bias it exists to hold. Both the magnitude clamp and
        the slew limit count -- either one means the correction this axis is
        contributing did not reach the airframe, which is precisely the
        condition the integrator must not keep charging against.
        """
        self._saturated = [bool(abs(float(delivered[i]) - float(wanted[i])) > tolerance)
                           for i in range(3)]

    def _to_body(self, world_velocity, yaw, yaw_rate, dt):
        # type: (np.ndarray, float, float, float) -> np.ndarray
        """Rotate the world command into the body frame. The last thing done.

        The heading used is the one the aircraft will *have* when the command
        takes effect, not the one it has now: a turning aircraft handed a
        rotation built on a stale heading flies a command that is skewed by
        ``yaw_rate * delay``, which at 1 rad/s and 0.18 s is over 10 degrees of
        steering error that appears only in turns.
        """
        heading = yaw + float(yaw_rate) * self.params.plant.yaw.delay_s
        cos, sin = math.cos(heading), math.sin(heading)
        return np.array([cos * world_velocity[0] + sin * world_velocity[1],
                         -sin * world_velocity[0] + cos * world_velocity[1],
                         world_velocity[2]], dtype=float)

    def _hold_station(self, measured, measured_velocity, yaw, dt):
        # type: (np.ndarray, np.ndarray, float, float) -> BodyTwistCommand
        """Fly to where the aircraft already is, because nothing else is asking.

        The hold point is latched the first time this happens, so the aircraft
        returns to it rather than ratcheting away from it. A velocity-commanded
        airframe given a zero twist does stop, unlike an acceleration-commanded
        one -- but it stops *wherever it drifted to*, so "stop" is still
        commanded as "hold this point".
        """
        if self._hold is None:
            self._hold = measured.copy()
        error = self._hold - measured
        clamped = self._clamp_error(error)
        correction = np.array([self._correct(i, float(error[i]), float(clamped[i]), dt)
                               for i in range(3)])
        command, saturated = limit_velocity(correction, self.params.limits,
                                            max_speed_xy=self.params.hold_speed_xy)
        command, rate_limited = slew_velocity(self._previous, command,
                                              self.params.limits, dt)
        self._previous = command.copy()
        self._note_saturation(correction, command)
        yaw_rate = self._yaw.update(None, 0.0, yaw, dt)
        body = self._to_body(command, yaw, yaw_rate, dt)
        return BodyTwistCommand(
            vx=float(body[0]), vy=float(body[1]), vz=float(body[2]),
            yaw_rate=float(yaw_rate),
            world_vx=float(command[0]), world_vy=float(command[1]),
            world_vz=float(command[2]),
            commanded_yaw=float(yaw),
            position_error_m=float(np.linalg.norm(error)),
            reference_time_s=self._feed.last_reference_time,
            trajectory_id=self.trajectory_id,
            holding=True, saturated=saturated, rate_limited=rate_limited)
