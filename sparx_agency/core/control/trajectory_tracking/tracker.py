"""Follow a planned trajectory on an aircraft that has physics.

A planner validated in a geometry-only simulator closes no loop at all: the
commanded state *is* the state, so "tracking" is a rename. FALCON is such a
planner -- and worse, while flying it replans from its **own previous curve**
rather than from the aircraft's measured position, so tracking error is
invisible to it and never corrected by replanning. Whatever gap opens between
the plan and the airframe is this module's to close, because nothing upstream
will.

The law is

.. code-block:: text

    a_cmd = a_ref  +  Kp*clamp(p_ref - p)  +  Kd*(v_ref - v)  +  Ki*int(p_ref - p)
            \\___/     \\_________________/     \\_____________/     \\______________/
          the plan,     back onto the         match the plan's      the standing
          read not      curve, bounded         speed; this is        bias, learned
          computed                             the damping           near the curve

and the reference it is measured against is the **nearest point on the curve**,
not the point at the current time -- with a separate along-track term to put the
schedule back, because projection deliberately forgets it.

That split is worth stating precisely, because the usual justification for it is
wrong here. Projection is normally sold as the cure for corner cutting, and
measured against FALCON's trajectories it is not: those curves are already
dynamically feasible, so a healthy inner loop tracks the time-indexed reference
around a bend just as well. What projection actually buys is recovery from a
displacement in *time* -- the aircraft holding while FALCON replans, then
resuming to find a time-indexed reference several seconds down the route and
being pulled at it across whatever lies between. See
``core.planning.trajectories.bspline.projection`` for the numbers.

Yaw is passed through, rate-limited, never servoed. FALCON chooses heading to
aim the depth camera at the frontier it wants next, so the heading *is* part of
the plan rather than a consequence of the route.

Frames: any right-handed world frame with +z up -- world ENU here (REP-103).
"""
from __future__ import annotations

import math
from typing import Optional

import numpy as np

from sparx_agency.core.common.types import normalize_angle
from sparx_agency.core.control.flatness.limits import limit_acceleration
from sparx_agency.core.control.reference.feed import TrajectoryFeed
from sparx_agency.core.control.trajectory_tracking.params import TrajectoryTrackerParams
from sparx_agency.core.control.trajectory_tracking.types import AccelerationCommand
from sparx_agency.core.planning.trackers.drift_pid.pid import AxisPid

_GOVERNOR_TAPER = 0.6
"""Fraction of the overspeed allowance used to fade the governor in.

Below 1.0 so the fade finishes before the ceiling rather than at it, which is
what lets an airframe with a lagging acceleration actually stop at the limit
instead of coasting through it.
"""


class TrajectoryTracker:
    """Turns a B-spline trajectory and a measured state into a wanted acceleration.

    One instance per aircraft. Stateful -- the integrators hold the learned
    standing bias, the projector holds where on the curve the aircraft was last
    tick, and the commanded heading slews from its own previous value -- so it
    must be stepped at a steady rate and :meth:`reset` whenever the aircraft
    stops flying the plan.

    Args:
        params: Gains, limits and freshness rules. The defaults are for an
            indoor multirotor flying a metre-per-second exploration trajectory.
    """

    def __init__(self, params=None):
        # type: (Optional[TrajectoryTrackerParams]) -> None
        self.params = params or TrajectoryTrackerParams()
        self._pid = (
            AxisPid(self.params.horizontal_pid),
            AxisPid(self.params.horizontal_pid),
            AxisPid(self.params.vertical_pid),
        )
        self._feed = TrajectoryFeed(self.params.reference_params())
        self._yaw_cmd = None        # type: Optional[float]
        # Overridden per tick by update(limits=...); see its docstring.
        self._limits = self.params.limits
        self._hold = None           # type: Optional[np.ndarray]
        # Per axis, not one flag for the whole command. `limit_acceleration`
        # gives horizontal away first precisely so altitude survives, so a
        # horizontal clamp must not be what stops the vertical integrator
        # learning the standing thrust bias it exists to hold.
        self._saturated = [False, False, False]

    def reset(self, yaw=None, hold_position=None):
        # type: (Optional[float], Optional[object]) -> None
        """Forget every accumulated correction and re-seed the commanded heading.

        Call whenever the aircraft stops flying the plan -- before handing over
        from a takeoff climb, after a landing, on a planner restart. The
        integrators hold a bias learned in a different flight regime, and
        carrying it across a handover puts a step into the first command of the
        new one.

        Args:
            yaw: Heading to start slewing from. None re-seeds on the first tick.
            hold_position: World ``(x, y, z)`` to hold while nothing is being
                followed. None latches wherever the aircraft is when that first
                happens.
        """
        for axis in self._pid:
            axis.reset()
        # Drops the held and queued curves too. Without that, reset(hold_position=X)
        # is silently ignored whenever a trajectory is still loaded -- update()
        # finds it usable and follows it instead of holding, so the one call a
        # caller makes to say "stop flying the plan" does not stop flying the
        # plan. The mission resets at handover, where any curve still held
        # belongs to a previous phase and must not be resumed.
        self._feed.reset()
        self._yaw_cmd = None if yaw is None else normalize_angle(float(yaw))
        self._hold = None if hold_position is None else np.asarray(hold_position, dtype=float)
        self._saturated = [False, False, False]

    def set_trajectory(self, trajectory):
        # type: (object) -> bool
        """Accept a newly planned trajectory.

        It is *queued*, not adopted. FALCON deliberately starts each curve a
        planning-time in the future so it joins smoothly onto the one still
        being flown, so switching the instant it arrives would jump the
        reference forward to a point the aircraft has not reached yet. The swap
        happens in :meth:`update`, when the new curve's own start time comes up.

        Args:
            trajectory: A :class:`~...trajectories.bspline.BsplineTrajectory`.

        Returns:
            True if it was queued. False rejects a trajectory whose id is not
            newer than what is already held -- a re-send or a misordered
            message, either of which would restart a curve mid-flight.
        """
        return self._feed.set_trajectory(trajectory)

    @property
    def trajectory_id(self):
        # type: () -> int
        """Id of the trajectory being flown, or -1 when none is."""
        return self._feed.trajectory_id

    @property
    def commanded_yaw(self):
        # type: () -> Optional[float]
        """The heading last put on the wire, or None before the first tick."""
        return self._yaw_cmd

    def update(self, position, velocity, yaw, dt, now_s, follow=True, limits=None):
        # type: (object, object, float, float, float, bool) -> AccelerationCommand
        """Advance one control tick.

        Args:
            position: Measured world ``(x, y, z)``, metres.
            velocity: Measured world ``(vx, vy, vz)``, m/s. Measured rather than
                differenced -- the damping term is only as good as this signal.
            yaw: Measured heading, radians CCW from +x.
            dt: Seconds since the previous call. Must be > 0.
            now_s: Current time, on the clock the trajectories' start times are
                stamped on.
            follow: False holds station instead of tracking.
            limits: Acceleration envelope for this tick, overriding
                ``params.limits``. The caller passes it when it knows something
                the tracker cannot -- ``AirframeController`` cuts the thrust
                ceiling to what the learned throttle can actually buy. It MUST
                be the same envelope the stage below then applies, or
                ``saturated`` reports on a clamp nobody performed and the
                integrator keeps charging against a correction that is being
                trimmed downstream. This is how a
                caller acts on FALCON condemning its own live trajectory: the
                aircraft brakes toward a latched point rather than carrying its
                momentum into the obstacle that was just found.

        Returns:
            The acceleration and heading to command, with the diagnostics that
            say whether the aircraft is actually flying the plan.

        Raises:
            ValueError: If ``dt`` is not positive.
        """
        self._limits = self.params.limits if limits is None else limits
        if dt <= 0.0:
            raise ValueError("TrajectoryTracker.update: dt must be > 0, got %r" % (dt,))
        measured = np.asarray(position, dtype=float).reshape(3)
        measured_velocity = np.asarray(velocity, dtype=float).reshape(3)
        yaw = normalize_angle(float(yaw))
        if self._yaw_cmd is None:
            self._yaw_cmd = yaw

        if self._feed.promote(now_s):
            # The new curve is a different parameterisation, so the derivative
            # memory built on the old one means nothing on it. The learned bias
            # is kept: the wind did not change because the plan did.
            for axis in self._pid:
                axis.reset_derivative()

        sample = None
        if follow:
            sample = self._feed.resolve(measured, now_s,
                                        lookahead_s=self.params.projection.lookahead_s)
        if sample is None:
            return self._hold_station(measured, measured_velocity, dt)

        self._hold = None
        reference = sample.point
        target = np.array(sample.position, dtype=float)
        planned_velocity = np.array(sample.velocity, dtype=float)
        error = target - measured
        feed_forward = np.array(sample.acceleration, dtype=float)
        jerk = (reference.jx, reference.jy, reference.jz)

        # Attitude-lag lead: the airframe reaches a commanded attitude a time
        # constant late, so the FEEDFORWARD -- the part of this command that IS
        # the attitude -- is sampled that far ahead of the reference. Feedback
        # stays at the reference; it corrects errors that exist, not errors
        # that are predicted. Past the end there is nothing ahead to lead into.
        if self.params.attitude_lead_s > 0.0 and not sample.past_end:
            trajectory = self._feed.trajectory
            led = trajectory.sample(min(sample.reference_time_s
                                        + self.params.attitude_lead_s,
                                        trajectory.duration))
            feed_forward = np.array([led.ax, led.ay, led.az], dtype=float)
            jerk = (led.jx, led.jy, led.jz)

        # Standing-force feedforward: the acceleration spent overcoming drag at
        # the PLANNED velocity. See the params for why planned, not measured.
        feed_forward += self._drag(planned_velocity)

        catchup = self._catchup(planned_velocity, sample.along_track_lag_m)
        damping = self._damping(planned_velocity + catchup, measured_velocity)
        clamped = self._clamp_error(error)
        correction = np.array([self._correct(i, float(error[i]), float(clamped[i]), dt)
                               for i in range(3)])
        wanted = self._limit_speed(feed_forward + damping + correction, measured_velocity,
                                   float(np.linalg.norm(planned_velocity)))
        command, saturated = limit_acceleration(wanted, self._limits)
        self._note_saturation(wanted, command)

        reference_yaw = yaw if reference.yaw is None else float(reference.yaw)
        self._yaw_cmd = self._slew_yaw(reference_yaw, dt)

        return AccelerationCommand(
            ax=float(command[0]), ay=float(command[1]), az=float(command[2]),
            yaw=self._yaw_cmd,
            yaw_rate=float(reference.yaw_rate or 0.0),
            jx=float(jerk[0]), jy=float(jerk[1]), jz=float(jerk[2]),
            position_error_m=sample.gap_m,
            along_track_lag_m=sample.along_track_lag_m,
            cross_track_error_m=sample.cross_track_error_m,
            yaw_error_rad=normalize_angle(reference_yaw - yaw),
            reference_time_s=sample.reference_time_s,
            trajectory_id=sample.trajectory_id,
            diverged=sample.gap_m > self.params.max_position_error_m,
            past_end=sample.past_end,
            saturated=saturated)

    def _note_saturation(self, wanted, delivered, tolerance=1e-9):
        # type: (np.ndarray, np.ndarray, float) -> None
        """Record, per axis, whether the command asked for survived the clamp.

        Per axis rather than one flag for the whole command, because the
        integrator gate reads it. ``limit_acceleration`` deliberately gives the
        horizontal axes away first so that altitude survives -- so a hard
        corner saturates horizontally on almost every tick, and a single shared
        flag would freeze the *vertical* integrator throughout, costing it the
        standing thrust bias it exists to hold. The two axes saturate for
        different reasons and must be gated separately.
        """
        self._saturated = [bool(abs(float(delivered[i]) - float(wanted[i])) > tolerance)
                           for i in range(3)]

    def _limit_speed(self, wanted, measured_velocity, planned_speed):
        # type: (np.ndarray, np.ndarray, float) -> np.ndarray
        """Refuse to accelerate past the plan's speed plus a margin.

        The position loop has no natural speed ceiling: a metre of error asks
        for ``kp * clamp`` of acceleration, and the damping term only balances
        it once the aircraft is ``kp * clamp / kd`` faster than the plan --
        about 0.9 m/s with these gains. That is not a tuning nicety. FALCON
        checks its trajectory against the map at the speed it planned, with a
        fixed clearance around it, and flying the same curve faster spends that
        clearance on stopping distance. Measured before this existed: 42% of a
        flight above 1.1 m/s on a 0.6 m/s plan, and it ended inside a desk.

        The governor **tapers in below the ceiling** rather than switching on at
        it, and that is not refinement for its own sake. An airframe's
        acceleration decays over a time constant, so a limiter that waits for
        the ceiling has already lost: by the time the command reverses, the
        aircraft has coasted past. Measured on a 1.00 m/s plan with a 1.50 m/s
        ceiling, closing a 1.3 m lag: no ceiling 1.99, braking only once the
        ceiling is passed 1.72, tapered 1.63.

        So over the last stretch below the ceiling the accelerating component is
        faded out, and above it the excess is actively braked off at the loop's
        own damping gain. Only the *accelerating* component is ever removed and
        braking is only ever added, so this can never stop the aircraft slowing
        down and cannot deadlock a recovery.

        It remains a **soft** ceiling -- nothing can stop an airframe instantly
        -- and the ceiling moves with the plan, so unlike an absolute clamp it
        can never hold the aircraft permanently behind a faster trajectory.
        """
        speed = float(np.linalg.norm(measured_velocity))
        ceiling = planned_speed + self.params.max_overspeed
        taper = max(self.params.max_overspeed * _GOVERNOR_TAPER, 1e-3)
        if speed <= ceiling - taper or speed <= 1e-6:
            return wanted

        direction = measured_velocity / speed
        along = float(np.dot(wanted, direction))
        # Fade the accelerating component out across the taper band, so the
        # command is already neutral by the time the ceiling arrives.
        fade = max(0.0, min(1.0, (speed - (ceiling - taper)) / taper))
        governed = wanted - fade * max(along, 0.0) * direction
        if speed <= ceiling:
            return governed
        return governed - self.params.velocity_damping_xy * (speed - ceiling) * direction

    def _catchup(self, planned_velocity, lag_m):
        # type: (np.ndarray, float) -> np.ndarray
        """Extra target speed along the tangent, to recover schedule.

        Projection asks *where* the aircraft is on the curve and deliberately
        forgets *when* it should have been there. This puts the timing back,
        and it does so along the direction of travel only -- so unlike a
        lookahead, or a straight time-indexed reference, it cannot pull the
        aircraft across a corner. Cross-track and along-track are corrected by
        different terms because they are different mistakes.

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
        clipped on its own. Clamping each axis independently -- which this did
        at first -- silently rotates the correction: an error of (5.0, 1.0) m
        clamps to (1.0, 1.0) and points 33.7 degrees away from the reference,
        and the worst case is a full 45. That is the opposite of what the clamp
        is for, and it is most wrong exactly when the aircraft is furthest off
        the plan and the correction matters most.

        Horizontal and vertical are separated rather than scaled as one 3-vector
        because they are not interchangeable: they have their own gains, their
        own limits, and losing altitude is not the same kind of mistake as
        drifting sideways. ``limit_acceleration`` splits them for the same
        reason.
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

    def _drag(self, planned_velocity):
        # type: (np.ndarray) -> np.ndarray
        """Acceleration spent overcoming drag at the planned velocity.

        The measured curve is linear-plus-offset (0.176*v + 0.121 m/s^2 at
        1 m/s on the Pegasus Iris); the offset acts along the direction of
        travel and vanishes with it, so a hover feeds nothing forward.
        """
        if self.params.drag_per_mps <= 0.0 and self.params.drag_offset_mps2 <= 0.0:
            return np.zeros(3)
        speed = float(np.linalg.norm(planned_velocity))
        if speed <= 1e-6:
            return np.zeros(3)
        direction = planned_velocity / speed
        magnitude = self.params.drag_per_mps * speed + self.params.drag_offset_mps2
        return direction * magnitude

    def _damping(self, planned_velocity, measured_velocity):
        # type: (np.ndarray, np.ndarray) -> np.ndarray
        """Gain on the velocity error -- the term that actually tracks.

        Deliberately not clamped and not derived from the position error. The
        position loop is held gentle precisely because this one is here.
        """
        gains = np.array([self.params.velocity_damping_xy,
                          self.params.velocity_damping_xy,
                          self.params.velocity_damping_z], dtype=float)
        return gains * (planned_velocity - measured_velocity)

    def _hold_station(self, measured, measured_velocity, dt):
        # type: (np.ndarray, np.ndarray, float) -> AccelerationCommand
        """Fly to where the aircraft already is, because nothing else is asking.

        The hold point is latched the first time this happens, so the aircraft
        returns to it rather than ratcheting away from it. Velocity control has
        no position feedback of its own and an aircraft sent zero acceleration
        keeps whatever velocity it had, so "stop" has to be commanded as "hold
        this point".
        """
        if self._hold is None:
            self._hold = measured.copy()
        error = self._hold - measured
        damping = self._damping(np.zeros(3), measured_velocity)
        clamped = self._clamp_error(error)
        correction = np.array([self._correct(i, float(error[i]), float(clamped[i]), dt)
                               for i in range(3)])
        # A hold has no plan, so the ceiling is the margin alone. An aircraft
        # returning to a latched point after being pushed off it has no business
        # doing so at cruise speed, and this is the case where it is most likely
        # to be near whatever pushed it.
        wanted = self._limit_speed(damping + correction, measured_velocity, 0.0)
        command, saturated = limit_acceleration(wanted, self._limits)
        self._note_saturation(wanted, command)
        return AccelerationCommand(
            ax=float(command[0]), ay=float(command[1]), az=float(command[2]),
            yaw=self._yaw_cmd if self._yaw_cmd is not None else 0.0,
            position_error_m=float(np.linalg.norm(error)),
            reference_time_s=self._feed.last_reference_time,
            trajectory_id=self.trajectory_id,
            holding=True, saturated=saturated)

    def _slew_yaw(self, reference_yaw, dt):
        # type: (float, float) -> float
        """Step the commanded heading toward the reference, rate-limited."""
        ceiling = self.params.max_yaw_rate * self.params.yaw_rate_margin
        step = ceiling * dt
        error = normalize_angle(reference_yaw - self._yaw_cmd)
        return normalize_angle(self._yaw_cmd + max(-step, min(step, error)))
