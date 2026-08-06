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
from sparx_agency.core.control.trajectory_tracking.params import TrajectoryTrackerParams
from sparx_agency.core.control.trajectory_tracking.types import AccelerationCommand
from sparx_agency.core.planning.trackers.drift_pid.pid import AxisPid
from sparx_agency.core.planning.trajectories.bspline.projection import TrajectoryProjector


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
        self._projector = TrajectoryProjector(self.params.projection)
        self._current = None        # type: Optional[object]
        self._pending = None        # type: Optional[object]
        self._yaw_cmd = None        # type: Optional[float]
        self._hold = None           # type: Optional[np.ndarray]
        self._saturated = False

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
        self._projector.reset()
        self._yaw_cmd = None if yaw is None else normalize_angle(float(yaw))
        self._hold = None if hold_position is None else np.asarray(hold_position, dtype=float)
        self._saturated = False

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
        newest = self._pending or self._current
        if newest is not None and trajectory.traj_id <= newest.traj_id:
            return False
        self._pending = trajectory
        return True

    @property
    def trajectory_id(self):
        # type: () -> int
        """Id of the trajectory being flown, or -1 when none is."""
        return -1 if self._current is None else int(self._current.traj_id)

    @property
    def commanded_yaw(self):
        # type: () -> Optional[float]
        """The heading last put on the wire, or None before the first tick."""
        return self._yaw_cmd

    def update(self, position, velocity, yaw, dt, now_s, follow=True):
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
            follow: False holds station instead of tracking. This is how a
                caller acts on FALCON condemning its own live trajectory: the
                aircraft brakes toward a latched point rather than carrying its
                momentum into the obstacle that was just found.

        Returns:
            The acceleration and heading to command, with the diagnostics that
            say whether the aircraft is actually flying the plan.

        Raises:
            ValueError: If ``dt`` is not positive.
        """
        if dt <= 0.0:
            raise ValueError("TrajectoryTracker.update: dt must be > 0, got %r" % (dt,))
        measured = np.asarray(position, dtype=float).reshape(3)
        measured_velocity = np.asarray(velocity, dtype=float).reshape(3)
        yaw = normalize_angle(float(yaw))
        if self._yaw_cmd is None:
            self._yaw_cmd = yaw

        self._promote(now_s)
        if not follow or not self._usable(now_s):
            return self._hold_station(measured, measured_velocity, dt)

        self._hold = None
        elapsed = self._current.elapsed(now_s)
        projected_time = self._project(measured, elapsed)
        # Once the schedule has run out the reference is pinned to the stopped
        # endpoint, whatever the projection says. This is not a tidying detail:
        # the search returns a time marginally *inside* the curve, and sampling
        # there hands back the plan's full cruise velocity as a feedforward. An
        # aircraft given that keeps flying at cruise past the end of a
        # trajectory -- measured, a metre and a half into space FALCON never
        # checked -- while the position term alone gently disagrees.
        past_end = elapsed >= self._current.duration
        reference_time = (self._current.duration if past_end
                          else min(projected_time + self.params.projection.lookahead_s,
                                   self._current.duration))
        reference = self._current.sample(reference_time)
        target = np.array([reference.x, reference.y, reference.z], dtype=float)
        planned_velocity = np.array([reference.vx, reference.vy, reference.vz], dtype=float)
        error = target - measured

        feed_forward = np.array([reference.ax, reference.ay, reference.az], dtype=float)
        # The schedule deficit is measured in SPACE, against where the plan says
        # the aircraft should be *now*, and not as a difference of times on this
        # curve. See _schedule_deficit: the obvious time-based version reads zero
        # in precisely the case that matters.
        gap, schedule_lag_m, cross = self._diagnose(measured, planned_velocity, elapsed)
        catchup = self._catchup(planned_velocity, schedule_lag_m)
        damping = self._damping(planned_velocity + catchup, measured_velocity)
        correction = np.array([self._correct(i, float(error[i]), dt) for i in range(3)])
        command, self._saturated = limit_acceleration(
            feed_forward + damping + correction, self.params.limits)

        reference_yaw = yaw if reference.yaw is None else float(reference.yaw)
        self._yaw_cmd = self._slew_yaw(reference_yaw, dt)

        along = schedule_lag_m
        return AccelerationCommand(
            ax=float(command[0]), ay=float(command[1]), az=float(command[2]),
            yaw=self._yaw_cmd,
            yaw_rate=float(reference.yaw_rate or 0.0),
            jx=reference.jx, jy=reference.jy, jz=reference.jz,
            position_error_m=gap,
            along_track_lag_m=along,
            cross_track_error_m=cross,
            yaw_error_rad=normalize_angle(reference_yaw - yaw),
            reference_time_s=reference_time,
            trajectory_id=int(self._current.traj_id),
            diverged=gap > self.params.max_position_error_m,
            past_end=past_end,
            saturated=self._saturated)

    def _promote(self, now_s):
        # type: (float) -> None
        """Swap in the queued trajectory once its own start time has arrived."""
        if self._pending is None:
            return
        if self._current is None or self._pending.start_time_s <= float(now_s):
            self._current = self._pending
            self._pending = None
            # The new curve is a different parameterisation, so last tick's
            # position along the old one means nothing on it.
            self._projector.reset()
            for axis in self._pid:
                axis.reset_derivative()

    def _usable(self, now_s):
        # type: (float) -> bool
        """Whether the held trajectory is worth following at ``now_s``."""
        if self._current is None:
            return False
        elapsed = self._current.elapsed(now_s)
        if elapsed < 0.0:
            return False
        return elapsed < self._current.duration + self.params.max_trajectory_age_s

    def _project(self, measured, elapsed):
        # type: (np.ndarray, float) -> float
        """Where on the curve the aircraft currently is, in seconds from its start."""
        if not self.params.use_projection:
            return min(max(0.0, elapsed), self._current.duration)
        return self._projector.project(self._current, measured)

    def _diagnose(self, measured, planned_velocity, elapsed):
        # type: (np.ndarray, np.ndarray, float) -> tuple
        """Split the gap from the plan into "late" and "sideways".

        One displacement, decomposed once: from the aircraft to where the plan
        says it should be **at this instant**, resolved along and across the
        direction of travel. So ``along**2 + cross**2 == gap**2`` always, and
        "1.3 m of gap" is always answerable as how much of it is each. The two
        halves are not equally dangerous -- being late is benign, being sideways
        is what hits walls -- and the control law treats them differently, so
        they must not be conflated.

        **Measured in space, deliberately.** The obvious way to find the lag is
        ``elapsed - projected_time``, a difference of two times on the current
        curve, and it is wrong in exactly the case the catch-up term exists for.
        FALCON does not plan the next curve from the aircraft; it plans it from
        its **own previous curve**, at ``now + replan_duration``. So a lagging
        aircraft is behind the new curve's *start*, a curve has no negative
        time, the projection clamps at zero, and the deficit disappears.
        Measured on a real flight: a true 1.30 m of lag read as 0.03 m, and the
        catch-up contributed 0.03 m/s instead of its 0.5 m/s ceiling. FALCON
        replans about four times a second, so no lag ever survived on one curve
        long enough to be seen.

        The distance to the *nearest* point on the curve is the other tempting
        definition of cross-track, and it fails the same way: with the aircraft
        directly behind the start of a straight curve it reports the full 1.3 m
        as cross-track, because the nearest point really is 1.3 m away. Honest
        about the curve, useless as a measure of being off the path.

        Returns:
            ``(gap_m, along_track_lag_m, cross_track_error_m)``. Positive
            ``along`` means late.
        """
        on_schedule = self._current.position_at(
            min(max(0.0, elapsed), self._current.duration))
        offset = on_schedule - measured
        gap = float(np.linalg.norm(offset))
        speed = float(np.linalg.norm(planned_velocity))
        if speed <= 1e-6:
            # A stationary reference has no direction of travel, so none of the
            # gap can be called lateness. Reporting it all as cross-track is the
            # safe reading: an offset from a hover point is not being late.
            return gap, 0.0, gap
        direction = planned_velocity / speed
        along = float(np.dot(offset, direction))
        cross = float(np.linalg.norm(offset - along * direction))
        return gap, along, cross

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

    def _correct(self, axis, error, dt):
        # type: (int, float, float) -> float
        """One axis of position feedback, on a clamped error.

        Integral separation on top of the clamp: outside ``integral_band_m`` the
        axis is not holding a standing bias, it is *travelling*. The integrator
        is also frozen while the command is saturated, because a correction that
        cannot be applied should not keep accumulating.
        """
        near = abs(error) <= self.params.integral_band_m
        clamp = self.params.position_error_clamp_m
        clamped = max(-clamp, min(clamp, error))
        return self._pid[axis].update(clamped, dt, integrate=near and not self._saturated)

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
        correction = np.array([self._correct(i, float(error[i]), dt) for i in range(3)])
        command, self._saturated = limit_acceleration(damping + correction, self.params.limits)
        return AccelerationCommand(
            ax=float(command[0]), ay=float(command[1]), az=float(command[2]),
            yaw=self._yaw_cmd if self._yaw_cmd is not None else 0.0,
            position_error_m=float(np.linalg.norm(error)),
            reference_time_s=self._projector.last_t,
            trajectory_id=self.trajectory_id,
            holding=True, saturated=self._saturated)

    def _slew_yaw(self, reference_yaw, dt):
        # type: (float, float) -> float
        """Step the commanded heading toward the reference, rate-limited."""
        ceiling = self.params.max_yaw_rate * self.params.yaw_rate_margin
        step = ceiling * dt
        error = normalize_angle(reference_yaw - self._yaw_cmd)
        return normalize_angle(self._yaw_cmd + max(-step, min(step, error)))
