"""Track a time-parameterised 3D reference on an aircraft that has physics.

A trajectory planner validated in a geometry-only simulator closes no loop at
all: the commanded state *is* the state, so "tracking" is a rename. Put the same
planner on an airframe -- simulated with real dynamics, or real -- and the
commanded state and the actual one come apart, from thrust lag, from a
mis-trimmed hover, from a draught, from an autopilot that smooths what it is
sent. Nothing in the planner notices, because the planner is told where the
aircraft *is*, replans from there, and emits another open-loop reference.

This is the layer that closes it. Given the planner's reference at time *t*
(position, velocity, acceleration, yaw) and the aircraft's measured state, it
emits a world-frame velocity:

.. code-block:: text

    v_cmd = v_ref + a_ref*lead  +  Kd*(v_ref - v_meas)  +  PI(clamp(p_ref - p_meas))
            \\_______________/     \\_________________/     \\____________________/
              the plan, led           what the inner            where it has
                                      loop has not yet          ended up, bounded
                                      delivered

The first two terms are the planner's intent replayed verbatim -- they are why
the aircraft flies the *shape* of the trajectory instead of chasing a point
along it. The third is the only part that knows the aircraft is real, and it is
capped well below the flight speed on purpose: a correction that can out-run the
trajectory is a correction that flies the aircraft, and then the planner's
dynamic-feasibility guarantees stop describing what the airframe does.

Yaw is passed through rather than servoed. An exploration planner chooses yaw to
point the sensor at what it wants to see next, so the heading *is* part of the
plan; the only thing done to it is a rate limit, which in normal operation is
inactive because the planner already rate-limited it.

Frames: any right-handed world frame with +z up. In this repo that is body-FLU /
world-ENU (REP-103). The tracker never touches the body frame, so it needs no
attitude beyond the heading.
"""
from __future__ import annotations

import math
from typing import Optional

from sparx_agency.core.common.types import TrajectoryPoint, normalize_angle
from sparx_agency.core.planning.trackers.drift_pid.pid import AxisPid
from sparx_agency.core.planning.trackers.reference_tracker_3d.params import (
    ReferenceTrackerParams,
)
from sparx_agency.core.planning.trackers.reference_tracker_3d.plan_state import (
    ENDPOINT, STALE, PlanState,
)
from sparx_agency.core.planning.trackers.reference_tracker_3d.progress import (
    limit_lead, split_error, unit,
)
from sparx_agency.core.planning.trackers.reference_tracker_3d.types import TrackedSetpoint


class ReferenceTracker3D:
    """Closes the loop between a planner's reference and an aircraft with physics.

    One instance per aircraft. It is stateful -- the PID integrators are the
    learned standing bias on each axis, and the commanded heading is slewed from
    its own previous value -- so it must be stepped at a steady rate and
    :meth:`reset` when the aircraft stops flying the plan.

    Args:
        params: Gains, limits and freshness rules. Defaults are for an indoor
            multirotor tracking a metre-per-second trajectory.
    """

    def __init__(self, params=None):
        # type: (Optional[ReferenceTrackerParams]) -> None
        self.params = params or ReferenceTrackerParams()
        self._pid = (
            AxisPid(self.params.horizontal_pid),
            AxisPid(self.params.horizontal_pid),
            AxisPid(self.params.vertical_pid),
        )
        self._yaw_cmd = None        # type: Optional[float]
        self._hold = None           # type: Optional[tuple]
        self._last_command = None   # type: Optional[tuple]
        self._last_reference_age = 0.0
        self._plan = PlanState(self.params.plan_state)
        #: Last direction the plan was actually travelling, for the error split
        #: and the lead limit. A finished plan reports zero velocity, and its
        #: last real heading is the only honest answer to "which way is ahead".
        self._travel = None         # type: Optional[tuple]
        #: Read-only breakdown of the last command, for diagnostics only.
        #:
        #: ``None`` before the first update, then a dict of world ``(x, y, z)``
        #: triples: ``feed_forward``, ``damping``, ``correction``, ``commanded``
        #: (their raw sum), ``clamped`` and ``smoothed`` (what was returned).
        #: Recording only the sum makes an over-aggressive gain indistinguishable
        #: from a large reference velocity. Nothing in the control law reads it.
        self.last_terms = None      # type: Optional[dict]

    def reset(self, yaw=None, hold_position=None):
        # type: (Optional[float], Optional[tuple]) -> None
        """Forget every accumulated correction and re-seed the commanded heading.

        Call this whenever the aircraft stops flying the plan -- before handing
        over from a takeoff climb, after a landing, on a planner restart. The
        integrators hold a bias learned in a different flight regime; carrying it
        across a handover puts a step into the first command of the new one.

        Args:
            yaw: Heading to start slewing from, radians. None re-seeds on the
                first reference instead.
            hold_position: World ``(x, y, z)`` to hold while no reference is
                fresh. None holds wherever the aircraft is when that first
                happens.
        """
        for axis in self._pid:
            axis.reset()
        self._yaw_cmd = None if yaw is None else normalize_angle(float(yaw))
        self._hold = None if hold_position is None else tuple(float(v) for v in hold_position)
        self._last_command = None
        self._last_reference_age = 0.0
        self._plan.reset()
        self._travel = None
        self.last_terms = None

    def on_new_trajectory(self):
        # type: () -> None
        """Drop the state that a newly committed trajectory invalidates.

        The integrators are deliberately *kept*: they hold a standing bias of
        the airframe, not of the trajectory, and FALCON commits a new curve
        roughly every 0.7 s -- resetting them that often would mean they never
        learn anything. What must go is the latched hold point, which belongs to
        a plan that no longer exists; carrying it across makes the aircraft fly
        back to the previous trajectory's endpoint after the new one has
        started.

        The remembered direction of travel goes with it, for the same reason:
        "which way is ahead" is a property of the curve, not of the airframe.
        """
        self._hold = None
        self._travel = None

    @property
    def commanded_yaw(self):
        # type: () -> Optional[float]
        """The heading last put on the wire, or None before the first update."""
        return self._yaw_cmd

    def update(self, reference, position, yaw, dt, velocity=None, reference_age=0.0,
               trajectory_id=None):
        # type: (Optional[TrajectoryPoint], tuple, float, float, Optional[tuple], float, object) -> TrackedSetpoint
        """Advance one control tick.

        The law is

        .. code-block:: text

            v_cmd = v_ref + a_ref*lead + Kd*(v_ref - v_meas) + PI(clamp(p_ref - p_meas))

        smoothed, then clamped. Each term is doing a different job and the
        middle one is doing most of the work -- see :mod:`~.params`.

        Args:
            reference: The planner's state for *now*: position, velocity,
                acceleration and yaw. None (or a reference older than
                ``reference_timeout_s``) makes the tracker hold station instead.
            position: The aircraft's measured world ``(x, y, z)``, metres.
            yaw: The aircraft's measured heading, radians CCW from +x.
            dt: Seconds since the previous call. Must be > 0.
            velocity: The aircraft's measured world ``(vx, vy, vz)``, m/s. Omit it
                and the damping term is simply absent -- the controller still
                works, less well. Pass it whenever it is measured rather than
                differenced, which on a simulator it always is.
            reference_age: How long ago ``reference`` was produced, seconds. The
                caller knows this and the tracker cannot: a reference arriving
                over a link is already old when it lands.
            trajectory_id: The planner's id for the curve this reference came
                from, when the caller has one. Used only to stop a freshly
                committed trajectory inheriting the previous one's
                finished-plan state.

        Returns:
            The velocity and heading to command, with the diagnostics that say
            whether the aircraft is actually flying the plan.

        Raises:
            ValueError: If ``dt`` is not positive.
        """
        if dt <= 0.0:
            raise ValueError("ReferenceTracker3D.update: dt must be > 0, got %r" % (dt,))

        measured = (float(position[0]), float(position[1]), float(position[2]))
        yaw = normalize_angle(float(yaw))
        if self._yaw_cmd is None:
            self._yaw_cmd = yaw

        self._last_reference_age = float(reference_age)
        verdict = self._plan.update(reference, reference_age,
                                    self.params.reference_timeout_s, dt,
                                    trajectory_id=trajectory_id)
        if verdict == STALE:
            return self._hold_station(measured, dt)
        if verdict == ENDPOINT:
            return self._hold_endpoint(reference, measured, dt)

        self._hold = None
        target = (float(reference.x), float(reference.y), float(reference.z))
        feed_forward = self._feed_forward(reference)
        error = tuple(target[i] - measured[i] for i in range(3))
        damping = self._damping(reference, velocity)

        travel = self._travel_direction(reference)
        # The aircraft being ahead of schedule is not an error to fly out of --
        # but only horizontally. The defect this bounds is backward flight on a
        # forward-facing airframe; the vertical axis has its own loop, its own
        # gains and its own clamp, and capping it against the direction of
        # TRAVEL would throttle a legitimate climb on a rising segment.
        bounded = self._limit_horizontal_lead(error, travel)
        correction = tuple(self._correct(i, bounded[i], error[i], dt)
                           for i in range(3))
        commanded = tuple(feed_forward[i] + damping[i] + correction[i] for i in range(3))
        clamped = self._clamp_velocity(commanded)
        command = self._smooth(clamped)
        self._record_terms(feed_forward, damping, correction, commanded, clamped, command)

        reference_yaw = yaw if reference.yaw is None else float(reference.yaw)
        self._yaw_cmd = self._slew_yaw(reference_yaw, dt)

        along, cross = split_error(error, travel)
        distance = math.sqrt(sum(component * component for component in error))
        return TrackedSetpoint(
            vx=command[0], vy=command[1], vz=command[2], yaw=self._yaw_cmd,
            position_error_m=distance,
            along_track_lag_m=along,
            cross_track_error_m=cross,
            yaw_error_rad=normalize_angle(reference_yaw - yaw),
            diverged=distance > self.params.max_position_error_m,
        )

    def _travel_direction(self, reference):
        # type: (TrajectoryPoint) -> Optional[tuple]
        """The plan's direction of travel, remembered across a stop.

        The reference's own velocity while it is moving, and the last one it had
        once it stops -- which is what makes the error split and the lead limit
        keep working past the end of a curve, where the velocity is zero and
        "ahead" would otherwise be undefined.
        """
        current = unit((reference.vx, reference.vy, reference.vz))
        if current is not None:
            self._travel = current
        return self._travel

    def _hold_endpoint(self, reference, measured, dt):
        # type: (TrajectoryPoint, tuple, float) -> TrackedSetpoint
        """Fly to the plan's last point and stop there.

        The plan has run out, so its final point is the last thing a planner
        actually vouched for: the aircraft finishes the path to it and holds,
        rather than either stopping short or treating a dead setpoint as a live
        one and chasing it at cruise. Beyond ``endpoint_reach_m`` the endpoint is
        too far to count as finishing the path -- nothing has confirmed the
        ground in between since the plan ended -- so the aircraft holds where it
        is and waits for a real trajectory.
        """
        endpoint = (float(reference.x), float(reference.y), float(reference.z))
        # Measured against what the aircraft is ACTUALLY flying to, which is the
        # latched hold once there is one. Checking only the incoming reference
        # lets an aircraft that has since been pushed away keep flying metres
        # back to a point no live plan has vouched for since: measured on the
        # recorded run, 93 ticks at 0.87 m/s toward a hold over 3 m away.
        target = self._hold if self._hold is not None else endpoint
        gap = math.sqrt(sum((target[i] - measured[i]) ** 2 for i in range(3)))
        if gap > self.params.plan_state.endpoint_reach_m:
            # Re-latch here. Bounded, not the ratcheting the plain hold warns
            # about: it can only fire once per excursion, and the new gap is 0.
            self._hold = None
            return self._hold_station(measured, dt, past_end=True)
        return self._hold_station(measured, dt, hold_point=endpoint, past_end=True)

    def _record_terms(self, feed_forward, damping, correction, commanded,
                      clamped, smoothed):
        # type: (tuple, tuple, tuple, tuple, tuple, tuple) -> None
        """Store the terms this tick already computed, for :attr:`last_terms`."""
        self.last_terms = {
            "feed_forward": feed_forward,
            "damping": damping,
            "correction": correction,
            "commanded": commanded,
            "clamped": clamped,
            "smoothed": smoothed,
        }

    def _limit_horizontal_lead(self, error, travel):
        # type: (tuple, Optional[tuple]) -> tuple
        """Bound the backward along-track error in the horizontal plane only.

        Args:
            error: World ``(dx, dy, dz)`` from the aircraft to the reference.
            travel: The plan's direction of travel, or None.

        Returns:
            The error with its horizontal backward component limited and its
            vertical component untouched.
        """
        if travel is None:
            return tuple(float(component) for component in error)
        flat = limit_lead((error[0], error[1], 0.0),
                          (travel[0], travel[1], 0.0),
                          self.params.max_lead_error_m)
        return (flat[0], flat[1], error[2])

    def _correct(self, axis, error, raw_error, dt):
        # type: (int, float, float, float) -> float
        """One axis of position feedback, on a clamped error.

        The clamp is the collision-avoidance property (see
        ``params.position_error_clamp_m``): it bounds how hard the loop pulls
        toward a distant reference without changing which way it pulls, so a
        lagging aircraft rounds an obstacle instead of cutting through it.

        Integral separation on top of that: outside ``integral_band_m`` the axis
        is not holding a standing bias, it is *travelling*. Integrating a large
        error charges a correction that only finishes arriving once the error is
        gone, and then pushes the aircraft past the reference by roughly that
        much -- measured as a 9 cm overshoot on a 1 m step, 1 cm with this gate.
        """
        # Judged on the RAW error: the lead limit shrinks a backward error to
        # at most max_lead_error_m (0.25), which is inside integral_band_m
        # (0.5), so testing the bounded value would let an aircraft far AHEAD of
        # its reference charge the integrator -- the exact windup the band
        # exists to prevent, on the one side the limiter touches.
        near = abs(raw_error) <= self.params.integral_band_m
        clamp = self.params.position_error_clamp_m
        clamped = max(-clamp, min(clamp, error))
        return self._pid[axis].update(clamped, dt, integrate=near)

    def _damping(self, reference, velocity):
        # type: (TrajectoryPoint, Optional[tuple]) -> tuple
        """Gain on the velocity error, which is what closes an inner loop's lag.

        Deliberately NOT clamped, and deliberately not derived from the position
        error. It is the term that does the tracking: the position loop is held
        gentle precisely because this one is here, and clamping it would put the
        lag back.
        """
        if velocity is None:
            return (0.0, 0.0, 0.0)
        gains = (self.params.velocity_damping_xy, self.params.velocity_damping_xy,
                 self.params.velocity_damping_z)
        planned = (float(reference.vx), float(reference.vy), float(reference.vz))
        return tuple(gains[i] * (planned[i] - float(velocity[i])) for i in range(3))

    def _smooth(self, command):
        # type: (tuple) -> tuple
        """Exponentially smooth the output, so the airframe is not stepped.

        Applied after the clamp rather than before, so the smoothed command is
        also inside the limits; a smoother that can overshoot its own ceiling is
        not a limit.
        """
        alpha = self.params.command_smoothing_alpha
        if alpha >= 1.0 or self._last_command is None:
            self._last_command = command
            return command
        smoothed = tuple(alpha * command[i] + (1.0 - alpha) * self._last_command[i]
                         for i in range(3))
        self._last_command = smoothed
        return smoothed

    def _feed_forward(self, reference):
        # type: (TrajectoryPoint) -> tuple
        """The planner's own velocity, plus its acceleration led by a fixed time."""
        lead = self.params.accel_lead_s
        return (
            float(reference.vx) + lead * float(reference.ax),
            float(reference.vy) + lead * float(reference.ay),
            float(reference.vz) + lead * float(reference.az),
        )

    def _hold_station(self, measured, dt, hold_point=None, past_end=False):
        # type: (tuple, float, Optional[tuple], bool) -> TrackedSetpoint
        """Fly to a single point and stay on it, because nothing is asking for more.

        Velocity control has no position feedback of its own, so "stop" has to be
        commanded as "hold this point" -- an aircraft sent zero velocity drifts,
        and was measured drifting three metres sideways in five seconds. The hold
        point is latched the first time this happens so the aircraft returns to it
        rather than ratcheting away from it.

        Args:
            measured: The aircraft's measured world position.
            dt: Seconds since the previous call.
            hold_point: Where to hold. Defaults to wherever the aircraft is,
                which is the right answer when nothing has vouched for anywhere
                else; the finished-plan path passes the plan's own last point.
            past_end: Whether this hold is a finished plan rather than a missing
                one. Reported, not acted on.
        """
        if self._hold is None:
            self._hold = tuple(measured) if hold_point is None else tuple(hold_point)
        error = tuple(self._hold[i] - measured[i] for i in range(3))
        correction = tuple(self._correct(i, error[i], error[i], dt) for i in range(3))
        clamped = self._clamp_velocity(correction)
        command = self._smooth(clamped)
        zero = (0.0, 0.0, 0.0)
        self._record_terms(zero, zero, correction, correction, clamped, command)
        distance = math.sqrt(sum(component * component for component in error))
        # Split against the last direction the plan travelled: this is the case
        # the explicit-direction split exists for, and reporting 0.0 lag here
        # would hide exactly the overshoot the settle band above reacts to.
        along, cross = split_error(error, self._travel)
        return TrackedSetpoint(
            vx=command[0], vy=command[1], vz=command[2],
            yaw=self._yaw_cmd if self._yaw_cmd is not None else 0.0,
            position_error_m=distance,
            along_track_lag_m=along, cross_track_error_m=cross,
            holding=True, past_end=past_end,
        )

    def _clamp_velocity(self, command):
        # type: (tuple) -> tuple
        """Cap the horizontal speed and the climb rate, preserving direction.

        The horizontal pair is scaled together rather than clipped per axis: per
        axis clipping turns an over-speed diagonal into a different heading,
        which is a steering error dressed up as a speed limit.
        """
        limits = self.params.limits
        vx, vy, vz = command
        speed = math.hypot(vx, vy)
        if speed > limits.max_speed_xy and speed > 0.0:
            scale = limits.max_speed_xy / speed
            vx, vy = vx * scale, vy * scale
        vz = max(-limits.max_speed_z, min(limits.max_speed_z, vz))
        return vx, vy, vz

    def _slew_yaw(self, reference_yaw, dt):
        # type: (float, float) -> float
        """Step the commanded heading toward the reference, rate-limited.

        The limit sits above the planner's own yaw-rate ceiling
        (``yaw_rate_margin``), so a planner that respects its own limits passes
        through untouched. It bites only when the reference heading *jumps* --
        which happens when the aircraft has fallen behind far enough that the
        planner replanned from somewhere else.
        """
        ceiling = self.params.limits.max_yaw_rate * self.params.yaw_rate_margin
        step = ceiling * dt
        error = normalize_angle(reference_yaw - self._yaw_cmd)
        return normalize_angle(self._yaw_cmd + max(-step, min(step, error)))
