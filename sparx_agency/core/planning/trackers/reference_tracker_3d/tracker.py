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

    v_cmd = v_ref  +  a_ref * accel_lead_s  +  PID(p_ref - p_meas)
            \\____/    \\___________________/    \\_________________/
          the plan       the plan's own          what physics did
                        curvature, led

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
        self._yaw_cmd = None      # type: Optional[float]
        self._hold = None         # type: Optional[tuple]
        self._last_reference_age = 0.0

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
        self._last_reference_age = 0.0

    @property
    def commanded_yaw(self):
        # type: () -> Optional[float]
        """The heading last put on the wire, or None before the first update."""
        return self._yaw_cmd

    def update(self, reference, position, yaw, dt, reference_age=0.0):
        # type: (Optional[TrajectoryPoint], tuple, float, float, float) -> TrackedSetpoint
        """Advance one control tick.

        Args:
            reference: The planner's state for *now*: position, velocity,
                acceleration and yaw. None (or a reference older than
                ``reference_timeout_s``) makes the tracker hold station instead.
            position: The aircraft's measured world ``(x, y, z)``, metres.
            yaw: The aircraft's measured heading, radians CCW from +x.
            dt: Seconds since the previous call. Must be > 0.
            reference_age: How long ago ``reference`` was produced, seconds. The
                caller knows this and the tracker cannot: a reference arriving
                over a link is already old when it lands.

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

        stale = reference is None or reference_age > self.params.reference_timeout_s
        self._last_reference_age = float(reference_age)
        if stale:
            return self._hold_station(measured, dt)

        self._hold = None
        target = (float(reference.x), float(reference.y), float(reference.z))
        feed_forward = self._feed_forward(reference)
        error = tuple(target[i] - measured[i] for i in range(3))

        correction = tuple(self._correct(i, error[i], dt) for i in range(3))
        command = tuple(feed_forward[i] + correction[i] for i in range(3))
        command = self._clamp_velocity(command)

        reference_yaw = yaw if reference.yaw is None else float(reference.yaw)
        self._yaw_cmd = self._slew_yaw(reference_yaw, dt)

        along, cross = _split_error(error, (reference.vx, reference.vy, reference.vz))
        distance = math.sqrt(sum(component * component for component in error))
        return TrackedSetpoint(
            vx=command[0], vy=command[1], vz=command[2], yaw=self._yaw_cmd,
            position_error_m=distance,
            along_track_lag_m=along,
            cross_track_error_m=cross,
            yaw_error_rad=normalize_angle(reference_yaw - yaw),
            diverged=distance > self.params.max_position_error_m,
        )

    def _correct(self, axis, error, dt):
        # type: (int, float, float) -> float
        """One axis of feedback, with the integrator gated by how far off we are.

        Integral separation: outside ``integral_band_m`` the axis is not holding
        a standing bias, it is *travelling*. Integrating a large error charges a
        correction that only finishes arriving once the error is gone, and then
        pushes the aircraft past the reference by roughly that much -- measured
        as a 9 cm overshoot on a 1 m step, which fell to 1 cm with this gate.
        """
        near = abs(error) <= self.params.integral_band_m
        return self._pid[axis].update(error, dt, integrate=near)

    def _feed_forward(self, reference):
        # type: (TrajectoryPoint) -> tuple
        """The planner's own velocity, plus its acceleration led by a fixed time."""
        lead = self.params.accel_lead_s
        return (
            float(reference.vx) + lead * float(reference.ax),
            float(reference.vy) + lead * float(reference.ay),
            float(reference.vz) + lead * float(reference.az),
        )

    def _hold_station(self, measured, dt):
        # type: (tuple, float) -> TrackedSetpoint
        """Fly to where the aircraft already is, because nothing else is asking.

        Velocity control has no position feedback of its own, so "stop" has to be
        commanded as "hold this point" -- an aircraft sent zero velocity drifts,
        and was measured drifting three metres sideways in five seconds. The hold
        point is latched the first time this happens so the aircraft returns to it
        rather than ratcheting away from it.
        """
        if self._hold is None:
            self._hold = measured
        error = tuple(self._hold[i] - measured[i] for i in range(3))
        command = self._clamp_velocity(tuple(
            self._correct(i, error[i], dt) for i in range(3)))
        distance = math.sqrt(sum(component * component for component in error))
        return TrackedSetpoint(
            vx=command[0], vy=command[1], vz=command[2],
            yaw=self._yaw_cmd if self._yaw_cmd is not None else 0.0,
            position_error_m=distance, holding=True,
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


def _split_error(error, reference_velocity):
    # type: (tuple, tuple) -> tuple
    """Split a position error into along-track lag and cross-track offset.

    Along-track is measured along the reference's direction of travel, so a
    positive value means the aircraft is behind where it should be. With the
    reference stationary there is no direction of travel and the whole error is
    reported as cross-track, which is the safe reading: an offset from a hover
    point is not lateness.

    Returns:
        ``(along_track_lag_m, cross_track_error_m)``.
    """
    speed = math.sqrt(sum(float(v) * float(v) for v in reference_velocity))
    magnitude = math.sqrt(sum(component * component for component in error))
    if speed <= 1e-6:
        return 0.0, magnitude
    direction = tuple(float(v) / speed for v in reference_velocity)
    # The aircraft is behind the reference when the error points the way the
    # reference is travelling, hence the sign convention: error = ref - measured.
    along = sum(error[i] * direction[i] for i in range(3))
    cross_squared = max(magnitude * magnitude - along * along, 0.0)
    return along, math.sqrt(cross_squared)
