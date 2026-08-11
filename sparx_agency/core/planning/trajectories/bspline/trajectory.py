"""One FALCON trajectory: a position curve, a yaw curve, and their derivatives.

FALCON publishes a trajectory as two independent B-splines -- where to be, and
which way to look -- plus the wall-clock instant the curve begins. Its
``traj_server`` then differentiates the position curve three times and the yaw
curve once, and samples all six at 100 Hz. This class is the same six curves,
built the same way, sampled on the caller's own clock instead.

The yaw curve being independent is not an implementation detail: on a multirotor
heading is decoupled from translation, so an exploration planner is free to aim
the sensor wherever it likes while flying wherever it likes. Nothing here mixes
the two, and neither should a controller.
"""
from __future__ import annotations

from typing import Optional, Sequence

import numpy as np

from sparx_agency.core.common.types import TrajectoryPoint, normalize_angle
from sparx_agency.core.planning.trajectories.bspline.non_uniform_bspline import (
    NonUniformBspline,
)


class BsplineTrajectory:
    """A position B-spline and a yaw B-spline, with every derivative a
    controller needs.

    Immutable once built. Derivatives are taken at construction -- three from
    the position curve and one from the yaw curve -- because that cost belongs
    once per trajectory, not once per control tick.

    Args:
        position: The position curve, in metres, D = 3.
        yaw: The heading curve, in radians, D = 1.
        start_time_s: When the curve begins, on the clock the caller will later
            sample against.
        traj_id: FALCON's monotonically increasing trajectory identifier. The
            only signal that says a *new* plan exists rather than a re-send.

    Raises:
        ValueError: If the position curve is not 3D or the yaw curve not 1D.
    """

    def __init__(self, position, yaw, start_time_s, traj_id):
        # type: (NonUniformBspline, NonUniformBspline, float, int) -> None
        if position.dimension != 3:
            raise ValueError("the position curve must be 3D, got %dD" % (position.dimension,))
        if yaw.dimension != 1:
            raise ValueError("the yaw curve must be 1D, got %dD" % (yaw.dimension,))
        self.position = position
        self.velocity = position.derivative()
        self.acceleration = self.velocity.derivative()
        self.jerk = self.acceleration.derivative()
        self.yaw = yaw
        self.yaw_rate = yaw.derivative()
        self.start_time_s = float(start_time_s)
        self.traj_id = int(traj_id)

    @classmethod
    def from_falcon(cls, order, knots, position_points, yaw_points, yaw_dt,
                    start_time_s, traj_id):
        # type: (int, Sequence, Sequence, Sequence, float, float, int) -> BsplineTrajectory
        """Rebuild the trajectory from the fields of a ``trajectory/Bspline`` message.

        The construction rules are FALCON's, and the asymmetry between the two
        curves is in the wire format: the position curve carries an **explicit
        knot vector**, the yaw curve only a **knot interval**. Read nothing more
        into that. FALCON has the machinery to reparameterise the position knots
        for velocity feasibility (``NonUniformBspline::reallocateTime``), which
        would make them non-uniform -- but nothing in the build this flies
        against ever calls it, so in practice the transmitted knots are evenly
        spaced and the explicit vector is generality, not information. An
        earlier version of this docstring claimed the optimiser reparameterises;
        it does not, and the flight data shows the consequence -- planned speeds
        exceed the configured limit, since feasibility is only a soft cost.

        Args:
            order: Spline degree of the position curve, 3 in practice.
            knots: The position curve's explicit knot vector.
            position_points: ``(N, 3)`` control points, metres.
            yaw_points: ``(M,)`` control points, radians.
            yaw_dt: The yaw curve's uniform knot interval, seconds.
            start_time_s: When the trajectory begins.
            traj_id: FALCON's trajectory identifier.

        Returns:
            The rebuilt trajectory.
        """
        position = NonUniformBspline(np.asarray(position_points, dtype=float).reshape(-1, 3),
                                     int(order), 0.1, knots=knots)
        # Degree 3 regardless of the message's `order` field, matching the C++:
        # the yaw curve's degree is not transmitted and is not the position
        # curve's to borrow.
        yaw = NonUniformBspline(np.asarray(yaw_points, dtype=float).reshape(-1, 1),
                                3, float(yaw_dt))
        return cls(position, yaw, start_time_s, traj_id)

    @property
    def duration(self):
        # type: () -> float
        """How long the position curve lasts, seconds.

        The yaw curve's own span may differ; it is clamped rather than
        extrapolated, which is what the C++ does.
        """
        return self.position.duration

    def elapsed(self, now_s):
        # type: (float) -> float
        """Seconds into the trajectory at wall-clock ``now_s``.

        Negative before the trajectory starts -- which is a normal state, not an
        error: FALCON deliberately begins each new curve a planning-time into
        the future so it joins the one still being flown.
        """
        return float(now_s) - self.start_time_s

    def is_active(self, now_s):
        # type: (float) -> bool
        """Whether ``now_s`` falls inside the trajectory's own time span."""
        elapsed = self.elapsed(now_s)
        return 0.0 <= elapsed < self.duration

    def sample(self, t):
        # type: (float) -> TrajectoryPoint
        """The full reference state at ``t`` seconds into the trajectory.

        Past the end the position and heading are held and every derivative goes
        to zero, so a controller that overruns brakes to a hover on the last
        point rather than extrapolating off the end of a plan into unmapped
        space. Before the start the first point is held, for the same reason.

        Args:
            t: Seconds since the trajectory began.

        Returns:
            Position, velocity, acceleration, jerk, yaw and yaw rate.
        """
        clamped = min(max(0.0, float(t)), self.duration)
        stopped = float(t) >= self.duration or float(t) < 0.0
        position = self.position.evaluate_at_time(clamped)
        heading = normalize_angle(float(self.yaw.evaluate_at_time(clamped)[0]))
        if stopped:
            zero = np.zeros(3, dtype=float)
            velocity, acceleration, jerk, heading_rate = zero, zero, zero, 0.0
        else:
            velocity = self.velocity.evaluate_at_time(clamped)
            acceleration = self.acceleration.evaluate_at_time(clamped)
            jerk = self.jerk.evaluate_at_time(clamped)
            heading_rate = float(self.yaw_rate.evaluate_at_time(clamped)[0])
        return TrajectoryPoint(
            t=clamped,
            x=float(position[0]), y=float(position[1]), z=float(position[2]),
            vx=float(velocity[0]), vy=float(velocity[1]), vz=float(velocity[2]),
            ax=float(acceleration[0]), ay=float(acceleration[1]), az=float(acceleration[2]),
            jx=float(jerk[0]), jy=float(jerk[1]), jz=float(jerk[2]),
            yaw=heading, yaw_rate=heading_rate)

    def sample_at(self, now_s):
        # type: (float) -> TrajectoryPoint
        """The reference state at a wall-clock instant. See :meth:`sample`."""
        return self.sample(self.elapsed(now_s))

    def position_at(self, t):
        # type: (float) -> np.ndarray
        """Just the position, for the projection search that calls it in a loop."""
        return self.position.evaluate_at_time(min(max(0.0, float(t)), self.duration))

    def with_start_time(self, start_time_s):
        # type: (float) -> BsplineTrajectory
        """The same curves, timed from a different instant.

        The curves are shared, not copied: only the instant they are counted
        from changes. This exists because a trajectory's start time is stamped
        on *the planner's* clock, and the aircraft flying it does not always
        share that clock -- a simulator running at a fraction of real time is
        handed a schedule it cannot keep, and lags the plan by a distance that
        grows with how far behind real time it is. Re-basing the start time onto
        the clock the aircraft actually experiences makes the schedule
        achievable again without touching the geometry.

        Args:
            start_time_s: The instant the trajectory begins, on the caller's clock.

        Returns:
            A new trajectory over the same curves.
        """
        return BsplineTrajectory(self.position, self.yaw, start_time_s, self.traj_id)
