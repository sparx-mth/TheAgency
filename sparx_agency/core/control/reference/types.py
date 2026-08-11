"""What the plan says the aircraft should be doing, and how far off it is.

One value object, produced once per control tick and consumed by whichever
backend is flying the aircraft. It exists so that "where on the curve are we,
and how wrong is that" is answered in exactly one place: the acceleration
backend and the velocity backend disagree about what to *command*, but they must
never disagree about what the plan *says*.

The error decomposition is the part worth reading. One displacement -- from the
aircraft to where the plan says it should be at this instant -- resolved along
and across the direction of travel, so ``along**2 + cross**2 == gap**2`` always
and a gap is always fully attributable. The two halves are not equally
dangerous: being late is benign, being sideways is what hits walls, and any
controller that conflates them will trade the cheap error for the expensive one.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from sparx_agency.core.common.types import TrajectoryPoint


@dataclass(frozen=True)
class ReferenceSample:
    """The plan's state at the reference point, with the error against it.

    Attributes:
        point: The full reference state -- position, velocity, acceleration,
            jerk, yaw and yaw rate -- sampled off the curve at
            ``reference_time_s``.
        reference_time_s: Where on the trajectory the reference was taken,
            seconds from its start. With projection enabled this is where the
            aircraft *is* on the plan, which is not the elapsed time and is the
            more useful number of the two.
        elapsed_s: Seconds since the trajectory began, on the flying clock.
            Where the aircraft *should* be, as opposed to where it is.
        gap_m: Distance from the aircraft to where the plan says it should be at
            this instant. Not to the nearest point on the curve -- see
            :mod:`~sparx_agency.core.control.reference.diagnosis`.
        along_track_lag_m: Component of that gap along the direction of travel.
            Positive means late.
        cross_track_error_m: Component perpendicular to it.
        trajectory_id: FALCON's id for the curve being flown.
        past_end: True once the schedule has run out and the reference is pinned
            to the stopped endpoint. Normal for a second between replans; a
            standing condition means the planner stopped.
        duration_s: The trajectory's own length, seconds.
    """

    point: TrajectoryPoint
    reference_time_s: float
    elapsed_s: float
    gap_m: float
    along_track_lag_m: float
    cross_track_error_m: float
    trajectory_id: int
    past_end: bool
    duration_s: float

    @property
    def position(self):
        # type: () -> tuple
        """The reference position as a plain ``(x, y, z)`` triple."""
        return self.point.x, self.point.y, self.point.z

    @property
    def velocity(self):
        # type: () -> tuple
        """The reference velocity as a plain ``(vx, vy, vz)`` triple."""
        return self.point.vx, self.point.vy, self.point.vz

    @property
    def acceleration(self):
        # type: () -> tuple
        """The reference acceleration as a plain ``(ax, ay, az)`` triple."""
        return self.point.ax, self.point.ay, self.point.az

    @property
    def jerk(self):
        # type: () -> tuple
        """The reference jerk as a plain ``(jx, jy, jz)`` triple."""
        return self.point.jx, self.point.jy, self.point.jz

    @property
    def yaw(self):
        # type: () -> Optional[float]
        """The reference heading, radians, or None if the curve carries none."""
        return self.point.yaw

    @property
    def yaw_rate(self):
        # type: () -> float
        """The reference yaw rate, rad/s."""
        return float(self.point.yaw_rate or 0.0)
