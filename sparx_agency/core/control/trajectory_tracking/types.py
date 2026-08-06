"""What the outer loop emits, and what it reports about itself.

The command is a **world-frame acceleration plus an absolute heading**. Not a
velocity, and not a position:

* A *position* setpoint hands the trajectory back to the autopilot's own
  position controller, which re-shapes it with its own smoother and its own
  limits. The plan's timing is lost, and on PX4 offboard specifically the
  aircraft was measured closing a one-metre gap at one centimetre per second.
* A *velocity* setpoint keeps the autopilot's velocity loop in the chain. That
  loop runs at tens of Hz off the same position estimate this one uses, so it
  adds a stage of lag without adding any information -- and that lag is the
  metre of tracking error this whole package exists to remove.
* An *acceleration* is the last quantity expressible in world coordinates before
  the airframe's own geometry takes over. It is also, for a multirotor, the same
  statement as an attitude.

Jerk rides along unchanged. The outer loop has no use for it, but the flatness
stage does -- it is the attitude feedforward -- and routing it through here keeps
the stages in a line instead of both reaching into the trajectory.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class AccelerationCommand:
    """One tick of outer-loop output, plus everything needed to judge it.

    Attributes:
        ax: World-frame acceleration command along +x, m/s^2, excluding gravity.
        ay: The same along +y.
        az: The same along +z (up).
        yaw: Absolute heading to command, radians CCW from +x. Already slewed,
            so a caller can pass it straight on.
        yaw_rate: Planned yaw rate, rad/s, for the attitude feedforward.
        jx: Trajectory jerk along +x, m/s^3, passed through for the flatness
            stage's rate feedforward. Zero while holding station.
        jy: The same along +y.
        jz: The same along +z.
        position_error_m: Distance from the aircraft to the reference point.
            The single number that says whether it is flying the plan.
        along_track_lag_m: Component of that error along the reference's
            direction of travel. Positive means late, which is benign.
        cross_track_error_m: Component perpendicular to it. This is the one that
            flies into walls -- same magnitude, entirely different meaning.
        yaw_error_rad: Reference heading minus measured, wrapped.
        reference_time_s: Where on the trajectory the reference was taken, in
            seconds from its start. With projection enabled this is where the
            aircraft *is* on the plan, which is not the same as the elapsed time
            and is the more useful number of the two.
        trajectory_id: FALCON's id for the trajectory being flown, or -1 when
            none is.
        diverged: True while the position error exceeds its ceiling. Advisory --
            the loop keeps trying.
        holding: True when no trajectory is being followed and the loop is
            holding station instead.
        past_end: True when the trajectory has run out and the aircraft is
            flying to its final point. Normal for a second between replans; a
            standing condition means the planner stopped.
        saturated: True when the command hit the airframe's acceleration limits.
    """

    ax: float
    ay: float
    az: float
    yaw: float
    yaw_rate: float = 0.0
    jx: float = 0.0
    jy: float = 0.0
    jz: float = 0.0
    position_error_m: float = 0.0
    along_track_lag_m: float = 0.0
    cross_track_error_m: float = 0.0
    yaw_error_rad: float = 0.0
    reference_time_s: float = 0.0
    trajectory_id: int = -1
    diverged: bool = False
    holding: bool = False
    past_end: bool = False
    saturated: bool = False

    def acceleration(self):
        # type: () -> tuple
        """The command as a plain ``(ax, ay, az)`` triple."""
        return self.ax, self.ay, self.az

    def jerk(self):
        # type: () -> tuple
        """The trajectory's jerk as a plain ``(jx, jy, jz)`` triple."""
        return self.jx, self.jy, self.jz
