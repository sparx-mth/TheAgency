"""What the 3D reference tracker emits, and what it reports about itself.

The command is a **world-frame velocity plus an absolute heading**, not a
position and not a body-frame twist. That is a deliberate choice about where the
loop is closed:

* A *position* setpoint hands the trajectory back to the autopilot's own
  position controller, which re-shapes it with its own smoother and its own
  limits -- the planner's timing is lost, and on PX4 offboard specifically the
  aircraft was measured closing a one-metre gap at one centimetre per second.
* A *body-frame* twist would have to be rotated by the aircraft's measured
  attitude, putting the yaw estimate inside the position loop.

A world velocity goes almost straight into the inner-loop velocity controller,
so the reference's own timing survives.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TrackedSetpoint:
    """One tick of tracker output, plus everything a caller needs to judge it.

    Attributes:
        vx: World-frame velocity command along +x, m/s.
        vy: World-frame velocity command along +y, m/s.
        vz: World-frame velocity command along +z (up), m/s.
        yaw: Absolute heading to command, radians CCW from +x. Already slewed, so
            a caller can send it verbatim.
        position_error_m: Distance from the aircraft to the reference position.
            The single number that says whether the aircraft is flying the plan.
        along_track_lag_m: Component of the position error along the reference
            velocity. Positive means the aircraft is *behind* the reference --
            the normal state while accelerating. Separated from the cross-track
            term because lag is benign and cross-track error is not: lag means
            "late", cross-track means "somewhere else".
        cross_track_error_m: Component of the position error perpendicular to the
            reference velocity. This is the one that flies into walls.
        yaw_error_rad: Signed difference between the reference heading and the
            aircraft's, wrapped to (-pi, pi].
        diverged: True while :attr:`position_error_m` exceeds the configured
            ceiling. Advisory; the tracker keeps flying.
        holding: True when no fresh reference was available and the tracker is
            holding station instead of following one.
    """

    vx: float
    vy: float
    vz: float
    yaw: float
    position_error_m: float = 0.0
    along_track_lag_m: float = 0.0
    cross_track_error_m: float = 0.0
    yaw_error_rad: float = 0.0
    diverged: bool = False
    holding: bool = False

    def velocity(self):
        # type: () -> tuple
        """The command as a plain ``(vx, vy, vz)`` triple."""
        return self.vx, self.vy, self.vz
