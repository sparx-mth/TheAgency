"""What the flatness conversion emits.

An **attitude** and a **specific thrust** -- thrust divided by mass, in m/s^2.
Specific rather than newtons because the conversion itself is mass-free: the
attitude a multirotor needs to produce a given acceleration is the same whatever
it weighs. Mass enters one layer later, in
:mod:`sparx_agency.core.control.thrust_model`, which is also where it belongs
because that is where the number stops being geometry and starts being a
throttle setting.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class AttitudeThrustCommand:
    """One tick of the flatness conversion.

    Attributes:
        qw: Scalar part of the desired attitude quaternion.
        qx: Desired attitude quaternion, x.
        qy: Desired attitude quaternion, y.
        qz: Desired attitude quaternion, z.
        specific_thrust_mps2: Thrust to command, divided by mass. Always
            positive -- a multirotor cannot pull.
        roll_rate: Body-frame angular rate feedforward about x, rad/s. Derived
            from the trajectory's jerk: the rate at which the thrust direction
            must rotate to stay on the plan. Zero when no jerk was supplied,
            which costs tracking rather than correctness.
        pitch_rate: The same about y.
        yaw_rate: The same about z, from the plan's yaw rate.
        tilt_rad: Angle between the commanded thrust axis and vertical. Reported
            because it is the single number that says how aggressive this
            command is, and because it is what the tilt limit acts on.
        saturated: True when the requested acceleration had to be reduced to
            respect the tilt or thrust ceiling. Advisory, but a controller that
            saturates continuously is being asked for a trajectory the airframe
            cannot fly.
    """

    qw: float
    qx: float
    qy: float
    qz: float
    specific_thrust_mps2: float
    roll_rate: float = 0.0
    pitch_rate: float = 0.0
    yaw_rate: float = 0.0
    tilt_rad: float = 0.0
    saturated: bool = False

    def quaternion_wxyz(self):
        # type: () -> tuple
        """The attitude as ``(w, x, y, z)`` -- MAVLink's ordering."""
        return self.qw, self.qx, self.qy, self.qz

    def body_rates(self):
        # type: () -> tuple
        """The angular-rate feedforward as ``(roll, pitch, yaw)``, rad/s."""
        return self.roll_rate, self.pitch_rate, self.yaw_rate
