"""Stand-in airframes for the stub flight, one per cut into PX4.

Neither is a physics engine. Each models exactly the one property of a real
aircraft that the control path above it has to fight, and nothing else -- which
is what makes the stub a four-minute check rather than a simulation.

* :class:`LaggingAircraft` -- takes a **velocity** and reaches it through a
  first-order lag. That lag is what PX4's velocity controller looks like from
  outside, and it is the whole of what the velocity cut has to close.
* :class:`AttitudeAircraft` -- takes a **tilt and a throttle** and works out its
  own acceleration. It has no velocity loop at all, because on the attitude cut
  there is not one; instead it lags the *thrust axis*, which is what PX4's
  attitude and rate loops look like from outside.

The second one is deliberately harsher than the first. A velocity-commanded body
cannot fall out of the sky when the thrust scale is wrong, and this one can --
which is the point, because so can the real aircraft.
"""
from __future__ import annotations

import math

import numpy as np

from sparx_agency.core.common.types import normalize_angle
from sparx_agency.core.control.constants import GRAVITY_MPS2
from sparx_agency.core.control.flatness import matrix_from_quaternion
from sparx_agency.robots.PEGASUS.adapters.camera_pose import camera_pose_world
from sparx_agency.robots.PEGASUS.adapters.vehicle import CAMERA_OFFSET_FLU

YAW_RATE = math.radians(60.0)
"""How fast either stand-in can turn. Matches ``MPC_YAWRAUTO_MAX``."""

CLIMB_TAU_S = 0.35
"""Velocity-command lag, used by both airframes before handover.

Both of them accept a velocity during the climb and the survey turn, because the
real aircraft does too: PX4 is in its velocity mode for those phases whichever
cut the exploration will use, and only switches to attitude at handover.
"""


def _sensor_position(position, quaternion_xyzw):
    """Where FALCON is told the aircraft is: at the camera, not the body.

    The same rule the real aircraft follows -- see ``isaac/sensing.py``'s
    ``nav_position``. Reproduced exactly so that a planning failure caused by the
    20 cm mount offset shows up in a stub run rather than on Isaac Sim.
    """
    translation, _quaternion = camera_pose_world(position, quaternion_xyzw,
                                                 CAMERA_OFFSET_FLU)
    return translation


class LaggingAircraft:
    """A velocity-commanded rigid body with first-order lag and no attitude.

    ``tau`` is how long the inner loop takes to reach a commanded velocity --
    the one property of a real airframe the velocity-cut outer loop actually has
    to fight. Everything else a multirotor does (tilt, rotor dynamics, ground
    effect) is left to the simulator that has physics.
    """

    def __init__(self, position, yaw, tau=CLIMB_TAU_S):
        # type: (object, float, float) -> None
        self.position = np.asarray(position, dtype=float).copy()
        self.velocity = np.zeros(3)
        self.acceleration = np.zeros(3)
        self.yaw = float(yaw)
        self.tau = float(tau)

    def step_velocity(self, velocity_command, yaw_command, dt):
        # type: (object, float, float) -> None
        """Advance one tick under a velocity and heading command."""
        alpha = dt / (self.tau + dt)
        previous = self.velocity.copy()
        self.velocity += alpha * (np.asarray(velocity_command, dtype=float) - self.velocity)
        self.acceleration = (self.velocity - previous) / dt
        self.position += self.velocity * dt
        self._slew_yaw(yaw_command, dt)

    def _slew_yaw(self, yaw_command, dt):
        # type: (float, float) -> None
        """Turn toward the commanded heading at the airframe's rate limit."""
        error = normalize_angle(float(yaw_command) - self.yaw)
        step = YAW_RATE * dt
        self.yaw = normalize_angle(self.yaw + max(-step, min(step, error)))

    @property
    def quaternion_xyzw(self):
        # type: () -> tuple
        """Attitude as a yaw-only quaternion, scalar last."""
        return (0.0, 0.0, math.sin(self.yaw / 2.0), math.cos(self.yaw / 2.0))

    @property
    def body_z(self):
        # type: () -> np.ndarray
        """Thrust axis. Always up: this body does not tilt."""
        return np.array([0.0, 0.0, 1.0])

    @property
    def nav_position(self):
        # type: () -> np.ndarray
        """The camera's world position -- what FALCON is told."""
        return _sensor_position(self.position, self.quaternion_xyzw)


class AttitudeAircraft:
    """A body that accepts a tilt and a throttle, and nothing else.

    Two things are modelled. The **thrust axis lags** the commanded one through
    ``tau``, standing in for PX4's attitude and rate loops. And the **thrust
    curve is a fact of the airframe**, not something the controller is told: it
    produces ``throttle * full_scale`` of specific thrust, and the controller has
    to work out ``full_scale`` for itself. Seed the thrust model wrongly and
    this aircraft sinks, exactly as the real one does.

    Heading is slewed rather than derived from the quaternion. The commanded
    attitude does encode a heading, but recovering it and integrating a yaw rate
    would be modelling the yaw loop, which is not what this rig is for.

    Args:
        position: Where it starts, world ENU.
        yaw: Initial heading, radians.
        tau: Time constant of the thrust-axis response, seconds.
        true_hover_throttle: The airframe's real thrust curve, expressed as the
            throttle that holds a hover. The controller does not get told.
    """

    def __init__(self, position, yaw, tau=0.18, true_hover_throttle=0.62):
        # type: (object, float, float, float) -> None
        self.position = np.asarray(position, dtype=float).copy()
        self.velocity = np.zeros(3)
        self.acceleration = np.zeros(3)
        self.body_z = np.array([0.0, 0.0, 1.0])
        self.yaw = float(yaw)
        self.tau = float(tau)
        self.climb_tau = CLIMB_TAU_S
        self.full_scale = GRAVITY_MPS2 / float(true_hover_throttle)

    def step_velocity(self, velocity_command, yaw_command, dt):
        # type: (object, float, float) -> None
        """Advance one tick under a velocity command, as PX4 does before handover.

        The climb and the survey turn are velocity-commanded on the real
        aircraft whichever cut the exploration will use, so the stand-in has to
        accept a velocity too. The thrust axis is relaxed toward level while this
        runs, which is what a gently climbing multirotor is doing.
        """
        alpha = dt / (self.climb_tau + dt)
        previous = self.velocity.copy()
        self.velocity += alpha * (np.asarray(velocity_command, dtype=float) - self.velocity)
        self.acceleration = (self.velocity - previous) / dt
        self.position += self.velocity * dt
        self.body_z += alpha * (np.array([0.0, 0.0, 1.0]) - self.body_z)
        self.body_z /= float(np.linalg.norm(self.body_z))
        self._slew_yaw(yaw_command, dt)

    def step_attitude(self, quaternion_wxyz, throttle, yaw_command, dt):
        # type: (object, float, float, float) -> None
        """Advance one tick under an attitude and throttle command.

        Args:
            quaternion_wxyz: Commanded attitude, world ENU with a body-FLU
                frame, scalar first.
            throttle: Commanded collective thrust, 0..1.
            yaw_command: Commanded heading, radians.
            dt: Tick length, seconds.
        """
        wanted = matrix_from_quaternion(quaternion_wxyz)[:, 2]
        alpha = dt / (self.tau + dt)
        self.body_z += alpha * (wanted - self.body_z)
        norm = float(np.linalg.norm(self.body_z))
        self.body_z = self.body_z / norm if norm > 0.0 else np.array([0.0, 0.0, 1.0])

        thrust = max(0.0, float(throttle)) * self.full_scale
        self.acceleration = thrust * self.body_z - np.array([0.0, 0.0, GRAVITY_MPS2])
        self.velocity += self.acceleration * dt
        self.position += self.velocity * dt
        # The floor. Without it a badly seeded thrust model produces an aircraft
        # sinking through the building, and the flight ends as a confusing
        # divergence rather than as the obvious "it could not hold altitude".
        if self.position[2] < 0.0:
            self.position[2] = 0.0
            self.velocity[2] = max(0.0, float(self.velocity[2]))

        self._slew_yaw(yaw_command, dt)

    def _slew_yaw(self, yaw_command, dt):
        # type: (float, float) -> None
        """Turn toward the commanded heading at the airframe's rate limit."""
        error = normalize_angle(float(yaw_command) - self.yaw)
        step = YAW_RATE * dt
        self.yaw = normalize_angle(self.yaw + max(-step, min(step, error)))

    @property
    def quaternion_xyzw(self):
        # type: () -> tuple
        """Attitude as a yaw-only quaternion, scalar last.

        Deliberately yaw-only, even though the body is tilting: this is what the
        aircraft *reports* to FALCON, and FALCON reads nothing but the heading
        out of it. Reporting the full attitude here would model a sensor the
        planner does not have.
        """
        return (0.0, 0.0, math.sin(self.yaw / 2.0), math.cos(self.yaw / 2.0))

    @property
    def nav_position(self):
        # type: () -> np.ndarray
        """The camera's world position -- what FALCON is told."""
        return _sensor_position(self.position, self.quaternion_xyzw)
