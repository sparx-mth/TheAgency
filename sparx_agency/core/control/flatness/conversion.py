"""Turn a wanted acceleration into the attitude that produces it.

A multirotor has four inputs -- one collective thrust and three torques -- for
six degrees of freedom, and all of its thrust points along one body axis. So it
cannot be told "accelerate this way" independently of "point that way": the two
are one statement. Given the acceleration you want, the direction the thrust
axis must point is fixed, and only the rotation *about* that axis is left free.
Heading takes that last freedom.

That is the whole content of differential flatness for this airframe, and it is
why this module has no gains, no state and no tuning: it is a change of
variables, not a controller.

.. code-block:: text

    thrust axis = desired acceleration + gravity      <- direction fixes tilt
    thrust size = |that|                              <- magnitude fixes throttle
    heading     = free rotation about the thrust axis <- the plan's yaw goes here

The angular-rate feedforward is the same idea differentiated once: as the
trajectory's acceleration changes -- that is, as its *jerk* is non-zero -- the
thrust axis has to rotate, and the rate at which it must rotate is computable
from the plan rather than waiting for an attitude error to appear. This is the
one place jerk is used, and the reason the B-spline is carried rather than its
samples.

Frames: world is any right-handed frame with +z up (REP-103 ENU here). Body is
FLU -- x forward, y left, z up -- so the thrust axis is body +z.
"""
from __future__ import annotations

import math
from typing import Optional

import numpy as np

from sparx_agency.core.control.constants import GRAVITY_MPS2
from sparx_agency.core.control.flatness.limits import AccelerationLimits, limit_acceleration
from sparx_agency.core.control.flatness.rotations import quaternion_from_matrix
from sparx_agency.core.control.flatness.types import AttitudeThrustCommand

_DEGENERATE = 1e-6


def acceleration_to_attitude(acceleration, yaw, jerk=None, yaw_rate=0.0, limits=None):
    # type: (object, float, Optional[object], float, Optional[AccelerationLimits]) -> AttitudeThrustCommand
    """Convert a desired world acceleration and heading into attitude and thrust.

    Args:
        acceleration: Desired world ``(ax, ay, az)``, m/s^2, **excluding**
            gravity. This is the acceleration the vehicle should have, so a
            hover is ``(0, 0, 0)`` rather than ``(0, 0, g)``.
        yaw: Desired heading, radians CCW from world +x.
        jerk: The trajectory's world jerk ``(jx, jy, jz)``, m/s^3, for the
            angular-rate feedforward. None leaves the rates at zero, which is
            correct but slower to track.
        yaw_rate: The trajectory's yaw rate, rad/s.
        limits: Ceilings to apply first. None applies the defaults; pass an
            explicit instance to tune per airframe.

    Returns:
        The attitude, specific thrust and rate feedforward to command.
    """
    limits = limits or AccelerationLimits()
    limited, saturated = limit_acceleration(acceleration, limits)

    # The thrust axis must carry the wanted acceleration AND hold up the
    # aircraft, so gravity is added here and nowhere else in the chain.
    thrust_vector = limited + np.array([0.0, 0.0, GRAVITY_MPS2], dtype=float)
    specific_thrust = float(np.linalg.norm(thrust_vector))
    if specific_thrust < _DEGENERATE:
        # Free fall. There is no attitude that means anything, so hold level at
        # the floor thrust and let the limits above stop this recurring.
        body_z = np.array([0.0, 0.0, 1.0], dtype=float)
        specific_thrust = limits.min_specific_thrust
    else:
        body_z = thrust_vector / specific_thrust

    rotation = _attitude_from_thrust_axis(body_z, float(yaw))
    roll_rate, pitch_rate, body_yaw_rate = _rate_feedforward(
        rotation, specific_thrust, jerk, float(yaw_rate))
    tilt = math.acos(max(-1.0, min(1.0, float(body_z[2]))))
    qw, qx, qy, qz = quaternion_from_matrix(rotation)
    return AttitudeThrustCommand(
        qw=qw, qx=qx, qy=qy, qz=qz,
        specific_thrust_mps2=specific_thrust,
        roll_rate=roll_rate, pitch_rate=pitch_rate, yaw_rate=body_yaw_rate,
        tilt_rad=tilt, saturated=saturated)


def _attitude_from_thrust_axis(body_z, yaw):
    # type: (np.ndarray, float) -> np.ndarray
    """Build a rotation matrix from the thrust direction and a heading.

    The thrust axis fixes two of the three degrees of freedom and the heading
    resolves the third -- but "the heading" has two reasonable definitions once
    the aircraft is tilted, and they differ by about half a degree at a ten
    degree lean. This uses the one where the **body x axis, projected onto the
    horizontal plane, points exactly along the commanded heading**, obtained by
    building the body x axis perpendicular to the commanded *left* direction.

    That choice is not cosmetic here. FALCON picks yaw to aim the depth camera
    at the frontier it wants to observe next, and the camera looks along body x.
    The other convention -- resolving through the commanded forward direction --
    leaves the camera pointing a fraction of a degree off wherever the aircraft
    happens to be leaning, which turns a sensing decision into a function of the
    manoeuvre. It also makes the commanded yaw disagree with the measured yaw
    the tracker compares it against.

    Args:
        body_z: Unit thrust direction in world coordinates.
        yaw: Desired heading, radians.

    Returns:
        A ``(3, 3)`` rotation matrix whose columns are the body axes in world
        coordinates.
    """
    left = np.array([-math.sin(yaw), math.cos(yaw), 0.0], dtype=float)
    body_x = np.cross(left, body_z)
    norm = float(np.linalg.norm(body_x))
    if norm < _DEGENERATE:
        # The thrust axis is horizontal and aligned with the commanded left,
        # so the construction above is undefined. Cannot happen inside the tilt
        # limit; resolved the other way round so the result stays a valid
        # rotation rather than a NaN.
        forward = np.array([math.cos(yaw), math.sin(yaw), 0.0], dtype=float)
        body_y = np.cross(body_z, forward)
        body_y /= max(float(np.linalg.norm(body_y)), _DEGENERATE)
        body_x = np.cross(body_y, body_z)
    else:
        body_x /= norm
        body_y = np.cross(body_z, body_x)
    return np.column_stack((body_x, body_y, body_z))


def _rate_feedforward(rotation, specific_thrust, jerk, yaw_rate):
    # type: (np.ndarray, float, Optional[object], float) -> tuple
    """Angular rates the plan implies, from its jerk.

    Differentiating "the thrust axis points along acceleration plus gravity"
    gives the rate at which that axis turns. Only the component of jerk
    *perpendicular* to the thrust axis rotates it -- the parallel component just
    changes the throttle -- which is what the projection below removes.

    Args:
        rotation: The commanded attitude, columns = body axes in world.
        specific_thrust: Commanded thrust over mass, m/s^2.
        jerk: World jerk, or None for no feedforward.
        yaw_rate: Planned yaw rate about world z, rad/s.

    Returns:
        ``(roll_rate, pitch_rate, yaw_rate)`` in the body frame, rad/s.
    """
    body_x, body_y, body_z = rotation[:, 0], rotation[:, 1], rotation[:, 2]
    # World z projected onto the body z axis: a tilted aircraft turning about
    # world z is turning more slowly about its own z.
    body_yaw_rate = yaw_rate * float(body_z[2])
    if jerk is None or specific_thrust < _DEGENERATE:
        return 0.0, 0.0, body_yaw_rate
    world_jerk = np.asarray(jerk, dtype=float).reshape(3)
    turning = (world_jerk - float(np.dot(body_z, world_jerk)) * body_z) / specific_thrust
    return -float(np.dot(turning, body_y)), float(np.dot(turning, body_x)), body_yaw_rate
