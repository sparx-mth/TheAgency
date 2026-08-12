"""Convert an attitude between this repo's frame convention and an autopilot's.

Everything in ``core`` works in **world ENU / body FLU** (REP-103): x east,
y north, z up, and the body's x forward, y left, z up. Autopilots work in
**NED / FRD**: x north, y east, z down, and the body's x forward, y right,
z down. Both are right-handed and both are standard; they are simply different,
and the difference is not a sign flip that can be patched at the call site.

Getting it wrong does not error. It produces an aircraft that rolls when told to
pitch, or flies a mirror image of its plan -- which reads exactly like a badly
tuned controller and is the reason this is a named function with its own tests
rather than three lines inside a MAVLink call.

The conversion is two basis changes, one on each side of the rotation:

.. code-block:: text

    R_ned_frd  =  S  ·  R_enu_flu  ·  D

    S = ENU -> NED on the world side:  (e, n, u) -> (n, e, -u)
    D = FLU -> FRD on the body side:   y and z negated

Both have determinant +1, so the product is a rotation and not a reflection --
which is the check that catches the tempting-but-wrong versions of this.
"""
from __future__ import annotations

import numpy as np

from sparx_agency.core.control.flatness.rotations import (
    matrix_from_quaternion, quaternion_from_matrix, rotation_about_z,
)

_WORLD_ENU_TO_NED = np.array([[0.0, 1.0, 0.0],
                              [1.0, 0.0, 0.0],
                              [0.0, 0.0, -1.0]], dtype=float)
"""Swap east and north, flip up to down. Determinant +1."""

_BODY_FLU_TO_FRD = np.array([[1.0, 0.0, 0.0],
                             [0.0, -1.0, 0.0],
                             [0.0, 0.0, -1.0]], dtype=float)
"""Left becomes right, up becomes down; forward is forward. Determinant +1."""


def world_attitude_to_ned_frd(quaternion_wxyz, yaw_offset=0.0):
    # type: (object, float) -> tuple
    """Convert a world-ENU / body-FLU attitude into the autopilot's NED / FRD one.

    Args:
        quaternion_wxyz: Desired attitude in world ENU with a body-FLU frame,
            ``(w, x, y, z)``.
        yaw_offset: Rotation about the **world z axis**, radians, applied before
            the conversion. This is where an autopilot's heading bias goes -- the
            angle between the simulator's world frame and the frame the
            autopilot's own estimator believes it is flying in. Pass the negated
            bias, matching what the velocity path subtracts from its heading.

    Returns:
        ``(w, x, y, z)`` in NED / FRD, ready for ``SET_ATTITUDE_TARGET``.
    """
    rotation = matrix_from_quaternion(quaternion_wxyz)
    if yaw_offset:
        rotation = rotation_about_z(float(yaw_offset)).dot(rotation)
    return quaternion_from_matrix(
        _WORLD_ENU_TO_NED.dot(rotation).dot(_BODY_FLU_TO_FRD))
