"""Differential flatness: a wanted acceleration is a wanted attitude.

See ``conversion.acceleration_to_attitude``.
"""
from sparx_agency.core.control.flatness.conversion import acceleration_to_attitude
from sparx_agency.core.control.flatness.frames import world_attitude_to_ned_frd
from sparx_agency.core.control.flatness.limits import AccelerationLimits, limit_acceleration
from sparx_agency.core.control.flatness.rotations import (
    matrix_from_quaternion, quaternion_from_matrix, rotation_about_z,
)
from sparx_agency.core.control.flatness.types import AttitudeThrustCommand

__all__ = [
    "acceleration_to_attitude",
    "world_attitude_to_ned_frd",
    "AccelerationLimits",
    "limit_acceleration",
    "AttitudeThrustCommand",
    "matrix_from_quaternion",
    "quaternion_from_matrix",
    "rotation_about_z",
]
