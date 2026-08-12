"""Fly a trajectory on an autopilot that only takes a velocity setpoint.

See ``servo.py`` for why the autopilot underneath is inverted rather than
treated as a black box, and ``plant.py`` for the three numbers per axis that
inversion needs.
"""
from sparx_agency.core.control.velocity_servo.limits import (
    VelocityLimits, limit_velocity, slew_velocity,
)
from sparx_agency.core.control.velocity_servo.params import VelocityServoParams
from sparx_agency.core.control.velocity_servo.plant import AxisPlant, VelocityPlant
from sparx_agency.core.control.velocity_servo.servo import VelocityServo
from sparx_agency.core.control.velocity_servo.types import BodyTwistCommand
from sparx_agency.core.control.velocity_servo.yaw import YawServo

__all__ = [
    "VelocityServo",
    "VelocityServoParams",
    "BodyTwistCommand",
    "VelocityLimits",
    "VelocityPlant",
    "AxisPlant",
    "YawServo",
    "limit_velocity",
    "slew_velocity",
]
