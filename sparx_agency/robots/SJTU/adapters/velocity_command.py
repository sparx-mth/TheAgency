"""Put a core velocity command on this platform's only control input.

The SJTU plugin exposes exactly one thing while flying: a ``geometry_msgs/Twist``
on ``/simple_drone/cmd_vel``, read in the yaw-aligned body frame. That is the
same shape as
:class:`~sparx_agency.core.control.velocity_servo.types.BodyTwistCommand`, which
is why this platform gets a velocity backend and not an attitude one -- there is
no attitude, rate, thrust or motor input to send.

**No ROS import anywhere in this module, at any scope.** It converts to plain
numbers and, if the caller has a message object, fills it in place. That keeps
the mapping -- which is where the sign errors live -- unit-testable in the plain
``.venv`` with no ROS 2 on the path, and it keeps this module importable from
tooling that only wants to reason about commands.

Two things this does that a naive field copy does not:

* **clamps to what the plugin will actually accept**, horizontal pair scaled
  together so the direction of travel survives. Clipping x and y independently
  turns a speed limit into a steering error: (1.4, 0.4) m/s clipped per axis
  flies eight degrees off where it was aimed, and the error grows with how
  saturated the command is. Same argument as
  ``core/control/velocity_servo/limits.py``, applied here because this clamp is
  against the *airframe's* saturation rather than the controller's own ceilings,
  and the two are configured from different places;
* **reports whether it clamped**, because a command that is quietly saturated
  every tick is a controller that is no longer in charge, and the aircraft looks
  identical either way.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

from sparx_agency.core.control.velocity_servo import BodyTwistCommand
from sparx_agency.robots.SJTU.adapters import topics

EMPTY_MESSAGE = "std_msgs/Empty"
BOOL_MESSAGE = "std_msgs/Bool"


@dataclass(frozen=True)
class BodyVelocityLimits:
    """What the plugin will accept before it saturates on its own terms.

    No defaults on purpose. These belong to the airframe, are loaded from
    ``config/airframe.yaml`` by
    :func:`~sparx_agency.robots.SJTU.adapters.plant_config.body_velocity_limits`,
    and a caller that has to type them by hand has almost certainly guessed
    them.

    Attributes:
        max_speed_xy: Horizontal ceiling, m/s. The plugin's ``velocityXYLimit``.
        max_speed_z: Vertical ceiling, m/s. Not enforced by the plugin -- its
            ``velocityZLimit`` is disabled -- so this one is genuinely ours.
        max_yaw_rate: Heading-rate ceiling, rad/s. The plugin's ``yawLimit``.
    """

    max_speed_xy: float
    max_speed_z: float
    max_yaw_rate: float

    def __post_init__(self):
        # type: () -> None
        """Reject a limit set that cannot bound anything."""
        for name in ("max_speed_xy", "max_speed_z", "max_yaw_rate"):
            value = getattr(self, name)
            if not value > 0.0:
                raise ValueError("%s must be > 0, got %r" % (name, value))


@dataclass(frozen=True)
class TwistFields:
    """The four numbers a ``geometry_msgs/Twist`` needs, plus what was clipped.

    The other five fields of a Twist (``linear`` has three, ``angular`` three)
    are left alone by design: ``angular.x``/``angular.y`` are read by the plugin
    only while the aircraft is *not* flying, and writing them from a flight
    command would arm a control path nobody intended.

    Attributes:
        linear_x: Body-forward speed, m/s.
        linear_y: Body-**left** speed, m/s. FLU, REP-103. The plugin agrees: it
            drives ``linear.y`` towards the heading-frame y velocity through a
            *negated* roll command, which is what makes positive y a left
            translation.
        linear_z: Climb rate, m/s, positive up.
        angular_z: Yaw rate, rad/s, positive counter-clockwise.
        speed_clamped: True when the horizontal or vertical ceiling reduced the
            command.
        yaw_rate_clamped: True when the heading-rate ceiling reduced it.
    """

    linear_x: float
    linear_y: float
    linear_z: float
    angular_z: float
    speed_clamped: bool = False
    yaw_rate_clamped: bool = False


@dataclass(frozen=True)
class Latch:
    """One of the plugin's non-twist inputs, described rather than published.

    Attributes:
        topic: Full topic name, from
            :mod:`~sparx_agency.robots.SJTU.adapters.topics`.
        message_type: ``"std_msgs/Empty"`` or ``"std_msgs/Bool"``. A string
            rather than a type object so this module stays ROS-free; the node
            that owns the publisher already imports the real message.
        payload: The boolean to send, or None for an ``Empty`` trigger.
        description: What it does, for an operator UI or a log line.
    """

    topic: str
    message_type: str
    payload: Optional[bool]
    description: str


def twist_fields(command, limits):
    # type: (BodyTwistCommand, BodyVelocityLimits) -> TwistFields
    """Convert a core body twist into the fields of a ``geometry_msgs/Twist``.

    Args:
        command: What the controller decided, body FLU.
        limits: The airframe's saturations, from ``config/airframe.yaml``.

    Returns:
        The four Twist fields, clamped, with flags saying whether they were.
    """
    horizontal_x, horizontal_y, speed_clamped = _clamp_horizontal(
        float(command.vx), float(command.vy), limits.max_speed_xy)
    vertical = _clamp(float(command.vz), limits.max_speed_z)
    speed_clamped = speed_clamped or vertical != float(command.vz)

    yaw_rate = _clamp(float(command.yaw_rate), limits.max_yaw_rate)
    return TwistFields(
        linear_x=horizontal_x,
        linear_y=horizontal_y,
        linear_z=vertical,
        angular_z=yaw_rate,
        speed_clamped=speed_clamped,
        yaw_rate_clamped=yaw_rate != float(command.yaw_rate),
    )


def fill_twist(message, fields):
    # type: (object, TwistFields) -> object
    """Write ``fields`` into a caller-supplied ``geometry_msgs/Twist``.

    The message is created and published by the node, which is the only place
    that may import ``geometry_msgs``. This function only writes into it, so the
    field-by-field mapping still lives beside the reasoning about it.

    Args:
        message: A ``geometry_msgs/Twist`` (or anything with ``linear`` and
            ``angular`` members carrying ``x``/``y``/``z``).
        fields: What to write.

    Returns:
        The same message, for chaining into a ``publish`` call.
    """
    message.linear.x = fields.linear_x
    message.linear.y = fields.linear_y
    message.linear.z = fields.linear_z
    message.angular.z = fields.angular_z
    return message


def zero_twist_fields():
    # type: () -> TwistFields
    """A full stop.

    Worth having by name: this platform has no disarm and no failsafe reachable
    from the outside, so "stop" is a zero twist that must keep being published.
    Publishing nothing does not stop the aircraft -- the plugin holds the last
    command it was given.
    """
    return TwistFields(0.0, 0.0, 0.0, 0.0)


def _clamp(value, ceiling):
    # type: (float, float) -> float
    """Symmetric saturation of a scalar."""
    return max(-ceiling, min(ceiling, value))


def _clamp_horizontal(vx, vy, ceiling):
    # type: (float, float, float) -> tuple
    """Scale the horizontal pair together so its direction survives the clamp.

    Returns:
        ``(vx, vy, clamped)``.
    """
    speed = math.hypot(vx, vy)
    if speed <= ceiling or speed <= 0.0:
        return vx, vy, False
    scale = ceiling / speed
    return vx * scale, vy * scale, True


TAKEOFF = Latch(
    topic=topics.TAKEOFF, message_type=EMPTY_MESSAGE, payload=None,
    description="Take off. Ignored unless /simple_drone/state reports LANDED.")

LAND = Latch(
    topic=topics.LAND, message_type=EMPTY_MESSAGE, payload=None,
    description="Land. Ignored unless the aircraft is FLYING.")

RESET = Latch(
    topic=topics.RESET, message_type=EMPTY_MESSAGE, payload=None,
    description="Return the model to its spawn pose and clear the PID integrators.")

POSITION_CONTROL_ON = Latch(
    topic=topics.POSCTRL, message_type=BOOL_MESSAGE, payload=True,
    description="Read cmd_vel.linear as an absolute world position, not a velocity.")

POSITION_CONTROL_OFF = Latch(
    topic=topics.POSCTRL, message_type=BOOL_MESSAGE, payload=False,
    description="Read cmd_vel.linear as a body velocity. The mode this stack flies in.")

VELOCITY_MODE_ON = Latch(
    topic=topics.DRONEVEL_MODE, message_type=BOOL_MESSAGE, payload=True,
    description="While not flying, read cmd_vel.angular.x/y as horizontal velocities.")

VELOCITY_MODE_OFF = Latch(
    topic=topics.DRONEVEL_MODE, message_type=BOOL_MESSAGE, payload=False,
    description="While not flying, read cmd_vel.angular.x/y as roll/pitch angles.")

LATCHES = (TAKEOFF, LAND, RESET, POSITION_CONTROL_ON, POSITION_CONTROL_OFF,
           VELOCITY_MODE_ON, VELOCITY_MODE_OFF)
"""Every non-twist input, in one tuple, so a node can declare its publishers from it."""
