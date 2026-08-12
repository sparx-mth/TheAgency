"""The command mapping, its sign conventions, and its clamp.

The mapping itself is four assignments, so the value of these tests is entirely
in the two things four assignments get wrong: which way ``linear.y`` points, and
what happens at the ceiling. Both are silent failures in flight -- a sign error
flies a mirror image of the plan and never raises, and a per-axis clamp steers
off course only while saturated, which is exactly when nobody is reading the
logs.
"""
from __future__ import annotations

import math

import pytest

from sparx_agency.core.control.velocity_servo import BodyTwistCommand
from sparx_agency.robots.SJTU.adapters import velocity_command as vc

LIMITS = vc.BodyVelocityLimits(max_speed_xy=2.0, max_speed_z=2.0, max_yaw_rate=1.5)


class _Vector3:
    """Stand-in for ``geometry_msgs/Vector3``, so no ROS import is needed."""

    def __init__(self):
        self.x = None
        self.y = None
        self.z = None


class _Twist:
    """Stand-in for ``geometry_msgs/Twist``."""

    def __init__(self):
        self.linear = _Vector3()
        self.angular = _Vector3()


def test_module_does_not_import_ros():
    """This module must stay importable in the plain ``.venv``.

    Asserted directly rather than trusted: an ``import rclpy`` added at module
    scope would make every consumer of the mapping -- including these tests --
    require a sourced ROS 2 environment, and the failure would show up as a
    collection error somewhere else entirely.
    """
    import sys
    assert "rclpy" not in sys.modules
    assert "geometry_msgs" not in sys.modules


def test_fields_pass_through_below_the_limits():
    """Nothing is scaled, reordered or renamed on the ordinary path."""
    fields = vc.twist_fields(
        BodyTwistCommand(vx=1.0, vy=0.5, vz=-0.25, yaw_rate=0.3), LIMITS)
    assert (fields.linear_x, fields.linear_y, fields.linear_z, fields.angular_z) == \
        (1.0, 0.5, -0.25, 0.3)
    assert not fields.speed_clamped
    assert not fields.yaw_rate_clamped


def test_positive_y_is_left():
    """Body FLU, REP-103: a positive ``vy`` must reach the wire as positive.

    The plugin drives ``cmd_vel.linear.y`` towards the heading-frame y velocity
    through a negated roll command, so its ``+y`` is the aircraft's own left.
    Nothing here may flip it -- the negation already happened in C++.
    """
    fields = vc.twist_fields(BodyTwistCommand(vx=0.0, vy=0.8, vz=0.0, yaw_rate=0.0),
                             LIMITS)
    assert fields.linear_y == pytest.approx(0.8)


def test_positive_yaw_rate_is_counter_clockwise():
    """``angular.z`` keeps its sign; the plugin's yaw PID is right-handed about world z."""
    fields = vc.twist_fields(BodyTwistCommand(vx=0.0, vy=0.0, vz=0.0, yaw_rate=0.4),
                             LIMITS)
    assert fields.angular_z == pytest.approx(0.4)


def test_horizontal_clamp_preserves_direction():
    """A saturated diagonal is scaled as a pair, not clipped per axis.

    (3.0, 1.0) clipped per axis against 2.0 would be (2.0, 1.0) -- 8.1 degrees
    off the commanded heading. Scaled as a pair the bearing is unchanged and
    only the speed is reduced.
    """
    command = BodyTwistCommand(vx=3.0, vy=1.0, vz=0.0, yaw_rate=0.0)
    fields = vc.twist_fields(command, LIMITS)

    assert math.hypot(fields.linear_x, fields.linear_y) == pytest.approx(2.0)
    assert math.atan2(fields.linear_y, fields.linear_x) == \
        pytest.approx(math.atan2(1.0, 3.0))
    assert fields.speed_clamped


def test_vertical_and_yaw_clamp_independently():
    """Climb and heading rate have their own ceilings and their own flags."""
    fields = vc.twist_fields(
        BodyTwistCommand(vx=0.0, vy=0.0, vz=-5.0, yaw_rate=4.0), LIMITS)
    assert fields.linear_z == pytest.approx(-2.0)
    assert fields.angular_z == pytest.approx(1.5)
    assert fields.speed_clamped
    assert fields.yaw_rate_clamped


def test_yaw_clamp_does_not_touch_the_translation():
    """A saturated turn must not slow the aircraft down as a side effect."""
    fields = vc.twist_fields(
        BodyTwistCommand(vx=1.0, vy=0.0, vz=0.0, yaw_rate=9.0), LIMITS)
    assert fields.linear_x == pytest.approx(1.0)
    assert not fields.speed_clamped
    assert fields.yaw_rate_clamped


def test_fill_twist_writes_only_the_four_controllable_fields():
    """``angular.x``/``angular.y`` are left untouched.

    The plugin reads them only while the aircraft is *not* flying, where they
    become roll/pitch or horizontal-velocity targets depending on
    ``dronevel_mode``. Writing them from a flight command arms a control path
    nobody asked for.
    """
    message = vc.fill_twist(
        _Twist(),
        vc.twist_fields(BodyTwistCommand(vx=0.7, vy=-0.2, vz=0.1, yaw_rate=-0.5),
                        LIMITS))

    assert (message.linear.x, message.linear.y, message.linear.z) == (0.7, -0.2, 0.1)
    assert message.angular.z == -0.5
    assert message.angular.x is None
    assert message.angular.y is None


def test_zero_twist_fields_is_a_full_stop():
    """The only "stop" this platform has, and it must keep being published."""
    fields = vc.zero_twist_fields()
    assert (fields.linear_x, fields.linear_y, fields.linear_z, fields.angular_z) == \
        (0.0, 0.0, 0.0, 0.0)


@pytest.mark.parametrize("kwargs", [
    {"max_speed_xy": 0.0, "max_speed_z": 2.0, "max_yaw_rate": 1.5},
    {"max_speed_xy": 2.0, "max_speed_z": -1.0, "max_yaw_rate": 1.5},
    {"max_speed_xy": 2.0, "max_speed_z": 2.0, "max_yaw_rate": 0.0},
])
def test_limits_reject_a_useless_ceiling(kwargs):
    """A non-positive ceiling would pin the axis at zero and fly nothing."""
    with pytest.raises(ValueError):
        vc.BodyVelocityLimits(**kwargs)


def test_every_latch_names_a_real_topic_and_a_real_message():
    """The latch table is what a node declares its publishers from."""
    from sparx_agency.robots.SJTU.adapters import topics

    known = {topics.TAKEOFF, topics.LAND, topics.RESET, topics.POSCTRL,
             topics.DRONEVEL_MODE}
    for latch in vc.LATCHES:
        assert latch.topic in known
        assert latch.message_type in (vc.EMPTY_MESSAGE, vc.BOOL_MESSAGE)
        # Empty carries no payload; Bool must carry one, or the publish is a
        # coin flip on whatever default the message type happens to have.
        assert (latch.payload is None) == (latch.message_type == vc.EMPTY_MESSAGE)
        assert latch.description
