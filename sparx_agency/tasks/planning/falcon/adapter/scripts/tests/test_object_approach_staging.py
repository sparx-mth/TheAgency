"""Behavioural tests for object_approach_node's STAGED APPROACH glue.

The mission flies to a vantage point and looks at the object from there, rather than
onto the object's catalogued (map-derived, imprecise) coordinate. The decision logic
lives in the pure, unit-tested core (``AimBearingPolicy``, ``VisualApproachStateMachine``);
what is tested HERE is the node's glue between them and ROS, which is exactly where a
fault would be silent -- a mission that quietly never aims looks identical to one that
aimed and saw nothing, and both end with the drone flying onto the raw coordinate.

Locked in:
  * the aim arms only when the goal really is a DIFFERENT place from the object, we
    know both, and we have a heading -- and never after an escalation;
  * a pose with no orientation (a localization source that never fills the quaternion
    in) reads as "no heading" rather than as yaw 0, so the drone is never aimed down
    an invented bearing;
  * the heading error is the bearing to the object minus the drone's yaw, wrapped;
  * escalation publishes the OBJECT's position as the new goal, exactly once, and
    latches so the mission cannot re-enter the aim for that target;
  * a live retarget (a new object position) re-arms the aim.

``rospy`` and the message packages are stubbed, as in test_mission_director.
"""
import math
import pathlib
import sys
import types

import pytest

_SCRIPTS = str(pathlib.Path(__file__).resolve().parents[1])
if _SCRIPTS not in sys.path:
    sys.path.insert(0, _SCRIPTS)

_PARAMS = {}


class _Pub:
    def __init__(self, topic):
        self.topic = topic
        self.msgs = []

    def publish(self, msg):
        self.msgs.append(msg)

    @property
    def last(self):
        return self.msgs[-1] if self.msgs else None


class _Point:
    def __init__(self, x=0.0, y=0.0, z=0.0):
        self.x, self.y, self.z = x, y, z


class _Quat:
    def __init__(self, x=0.0, y=0.0, z=0.0, w=1.0):
        self.x, self.y, self.z, self.w = x, y, z, w


class _Time:
    t = 0.0

    def __init__(self, secs=0.0):
        self.secs = float(secs)

    def to_sec(self):
        return self.secs

    def __sub__(self, other):
        return _Time(self.secs - other.secs)

    @staticmethod
    def now():
        return _Time(_Time.t)


class _Msg:
    def __init__(self, **kw):
        for k, v in kw.items():
            setattr(self, k, v)


def _install_stubs():
    rospy = types.ModuleType("rospy")
    rospy.init_node = lambda *a, **k: None
    rospy.get_param = lambda name, default=None: _PARAMS.get(name, default)
    rospy.Publisher = lambda topic, *a, **k: _Pub(topic)
    rospy.Subscriber = lambda *a, **k: None
    rospy.Timer = lambda *a, **k: None
    rospy.Duration = lambda *a, **k: None
    rospy.Time = _Time
    rospy.spin = lambda: None
    rospy.is_shutdown = lambda: True
    for fn in ("loginfo", "logwarn", "logerr", "logfatal", "logwarn_throttle",
               "loginfo_throttle", "logerr_throttle"):
        setattr(rospy, fn, lambda *a, **k: None)
    rospy.ROSInterruptException = type("ROSInterruptException", (Exception,), {})
    sys.modules["rospy"] = rospy

    def _mod(name, **attrs):
        m = types.ModuleType(name)
        for k, v in attrs.items():
            setattr(m, k, v)
        sys.modules[name] = m
        return m

    geo = _mod("geometry_msgs")
    geo.msg = _mod("geometry_msgs.msg", Point=_Point, Pose=_Msg, PoseStamped=_Msg,
                   Twist=lambda: _Msg(linear=_Msg(x=0.0, y=0.0, z=0.0),
                                      angular=_Msg(x=0.0, y=0.0, z=0.0)))
    sens = _mod("sensor_msgs")
    sens.msg = _mod("sensor_msgs.msg", CameraInfo=_Msg, Image=_Msg)
    std = _mod("std_msgs")
    std.msg = _mod("std_msgs.msg", Bool=_Msg, String=_Msg)


_install_stubs()
import object_approach_node as oan  # noqa: E402


def _make(**params):
    _PARAMS.clear()
    # lock_mode 'detector' builds NO box tracker, and that is deliberate here: the
    # MedianFlow one reaches for real OpenCV constants in its constructor, while a
    # sibling test module installs a fake cv2 into the sys.modules that pytest shares
    # for the whole session -- so with the default mode this file would pass alone and
    # fail in a full run, purely on collection order. Nothing about the staged approach
    # involves box tracking, so the tracker-free mode is the honest choice, not a dodge.
    _PARAMS.update({"~lock_mode": "detector"})
    _PARAMS.update({("~" + k): v for k, v in params.items()})
    return oan.ObjectApproachNode()


def _staged(goal=(0.0, -2.0), obj=(2.0, -2.0), pose=(0.0, -2.0), yaw=0.0, **params):
    """A node standing at the vantage point with a known object elsewhere."""
    node = _make(goal_x=goal[0], goal_y=goal[1], **params)
    node._object_position_cb(_Point(x=obj[0], y=obj[1]))
    node._pose_xy = pose
    node._pose_yaw = yaw
    return node


# ── arming the aim ────────────────────────────────────────────────────
def test_aim_arms_when_the_goal_is_a_staging_point():
    assert _staged()._aim_ready() is True


def test_aim_does_not_arm_without_an_object_position():
    """No director (or a plain coordinate run): behave exactly as before."""
    node = _make(goal_x=0.0, goal_y=-2.0)
    node._pose_xy, node._pose_yaw = (0.0, -2.0), 0.0
    assert node._aim_ready() is False


def test_aim_does_not_arm_when_the_goal_already_is_the_object():
    """Staging off: the goal IS the object, so there is nothing to aim at from it --
    arriving there must go straight to the ordinary land/scan ending."""
    assert _staged(goal=(2.0, -2.0), obj=(2.0, -2.0))._aim_ready() is False


def test_aim_does_not_arm_for_an_object_inside_the_arrival_radius():
    """Closer than 'arrived' means the goal and the object are the same place as far
    as the mission can tell; turning to look at it would be noise."""
    node = _staged(goal=(0.0, -2.0), obj=(0.3, -2.0), arrive_radius_m=0.6)
    assert node._aim_ready() is False


def test_aim_can_be_switched_off():
    assert _staged(aim_before_direct=False)._aim_ready() is False


def test_aim_does_not_arm_without_a_heading():
    """A pose that carries no orientation cannot be aimed from."""
    node = _staged()
    node._pose_yaw = None
    assert node._aim_ready() is False


# ── the heading a pose actually carries ───────────────────────────────
def test_yaw_is_read_from_the_quaternion():
    q = _Quat(0.0, 0.0, math.sin(math.pi / 4), math.cos(math.pi / 4))   # +90 deg
    assert oan.ObjectApproachNode._yaw_of(q) == pytest.approx(math.pi / 2)


def test_a_degenerate_quaternion_is_absent_not_zero():
    """All-zero is a field nobody filled in, NOT "facing along +x". Reading it as 0
    would aim the drone down whatever bearing that happens to be, confidently."""
    assert oan.ObjectApproachNode._yaw_of(_Quat(0.0, 0.0, 0.0, 0.0)) is None


# ── where to turn ─────────────────────────────────────────────────────
def test_heading_error_is_the_bearing_minus_the_yaw():
    # Standing at the origin facing +x, with the object due +y: turn +90 deg (left).
    node = _staged(goal=(0.0, 0.0), obj=(0.0, 3.0), pose=(0.0, 0.0), yaw=0.0)
    assert node._heading_error_to_object() == pytest.approx(math.pi / 2)


def test_heading_error_wraps_the_short_way_round():
    """Facing +y (yaw 90 deg) with the object due -x (bearing ~-180 deg): the raw
    difference is -270 deg, which would turn the drone three-quarters of the way round
    the wrong way. Wrapped, it is the +90 deg turn that actually faces it."""
    node = _staged(goal=(0.0, 0.0), obj=(-1.0, -0.0001), pose=(0.0, 0.0),
                   yaw=math.pi / 2)
    err = node._heading_error_to_object()
    assert err == pytest.approx(math.pi / 2, abs=1e-3)
    assert abs(err) <= math.pi


def test_heading_error_is_none_without_a_pose():
    node = _staged()
    node._pose_xy = None
    assert node._heading_error_to_object() is None


# ── escalation: give up aiming, fly at the coordinate ─────────────────
def test_escalation_publishes_the_object_as_the_new_goal():
    node = _staged(obj=(2.0, -3.0))
    node._escalate_to_object()
    assert (node.goal_pub.last.x, node.goal_pub.last.y) == (2.0, -3.0)
    assert node._goal_xy == (2.0, -3.0)


def test_escalation_disarms_the_aim_so_arrival_ends_the_mission():
    """After escalating, the goal IS the object -- arriving there must land/scan, not
    aim again, or the mission would loop between the two forever."""
    node = _staged()
    node._escalate_to_object()
    assert node._aim_ready() is False


def test_escalation_is_idempotent():
    node = _staged()
    node._escalate_to_object()
    node._escalate_to_object()
    assert len(node.goal_pub.msgs) == 1


def test_escalation_without_an_object_position_does_nothing():
    node = _make(goal_x=0.0, goal_y=-2.0)
    node._escalate_to_object()
    assert node.goal_pub.msgs == []


def test_retarget_re_arms_the_aim_after_an_escalation():
    """Picking another object starts a fresh mission for it: it gets its own look
    from the vantage point, not the previous target's spent escalation."""
    node = _staged()
    node._escalate_to_object()
    assert node._aim_ready() is False
    node._goal_xy = (0.0, -2.0)                       # director republishes the goal
    node._object_position_cb(_Point(x=-2.0, y=1.0))   # ...and the new object
    assert node._object_xy == (-2.0, 1.0)
    assert node._aim_ready() is True


def test_a_relatched_object_position_does_not_clear_the_escalation():
    """The director's publisher is latched, so the SAME position can be re-delivered.
    Treating that as a retarget would un-escalate and send the drone back to aim."""
    node = _staged(obj=(2.0, -2.0))
    node._escalate_to_object()
    node._object_position_cb(_Point(x=2.0, y=-2.0))   # same value, re-delivered
    assert node._escalated is True
    assert node._aim_ready() is False
