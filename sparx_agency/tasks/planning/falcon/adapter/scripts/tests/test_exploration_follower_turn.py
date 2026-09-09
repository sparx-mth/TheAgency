"""End-to-end tests of the follower's turn-in-place behaviour.

The gate itself is unit-tested in ``test_heading_gate.py``. What is tested here
is the thing that actually flies: the real
``FalconExplorationFollowerNode._step`` composing the tracker, the gate, the
course limiter, the park scan, the escape reflex and the pulse shaper -- because
every defect this file was written for lived in the *composition*, not in a
part. The turn-in-place gate and the reverse clamp are the aircraft's only
protection against blind backward flight, and neither was reachable by a test
until now.

``rospy`` and the message packages are not installed on a development host, so
they are stubbed just far enough to import and step the node. Nothing about the
control path is stubbed: the tracker, the gate, the shaper and every gate
decision are the real objects.
"""
import math
import pathlib
import sys
import types

import pytest

_SCRIPTS = str(pathlib.Path(__file__).resolve().parents[1])
if _SCRIPTS not in sys.path:
    sys.path.insert(0, _SCRIPTS)

_CLOCK = [1000.0]
_PARAMS = {}


class _Time(object):
    def __init__(self, secs=0.0):
        self.secs = float(secs)

    def to_sec(self):
        return self.secs

    def __sub__(self, other):
        return _Time(self.secs - other.secs)

    def __add__(self, other):
        return _Time(self.secs + other.to_sec())

    def __lt__(self, other):
        return self.secs < other.to_sec()

    @staticmethod
    def now():
        return _Time(_CLOCK[0])


class _Publisher(object):
    def __init__(self, *_a, **_k):
        self.sent = []

    def publish(self, msg):
        self.sent.append(msg)


class _Vector3(object):
    def __init__(self):
        self.x = self.y = self.z = 0.0


class _Twist(object):
    def __init__(self):
        self.linear = _Vector3()
        self.angular = _Vector3()


def _install_ros_stubs():
    """Make the ROS imports resolve, without stubbing any control code."""
    rospy = types.ModuleType("rospy")
    rospy.Time = _Time
    rospy.Duration = lambda secs=0.0: _Time(secs)
    rospy.Publisher = _Publisher
    rospy.Subscriber = lambda *a, **k: None
    rospy.Timer = lambda *a, **k: None
    rospy.init_node = lambda *a, **k: None
    rospy.get_param = lambda name, default=None: _PARAMS.get(name, default)
    for name in ("loginfo", "logwarn", "logerr", "logdebug", "spin",
                 "loginfo_throttle", "logwarn_throttle", "logerr_throttle"):
        setattr(rospy, name, lambda *a, **k: None)
    rospy.is_shutdown = lambda: False
    sys.modules["rospy"] = rospy

    geometry = types.ModuleType("geometry_msgs")
    geometry_msg = types.ModuleType("geometry_msgs.msg")
    geometry_msg.Twist, geometry_msg.Vector3 = _Twist, _Vector3
    geometry.msg = geometry_msg
    sys.modules["geometry_msgs"], sys.modules["geometry_msgs.msg"] = geometry, geometry_msg

    for pkg, names in (("nav_msgs", ("Odometry",)),
                       ("std_msgs", ("String",)),
                       ("quadrotor_msgs", ("PositionCommand",))):
        mod = types.ModuleType(pkg)
        msg = types.ModuleType(pkg + ".msg")
        for n in names:
            setattr(msg, n, type(n, (object,), {
                "__init__": lambda self, **kw: self.__dict__.update(kw),
                "TRAJECTORY_STATUS_READY": 1}))
        mod.msg = msg
        sys.modules[pkg], sys.modules[pkg + ".msg"] = mod, msg


_install_ros_stubs()

from sparx_agency.core.common.types import TrajectoryPoint  # noqa: E402

import falcon_exploration_follower_node as fef  # noqa: E402

DT = 0.05


def _node(**params):
    """A follower with odometry, attitude and demo mode already established."""
    _PARAMS.clear()
    _PARAMS.update(dict(("~" + k, v) for k, v in params.items()))
    _CLOCK[0] = 1000.0
    node = fef.FalconExplorationFollowerNode()
    node._pose = (0.0, 0.0, 1.5)
    node._yaw = 0.0
    node._velocity = (0.0, 0.0, 0.0)
    node._attitude_at = _Time(_CLOCK[0])
    node._roll_deg = node._pitch_deg = 0.0
    node.current_demo_mode = fef.MODE_EXPLORING
    node._prev_tick_t = _CLOCK[0]
    return node


def _reference_at(node, x, y, z=1.5, vx=0.0, vy=0.0, moving=True):
    """Install a reference the way the pos_cmd callback would."""
    node._reference = TrajectoryPoint(t=_CLOCK[0], x=x, y=y, z=z,
                                      vx=vx, vy=vy, vz=0.0, yaw=0.0)
    node._reference_stamp = _Time(_CLOCK[0])
    node._reference_ready = True
    node._reference_traj_id = 1
    if not moving:
        node._reference = TrajectoryPoint(t=_CLOCK[0], x=x, y=y, z=z, yaw=0.0)


def _step(node, dt=DT):
    """Advance the clock and run one real control tick."""
    _CLOCK[0] += dt
    node._attitude_at = _Time(_CLOCK[0])
    node._reference_stamp = _Time(_CLOCK[0])
    node._tick(None)
    return node.cmd_pub.sent[-1] if node.cmd_pub.sent else None


def test_a_demand_behind_the_nose_publishes_no_translation():
    """Requirement 5: turn toward the target before initiating forward motion."""
    node = _node(max_speed_xy=1.0)
    _reference_at(node, -3.0, 0.0, vx=-0.5)      # 3 m directly behind
    sent = None
    for _ in range(5):
        sent = _step(node)
    assert node._heading_decision.turning
    assert sent.linear.x == pytest.approx(0.0)
    assert sent.linear.y == pytest.approx(0.0)
    assert abs(sent.angular.z) > 0.0             # ...but it is turning


def test_the_committed_turn_may_outrun_the_course_slew_limiter():
    """Requirement 3: the turn is what the raised yaw ceiling actually buys.

    The course limiter stays at its measured-safe 45 deg/s for ordinary
    steering (LOOP_BUGS B31 forbids raising it); a committed turn is the one
    case that bypasses it, because its target heading is held for the whole
    manoeuvre rather than chased.
    """
    # force_mode="none" is what every launch sets; see the next test for why
    # that matters to this particular assertion.
    node = _node(max_speed_xy=1.0, max_yaw_rate_deg=90.0, course_slew_deg_s=45.0,
                 force_mode="none")
    _reference_at(node, -3.0, 0.0, vx=-0.5)
    sent = None
    for _ in range(5):
        sent = _step(node)
    assert node._heading_decision.turning
    assert abs(sent.angular.z) > math.radians(45.0)          # past the limiter
    assert abs(sent.angular.z) <= math.radians(90.0) + 1e-9  # inside the ceiling


def test_in_fixed_force_mode_the_pulse_shaper_and_not_the_ceiling_sets_yaw():
    """A trap worth pinning: raising the yaw ceiling buys nothing in this mode.

    ``~force_mode`` defaults to "fixed" in the node and is set to "none" by
    every launch file. In "fixed" the shaper replaces each axis magnitude with
    ``~fixed_wz_deg`` (0.7 rad/s = 40 deg/s by default), which is BELOW the
    course-slew limiter and far below the 90 deg/s ceiling -- so a committed
    turn runs at 40 deg/s no matter what ``max_yaw_rate_deg`` says. Anyone
    reading a flight log from an ad hoc launch needs to know that before
    concluding the turn ignores its own ceiling.
    """
    node = _node(max_speed_xy=1.0, max_yaw_rate_deg=90.0, force_mode="fixed")
    _reference_at(node, -3.0, 0.0, vx=-0.5)
    sent = None
    for _ in range(5):
        sent = _step(node)
    assert node._heading_decision.turning
    assert abs(sent.angular.z) == pytest.approx(0.7, abs=1e-6)


def test_the_turn_does_not_reverse_the_yaw_command_when_it_releases():
    """The course limiter must be dragged along, or release commands a reversal.

    Found by replaying this node: a 180 deg demand released at +63 deg/s and
    inverted to -52 deg/s on the very next tick, turning the aircraft away from
    the target it had just stopped to face, and re-engaging the gate.
    """
    node = _node(max_speed_xy=1.0, max_yaw_rate_deg=90.0, course_slew_deg_s=45.0)
    _reference_at(node, -3.0, 0.0, vx=-0.5)
    rates, released = [], False
    for _ in range(120):
        sent = _step(node)
        # Close the yaw loop: the aircraft actually turns at what it is told.
        node._yaw = math.atan2(math.sin(node._yaw + sent.angular.z * DT),
                               math.cos(node._yaw + sent.angular.z * DT))
        if node._heading_decision.turning:
            rates.append(sent.angular.z)
        elif rates:
            released = True
            # The tick after release must not command the opposite sign.
            assert sent.angular.z * rates[-1] >= -1e-6, (
                "yaw reversed from %.3f to %.3f at release"
                % (rates[-1], sent.angular.z))
            break
    assert released, "the turn never released"


def test_normal_flight_never_publishes_backward_translation():
    """The property the clamp exists to hold, checked at the wire."""
    node = _node(max_speed_xy=1.0)
    for angle_deg in range(0, 360, 15):
        angle = math.radians(angle_deg)
        _reference_at(node, 3.0 * math.cos(angle), 3.0 * math.sin(angle),
                      vx=0.5 * math.cos(angle), vy=0.5 * math.sin(angle))
        sent = _step(node)
        assert sent.linear.x >= 0.0, "published %.3f m/s backward at %d deg" % (
            sent.linear.x, angle_deg)


def test_the_escape_reflex_is_still_allowed_to_reverse():
    """Backing out of a contact is the one known-clear backward motion."""
    node = _node(max_speed_xy=1.0, escape_speed_mps=0.3)
    _reference_at(node, 3.0, 0.0, vx=0.5)
    node._escape_until = _CLOCK[0] + 10.0        # force an escape in progress
    sent = _step(node)
    assert sent.linear.x < 0.0


def test_the_park_scan_does_not_fight_a_turn():
    """Two owners of the yaw axis would leave the aircraft sweeping, not turning."""
    node = _node(max_speed_xy=1.0, park_scan_rate=0.5, park_scan_after_s=0.0)
    _reference_at(node, -3.0, 0.0, vx=-0.5)
    for _ in range(5):
        _step(node)
    assert node._heading_decision.turning
    assert node._parked_since is None            # the scan never armed
