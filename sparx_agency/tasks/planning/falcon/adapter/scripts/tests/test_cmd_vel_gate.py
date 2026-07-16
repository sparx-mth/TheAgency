"""Behavioural tests for cmd_vel_gate_node -- the GO gate.

``rospy`` + the message packages are stubbed in ``sys.modules`` before the node is
imported, so the real gate logic runs headless.

The contract locked in here (this node decides whether a drone can move, so the
contract is a safety one):
  * CLOSED means SILENCE -- not forwarded, and no zeros either. A zero stream is a
    command to hold still, which is exactly what fights a manual takeoff;
  * ~start_go defaults TRUE, so every launch that predates this node is unchanged;
  * GO forwards the twist BYTE-FOR-BYTE (the gate must not editorialise);
  * closing mid-flight emits exactly ONE zero twist (a stop, not a hold), and nothing
    after it;
  * a repeated GO/GO or STOP/STOP changes nothing (idempotent);
  * in==out is refused: the gate would feed itself.
"""
import pathlib
import sys
import types

import pytest

_SCRIPTS = str(pathlib.Path(__file__).resolve().parents[1])
if _SCRIPTS not in sys.path:
    sys.path.insert(0, _SCRIPTS)


class _Pub:
    def __init__(self, topic, *a, **k):
        self.topic = topic
        self.msgs = []

    def publish(self, msg):
        self.msgs.append(msg)


class _Twist:
    def __init__(self):
        self.linear = types.SimpleNamespace(x=0.0, y=0.0, z=0.0)
        self.angular = types.SimpleNamespace(x=0.0, y=0.0, z=0.0)


class _Bool:
    def __init__(self, data=False):
        self.data = data


class _String:
    def __init__(self, data=""):
        self.data = data


_PARAMS = {}
_SUBS = {}


def _get_param(name, default=None):
    return _PARAMS.get(name, default)


def _subscriber(topic, _type, cb, **kw):
    _SUBS[topic] = cb
    return object()


_rospy = types.ModuleType("rospy")
_rospy.init_node = lambda *a, **k: None
_rospy.get_param = _get_param
_rospy.Publisher = _Pub
_rospy.Subscriber = _subscriber
_rospy.Timer = lambda *a, **k: None
_rospy.Duration = lambda *a, **k: None
_rospy.spin = lambda: None
_rospy.loginfo = lambda *a, **k: None
_rospy.logwarn = lambda *a, **k: None
_rospy.loginfo_throttle = lambda *a, **k: None
_rospy.logwarn_throttle = lambda *a, **k: None
_rospy.ROSInterruptException = type("ROSInterruptException", (Exception,), {})
sys.modules["rospy"] = _rospy

_geom = types.ModuleType("geometry_msgs.msg")
_geom.Twist = _Twist
sys.modules["geometry_msgs"] = types.ModuleType("geometry_msgs")
sys.modules["geometry_msgs.msg"] = _geom

_std = types.ModuleType("std_msgs.msg")
_std.Bool = _Bool
_std.String = _String
sys.modules["std_msgs"] = types.ModuleType("std_msgs")
sys.modules["std_msgs.msg"] = _std

import cmd_vel_gate_node as gate  # noqa: E402


def _make(**params):
    _PARAMS.clear()
    _SUBS.clear()
    _PARAMS.update({"~%s" % k: v for k, v in params.items()})
    return gate.CmdVelGateNode()


def _twist(vx=1.0, wz=0.5):
    t = _Twist()
    t.linear.x = vx
    t.angular.z = wz
    return t


def _send(node, msg):
    _SUBS[node.in_topic](msg)


def _go(node, value):
    _SUBS[node.go_topic](_Bool(value))


# ── Closed means silence ────────────────────────────────────────────────────
def test_closed_gate_forwards_nothing():
    node = _make(start_go=False)
    for _ in range(5):
        _send(node, _twist())
    assert node.cmd_pub.msgs == []


def test_closed_gate_does_not_emit_zeros_either():
    """A zero stream is a command to hold still -- it would fight a manual takeoff."""
    node = _make(start_go=False)
    _send(node, _twist())
    assert node.cmd_pub.msgs == [], "the gate must be SILENT while closed, not zeroing"


def test_closed_gate_counts_what_it_blocked():
    node = _make(start_go=False)
    for _ in range(3):
        _send(node, _twist())
    assert node._blocked == 3


# ── Default is open: existing launches unchanged ────────────────────────────
def test_start_go_defaults_true_so_old_launches_are_unchanged():
    node = _make()
    _send(node, _twist())
    assert len(node.cmd_pub.msgs) == 1


@pytest.mark.parametrize("value,expected", [
    ("false", False), ("true", True), (False, False), (True, True),
    ("0", False), ("1", True),
])
def test_start_go_accepts_roslaunch_string_booleans(value, expected):
    """roslaunch passes params as the strings 'true'/'false'."""
    assert _make(start_go=value).go is expected


# ── GO opens it ─────────────────────────────────────────────────────────────
def test_go_opens_the_gate():
    node = _make(start_go=False)
    _send(node, _twist())
    _go(node, True)
    _send(node, _twist())
    assert len(node.cmd_pub.msgs) == 1, "only the post-GO command should pass"


def test_forwarded_twist_is_the_same_object_unmodified():
    node = _make(start_go=True)
    msg = _twist(vx=0.7, wz=-0.3)
    _send(node, msg)
    out = node.cmd_pub.msgs[0]
    assert out is msg and out.linear.x == 0.7 and out.angular.z == -0.3


# ── Closing mid-flight stops, it does not hold ──────────────────────────────
def test_closing_emits_exactly_one_zero_then_silence():
    node = _make(start_go=True)
    _send(node, _twist())
    _go(node, False)
    zero = node.cmd_pub.msgs[-1]
    assert zero.linear.x == 0.0 and zero.angular.z == 0.0
    before = len(node.cmd_pub.msgs)
    for _ in range(3):
        _send(node, _twist())
    assert len(node.cmd_pub.msgs) == before, "nothing may follow the stop"


def test_zero_on_close_can_be_disabled():
    node = _make(start_go=True, zero_on_close=False)
    _send(node, _twist())
    n = len(node.cmd_pub.msgs)
    _go(node, False)
    assert len(node.cmd_pub.msgs) == n


# ── Idempotence ─────────────────────────────────────────────────────────────
def test_repeated_go_is_idempotent():
    node = _make(start_go=False)
    _go(node, True)
    _go(node, True)
    _send(node, _twist())
    assert len(node.cmd_pub.msgs) == 1


def test_repeated_stop_emits_only_one_zero():
    node = _make(start_go=True)
    _go(node, False)
    _go(node, False)
    assert len(node.cmd_pub.msgs) == 1


def test_go_then_stop_then_go_again():
    node = _make(start_go=False)
    _go(node, True)
    _send(node, _twist())
    _go(node, False)          # + one zero
    _send(node, _twist())     # blocked
    _go(node, True)
    _send(node, _twist())
    passed = [m for m in node.cmd_pub.msgs if m.linear.x != 0.0]
    assert len(passed) == 2


# ── Wiring ──────────────────────────────────────────────────────────────────
def test_self_feeding_topics_are_refused():
    with pytest.raises(ValueError, match="feed itself"):
        _make(in_topic="/cmd_vel", out_topic="/cmd_vel")


def test_topics_default_off_the_drone_namespace():
    node = _make(drone_ns="/x")
    assert node.in_topic == "/x/cmd_vel_raw" and node.out_topic == "/x/cmd_vel"


def test_status_is_latched_and_reflects_the_gate():
    node = _make(start_go=False)
    assert "HELD" in node.status_pub.msgs[-1].data
    _go(node, True)
    assert "GO" in node.status_pub.msgs[-1].data
