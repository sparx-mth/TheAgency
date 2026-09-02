"""Offline wiring tests for RoosterTwistControlNode, no ROS required.

Stubs rclpy and the message packages, then drives the real node tick by tick.
This guards what the pure-servo unit tests cannot: the sign conventions on the
wire (REP103 left-positive vs the FCU's right-positive y and r), the per-axis
slew/warm-start behaviour, the 900-count ceilings, mode resolution, and the
legacy arm's bit-for-bit fidelity. The lateral sign in particular has no other
automated guard, and getting it wrong turns every cross-track correction into
a divergent push toward the obstacle.

Skipped under a real ROS environment (the stubs would fight the real rclpy).
"""

import importlib
import json
import math
import sys
import types

import pytest

try:                                    # pragma: no cover - env probe only
    import rclpy as _real_rclpy         # noqa: F401
    _HAVE_ROS = True
except ImportError:
    _HAVE_ROS = False

pytestmark = pytest.mark.skipif(
    _HAVE_ROS, reason="real rclpy present; offline stubs would conflict")


# ── Stub ROS modules, installed before the adapter is imported ──────────

class _TimeDelta:
    def __init__(self, nanoseconds):
        self.nanoseconds = int(nanoseconds)

    def __gt__(self, other):
        return self.nanoseconds > other.nanoseconds

    def __lt__(self, other):
        return self.nanoseconds < other.nanoseconds


class _Time:
    def __init__(self, nanoseconds):
        self.nanoseconds = int(nanoseconds)

    def __sub__(self, other):
        return _TimeDelta(self.nanoseconds - other.nanoseconds)


class _Clock:
    def __init__(self):
        self.t_ns = 0

    def now(self):
        return _Time(self.t_ns)

    def advance(self, seconds):
        self.t_ns += int(seconds * 1e9)


class _Logger:
    def __init__(self):
        self.lines = []

    def _log(self, msg, **kw):
        self.lines.append(str(msg))

    info = warn = error = debug = _log


class _Pub:
    def __init__(self, topic):
        self.topic = topic
        self.msgs = []

    def publish(self, msg):
        self.msgs.append(msg)


class _Node:
    def __init__(self, name):
        self._clock = _Clock()
        self._logger = _Logger()
        self.pubs = {}

    def get_clock(self):
        return self._clock

    def get_logger(self):
        return self._logger

    def create_publisher(self, msg_type, topic, qos):
        self.pubs[topic] = _Pub(topic)
        return self.pubs[topic]

    def create_subscription(self, msg_type, topic, cb, qos):
        return object()

    def create_timer(self, period, cb):
        return object()

    def destroy_node(self):
        pass


class _V3:
    def __init__(self):
        self.x = self.y = self.z = 0.0


class _Quat:
    def __init__(self):
        self.x = self.y = self.z = 0.0
        self.w = 1.0


class _Twist:
    def __init__(self):
        self.linear = _V3()
        self.angular = _V3()


class _PoseStamped:
    def __init__(self):
        self.pose = types.SimpleNamespace(position=_V3(), orientation=_Quat())


class _TwistStamped:
    def __init__(self):
        self.twist = _Twist()


class _String:
    def __init__(self, data=""):
        self.data = data


def _install_stubs():
    rclpy = types.ModuleType("rclpy")
    rclpy.init = lambda args=None: None
    rclpy.shutdown = lambda: None
    node_mod = types.ModuleType("rclpy.node")
    node_mod.Node = _Node
    dur_mod = types.ModuleType("rclpy.duration")
    dur_mod.Duration = lambda seconds=0.0: _TimeDelta(seconds * 1e9)
    geo = types.ModuleType("geometry_msgs")
    geo_msg = types.ModuleType("geometry_msgs.msg")
    geo_msg.Twist, geo_msg.PoseStamped, geo_msg.TwistStamped = (
        _Twist, _PoseStamped, _TwistStamped)
    std = types.ModuleType("std_msgs")
    std_msg = types.ModuleType("std_msgs.msg")
    std_msg.String = _String
    for name, mod in (("rclpy", rclpy), ("rclpy.node", node_mod),
                      ("rclpy.duration", dur_mod), ("geometry_msgs", geo),
                      ("geometry_msgs.msg", geo_msg), ("std_msgs", std),
                      ("std_msgs.msg", std_msg)):
        sys.modules.setdefault(name, mod)


if not _HAVE_ROS:
    _install_stubs()
    adapter = importlib.import_module(
        "sparx_agency.robots.ROBOTICAN.adapters.rooster_twist_control_adapter")
    from sparx_agency.robots.ROBOTICAN.rooster_axis_curve import (
        ROOSTER_HORIZONTAL_CURVE as CURVE,
    )


def make_node(**kw):
    return adapter.RoosterTwistControlNode(**kw)


def tick(node, vx=0.0, vy=0.0, wz=0.0, world_v=None, yaw=0.0):
    """One 50 ms control tick with the command streamed like the follower's."""
    if world_v is not None:
        tv = _TwistStamped()
        tv.twist.linear.x, tv.twist.linear.y = world_v
        node._velocity_callback(tv)
        ps = _PoseStamped()
        ps.pose.orientation.z = math.sin(yaw / 2.0)
        ps.pose.orientation.w = math.cos(yaw / 2.0)
        node._pose_callback(ps)
    tw = _Twist()
    tw.linear.x, tw.linear.y, tw.angular.z = vx, vy, wz
    node.cmd_vel_callback(tw)
    node.command_timer_callback()
    node.get_clock().advance(0.05)


def last_move(node):
    for msg in reversed(node.pubs["/R1/cmd_nav"].msgs):
        d = json.loads(msg.data)
        if d["action"] == "move":
            return d["axes"]
    return None


def test_curve_mode_resolution_and_lateral_opt_in():
    node = make_node()
    assert node.min_command_mps == 0.05
    assert node.max_lateral_axis == 0.0     # opt-in: safe for every caller
    assert sorted(node._servos) == ["x"]
    node = make_node(max_lateral_axis=900.0)
    assert sorted(node._servos) == ["x", "y"]


def test_forward_lands_on_the_curve():
    node = make_node()
    for _ in range(15):
        tick(node, vx=0.4, world_v=(0.4, 0.0))
    assert abs(last_move(node)["x"] - CURVE.axis_for(0.4)) <= 1.0


def test_left_command_gives_negative_stick_y():
    """REP103 +y (left) must come out as stick y<0 (the FCU 'left' sign)."""
    node = make_node(max_lateral_axis=900.0)
    for _ in range(40):                      # lateral slews at 20 counts/tick
        tick(node, vy=0.3, world_v=(0.0, 0.3))
    ax = last_move(node)
    assert abs(ax["y"] + CURVE.axis_for(0.3)) <= 1.0
    assert ax["x"] == 0


def test_left_yaw_gives_negative_stick_r():
    node = make_node()
    for _ in range(10):
        tick(node, wz=0.5, world_v=(0.0, 0.0))
    assert last_move(node)["r"] < -100


def test_world_to_body_rotation_feeds_the_servos():
    node = make_node()
    for _ in range(15):
        tick(node, vx=0.4, world_v=(0.0, 0.4), yaw=math.pi / 2.0)
    assert abs(last_move(node)["x"] - CURVE.axis_for(0.4)) <= 2.0


def test_both_axes_cap_at_900():
    node = make_node(max_lateral_axis=900.0)
    for _ in range(80):
        tick(node, vx=2.0, vy=2.0, world_v=(0.3, 0.3))
    ax = last_move(node)
    assert ax["x"] == 900 and ax["y"] == -900.0


def test_stale_feedback_degrades_to_open_loop_curve():
    node = make_node()
    for _ in range(15):
        tick(node, vx=0.4)                  # no velocity feedback at all
    assert abs(last_move(node)["x"] - CURVE.axis_for(0.4)) <= 1.0
    assert any("stale" in ln for ln in node.get_logger().lines)


def test_command_silence_stops():
    node = make_node()
    tick(node, vx=0.4, world_v=(0.4, 0.0))
    node.get_clock().advance(0.5)
    node.command_timer_callback()
    assert json.loads(node.pubs["/R1/cmd_nav"].msgs[-1].data)["action"] == "stop"


def test_lateral_ramps_gently_while_forward_stays_agile():
    """The lateral slew (400/s = 20 counts/tick) must be slower than forward's
    (1200/s = 60/tick): banks develop over ~1.5 s instead of 0.5 s."""
    node = make_node(max_lateral_axis=900.0)
    tick(node, vx=0.4, vy=-0.4, world_v=(0.4, -0.4))
    ax = last_move(node)
    assert abs(ax["x"]) == 60                # forward: 60 counts on tick 1
    assert abs(ax["y"]) == 20                # lateral: 20 counts on tick 1
    for _ in range(40):                      # 2 s: both fully developed
        tick(node, vx=0.4, vy=-0.4, world_v=(0.4, -0.4))
    ax = last_move(node)
    assert abs(abs(ax["y"]) - CURVE.axis_for(0.4)) <= 1.0


def test_a_brief_lateral_flip_never_develops_amplitude():
    """A sub-second sign flip is absorbed by the slew -- the weave filter."""
    node = make_node(max_lateral_axis=900.0)
    for _ in range(40):
        tick(node, vy=0.4, world_v=(0.0, 0.4))
    for _ in range(6):                       # 0.3 s flipped demand
        tick(node, vy=-0.4, world_v=(0.0, -0.1))
    ax = last_move(node)
    # 6 ticks x 20 counts: moved at most 120 counts off its prior value,
    # nowhere near a full opposite-side bank.
    assert ax["y"] < -(CURVE.axis_for(0.4) - 130)


def test_legacy_arm_is_the_pre_change_controller():
    node = make_node(legacy_feedforward=True)
    assert node.min_command_mps == 0.15
    assert node.max_lateral_axis == 0.0
    assert sorted(node._servos) == ["x"]
    # First motion tick warm-starts to the standing dead-band edge.
    tick(node, vx=0.4, world_v=(0.0, 0.0))
    assert 620 <= last_move(node)["x"] <= 700
    # Once moving, the (412, 1.847) regime pair rules the feedforward.
    for _ in range(10):
        tick(node, vx=0.4, world_v=(0.4, 0.0))
    assert abs(last_move(node)["x"] - (412.0 + 0.4 / 1.847 * 588.0)) <= 40.0
    # Lateral demand is dropped outright.
    for _ in range(10):
        tick(node, vy=0.4, world_v=(0.0, 0.4))
    assert last_move(node)["y"] == 0.0
