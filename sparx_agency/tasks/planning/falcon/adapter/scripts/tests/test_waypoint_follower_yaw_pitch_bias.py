"""Tests for the turn-in-place PITCH bias on the waypoint follower's output.

A pure-yaw twist (``angular.z`` alone) leaves this platform flat, so the turn
bites late and then coasts. ``~yaw_pitch_bias`` rides a little forward speed on
exactly those twists. What is locked in here:

  * a pure-yaw twist leaves with the bias as ``linear.x``,
  * a twist that already commands forward motion is untouched (the bias is never
    ADDED to a leg -- it only fills in an idle forward axis),
  * a stop stays a stop: no yaw, no bias,
  * ``yaw_pitch_bias: 0.0`` restores the old pure-yaw behaviour,
  * the bias reaches BOTH publish paths (one-axis and multi-axis), and is applied
    after the command-commitment gate, so the gate still sees the turn as a turn.

The node is a ROS1 adapter, so ``rospy`` and the message packages are stubbed
before it is imported -- the same trick, and the same bare-instance style, as
``test_waypoint_follower_thinking``.
"""
import pathlib
import sys

import pytest

_SCRIPTS = str(pathlib.Path(__file__).resolve().parents[1])
if _SCRIPTS not in sys.path:
    sys.path.insert(0, _SCRIPTS)

from test_waypoint_follower_thinking import _install_stubs  # noqa: E402

_install_stubs()

import waypoint_follower_node as wfn  # noqa: E402

BIAS = 0.05


class _Vec:
    def __init__(self):
        self.x = self.y = self.z = 0.0


class _Twist:
    def __init__(self):
        self.linear = _Vec()
        self.angular = _Vec()


class _Pub:
    """Records the twists that would have gone to the drone."""

    def __init__(self):
        self.sent = []

    def publish(self, msg):
        self.sent.append(msg)


def _node(bias=BIAS, commit_ticks=1):
    """A bare node wired with just what the publish path touches."""
    n = wfn.WaypointFollowerNode.__new__(wfn.WaypointFollowerNode)
    n.yaw_pitch_bias = bias
    n.cmd_stop_eps = 1e-3
    n.cmd_commit_ticks = commit_ticks
    n._cmd_cat = None
    n._cmd_run = 0
    n._cmd_vx = n._cmd_wz = 0.0
    n.cmd_vel_pub = _Pub()
    n._log_file = None
    return n


@pytest.fixture(autouse=True)
def _twist(monkeypatch):
    monkeypatch.setattr(wfn, "Twist", _Twist)


# ─── the rule itself ────────────────────────────────────────────────────────
def test_a_pure_yaw_twist_carries_the_pitch_bias():
    n = _node()
    n._publish_twist(0.0, 0.7)
    sent = n.cmd_vel_pub.sent[-1]
    assert sent.linear.x == pytest.approx(BIAS)
    assert sent.angular.z == pytest.approx(0.7)


def test_the_bias_is_positive_whichever_way_the_turn_goes():
    n = _node()
    n._publish_twist(0.0, -0.7)
    assert n.cmd_vel_pub.sent[-1].linear.x == pytest.approx(BIAS)


def test_a_commanded_forward_leg_is_left_alone():
    """The bias fills an idle forward axis; it never adds to a flight speed."""
    n = _node()
    n._publish_twist(0.25, 0.7)
    assert n.cmd_vel_pub.sent[-1].linear.x == pytest.approx(0.25)


def test_a_stop_stays_a_stop():
    n = _node()
    n._publish_twist(0.0, 0.0)
    assert n.cmd_vel_pub.sent[-1].linear.x == pytest.approx(0.0)


def test_zero_bias_restores_the_pure_yaw_twist():
    n = _node(bias=0.0)
    n._publish_twist(0.0, 0.7)
    assert n.cmd_vel_pub.sent[-1].linear.x == pytest.approx(0.0)


# ─── both publish paths, and the gate underneath ────────────────────────────
def test_the_multi_axis_path_gets_the_bias_too():
    n = _node()
    n._publish_twist_multi(0.0, 0.0, 0.7)
    assert n.cmd_vel_pub.sent[-1].linear.x == pytest.approx(BIAS)


def test_the_multi_axis_crab_is_untouched_by_the_bias():
    n = _node()
    n._publish_twist_multi(0.0, 0.2, 0.7)
    sent = n.cmd_vel_pub.sent[-1]
    assert sent.linear.x == pytest.approx(BIAS)
    assert sent.linear.y == pytest.approx(0.2)


def test_the_commitment_gate_still_sees_a_turn_as_a_turn():
    """The bias is applied AFTER the gate: were it applied before, every turn
    would categorise as forward+yaw and a held turn would repeat the bias as if
    it were a commanded leg."""
    n = _node(commit_ticks=3)
    n._publish_twist(0.0, 0.7)            # tick 1 of an under-committed turn
    n._publish_twist(0.0, 0.0)            # a stop too early -- the turn is held
    assert n._cmd_cat == (0, 1)           # yaw only, the bias did not show up
    assert n._cmd_vx == pytest.approx(0.0)
    held = n.cmd_vel_pub.sent[-1]
    assert held.angular.z == pytest.approx(0.7)
    assert held.linear.x == pytest.approx(BIAS)
