"""Tests for the regime gate between drift_pid and the rotation supervisor.

The supervisor's job is the map discipline around REAL turns: freeze while
rotating, stop-and-re-observe between segments. drift_pid also yaws while
TRACKING -- mid-leg heading trims whose rate can cross the supervisor's
``wz_turn_on`` -- and on the 2026-07-21 flight that armed the supervisor on a
perfectly tracking drone, which it then stopped mid-corridor for a stationary
re-observe (frame 88: TRACK, on the line, nose on the next point -> HOLD, coast,
sideways drift, re-acquire). The gate feeds the supervisor a zero yaw rate
unless the follower says it is actually rotating (TURN or ESCAPE), so a trim
can never be mistaken for a turn. Other holonomic trackers have no regime
signal and keep the old passthrough.

The node is a ROS1 adapter, so ``rospy`` and the message packages are stubbed
before it is imported -- same trick as ``test_waypoint_follower_thinking``.
"""
import pathlib
import sys
import types

_SCRIPTS = str(pathlib.Path(__file__).resolve().parents[1])
if _SCRIPTS not in sys.path:
    sys.path.insert(0, _SCRIPTS)

from test_waypoint_follower_thinking import _install_stubs  # noqa: E402

_install_stubs()

import waypoint_follower_node as wfn  # noqa: E402


def _node(kind, state, last_wz=0.31):
    """A bare node wired with just what the gate reads."""
    n = wfn.WaypointFollowerNode.__new__(wfn.WaypointFollowerNode)
    n.controller_kind = kind
    n.follower = types.SimpleNamespace(state=state)
    n._last_pub_wz = last_wz
    return n


def test_a_track_trim_is_not_a_turn():
    assert _node("drift_pid", "TRACK")._supervisor_cmd_wz() == 0.0


def test_a_hold_heading_trim_is_not_a_turn_either():
    assert _node("drift_pid", "HOLD")._supervisor_cmd_wz() == 0.0


def test_a_real_turn_reaches_the_supervisor():
    assert _node("drift_pid", "TURN")._supervisor_cmd_wz() == 0.31


def test_an_escape_keeps_the_map_discipline():
    """Escape yaw probes rotate blind into unseen space -- exactly what the
    freeze + re-observe discipline exists for."""
    assert _node("drift_pid", "ESCAPE")._supervisor_cmd_wz() == 0.31


def test_other_holonomic_trackers_keep_the_old_passthrough():
    assert _node("pure_pursuit", "TRACK")._supervisor_cmd_wz() == 0.31
