"""Tests for the waypoint_follower's drone-thinking narration.

The follower carries most of the operator's train of thought -- which waypoint it
is aligning to, which it is flying at, when it crossed one, when it stopped to
turn -- so these tests pin that narration.

The node is a ROS1 adapter, so ``rospy`` and the message packages are stubbed in
``sys.modules`` before it is imported. The narration methods are pure functions of
``FollowerCommand`` + the follower's active path, so the tests drive them on a
bare instance (``__new__``) rather than standing up the whole node: no ROS graph,
no timers, no bring-up machine -- just the decision -> sentence mapping.

What is locked in here:
  * waypoint positions come from the follower's ACTIVE (re-anchored) path, never
    from the raw path the node received -- indexing the wrong one silently prints
    the wrong coordinates,
  * a waypoint hand-off is narrated once, on the edge, and never manufactured by
    a re-anchor resetting the index,
  * a re-published route (the planners stream them) does NOT restart the
    narration, which would replay the drone's whole train of thought on repeat.
"""
import pathlib
import sys
import types

import pytest

_SCRIPTS = str(pathlib.Path(__file__).resolve().parents[1])
if _SCRIPTS not in sys.path:
    sys.path.insert(0, _SCRIPTS)


# ─── stub rospy + message packages BEFORE importing the node ────────────────
class _Time:
    t = 0.0

    def __init__(self, secs=0.0):
        self.secs = float(secs)

    def to_sec(self):
        return self.secs

    @staticmethod
    def now():
        return _Time(_Time.t)


def _install_stubs():
    rospy = types.ModuleType("rospy")
    rospy.init_node = lambda *a, **k: None
    rospy.get_param = lambda name, default=None: default
    rospy.has_param = lambda name: False
    rospy.Publisher = lambda *a, **k: types.SimpleNamespace(
        publish=lambda msg: None)
    rospy.Subscriber = lambda *a, **k: None
    rospy.Timer = lambda *a, **k: None
    rospy.Time = _Time
    rospy.Duration = lambda *a, **k: None
    rospy.spin = lambda: None
    rospy.signal_shutdown = lambda *a, **k: None
    rospy.is_shutdown = lambda: True
    for fn in ("loginfo", "logwarn", "logerr", "logfatal", "loginfo_throttle",
               "logwarn_throttle", "logerr_throttle"):
        setattr(rospy, fn, lambda *a, **k: None)
    rospy.ROSInterruptException = type("ROSInterruptException", (Exception,), {})
    sys.modules["rospy"] = rospy

    tf = types.ModuleType("tf")
    tft = types.ModuleType("tf.transformations")
    tft.euler_from_quaternion = lambda q: (0.0, 0.0, 0.0)
    tft.quaternion_from_euler = lambda r, p, y: (0.0, 0.0, 0.0, 1.0)
    tf.transformations = tft
    sys.modules["tf"] = tf
    sys.modules["tf.transformations"] = tft

    for pkg, names in [
        ("geometry_msgs", ["PointStamped", "Pose", "PoseStamped", "Twist"]),
        ("nav_msgs", ["OccupancyGrid", "Path"]),
        ("std_msgs", ["Bool", "Empty", "Float32", "Int8", "String"]),
    ]:
        root = types.ModuleType(pkg)
        mod = types.ModuleType(pkg + ".msg")
        for n in names:
            setattr(mod, n, type(n, (object,), {}))
        root.msg = mod
        sys.modules[pkg] = root
        sys.modules[pkg + ".msg"] = mod


_install_stubs()

import waypoint_follower_node as wfn  # noqa: E402
from sparx_agency.core.common.types import ControlCommand, Pose2D  # noqa: E402
from sparx_agency.core.planning.trackers.waypoint_follower import (  # noqa: E402
    ControlAxis, FollowerCommand, FollowerState)


class _FakeThinker:
    """Records narration instead of publishing it, and honours forget()."""

    def __init__(self):
        self.said = []
        self.forgotten = []

    def say(self, text, category="nav", level="info", key=None,
            repeat_after_s=None):
        self.said.append(text)
        return True

    def forget(self, key=None):
        self.forgotten.append(key)


def _node(active_path, controller="waypoint"):
    """A bare node wired with just what the narration methods touch."""
    n = wfn.WaypointFollowerNode.__new__(wfn.WaypointFollowerNode)
    n.thinker = _FakeThinker()
    n.controller_kind = controller
    n._prev_wp_idx = None
    n._path_pts = []
    n.follower = types.SimpleNamespace(active_path=list(active_path))
    return n


def _cmd(wp_idx, num_waypoints, axis=None, state=FollowerState.ADVANCE,
         done=False):
    return FollowerCommand(
        command=ControlCommand.velocity(0.0, 0.0, 0.0, 0.0, tracker="t"),
        state=state, required_axis=axis, freeze=None, done=done,
        wp_idx=wp_idx, num_waypoints=num_waypoints)


PATH = [(1.0, 2.0), (3.0, 4.0), (5.0, 6.0)]


# ─── the operator's asks: aligning / flying / reached / stopping ────────────
def test_aligning_names_the_waypoint_and_its_position():
    n = _node(PATH)
    n._narrate_nav(_cmd(0, 3, axis=ControlAxis.YAW, state=FollowerState.YAW_ALIGN))
    assert n.thinker.said == ["Aligning to waypoint 1/3 (x=1.00, y=2.00)"]


def test_flying_forward_names_the_waypoint_and_its_position():
    n = _node(PATH)
    n._narrate_nav(_cmd(1, 3, axis=ControlAxis.FORWARD))
    assert n.thinker.said == ["Flying forward to waypoint 2/3 (x=3.00, y=4.00)"]


def test_waypoint_position_comes_from_the_active_path_not_the_received_path():
    # set_path re-anchors and DROPS passed waypoints, so the follower's index
    # space is a suffix of what the node received. Indexing the received path
    # would print a real-looking but wrong position.
    n = _node([(5.0, 6.0)])                       # re-anchored: only the last left
    n._path_pts = [Pose2D(1.0, 2.0), Pose2D(3.0, 4.0), Pose2D(5.0, 6.0)]
    n._narrate_nav(_cmd(0, 1, axis=ControlAxis.FORWARD))
    assert n.thinker.said == ["Flying forward to waypoint 1/1 (x=5.00, y=6.00)"]


def test_crossing_a_waypoint_is_narrated_once_on_the_edge():
    n = _node(PATH)
    n._narrate_nav(_cmd(0, 3, axis=ControlAxis.FORWARD))
    n._narrate_nav(_cmd(1, 3, axis=ControlAxis.FORWARD))
    assert "Reached waypoint 1, heading for waypoint 2" in n.thinker.said


def test_no_handoff_is_narrated_on_the_first_tick():
    # Nothing was "reached" merely because the node started observing.
    n = _node(PATH)
    n._narrate_nav(_cmd(0, 3, axis=ControlAxis.FORWARD))
    assert not any("Reached waypoint" in s for s in n.thinker.said)


def test_a_reanchor_resetting_the_index_never_fakes_a_handoff():
    # set_path resets wp_idx to 0; going BACKWARDS is not an arrival.
    n = _node(PATH)
    n._narrate_nav(_cmd(2, 3, axis=ControlAxis.FORWARD))
    n.thinker.said = []
    n._narrate_nav(_cmd(0, 3, axis=ControlAxis.FORWARD))
    assert not any("Reached waypoint" in s for s in n.thinker.said)


def test_stopping_to_turn_is_narrated_while_braking():
    n = _node(PATH)
    n._narrate_nav(_cmd(1, 3, axis=None, state=FollowerState.BRAKE))
    assert n.thinker.said == ["Stopping to turn"]


def test_reaching_the_goal_is_narrated():
    n = _node(PATH)
    n._narrate_nav(_cmd(2, 3, done=True))
    assert n.thinker.said == ["Reached the goal -- route complete"]


def test_an_index_past_the_last_waypoint_narrates_nothing():
    n = _node(PATH)
    n._narrate_nav(_cmd(3, 3, axis=ControlAxis.FORWARD))
    assert n.thinker.said == []


def test_pure_pursuit_skips_the_per_waypoint_lines():
    # Its wp_idx walks hundreds of spline samples, so per-waypoint narration
    # would be a per-tick counter, not a decision.
    n = _node(PATH, controller="pure_pursuit")
    n._narrate_nav(_cmd(1, 3, axis=ControlAxis.FORWARD))
    assert n.thinker.said == []


def test_pure_pursuit_still_narrates_the_goal():
    n = _node(PATH, controller="pure_pursuit")
    n._narrate_nav(_cmd(2, 3, done=True))
    assert n.thinker.said == ["Reached the goal -- route complete"]


# ─── re-published routes must not restart the narration ────────────────────
def test_same_route_recognises_a_republished_route():
    pts = [Pose2D(0.0, 0.0), Pose2D(1.0, 1.0)]
    assert wfn.WaypointFollowerNode._same_route(pts, list(pts))


def test_same_route_tolerates_float_noise():
    a = [Pose2D(0.0, 0.0), Pose2D(1.0, 1.0)]
    b = [Pose2D(0.0, 0.0), Pose2D(1.0 + 1e-9, 1.0)]
    assert wfn.WaypointFollowerNode._same_route(a, b)


@pytest.mark.parametrize("other, why", [
    ([Pose2D(0.0, 0.0), Pose2D(2.0, 2.0)], "a waypoint moved"),
    ([Pose2D(0.0, 0.0)], "the route got shorter"),
    ([], "there was no previous route"),
])
def test_same_route_rejects_a_genuinely_different_route(other, why):
    pts = [Pose2D(0.0, 0.0), Pose2D(1.0, 1.0)]
    assert not wfn.WaypointFollowerNode._same_route(pts, other), why


def test_same_route_rejects_a_moved_waypoint_beyond_tolerance():
    a = [Pose2D(0.0, 0.0), Pose2D(1.0, 1.0)]
    b = [Pose2D(0.0, 0.0), Pose2D(1.02, 1.0)]
    assert not wfn.WaypointFollowerNode._same_route(a, b)
