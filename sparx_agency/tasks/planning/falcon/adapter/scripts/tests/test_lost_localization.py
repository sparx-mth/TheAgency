"""Behavioural tests for lost_localization_node -- the pose-went-cold recovery.

``rospy`` + the message packages are stubbed in ``sys.modules`` before the node is
imported, so the real node logic runs headless against a settable fake clock. The
escalation itself is tested in
``core/planning/lost_localization/tests/test_state_machine.py``; what is locked in
HERE is the ROS boundary, where every one of these is a way to get it wrong:

  * staleness is measured from message ARRIVAL, and the node watches the STAMPED
    topic -- ``header.stamp`` is a different machine's camera clock, and
    ``/gt_pose`` has no header at all;
  * a pose that never arrives at all is a WIRING fault, not a lost drone: the node
    stays inert rather than flying a blind ladder on boot;
  * a stop is published as ZEROS, never as silence -- the platform holds its last
    command, so going quiet leaves the drone flying it;
  * while localization is healthy the node publishes NOTHING (the follower owns
    cmd_vel);
  * taking over REQUIRES releasing: the follower is passive while demo_mode reads
    'recovery', so the node must actively request a mode back or the drone never
    flies again;
  * commands go to cmd_vel_raw, so recovery is still behind the GO gate;
  * an unconfirmed mode does not mute the recovery (default), because a demo
    manager that never learned 'recovery' would otherwise disable it silently.
"""
import math
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


class _PoseStamped:
    def __init__(self):
        self.header = types.SimpleNamespace(stamp=None, frame_id="world")
        self.pose = types.SimpleNamespace(
            position=types.SimpleNamespace(x=0.0, y=0.0, z=0.0),
            orientation=types.SimpleNamespace(x=0.0, y=0.0, z=0.0, w=1.0))


class _Float32:
    def __init__(self, data=0.0):
        self.data = data


class _String:
    def __init__(self, data=""):
        self.data = data


class _Time:
    """A settable fake clock; arithmetic mirrors rospy.Time/Duration."""

    now_s = 0.0

    def __init__(self, secs=0.0):
        self.secs = float(secs)

    @classmethod
    def now(cls):
        return cls(cls.now_s)

    def __sub__(self, other):
        return _Duration(self.secs - other.secs)

    def to_sec(self):
        return self.secs


class _Duration:
    def __init__(self, secs=0.0, *a, **k):
        self.secs = float(secs)

    def to_sec(self):
        return self.secs


_PARAMS = {}
_SUBS = {}
_TIMERS = []


def _subscriber(topic, _type, cb, **kw):
    _SUBS[topic] = cb
    return object()


def _timer(_period, cb, **kw):
    _TIMERS.append(cb)
    return object()


_rospy = types.ModuleType("rospy")
_rospy.init_node = lambda *a, **k: None
_rospy.get_param = lambda name, default=None: _PARAMS.get(name, default)
_rospy.Publisher = _Pub
_rospy.Subscriber = _subscriber
_rospy.Timer = _timer
_rospy.Duration = _Duration
_rospy.Time = _Time
_rospy.spin = lambda: None
_rospy.loginfo = lambda *a, **k: None
_rospy.logwarn = lambda *a, **k: None
_rospy.logerr = lambda *a, **k: None
_rospy.loginfo_throttle = lambda *a, **k: None
_rospy.logwarn_throttle = lambda *a, **k: None
_rospy.ROSInterruptException = type("ROSInterruptException", (Exception,), {})
sys.modules["rospy"] = _rospy

_geom = types.ModuleType("geometry_msgs.msg")
_geom.Twist = _Twist
_geom.PoseStamped = _PoseStamped
sys.modules["geometry_msgs"] = types.ModuleType("geometry_msgs")
sys.modules["geometry_msgs.msg"] = _geom

_std = types.ModuleType("std_msgs.msg")
_std.Float32 = _Float32
_std.String = _String
sys.modules["std_msgs"] = types.ModuleType("std_msgs")
sys.modules["std_msgs.msg"] = _std

import lost_localization_node as lost  # noqa: E402

DT = 0.1


def _make(**params):
    _PARAMS.clear()
    _SUBS.clear()
    _TIMERS[:] = []
    _Time.now_s = 100.0                  # not 0: catches uninitialised-clock bugs
    defaults = {"drone_ns": "/xtend", "stale_s": 0.3, "ladder_s": 1.0,
                "rate_hz": 1.0 / DT}
    defaults.update(params)
    _PARAMS.update({"~%s" % k: v for k, v in defaults.items()})
    node = lost.LostLocalizationNode()
    node.current_demo_mode = "fly_straight"
    return node


def _tick(node, n=1, *, tag=False, bearing=None):
    """Advance the fake clock n ticks, optionally delivering a pose/bearing first."""
    for _ in range(n):
        _Time.now_s += DT
        if tag:
            _SUBS[node.pose_topic](_PoseStamped())
        if bearing is not None:
            _SUBS[node.bearing_topic](_Float32(bearing))
        _TIMERS[0](None)


def _cmds(node):
    return node.cmd_pub.msgs


def _modes(node):
    return [m.data for m in node.demo_req_pub.msgs]


# ── The ROS boundary ────────────────────────────────────────────────
def test_watches_the_stamped_topic_not_gt_pose():
    """/gt_pose is a bare Pose with no header -- it cannot answer 'how old?'."""
    node = _make()
    assert node.pose_topic == "/xtend/localization"
    assert node.pose_topic in _SUBS


def test_commands_go_through_the_go_gate():
    """cmd_vel_raw, not cmd_vel: recovery must not bypass the GO kill switch."""
    node = _make()
    assert node.cmd_vel_topic == "/xtend/cmd_vel_raw"


def test_arrival_time_drives_staleness_not_header_stamp():
    """A pose whose header.stamp is ancient (or 0) is still FRESH on arrival.

    The stamp is the drone's camera clock from another machine; the only thing
    this node needs from a pose is that it turned up.
    """
    node = _make()
    stale_stamped = _PoseStamped()
    stale_stamped.header.stamp = _Time(0.0)      # decades old / never set
    for _ in range(20):
        _Time.now_s += DT
        _SUBS[node.pose_topic](stale_stamped)
        _TIMERS[0](None)
    assert _cmds(node) == [], "a live stream must leave the follower alone"


def test_healthy_localization_publishes_nothing():
    node = _make()
    _tick(node, 20, tag=True)
    assert _cmds(node) == []
    assert _modes(node) == [], "an idle recovery must not touch demo_mode either"


def test_never_bootstrapped_stays_inert():
    """No pose has EVER arrived => a wiring fault, not a lost drone.

    A bridge config that does not carry the pose topic would otherwise put the
    drone into a blind back-up the moment it boots.
    """
    node = _make()
    _tick(node, 100)                     # never a single pose
    assert _cmds(node) == []
    assert _modes(node) == []


# ── Stopping ────────────────────────────────────────────────────────
def test_cold_pose_stops_the_drone_with_zeros():
    node = _make()
    _tick(node, 1, tag=True)
    _tick(node, 5)                       # 0.5s with no pose
    assert _cmds(node), "a cold pose must produce a stop, not silence"
    for m in _cmds(node):
        assert (m.linear.x, m.linear.y, m.linear.z, m.angular.z) == (0, 0, 0, 0)


def test_cold_pose_claims_the_recovery_mode():
    node = _make()
    _tick(node, 1, tag=True)
    _tick(node, 5)
    assert lost.MODE_RECOVERY in _modes(node)


def test_stop_precedes_the_ladder():
    """0.3s stops; only 1.0s starts flying backwards."""
    node = _make(stale_s=0.3, ladder_s=1.0)
    _tick(node, 1, tag=True)
    _tick(node, 8)                       # 0.8s: cold, but not ladder-cold
    assert all(m.linear.x == 0.0 for m in _cmds(node)), \
        "must not fly backwards before ladder_s"


def test_ladder_flies_backwards_after_the_ladder_threshold():
    node = _make(stale_s=0.3, ladder_s=0.5, back_speed=0.25)
    _tick(node, 1, tag=True)
    _tick(node, 12)
    assert any(m.linear.x == pytest.approx(-0.25) for m in _cmds(node))


def test_climb_rung_sets_linear_z():
    """This node is the one Twist assembler that does NOT hardwire linear.z."""
    node = _make(stale_s=0.3, ladder_s=0.5, back_duration_s=0.2, dwell_s=0.2,
                 climb_speed=0.2)
    _tick(node, 1, tag=True)
    _tick(node, 60)
    assert any(m.linear.z == pytest.approx(0.2) for m in _cmds(node))


def test_climb_can_be_disabled_for_a_platform_that_drops_linear_z():
    node = _make(stale_s=0.3, ladder_s=0.5, climb_enabled=False,
                 back_duration_s=0.2, dwell_s=0.2)
    _tick(node, 1, tag=True)
    _tick(node, 80)
    assert all(m.linear.z == 0.0 for m in _cmds(node))


# ── Handing back ────────────────────────────────────────────────────
def test_release_requests_a_mode_back_or_the_follower_stays_passive_forever():
    """The follower is passive while demo_mode reads 'recovery'.

    Going quiet is NOT enough to hand back -- the node must actively request a
    non-recovery mode, or the drone never flies again.
    """
    node = _make(stale_s=0.3, ladder_s=0.5, exit_confirm_poses=2)
    _tick(node, 1, tag=True)
    _tick(node, 10)                      # recovery takes over
    assert lost.MODE_RECOVERY in _modes(node)
    _tick(node, 5, tag=True)             # tag comes back
    assert _modes(node)[-1] == lost.MODE_RELEASE


def test_release_sends_enough_zeros_to_actually_stop_the_platform():
    """One zero does NOT stop this drone.

    The XTEND converter ignores the first zero Twist after a motion command
    (zero_stop_required_count=2), and cmd_vel_gate's queue_size=1 can coalesce two
    published in the same tick. So the hand-back must send several zeros on
    SEPARATE ticks -- otherwise the drone keeps flying the rung it was on until
    the follower wakes up, which needs a demo-mode round-trip.
    """
    for confirm in (1, 2, 3):            # must not depend on the exit debounce
        node = _make(stale_s=0.3, ladder_s=0.5, exit_confirm_poses=confirm,
                     release_zero_ticks=3)
        _tick(node, 1, tag=True)
        _tick(node, 10)                  # flying backwards on back#1
        n_before = len(_cmds(node))
        _tick(node, 8, tag=True)
        tail = _cmds(node)[n_before:]
        zeros = [m for m in tail if m.linear.x == 0.0]
        assert len(zeros) >= 2, (
            "exit_confirm_poses=%d released with only %d zero(s) -- the converter "
            "discards a lone zero and the drone flies on" % (confirm, len(zeros)))
        assert all(m.linear.x == 0.0 for m in tail), \
            "nothing but zeros may be sent once we are handing back"


def test_a_single_zero_release_is_refused_at_construction():
    with pytest.raises(ValueError, match="release_zero_ticks"):
        _make(release_zero_ticks=1)


def test_release_finishes_and_then_goes_quiet():
    node = _make(stale_s=0.3, ladder_s=0.5)
    _tick(node, 1, tag=True)
    _tick(node, 10)
    _tick(node, 10, tag=True)            # exits and completes the zero burst
    n_before = len(_cmds(node))
    _tick(node, 20, tag=True)            # healthy for a long time
    assert len(_cmds(node)) == n_before, "a released node must publish nothing"


def test_idle_node_never_touches_demo_mode():
    """Never having taken over, we must not fight the follower's own modes."""
    node = _make()
    _tick(node, 30, tag=True)
    assert _modes(node) == []


# ── Giving up ───────────────────────────────────────────────────────
def test_exhausted_ladder_requests_the_land():
    node = _make(stale_s=0.3, ladder_s=0.5, back_repeats=0, climb_enabled=False,
                 turn_enabled=False)
    _tick(node, 1, tag=True)
    _tick(node, 10)
    assert _modes(node)[-1] == lost.MODE_FINISH
    assert all(m.linear.x == 0.0 for m in _cmds(node))


def test_land_is_never_released_even_if_the_tag_returns():
    node = _make(stale_s=0.3, ladder_s=0.5, back_repeats=0, climb_enabled=False,
                 turn_enabled=False)
    _tick(node, 1, tag=True)
    _tick(node, 10)                      # gives up -> finish
    _tick(node, 20, tag=True)            # tag returns mid-land
    assert lost.MODE_RELEASE not in _modes(node), \
        "a land in progress must not be handed back to the follower"


# ── Mode handshake ──────────────────────────────────────────────────
def test_unconfirmed_mode_does_not_mute_the_recovery_by_default():
    """A demo manager that never learned 'recovery' drops the request silently.

    Waiting for a confirmation that can never arrive would make recovery a
    no-op exactly when it is needed, so by default we command anyway.
    """
    node = _make(stale_s=0.3, ladder_s=0.5)
    node.current_demo_mode = "fly_straight"      # never grants 'recovery'
    _tick(node, 1, tag=True)
    _tick(node, 12)
    assert _cmds(node), "recovery must still act on a platform that ignores the mode"


def test_require_mode_confirm_holds_until_granted():
    node = _make(stale_s=0.3, ladder_s=0.5, require_mode_confirm=True)
    node.current_demo_mode = "fly_straight"
    _tick(node, 1, tag=True)
    _tick(node, 12)
    assert _cmds(node) == [], "opted in, we wait for the platform to grant the mode"
    node.current_demo_mode = lost.MODE_RECOVERY
    _tick(node, 2)
    assert _cmds(node), "and command once it is granted"


# ── The sweep's heading source ──────────────────────────────────────
def test_stale_bearing_is_not_trusted_to_close_the_sweep():
    node = _make(bearing_max_age_s=0.5)
    _tick(node, 1, tag=True, bearing=1.0)
    assert node._fresh_bearing(_Time.now()) == pytest.approx(1.0)
    _tick(node, 10, tag=True)            # bearing goes quiet for 1.0s
    assert node._fresh_bearing(_Time.now()) is None


def test_bearing_topic_can_be_disabled():
    node = _make(bearing_topic="")
    assert node._fresh_bearing(_Time.now()) is None


# ── The GO gate ─────────────────────────────────────────────────────
def test_closed_gate_makes_recovery_inert():
    """A shut gate means a human is flying: do not run a ladder at them.

    Every Twist would be dropped by the gate, so the ladder would tick through
    its rungs against a drone that never moved -- and then land it, which is NOT
    gated (the land goes out as a demo-mode request and the manager drives
    cmd_nav directly).
    """
    node = _make(stale_s=0.3, ladder_s=0.5)
    _SUBS[node.go_status_topic](_String("HELD -- no commands sent; publish true"))
    _tick(node, 1, tag=True)
    _tick(node, 100)                     # long enough for the whole ladder
    assert _cmds(node) == [], "recovery must not command while the gate is shut"
    assert lost.MODE_FINISH not in _modes(node), \
        "and must never land a drone a human is flying"


def test_recovery_arms_once_the_gate_opens():
    node = _make(stale_s=0.3, ladder_s=0.5)
    _SUBS[node.go_status_topic](_String("HELD -- no commands sent"))
    _tick(node, 1, tag=True)
    _tick(node, 20)
    assert _cmds(node) == []
    _SUBS[node.go_status_topic](_String("GO -- commands reaching the drone"))
    _tick(node, 12)
    assert _cmds(node), "an open gate must re-arm the recovery"


def test_an_unwired_gate_does_not_disable_recovery():
    """No gate status ever published => assume GO, not HELD.

    Failing closed here would silently disable the recovery on every launch that
    does not run the gate.
    """
    node = _make(stale_s=0.3, ladder_s=0.5)
    _tick(node, 1, tag=True)
    _tick(node, 12)                      # no go_status message ever
    assert _cmds(node)


# ── The bearing's units ─────────────────────────────────────────────
def test_bearing_in_degrees_is_detected_and_converted():
    """Fed degrees while believing radians, the sweep would close after ~9 deg."""
    node = _make(bearing_units="auto")
    _tick(node, 1, tag=True, bearing=336.0)      # cannot be radians
    assert node._fresh_bearing(_Time.now()) == pytest.approx(math.radians(336.0))


def test_bearing_unit_detection_latches():
    """It must not change its mind mid-sweep: that would corrupt the angle sum."""
    node = _make(bearing_units="auto")
    _tick(node, 1, tag=True, bearing=336.0)      # decides: degrees
    _tick(node, 1, tag=True, bearing=3.0)        # small, but still degrees
    assert node._fresh_bearing(_Time.now()) == pytest.approx(math.radians(3.0))


def test_bearing_units_can_be_pinned():
    node = _make(bearing_units="rad")
    _tick(node, 1, tag=True, bearing=336.0)      # operator says radians: obey
    assert node._fresh_bearing(_Time.now()) == pytest.approx(336.0)
    with pytest.raises(ValueError, match="bearing_units"):
        _make(bearing_units="furlongs")


def test_turn_dir_accepts_operator_words():
    assert lost._turn_dir("left") == 1
    assert lost._turn_dir("right") == -1
    with pytest.raises(ValueError):
        lost._turn_dir("sideways")
