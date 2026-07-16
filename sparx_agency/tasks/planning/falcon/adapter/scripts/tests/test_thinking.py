"""Tests for the Thinker narration helper.

Every nav node narrates through this, and the BEV viewer's log is only as good as
what comes out the other side -- so these tests drive the real publisher against
the real codec the viewer parses with, and pin the property the nodes depend on:
that narrating from inside a control loop is safe.

``rospy`` is stubbed with a controllable clock so the gate's timing is exercised
deterministically. The stub is applied to ``thinking``'s OWN ``rospy`` reference
rather than only to ``sys.modules``: ``thinking.py`` binds the module object at
import, and a sibling test module in this directory may well have imported it
first (importing any node imports ``thinking``). Re-binding ``sys.modules`` after
that point would leave ``thinking.rospy`` pointing at the other file's stub, and
these tests would pass alone but fail as part of the suite.
"""
import pathlib
import sys
import types

import pytest

_SCRIPTS = str(pathlib.Path(__file__).resolve().parents[1])
if _SCRIPTS not in sys.path:
    sys.path.insert(0, _SCRIPTS)


class _Clock:
    t = 0.0


class _Time:
    def __init__(self, secs=0.0):
        self.secs = float(secs)

    def to_sec(self):
        return self.secs

    @staticmethod
    def now():
        return _Time(_Clock.t)


class _Pub:
    def __init__(self, topic, *a, **k):
        self.topic = topic
        self.msgs = []

    def publish(self, msg):
        self.msgs.append(msg)


class _String:
    def __init__(self, data=""):
        self.data = data


_PARAMS = {}
_PUBS = []
_LOGS = []


def _publisher(topic, *a, **k):
    p = _Pub(topic)
    _PUBS.append(p)
    return p


def _install_stubs():
    """Put a minimal rospy/std_msgs in sys.modules so ``import thinking`` works.

    Only enough to get the import through -- the per-test fixture is what makes
    the stub authoritative, since a sibling test may have imported thinking
    against its own stub long before this module was collected.
    """
    if "rospy" not in sys.modules:
        rospy = types.ModuleType("rospy")
        rospy.get_param = lambda name, default=None: _PARAMS.get(name, default)
        rospy.Time = _Time
        rospy.Publisher = _publisher
        rospy.loginfo = lambda *a, **k: _LOGS.append(("info", a))
        rospy.logwarn = lambda *a, **k: _LOGS.append(("warn", a))
        sys.modules["rospy"] = rospy
    if "std_msgs.msg" not in sys.modules:
        std = types.ModuleType("std_msgs.msg")
        std.String = _String
        root = types.ModuleType("std_msgs")
        root.msg = std
        sys.modules["std_msgs"] = root
        sys.modules["std_msgs.msg"] = std


_install_stubs()

import thinking  # noqa: E402
from sparx_agency.core.common.thought_message import (  # noqa: E402
    parse_thought_message)


@pytest.fixture(autouse=True)
def _reset(monkeypatch):
    _PARAMS.clear()
    del _PUBS[:]
    del _LOGS[:]
    _Clock.t = 0.0
    # Patch what `thinking` actually resolves at call time, not sys.modules: it
    # bound `rospy` and `String` at import, possibly against a sibling's stub.
    monkeypatch.setattr(thinking.rospy, "get_param",
                        lambda name, default=None: _PARAMS.get(name, default))
    monkeypatch.setattr(thinking.rospy, "Publisher", _publisher)
    monkeypatch.setattr(thinking.rospy, "Time", _Time)
    monkeypatch.setattr(thinking.rospy, "loginfo",
                        lambda *a, **k: _LOGS.append(("info", a)))
    monkeypatch.setattr(thinking.rospy, "logwarn",
                        lambda *a, **k: _LOGS.append(("warn", a)))
    monkeypatch.setattr(thinking, "String", _String)
    yield


def _thinker(**params):
    _PARAMS.update(params)
    return thinking.Thinker("test_node")


def _published(t):
    return [parse_thought_message(m.data) for m in t._pub.msgs]


# ─── the wire contract with the viewer ─────────────────────────────────────
def test_a_thought_reaches_the_wire_as_the_viewer_will_parse_it():
    t = _thinker()
    _Clock.t = 12.5
    assert t.say("Stopping to turn")
    got = _published(t)
    assert len(got) == 1
    assert got[0].text == "Stopping to turn"
    assert got[0].category == "nav"
    assert got[0].level == "info"
    assert got[0].source == "test_node"
    assert got[0].stamp == 12.5


def test_category_and_level_reach_the_wire():
    t = _thinker()
    t.say("No localization", category="sensor", level="warn")
    got = _published(t)[0]
    assert (got.category, got.level) == ("sensor", "warn")


def test_it_publishes_on_the_shared_topic_by_default():
    assert _thinker()._pub.topic == thinking.THINKING_TOPIC


def test_the_topic_is_a_rosparam():
    assert _thinker(**{"~thinking_topic": "/other"})._pub.topic == "/other"


def test_the_log_is_not_latched():
    # A latched log would replay a backlog of old thoughts at a late-joining
    # viewer as if they were being thought now.
    calls = []
    thinking.rospy.Publisher = lambda topic, *a, **k: (
        calls.append(k) or _Pub(topic))
    try:
        _thinker()
    finally:
        _install_stubs()
    assert not calls[0].get("latch", False)


# ─── the property the nodes rely on: narrating from a control loop ─────────
def test_an_unchanged_thought_is_published_once():
    t = _thinker()
    for i in range(20):                       # a 5 Hz loop, 4 seconds
        _Clock.t = i * 0.2
        t.say("Flying forward to waypoint 3")
    assert len(t._pub.msgs) == 1


def test_a_changed_thought_publishes_again():
    t = _thinker()
    t.say("Aligning to waypoint 3")
    t.say("Flying forward to waypoint 3")
    assert [g.text for g in _published(t)] == [
        "Aligning to waypoint 3", "Flying forward to waypoint 3"]


def test_say_reports_whether_it_published():
    t = _thinker()
    assert t.say("Stopping to turn")
    assert not t.say("Stopping to turn")


def test_categories_are_independent_slots_by_default():
    # The motion story must not suppress the sensor story.
    t = _thinker()
    assert t.say("Flying forward", category="nav")
    assert t.say("No localization", category="sensor", level="warn")
    assert not t.say("Flying forward", category="nav")


def test_an_explicit_key_separates_two_stories_in_one_category():
    t = _thinker()
    assert t.say("Leg started", category="plan", key="leg")
    assert t.say("NavDP is healthy", category="plan", key="health")


def test_repeat_after_restates_a_persistent_state():
    t = _thinker()
    assert t.say("No localization", category="sensor", level="warn",
                 repeat_after_s=5.0)
    _Clock.t = 4.9
    assert not t.say("No localization", category="sensor", level="warn",
                     repeat_after_s=5.0)
    _Clock.t = 5.0
    assert t.say("No localization", category="sensor", level="warn",
                 repeat_after_s=5.0)


def test_the_instance_default_repeat_applies():
    t = thinking.Thinker("test_node", repeat_after_s=2.0)
    assert t.say("hold")
    _Clock.t = 1.9
    assert not t.say("hold")
    _Clock.t = 2.0
    assert t.say("hold")


def test_forget_lets_a_restarted_story_narrate_again():
    t = _thinker()
    t.say("Aligning to waypoint 1")
    t.forget()
    assert t.say("Aligning to waypoint 1")


def test_forget_targets_one_slot():
    t = _thinker()
    t.say("Flying forward", category="nav")
    t.say("Map frozen", category="map")
    t.forget("nav")
    assert t.say("Flying forward", category="nav")
    assert not t.say("Map frozen", category="map")


# ─── switches, echo, and failing loudly ────────────────────────────────────
def test_narration_can_be_disabled():
    t = _thinker(**{"~thinking": False})
    assert not t.say("Stopping to turn")
    assert t._pub is None


def test_an_empty_topic_means_off_not_a_publisher_on_an_empty_name():
    # '' disables the viewer's log window, so it must mean the same on a node
    # rather than dying inside rospy.Publisher("") on an obscure name error.
    t = _thinker(**{"~thinking_topic": ""})
    assert t._pub is None
    assert not t.say("Stopping to turn")


def test_thoughts_are_echoed_to_rosout_at_a_matching_level():
    t = _thinker()
    t.say("Stopping to turn")
    t.say("No localization", category="sensor", level="warn")
    assert [lvl for lvl, _ in _LOGS if lvl == "warn"]
    assert [lvl for lvl, _ in _LOGS if lvl == "info"]


def test_the_echo_can_be_disabled_without_silencing_the_topic():
    t = _thinker(**{"~thinking_echo": False})
    del _LOGS[:]
    assert t.say("Stopping to turn")
    assert _LOGS == []


def test_an_unknown_category_raises_rather_than_mislabelling():
    with pytest.raises(ValueError, match="unknown thought category"):
        _thinker().say("hi", category="weather")


def test_a_mislabelled_thought_keeps_raising_every_tick():
    # The gate records the text it is asked about, so validating AFTER gating
    # would make the bug raise on the first tick and then be swallowed as a
    # "repeat" on every tick after -- loud once, then silent forever, from a
    # node narrating in a loop.
    t = _thinker()
    for _ in range(3):
        with pytest.raises(ValueError, match="unknown thought category"):
            t.say("Stopping to turn", category="weather")


def test_a_rejected_thought_does_not_poison_the_slot():
    # The failed call must not leave its text recorded on the slot, or the
    # corrected line would be gated away as a repeat.
    t = _thinker()
    with pytest.raises(ValueError):
        t.say("Stopping to turn", category="weather")
    assert t.say("Stopping to turn", category="nav")


def test_an_unknown_level_raises():
    with pytest.raises(ValueError, match="unknown thought level"):
        _thinker().say("hi", level="fatal")


def test_empty_text_raises():
    with pytest.raises(ValueError, match="empty"):
        _thinker().say("   ")
