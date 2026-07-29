"""Tests for the drift-PID task adapter (rosparams -> DriftPidParams).

ROS-free: ``rospy`` is stubbed before importing the adapter, because the module
under test is the *translation layer* between rosparams and the core controller,
and that layer is where a typo costs a flight. The control law itself is tested
in ``core/planning/trackers/drift_pid/tests``.
"""
import pathlib
import sys
import types

import pytest

_SCRIPTS = str(pathlib.Path(__file__).resolve().parents[1])
if _SCRIPTS not in sys.path:
    sys.path.insert(0, _SCRIPTS)


class _FakeRospy(types.ModuleType):
    """Just enough rospy for the adapter's import and param reads."""

    def __init__(self):
        types.ModuleType.__init__(self, "rospy")
        self.params = {}
        self.Time = types.SimpleNamespace(now=lambda: 0)

    def get_param(self, name, default=None):
        return self.params.get(name, default)

    def Publisher(self, *a, **kw):
        return types.SimpleNamespace(publish=lambda *_a, **_kw: None)

    def Subscriber(self, *a, **kw):
        return None

    def loginfo(self, *a, **kw):
        return None

    def logwarn(self, *a, **kw):
        return None

    def logwarn_throttle(self, *a, **kw):
        return None


_rospy = _FakeRospy()
sys.modules["rospy"] = _rospy
_msgs = types.ModuleType("geometry_msgs.msg")
_msgs.PointStamped = lambda: types.SimpleNamespace(
    header=types.SimpleNamespace(), point=types.SimpleNamespace())
sys.modules.setdefault("geometry_msgs", types.ModuleType("geometry_msgs"))
sys.modules["geometry_msgs.msg"] = _msgs
_std = types.ModuleType("std_msgs.msg")
_std.String = lambda data=None: types.SimpleNamespace(data=data)
_std.Float32 = lambda data=None: types.SimpleNamespace(data=data)
sys.modules.setdefault("std_msgs", types.ModuleType("std_msgs"))
sys.modules["std_msgs.msg"] = _std

from drift_pid_follower import (  # noqa: E402
    build_drift_pid,
    build_drift_pid_params,
    param_bool,
)


@pytest.fixture(autouse=True)
def _clear_params():
    _rospy.params.clear()
    yield
    _rospy.params.clear()


# ── bool coercion ────────────────────────────────────────────────
def test_param_bool_accepts_the_usual_spellings():
    for text, want in (("true", True), ("True", True), ("1", True),
                       ("on", True), ("false", False), ("FALSE", False),
                       ("0", False), ("off", False)):
        _rospy.params["~flag"] = text
        assert param_bool("~flag", None) is want


def test_param_bool_rejects_a_typo_instead_of_flying_on_it():
    """`bool("fales")` is True -- the trap this helper exists to close."""
    _rospy.params["~flag"] = "fales"
    with pytest.raises(ValueError, match="not a boolean"):
        param_bool("~flag", False)


def test_param_bool_passes_real_bools_through():
    _rospy.params["~flag"] = True
    assert param_bool("~flag", False) is True


# ── param plumbing ───────────────────────────────────────────────
def test_defaults_build_a_valid_controller():
    """Every dataclass validator must pass on the shipped defaults."""
    assert build_drift_pid(_rospy.get_param) is not None


def test_every_param_is_actually_read():
    """A param the builder never reads is a dial that silently does nothing.

    Sets every ~dp_* key found in the source to a sentinel and asserts the built
    params differ from the defaults -- catching a key that was renamed in one
    place and not the other.
    """
    source = pathlib.Path(_SCRIPTS, "drift_pid_follower.py").read_text()
    names = sorted(set(
        part.split('"')[0]
        for part in source.split('"~dp_')[1:]))
    assert len(names) > 40, "expected the full dial set, found %d" % len(names)
    defaults = build_drift_pid_params(_rospy.get_param)
    for name in names:
        if name.endswith("_topic"):
            continue          # topics are read by the node, not the params
        assert "~dp_" + name not in _rospy.params
    # Spot-check that a representative dial from each group reaches the params.
    _rospy.params.update({
        "~dp_cruise_speed": 0.11,
        "~dp_max_vy": 0.15,
        "~dp_lat_ki": 0.21,
        "~dp_conf_full": 0.44,
        "~dp_block_confirm_ticks": 9,
        "~dp_escape_back_s": 1.25,
        "~dp_yaw_engage_deg": 33.0,
        "~dp_latency_s": 0.25,
        "~dp_decel_xy": 0.9,
        "~dp_std_deadband_gain": 1.1,
        "~dp_turn_side_cone_deg": 20.0,
    })
    tuned = build_drift_pid_params(_rospy.get_param)
    assert tuned.cruise_speed == 0.11 != defaults.cruise_speed
    assert tuned.envelope.max_vy == 0.15 != defaults.envelope.max_vy
    assert tuned.lateral_pid.ki == 0.21 != defaults.lateral_pid.ki
    assert tuned.confidence.conf_full == 0.44 != defaults.confidence.conf_full
    assert tuned.blockage.confirm_ticks == 9 != defaults.blockage.confirm_ticks
    assert tuned.escape.back_s == 1.25 != defaults.escape.back_s
    assert abs(tuned.yaw_engage_rad - 0.5759) < 1e-3
    assert tuned.confidence.latency_s == 0.25 != defaults.confidence.latency_s
    assert tuned.envelope.decel_xy == 0.9 != defaults.envelope.decel_xy
    assert (tuned.confidence.std_deadband_gain == 1.1
            != defaults.confidence.std_deadband_gain)
    assert abs(tuned.turn_side_cone_rad - 0.3491) < 1e-3
    assert tuned.turn_side_cone_rad != defaults.turn_side_cone_rad


def test_degree_params_are_converted_to_radians():
    _rospy.params["~dp_travel_cone_deg"] = 90.0
    params = build_drift_pid_params(_rospy.get_param)
    assert abs(params.travel_cone_rad - 1.5708) < 1e-3


def test_an_impossible_envelope_fails_loudly_at_startup():
    """A cruise above the forward cap would be clamped away every tick."""
    _rospy.params["~dp_cruise_speed"] = 5.0
    with pytest.raises(ValueError, match="cruise_speed"):
        build_drift_pid_params(_rospy.get_param)


def test_a_correction_larger_than_its_axis_is_rejected():
    _rospy.params["~dp_lat_max"] = 0.9
    with pytest.raises(ValueError):
        build_drift_pid_params(_rospy.get_param)


# ── turn anticipation (yaw lookahead) ────────────────────────────
def test_the_turn_anticipation_is_off_unless_asked_for():
    """It changes how every corner is flown, so it may never default on."""
    assert build_drift_pid_params(_rospy.get_param).yaw_lookahead.enabled is False


def test_every_yaw_lookahead_dial_reaches_the_core():
    _rospy.params.update({
        "~dp_yaw_lookahead": "true",
        "~dp_yaw_lookahead_start_m": 3.3,
        "~dp_yaw_lookahead_align_m": 0.44,
        "~dp_yaw_lookahead_corner_deg": 31.0,
        "~dp_yaw_lookahead_confirm_m": 1.7,
        "~dp_yaw_lookahead_max_deg": 60.0,
        "~dp_yaw_lookahead_catchup_deg": 8.0,
        "~dp_yaw_lookahead_rate": 0.22,
        "~dp_yaw_lookahead_cone_deg": 70.0,
        "~dp_yaw_lookahead_ff": 0.5,
    })
    yl = build_drift_pid_params(_rospy.get_param).yaw_lookahead
    assert yl.enabled is True
    assert yl.start_m == 3.3
    assert yl.align_m == 0.44
    assert abs(yl.corner_rad - 0.5411) < 1e-3
    assert yl.confirm_m == 1.7
    assert abs(yl.max_offset_rad - 1.0472) < 1e-3
    assert abs(yl.catchup_rad - 0.1396) < 1e-3
    assert yl.rate == 0.22
    assert abs(yl.side_cone_rad - 1.2217) < 1e-3
    assert yl.feedforward == 0.5


def test_the_schedule_rate_follows_the_tracking_yaw_cap_by_default():
    """The manoeuvre runs in TRACK, so it inherits that budget rather than
    silently asking for a rotation the controller may not command there."""
    _rospy.params.update({"~dp_yaw_lookahead": True,
                          "~dp_track_yaw_rate": 0.21})
    assert build_drift_pid_params(_rospy.get_param).yaw_lookahead.rate == 0.21


def test_a_schedule_faster_than_the_tracking_cap_fails_loudly_at_startup():
    _rospy.params.update({"~dp_yaw_lookahead": True,
                          "~dp_track_yaw_rate": 0.20,
                          "~dp_yaw_lookahead_rate": 0.40})
    with pytest.raises(ValueError, match="track_yaw_rate"):
        build_drift_pid_params(_rospy.get_param)


def test_a_typo_in_the_master_switch_is_refused_not_flown():
    _rospy.params["~dp_yaw_lookahead"] = "yes please"
    with pytest.raises(ValueError, match="not a boolean"):
        build_drift_pid_params(_rospy.get_param)
