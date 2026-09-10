"""The action types accept what their docstrings promise and refuse what they forbid.

A spec whose step disagrees with the simulator makes every perfect-execution
prediction wrong, and a bare integer means a different action to each
simulator. Neither crashes anything downstream -- they move SR and SPL -- so
each documented refusal of ``DiscreteAction``, ``DiscreteActionSpec`` and
``AgentDecision`` is exercised here, with the error class the docstring names.

Python 3.8 syntax, standard library only.
"""
from __future__ import annotations

import math
from types import MappingProxyType

import pytest

from sparx_agency.core.planning.objnav.errors import ObjNavError
from sparx_agency.core.planning.objnav.tests.helpers import INF, NAN
from sparx_agency.core.planning.objnav.types.actions import (
    ALL_ACTIONS,
    LOOK_ACTIONS,
    NAVIGATION_ACTIONS,
    DiscreteAction,
    DiscreteActionSpec,
)
from sparx_agency.core.planning.objnav.types.decision import AgentDecision


# -- DiscreteAction and DiscreteActionSpec -------------------------------------

def test_action_values_follow_habitats_objectnav_order():
    """The integers are Habitat's; a reordering would silently remap every action."""
    assert [(a.name, int(a)) for a in DiscreteAction] == [
        ("STOP", 0), ("MOVE_FORWARD", 1), ("TURN_LEFT", 2), ("TURN_RIGHT", 3),
        ("LOOK_UP", 4), ("LOOK_DOWN", 5)]
    assert ALL_ACTIONS == tuple(DiscreteAction)
    assert set(NAVIGATION_ACTIONS).union(LOOK_ACTIONS) == set(ALL_ACTIONS)


def test_the_default_spec_is_the_papers_geometry():
    """0.25 m, 30 deg, 30 deg with tilt: what the target papers evaluate with."""
    spec = DiscreteActionSpec()
    assert (spec.forward_step_m, spec.turn_angle_deg, spec.tilt_angle_deg) == (
        0.25, 30.0, 30.0)
    assert spec.actions == ALL_ACTIONS and spec.has_camera_tilt
    assert spec.turn_angle_rad == pytest.approx(math.pi / 6)
    assert spec.tilt_angle_rad == pytest.approx(math.pi / 6)
    assert spec.min_pitch_rad is None and spec.max_pitch_rad is None


@pytest.mark.parametrize("field", ["forward_step_m", "turn_angle_deg",
                                   "tilt_angle_deg"])
@pytest.mark.parametrize("value", [0.0, -0.25, NAN, INF, True, "0.25"])
def test_a_non_positive_or_non_numeric_step_geometry_is_refused(field, value):
    """A wrong step makes every perfect-execution prediction wrong; ``True`` would pass as 1."""
    with pytest.raises(ObjNavError, match=field):
        DiscreteActionSpec(**{field: value})


def test_a_turn_beyond_half_a_revolution_is_refused():
    """A 181-degree left turn is a 179-degree right turn under another name."""
    with pytest.raises(ObjNavError, match="at most 180"):
        DiscreteActionSpec(turn_angle_deg=181.0)
    assert DiscreteActionSpec(turn_angle_deg=180.0).turn_angle_rad == pytest.approx(
        math.pi)


def test_a_tilt_beyond_a_right_angle_is_refused():
    """No camera pitches past straight down in one step."""
    with pytest.raises(ObjNavError, match="at most 90"):
        DiscreteActionSpec(tilt_angle_deg=91.0)


@pytest.mark.parametrize("actions,message", [
    (NAVIGATION_ACTIONS + (DiscreteAction.LOOK_UP,), "half of the LOOK pair"),
    (NAVIGATION_ACTIONS + (DiscreteAction.LOOK_DOWN,), "half of the LOOK pair"),
    (NAVIGATION_ACTIONS[:3], "lacks TURN_RIGHT"),
    ((DiscreteAction.STOP, DiscreteAction.MOVE_FORWARD), "lacks TURN_LEFT, TURN_RIGHT"),
    (NAVIGATION_ACTIONS + (DiscreteAction.MOVE_FORWARD,), "repeats"),
    ((), "lacks STOP"),
    (NAVIGATION_ACTIONS + (4, 5), "DiscreteAction members"),
], ids=["only-look-up", "only-look-down", "missing-turn", "missing-both-turns",
        "duplicate", "empty", "bare-ints"])
def test_a_malformed_action_set_is_refused(actions, message):
    """Every ObjectNav benchmark has the four navigation actions and both LOOKs or neither."""
    with pytest.raises(ObjNavError, match=message):
        DiscreteActionSpec(actions=actions)


def test_actions_are_stored_as_a_sorted_tuple_whatever_the_input_order():
    """Two specs for one benchmark must compare equal however the adapter listed them."""
    spec = DiscreteActionSpec(actions=list(reversed(ALL_ACTIONS)))
    assert spec.actions == ALL_ACTIONS
    assert spec == DiscreteActionSpec()
    assert hash(spec) == hash(DiscreteActionSpec())


def test_a_spec_without_camera_tilt_allows_no_look_action():
    """RoboTHOR-style benchmarks without tilt must never be sent a LOOK."""
    spec = DiscreteActionSpec(actions=NAVIGATION_ACTIONS)
    assert not spec.has_camera_tilt
    assert not spec.allows(DiscreteAction.LOOK_UP)
    assert not spec.allows(DiscreteAction.LOOK_DOWN)
    assert all(spec.allows(a) for a in NAVIGATION_ACTIONS)


@pytest.mark.parametrize("low,high", [(30.0, -30.0), (10.0, 10.0)])
def test_an_inverted_or_empty_pitch_range_is_refused(low, high):
    """REP-103 puts up at negative pitch; a sign flip in an adapter shows as an inverted range."""
    with pytest.raises(ObjNavError, match="inverted"):
        DiscreteActionSpec(min_pitch_deg=low, max_pitch_deg=high)


@pytest.mark.parametrize("field", ["min_pitch_deg", "max_pitch_deg"])
@pytest.mark.parametrize("value", [91.0, -91.0, NAN, True])
def test_a_pitch_limit_beyond_a_right_angle_is_refused(field, value):
    """A limit past vertical is a unit error (radians typed as degrees, or the reverse)."""
    with pytest.raises(ObjNavError, match=field):
        DiscreteActionSpec(**{field: value})


def test_pitch_limits_convert_to_radians_and_may_be_one_sided():
    """AI2-THOR clamps both ways; a simulator may clamp only one."""
    spec = DiscreteActionSpec(min_pitch_deg=-30.0, max_pitch_deg=30.0)
    assert spec.min_pitch_rad == pytest.approx(-math.pi / 6)
    assert spec.max_pitch_rad == pytest.approx(math.pi / 6)
    one_sided = DiscreteActionSpec(max_pitch_deg=60.0)
    assert one_sided.min_pitch_rad is None
    assert one_sided.max_pitch_rad == pytest.approx(math.pi / 3)


# -- AgentDecision ------------------------------------------------------------------------

def test_a_decision_holds_an_action_and_its_own_info():
    """A shared default dict would merge the diagnostics of different steps."""
    first = AgentDecision(DiscreteAction.MOVE_FORWARD)
    second = AgentDecision(DiscreteAction.STOP)
    first.info["status"] = "forward"
    assert second.info == {}


@pytest.mark.parametrize("action", [1, 0, None, "MOVE_FORWARD"])
def test_a_decision_refuses_a_bare_int_or_anything_but_an_action(action):
    """A bare ``3`` is TURN_RIGHT to Habitat and something else to AI2-THOR."""
    with pytest.raises(ObjNavError, match="DiscreteAction"):
        AgentDecision(action)


@pytest.mark.parametrize("info", [None, [("status", "forward")], "forward"],
                         ids=["none", "pairs", "string"])
def test_a_decision_refuses_info_that_is_not_a_mapping(info):
    """A caller reading the decisions' info would crash on it, far from the agent that built it."""
    with pytest.raises(ObjNavError, match="info"):
        AgentDecision(DiscreteAction.STOP, info)


def test_a_decision_stores_its_info_as_a_dict_copy_of_any_mapping():
    """The agent may reuse its dict next step; a returned decision must not change with it."""
    source = {"status": "forward"}
    decision = AgentDecision(DiscreteAction.MOVE_FORWARD,
                             MappingProxyType(source))
    source["status"] = "turn"
    assert type(decision.info) is dict and decision.info == {"status": "forward"}
