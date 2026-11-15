"""``HeadlessAgentParams`` defaults to a safe idle action and refuses one that would end the episode or is not a turn.

An idle STOP would end the episode on the converter's say-so, not the
policy's; an idle MOVE_FORWARD into a wall is never reported blocked, and an
idle LOOK at the pitch limit is refused by AI2-THOR; and a bare integer is
TURN_LEFT only in Habitat's numbering. So each documented refusal is exercised
here, with the error class the docstring names.

Python 3.8 syntax, standard library only.
"""
from __future__ import annotations

import pytest

from sparx_agency.core.planning.objnav.action_converter.params import (
    ActionConverterParams,
)
from sparx_agency.core.planning.objnav.agent.params import HeadlessAgentParams
from sparx_agency.core.planning.objnav.errors import ObjNavError
from sparx_agency.core.planning.objnav.types.actions import DiscreteAction


def test_agent_params_default_to_turning_left_when_idle():
    """Turning in place always brings new views and never hits anything."""
    params = HeadlessAgentParams()
    assert params.converter == ActionConverterParams()
    assert params.idle_action == DiscreteAction.TURN_LEFT


def test_agent_params_refuse_stop_as_the_idle_action():
    """An idle STOP would end the episode on the converter's say-so, not the policy's."""
    with pytest.raises(ObjNavError, match="cannot be STOP"):
        HeadlessAgentParams(idle_action=DiscreteAction.STOP)


@pytest.mark.parametrize("idle", [DiscreteAction.MOVE_FORWARD,
                                  DiscreteAction.LOOK_UP,
                                  DiscreteAction.LOOK_DOWN])
def test_agent_params_refuse_an_idle_action_that_is_not_a_turn(idle):
    """An idle MOVE_FORWARD into a wall is never reported blocked; an idle LOOK at the limit is refused."""
    with pytest.raises(ObjNavError, match="TURN_LEFT or TURN_RIGHT"):
        HeadlessAgentParams(idle_action=idle)


def test_agent_params_accept_either_turn_as_the_idle_action():
    """Both turns are in every action set, and turning in place never hits anything."""
    for idle in (DiscreteAction.TURN_LEFT, DiscreteAction.TURN_RIGHT):
        assert HeadlessAgentParams(idle_action=idle).idle_action == idle


@pytest.mark.parametrize("idle", [2, "TURN_LEFT", None])
def test_agent_params_refuse_an_idle_action_that_is_not_an_action(idle):
    """A bare ``2`` is TURN_LEFT only in Habitat's numbering."""
    with pytest.raises(ObjNavError, match="idle_action"):
        HeadlessAgentParams(idle_action=idle)


def test_agent_params_refuse_a_converter_of_the_wrong_type():
    """A dict of settings would be silently ignored by attribute access."""
    with pytest.raises(ObjNavError, match="converter"):
        HeadlessAgentParams(converter={"lookahead_m": 0.5})
