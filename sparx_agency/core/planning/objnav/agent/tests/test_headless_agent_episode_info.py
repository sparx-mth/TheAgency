"""The headless agent's per-episode report serialises, adds up, and starts over with every episode.

Each test drives a real agent -- a real converter inside -- with the scripted
policy and the two-category mapper of
:mod:`~sparx_agency.core.planning.objnav.agent.tests.helpers`, and checks one
promise the harness relies on: the report stored beside every score is plain
JSON whose counts add up, carries the policy's own report, and never carries a
count or a blocked flag over from the previous episode.

Python 3.8 syntax, numpy.
"""
from __future__ import annotations

import json
import math

import pytest

from sparx_agency.core.planning.objnav.action_converter.types import STATUSES
from sparx_agency.core.planning.objnav.agent.headless_agent import (
    HeadlessObjNavAgent,
)
from sparx_agency.core.planning.objnav.agent.tests.helpers import (
    AHEAD,
    FORWARD,
    NO_TILT,
    ScriptedPolicy,
    TinyMapper,
    agent_with,
    episode,
    observation,
    started,
)
from sparx_agency.core.planning.objnav.errors import CommandError, ObjNavError
from sparx_agency.core.planning.objnav.types.actions import DiscreteAction
from sparx_agency.core.planning.objnav.types.command import NavigationCommand
from sparx_agency.core.planning.objnav.types.pose import AgentPose


class ReportingPolicy(ScriptedPolicy):
    """A scripted policy that keeps its own per-episode report."""

    name = "reporting"

    def episode_info(self):
        return {"plans": self.planned, "rooms": ["kitchen"]}


# -- episode_info -------------------------------------------------------------------

def test_episode_info_is_json_serialisable_and_counts_statuses():
    """The harness stores it beside every score; it must serialise and add up."""
    agent = started(AHEAD, AHEAD, AHEAD, NavigationCommand.hold(),
                    NavigationCommand.stop_here())
    agent.act(observation(step=0))                           # forward
    agent.act(observation(step=1))                           # forward, blocked
    agent.act(observation(step=2, pose=AgentPose(3.0, 0.0)))  # arrived
    agent.act(observation(step=3, pose=AgentPose(3.0, 0.0)))  # empty
    agent.act(observation(step=4, pose=AgentPose(3.0, 0.0)))  # stop
    info = agent.episode_info()
    assert json.loads(json.dumps(info, allow_nan=False)) == info
    assert info["agent"] == "headless/scripted"
    assert (info["steps"], info["idle_steps"], info["blocked_forward"]) == (
        5, 2, 1)
    assert list(info["statuses"]) == list(STATUSES)
    assert info["statuses"] == dict(
        {status: 0 for status in STATUSES},
        forward=2, arrived=1, empty=1, stop=1)
    assert info["policy"] == {}


def test_episode_info_carries_the_policys_own_report():
    """A method's diagnostics (rooms visited, plans made) belong beside its score."""
    agent = HeadlessObjNavAgent(ReportingPolicy(AHEAD), TinyMapper())
    agent.reset(episode())
    agent.act(observation())
    assert agent.episode_info()["policy"] == {"plans": 1, "rooms": ["kitchen"]}


def test_episode_info_before_reset_is_refused():
    """A report with no episode behind it would be logged against the wrong one."""
    with pytest.raises(ObjNavError):
        agent_with().episode_info()


# -- a new episode ----------------------------------------------------------------

def test_a_new_episode_gets_a_fresh_converter_and_fresh_counters():
    """A stale converter reports a blocked step that never happened in this episode."""
    agent = started(AHEAD)
    agent.act(observation(step=0))
    assert agent.act(observation(step=1)).info["forward_blocked"] is True
    agent.reset(episode(episode_id="ep1"))
    assert agent.episode_info()["steps"] == 0
    first = agent.act(observation(step=0))
    assert first.action == FORWARD
    assert first.info["forward_blocked"] is False
    assert agent.episode_info()["blocked_forward"] == 0


def test_a_new_episode_converts_with_its_own_action_spec():
    """A converter kept from a tilting benchmark would emit LOOKs on one without them."""
    agent = started(NavigationCommand.hold(camera_pitch=math.radians(30.0)))
    assert agent.act(observation()).action == DiscreteAction.LOOK_DOWN
    agent.reset(episode(episode_id="ep1", action_spec=NO_TILT))
    with pytest.raises(CommandError):
        agent.act(observation())
