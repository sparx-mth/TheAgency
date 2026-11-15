"""The headless agent is built from the right parts and starts an episode only when it can run it.

Each test builds or resets a real agent -- a real converter inside -- with the
scripted policy and the two-category mapper of
:mod:`~sparx_agency.core.planning.objnav.agent.tests.helpers`, and checks one
promise the harness relies on: results are logged under a name that says which
method ran, the policy is told the mapped target, a category or a mapper that
could not run the episode is refused at reset, before the first step, and an
idle action that is not a turn is refused before there is any episode.

Python 3.8 syntax, numpy.
"""
from __future__ import annotations

import pytest

from sparx_agency.core.planning.objnav.agent.headless_agent import (
    HeadlessObjNavAgent,
)
from sparx_agency.core.planning.objnav.agent.params import HeadlessAgentParams
from sparx_agency.core.planning.objnav.agent.tests.helpers import (
    AHEAD,
    HABITAT,
    LEFT,
    NO_TILT,
    RIGHT,
    ScriptedPolicy,
    TinyMapper,
    agent_with,
    episode,
    observation,
    started,
)
from sparx_agency.core.planning.objnav.errors import (
    ObjNavError,
    UnknownCategoryError,
)
from sparx_agency.core.planning.objnav.types.actions import DiscreteAction


# -- construction ---------------------------------------------------------------

def test_the_name_defaults_to_headless_slash_the_policy_name():
    """Results are logged under the agent's name, so it must say which method ran."""
    assert agent_with().name == "headless/scripted"
    assert agent_with(name="ours").name == "ours"


def test_arguments_of_the_wrong_kind_are_refused():
    """A mapper passed as the policy would otherwise fail on the first plan() call."""
    with pytest.raises(TypeError):
        HeadlessObjNavAgent(TinyMapper(), TinyMapper())
    with pytest.raises(TypeError):
        HeadlessObjNavAgent(ScriptedPolicy(), {"tv_monitor": "tv"})
    with pytest.raises(TypeError):
        HeadlessObjNavAgent(ScriptedPolicy(), TinyMapper(),
                            params={"idle_action": LEFT})


def test_a_blank_agent_or_policy_name_is_refused():
    """Two unnamed agents in one results table could not be told apart."""
    with pytest.raises(ObjNavError):
        agent_with(name="  ")
    unnamed = ScriptedPolicy()
    unnamed.name = ""
    with pytest.raises(ObjNavError):
        HeadlessObjNavAgent(unnamed, TinyMapper())
    assert HeadlessObjNavAgent(unnamed, TinyMapper(), name="ours").name == "ours"


# -- reset ----------------------------------------------------------------------

def test_reset_hands_the_policy_the_mapped_target():
    """The policy searches for what the mapper says, in our vocabulary, not the dataset's."""
    policy = ScriptedPolicy()
    agent = HeadlessObjNavAgent(policy, TinyMapper())
    run = episode()
    agent.reset(run)
    assert policy.resets == [(run, TinyMapper.TARGETS["tv_monitor"])]
    assert agent.episode is run
    assert agent.target is TinyMapper.TARGETS["tv_monitor"]
    with pytest.raises(AttributeError):
        agent.target = TinyMapper.TARGETS["chair"]


def test_an_uncovered_category_is_refused_at_reset_naming_the_mapper_and_its_categories():
    """A missing table row must fail before the first step, not as 500 wasted ones."""
    policy = ScriptedPolicy()
    agent = HeadlessObjNavAgent(policy, TinyMapper())
    with pytest.raises(UnknownCategoryError) as caught:
        agent.reset(episode(target_category="sofa"))
    message = str(caught.value)
    assert "'tiny'" in message and "'sofa'" in message
    assert "'chair'" in message and "'tv_monitor'" in message
    assert policy.resets == []


def test_a_mapper_that_answers_with_another_target_is_refused():
    """The policy would search for the wrong object, and every episode would just fail."""
    class Crossed(TinyMapper):
        def target_labels(self, category):
            return self.TARGETS["chair"]

    class Untyped(TinyMapper):
        def target_labels(self, category):
            return {"query": "tv"}

    with pytest.raises(ObjNavError):
        HeadlessObjNavAgent(ScriptedPolicy(), Crossed()).reset(episode())
    with pytest.raises(TypeError):
        HeadlessObjNavAgent(ScriptedPolicy(), Untyped()).reset(episode())


def test_an_idle_action_that_is_not_a_turn_is_refused_before_any_episode():
    """Found when the params are built, not at a reset or some episode's first idle step; a turn runs on every spec."""
    with pytest.raises(ObjNavError, match="LOOK_UP"):
        HeadlessAgentParams(idle_action=DiscreteAction.LOOK_UP)
    agent = agent_with(params=HeadlessAgentParams(idle_action=RIGHT))
    for spec in (HABITAT, NO_TILT):
        agent.reset(episode(action_spec=spec))
        assert agent.episode.action_spec == spec


def test_reset_refuses_something_that_is_not_an_episode():
    """An episode id string instead of the episode is a harness bug, named as one."""
    with pytest.raises(TypeError):
        agent_with().reset("ep0")


def test_a_failed_reset_leaves_no_episode_to_act_in():
    """Acting on after a refused reset would silently continue the previous episode."""
    agent = started(AHEAD)
    agent.act(observation(step=0))
    with pytest.raises(UnknownCategoryError):
        agent.reset(episode(episode_id="ep1", target_category="sofa"))
    assert agent.episode is None and agent.target is None
    with pytest.raises(ObjNavError):
        agent.act(observation(step=1))
