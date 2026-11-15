"""The four contracts demand exactly what they document, and no more.

Every benchmark adapter, agent and label table subclasses these, in branches
this package never sees. So the abstract surface is pinned here: an abstract
method added quietly would break every one of them at instantiation, and one
removed quietly would let an adapter ship without it and fail mid-run. The
``SearchPolicy`` protocol is structural, and a structural check that accepts
an object without ``plan`` would let the agent call a method that is not
there on the first step of a benchmark.

Python 3.8 syntax, standard library only.
"""
from __future__ import annotations

import json

import pytest

from sparx_agency.core.planning.objnav.interfaces.agent import ObjNavAgent
from sparx_agency.core.planning.objnav.interfaces.env import ObjNavEnv
from sparx_agency.core.planning.objnav.interfaces.label_mapper import LabelMapper
from sparx_agency.core.planning.objnav.interfaces.policy import SearchPolicy


def _env_methods():
    """A complete ObjNavEnv implementation, as a class namespace."""
    return {
        "name": "minimal",
        "episode_ids": lambda self: ("e0",),
        "reset": lambda self, episode_id: (None, None),
        "step": lambda self, action: None,
        "episode_over": property(lambda self: True),
        "measure": lambda self: None,
    }


def _agent_methods():
    """A complete ObjNavAgent implementation, as a class namespace."""
    return {
        "name": "minimal",
        "reset": lambda self, episode: None,
        "act": lambda self, observation: None,
    }


def _mapper_methods():
    """A complete LabelMapper implementation, as a class namespace."""
    return {
        "name": "minimal",
        "categories": lambda self: ("chair", "tv_monitor"),
        "target_labels": lambda self, category: None,
        "vocabulary": lambda self: ("chair", "tv"),
    }


#: Each ABC, the namespace that completes it, and its documented abstract surface.
CONTRACTS = (
    (ObjNavEnv, _env_methods,
     {"episode_ids", "reset", "step", "episode_over", "measure"}),
    (ObjNavAgent, _agent_methods, {"reset", "act"}),
    (LabelMapper, _mapper_methods, {"categories", "target_labels", "vocabulary"}),
)

CONTRACT_IDS = [base.__name__ for base, _, _ in CONTRACTS]

#: Every (ABC, abstract method) pair, for the leave-one-out test.
LEAVE_ONE_OUT = [(base, methods, name) for base, methods, abstract in CONTRACTS
                 for name in sorted(abstract)]


def _subclass(base, namespace):
    return type("Minimal" + base.__name__, (base,), namespace)


class ScriptedPolicy:
    """A search policy that inherits nothing: the protocol is structural."""

    name = "scripted"

    def reset(self, episode, target) -> None:
        self.target = target

    def plan(self, observation):
        return None


class PolicyWithoutPlan:
    """Has a name and reset, but cannot answer the one question a step asks."""

    name = "no_plan"

    def reset(self, episode, target) -> None:
        pass


class PolicyWithoutName:
    """Plans, but has no name to log results under."""

    def reset(self, episode, target) -> None:
        pass

    def plan(self, observation):
        return None


# -- the abstract base classes -------------------------------------------------

@pytest.mark.parametrize("base,methods,abstract", CONTRACTS, ids=CONTRACT_IDS)
def test_the_abstract_surface_is_exactly_the_documented_one(base, methods, abstract):
    """A new abstract method would break every adapter in every branch at once."""
    assert set(base.__abstractmethods__) == abstract


@pytest.mark.parametrize("base,methods,abstract", CONTRACTS, ids=CONTRACT_IDS)
def test_an_abstract_base_cannot_be_instantiated(base, methods, abstract):
    """The contract is a promise to implement, never a usable default."""
    with pytest.raises(TypeError):
        base()


@pytest.mark.parametrize("base,methods,missing", LEAVE_ONE_OUT,
                         ids=["%s-%s" % (b.__name__, m) for b, _, m in LEAVE_ONE_OUT])
def test_a_subclass_missing_one_abstract_method_cannot_be_instantiated(
        base, methods, missing):
    """An adapter without ``measure`` must fail when built, not at the end of an episode."""
    namespace = methods()
    del namespace[missing]
    # Word boundaries: a bare "act" would also match the word "abstract".
    with pytest.raises(TypeError, match=r"\b%s\b" % missing):
        _subclass(base, namespace)()


@pytest.mark.parametrize("base,methods,abstract", CONTRACTS, ids=CONTRACT_IDS)
def test_a_minimal_subclass_can_be_instantiated(base, methods, abstract):
    """The documented surface is all an implementation needs."""
    instance = _subclass(base, methods())()
    assert isinstance(instance, base)
    assert instance.name == "minimal"


@pytest.mark.parametrize("base", [ObjNavEnv, ObjNavAgent, LabelMapper])
def test_the_name_defaults_to_empty(base):
    """An unnamed implementation is visibly unnamed, not silently called something."""
    assert base.name == ""


def test_env_close_is_optional_and_holds_nothing():
    """A simulator without resources need not implement ``close``."""
    env = _subclass(ObjNavEnv, _env_methods())()
    assert env.close() is None
    assert env.episode_over is True


def test_agent_episode_info_defaults_to_an_empty_json_dict():
    """The harness logs it beside every score, so the default must serialise."""
    agent = _subclass(ObjNavAgent, _agent_methods())()
    info = agent.episode_info()
    assert info == {}
    assert json.dumps(info, allow_nan=False) == "{}"


def test_agent_episode_info_is_not_shared_between_calls():
    """A shared default dict would carry one episode's notes into the next."""
    agent = _subclass(ObjNavAgent, _agent_methods())()
    agent.episode_info()["leak"] = 1
    assert agent.episode_info() == {}


def test_label_mapper_covers_exactly_its_categories():
    """``covers`` is the reset-time check that stops a search for the wrong object."""
    mapper = _subclass(LabelMapper, _mapper_methods())()
    assert mapper.covers("chair")
    assert mapper.covers("tv_monitor")
    assert not mapper.covers("sofa")


def test_label_mapper_covers_is_verbatim():
    """A mis-cased category is an adapter bug, so it must not be normalised away."""
    mapper = _subclass(LabelMapper, _mapper_methods())()
    assert not mapper.covers("TV_Monitor")
    assert not mapper.covers("tv monitor")


# -- the search policy protocol --------------------------------------------------

def test_an_object_with_name_reset_and_plan_is_a_search_policy():
    """Any method plugs in without inheriting from this package."""
    assert isinstance(ScriptedPolicy(), SearchPolicy)


def test_an_object_without_plan_is_not_a_search_policy():
    """The agent calls ``plan`` every step; its absence must show at construction."""
    assert not isinstance(PolicyWithoutPlan(), SearchPolicy)


def test_an_object_without_a_name_is_not_a_search_policy():
    """Results are logged under the policy's name; an unnamed one cannot be compared."""
    assert not isinstance(PolicyWithoutName(), SearchPolicy)


def test_a_name_set_per_instance_satisfies_the_protocol():
    """A policy named from its configuration sets ``name`` in ``__init__``."""
    policy = PolicyWithoutName()
    policy.name = "configured"
    assert isinstance(policy, SearchPolicy)
