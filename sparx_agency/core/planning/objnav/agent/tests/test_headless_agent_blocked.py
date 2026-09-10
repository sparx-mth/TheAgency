"""A policy that listens is told of every blocked forward step, before it plans; one that does not still runs.

The hook is optional and feature-detected
(:mod:`~sparx_agency.core.planning.objnav.interfaces.policy`): a policy that
ignores a blocked step keeps sending the same path into the same wall until
the step budget runs out, so the agent must call ``notify_blocked`` exactly
when the converter's ``forward_blocked`` says so -- no more, no less, and
before ``plan()`` so the replan can use it -- and count the calls in the
episode report.

Python 3.8 syntax, numpy.
"""
from __future__ import annotations

import pytest

from sparx_agency.core.planning.objnav.agent.headless_agent import (
    HeadlessObjNavAgent,
)
from sparx_agency.core.planning.objnav.agent.tests.helpers import (
    AHEAD,
    FORWARD,
    ORIGIN,
    ScriptedPolicy,
    TinyMapper,
    episode,
    observation,
    started,
)
from sparx_agency.core.planning.objnav.types.command import NavigationCommand
from sparx_agency.core.planning.objnav.types.pose import AgentPose


class ListeningPolicy(ScriptedPolicy):
    """A scripted policy that records every notification and plan, in order."""

    name = "listening"

    def __init__(self, *commands):
        super().__init__(*commands)
        self.events = []

    def notify_blocked(self, observation):
        self.events.append(("blocked", observation))

    def plan(self, observation):
        self.events.append(("plan", observation))
        return super().plan(observation)


def listening(*commands):
    """A started agent around ``ListeningPolicy(*commands)``, and that policy."""
    policy = ListeningPolicy(*commands)
    agent = HeadlessObjNavAgent(policy, TinyMapper())
    agent.reset(episode())
    return agent, policy


def blocked_steps(policy):
    """The steps of the observations the policy was notified with."""
    return [obs.step for kind, obs in policy.events if kind == "blocked"]


def test_the_policy_is_notified_exactly_when_the_last_forward_step_was_blocked():
    """Blocked twice at the origin, then a step that moves: two calls, at steps 1 and 2."""
    agent, policy = listening(AHEAD)
    assert agent.act(observation(step=0)).action == FORWARD       # steps out
    agent.act(observation(step=1))                                # blocked
    agent.act(observation(step=2))                                # blocked again
    agent.act(observation(step=3, pose=AgentPose(0.25, 0.0)))     # it moved
    assert blocked_steps(policy) == [1, 2]
    info = agent.episode_info()
    assert info["blocked_notifications"] == 2
    assert info["blocked_notifications"] == info["blocked_forward"]


def test_the_hook_is_called_before_plan_with_the_same_observation():
    """The replan must see the wall in the very plan() that follows the notification."""
    agent, policy = listening(AHEAD)
    agent.act(observation(step=0))
    frame = observation(step=1)
    agent.act(frame)
    assert [kind for kind, _ in policy.events] == ["plan", "blocked", "plan"]
    assert policy.events[1][1] is frame and policy.events[2][1] is frame


def test_turns_and_idle_steps_are_never_notified():
    """Only a MOVE_FORWARD that did not move is a wall; turning in place is supposed to stay put."""
    agent, policy = listening(NavigationCommand.follow([(-3.0, 0.0)]),
                              NavigationCommand.hold())
    for step in range(4):
        agent.act(observation(step=step, pose=ORIGIN))
    assert blocked_steps(policy) == []
    assert agent.episode_info()["blocked_notifications"] == 0


def test_a_policy_without_the_hook_still_runs_and_is_never_notified():
    """The hook is optional; lacking it must not break the loop or the counts."""
    agent = started(AHEAD)
    agent.act(observation(step=0))
    assert agent.act(observation(step=1)).action == FORWARD
    info = agent.episode_info()
    assert (info["blocked_forward"], info["blocked_notifications"]) == (1, 0)


def test_a_notify_blocked_attribute_that_is_not_callable_counts_as_absent():
    """Feature detection asks for a callable, as it does for episode_info."""
    policy = ScriptedPolicy(AHEAD)
    policy.notify_blocked = None
    agent = HeadlessObjNavAgent(policy, TinyMapper())
    agent.reset(episode())
    agent.act(observation(step=0))
    agent.act(observation(step=1))
    assert agent.episode_info()["blocked_notifications"] == 0


def test_a_hook_that_returns_something_is_refused_and_still_counted():
    """A command returned from the hook would be silently ignored; the agent follows plan() only."""
    class Answering(ListeningPolicy):
        def notify_blocked(self, observation):
            return NavigationCommand.follow([(0.0, 3.0)])

    policy = Answering(AHEAD)
    agent = HeadlessObjNavAgent(policy, TinyMapper())
    agent.reset(episode())
    agent.act(observation(step=0))
    with pytest.raises(TypeError, match="notify_blocked"):
        agent.act(observation(step=1))
    assert agent.episode_info()["blocked_notifications"] == 1


def test_a_new_episode_starts_the_notification_count_over():
    """A count carried over would credit this episode with another's walls."""
    agent, policy = listening(AHEAD)
    agent.act(observation(step=0))
    agent.act(observation(step=1))
    assert agent.episode_info()["blocked_notifications"] == 1
    agent.reset(episode(episode_id="ep1"))
    assert agent.episode_info()["blocked_notifications"] == 0
    agent.act(observation(step=0))
    assert blocked_steps(policy) == [1]
