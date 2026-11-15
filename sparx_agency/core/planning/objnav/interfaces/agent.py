"""The :class:`ObjNavAgent` contract: one discrete action per observation.

This is the whole of what a benchmark harness knows about a method. No ROS, no
simulator, no clock: the runner hands the agent an episode, then an
observation at a time, and executes whatever action comes back::

    agent.reset(episode)
    while not env.episode_over:
        decision = agent.act(observation)
        observation = env.step(decision.action)

An ABC rather than a Protocol, following the repo's rule: swappable backends
selected by name are ``abc.ABC`` (``NavigationPolicy``, ``DetectionModel``);
pipeline stages are ``Protocol``. Agents are compared against each other by
name, so they are the first kind.

Python 3.8 syntax, standard library only.
"""
from __future__ import annotations

import abc
from typing import Any, Dict


class ObjNavAgent(abc.ABC):
    """An ObjectNav agent driven one step at a time.

    Attributes:
        name: The identifier every result row is logged under. Two agents
            compared in one table must not share it.
    """

    name = ""

    @abc.abstractmethod
    def reset(self, episode):
        """Start an episode; forget everything about the previous one.

        Args:
            episode: The :class:`~sparx_agency.core.planning.objnav.types.ObjNavEpisode`
                about to run. Public information only.

        Raises:
            UnknownCategoryError: The agent cannot search for this category.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def act(self, observation):
        """Choose the next action.

        Args:
            observation: The current
                :class:`~sparx_agency.core.planning.objnav.types.ObjNavObservation`.

        Returns:
            An :class:`~sparx_agency.core.planning.objnav.types.AgentDecision`
            whose action the benchmark accepts.

        Raises:
            ObservationError: The observation does not belong to the current
                episode, or arrived out of order.
        """
        raise NotImplementedError

    def episode_info(self) -> Dict[str, Any]:
        """Per-episode diagnostics to log beside the score, JSON-serialisable.

        Called once, after the episode ends. The default has nothing to say.
        """
        return {}
