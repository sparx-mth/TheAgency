"""Drive a search policy through a benchmark, one discrete action per observation.

:class:`HeadlessObjNavAgent` is the :class:`ObjNavAgent` every method is
benchmarked through. It translates the episode's dataset category with a
:class:`LabelMapper`, asks the :class:`SearchPolicy` where to go, and has a
:class:`DiscreteActionConverter` turn the answer into the benchmark's action --
no ROS, no simulator, no clock, so one method is scored on Habitat and
AI2-THOR by the same code. The quiet failures it is built against:

* **A search for the wrong object, or a frame from somewhere else.** Refused
  at the boundary, before the policy sees anything
  (:mod:`~sparx_agency.core.planning.objnav.agent.episode_checks`).
* **A STOP nobody asked for.** When the converter has nothing to do (the path
  is walked, or the policy asked for nothing) the step is spent on the idle
  action, never STOP: ending the episode is the policy's decision alone (a
  policy that wants to stop once its command is done says so with
  ``stop_on_arrival``). The idle action is a turn, which every action set
  has, so it is never refused mid-run.
* **A wall the policy never hears about.** When the last MOVE_FORWARD did not
  move the agent -- the converter's ``forward_blocked``, the one definition --
  the policy's optional ``notify_blocked(observation)`` is called before
  ``plan()``, so it can mark what its map does not show and replan. A policy
  without it still runs; :meth:`HeadlessObjNavAgent.episode_info` counts the
  calls.
* **State that outlives its episode.** Each episode gets a new converter built
  from its own action spec, so a path cursor, a blocked flag or another
  benchmark's step geometry never carries over.

What the agent records per step and per episode is built in
:mod:`~sparx_agency.core.planning.objnav.agent.episode_log`.

Python 3.8 syntax; numpy arrives only through the observation type.
"""
from __future__ import annotations

from typing import Any, Dict, Optional

from sparx_agency.core.planning.objnav.action_converter.converter import (
    DiscreteActionConverter,
)
from sparx_agency.core.planning.objnav.agent.episode_checks import (
    check_observation,
    map_target,
)
from sparx_agency.core.planning.objnav.agent.episode_log import (
    EpisodeLog,
    policy_report,
    step_info,
)
from sparx_agency.core.planning.objnav.agent.params import HeadlessAgentParams
from sparx_agency.core.planning.objnav.errors import (
    ObjNavError,
    ObjNavInternalError,
)
from sparx_agency.core.planning.objnav.interfaces.agent import ObjNavAgent
from sparx_agency.core.planning.objnav.interfaces.label_mapper import LabelMapper
from sparx_agency.core.planning.objnav.interfaces.policy import SearchPolicy
from sparx_agency.core.planning.objnav.types.command import NavigationCommand
from sparx_agency.core.planning.objnav.types.decision import AgentDecision
from sparx_agency.core.planning.objnav.types.episode import ObjNavEpisode
from sparx_agency.core.planning.objnav.types.observation import ObjNavObservation
from sparx_agency.core.planning.objnav.types.target import TargetLabels


class HeadlessObjNavAgent(ObjNavAgent):
    """A search policy, a label mapper and an action converter behind the step API.

    Stateful per episode -- the episode, its target, its converter, the
    previous observation's step and the :meth:`episode_info` counters -- and
    :meth:`reset` starts all of it over.

    Args:
        policy: Decides where to go; any :class:`SearchPolicy`.
        label_mapper: Translates the benchmark's dataset categories.
        params: Tuning; the defaults when None.
        name: The name results are logged under; ``"headless/<policy name>"``
            when None.

    Raises:
        TypeError: If ``policy``, ``label_mapper`` or ``params`` is of the
            wrong kind.
        ObjNavError: If ``name`` is not a non-blank string, or is None and the
            policy has no non-blank string name to derive it from.
    """

    def __init__(self, policy, label_mapper: LabelMapper,
                 params: Optional[HeadlessAgentParams] = None,
                 name: Optional[str] = None) -> None:
        if not isinstance(policy, SearchPolicy):
            raise TypeError(
                "policy must be a SearchPolicy -- an object with a name, "
                "reset(episode, target) and plan(observation) -- got %r"
                % (policy,))
        if not isinstance(label_mapper, LabelMapper):
            raise TypeError("label_mapper must be a LabelMapper, got %r"
                            % (label_mapper,))
        if params is None:
            params = HeadlessAgentParams()
        if not isinstance(params, HeadlessAgentParams):
            raise TypeError("params must be HeadlessAgentParams, got %r"
                            % (params,))
        if name is None:
            name = "headless/%s" % (_policy_name(policy),)
        if not isinstance(name, str) or not name.strip():
            raise ObjNavError(
                "the agent name must be a non-blank string -- every result "
                "row is logged under it -- got %r" % (name,))
        self._policy = policy
        self._label_mapper = label_mapper
        self._params = params
        self._name = name
        self._forget_episode()

    @property
    def name(self) -> str:
        """The name every result row is logged under."""
        return self._name

    @property
    def episode(self) -> Optional[ObjNavEpisode]:
        """The running episode, or None before a successful reset."""
        return self._episode

    @property
    def target(self) -> Optional[TargetLabels]:
        """The running episode's target, or None before a successful reset."""
        return self._target

    def reset(self, episode: ObjNavEpisode) -> None:
        """Start ``episode``: map its target, build its converter, reset the policy.

        A reset that raises leaves the agent with no episode, so :meth:`act`
        refuses to run rather than continue the previous one.

        Args:
            episode: The episode about to run.

        Raises:
            TypeError: If ``episode`` is not an :class:`ObjNavEpisode`, or the
                mapper answers with something other than :class:`TargetLabels`.
            UnknownCategoryError: If the mapper does not cover the episode's
                category; the message names the mapper and its categories.
            ObjNavError: If the mapper answers with another category's target,
                or the converter tuning does not fit its action geometry.
        """
        self._forget_episode()
        if not isinstance(episode, ObjNavEpisode):
            raise TypeError("reset() needs an ObjNavEpisode, got %r"
                            % (episode,))
        target = map_target(self._label_mapper, episode)
        converter = DiscreteActionConverter(episode.action_spec,
                                            self._params.converter)
        self._policy.reset(episode, target)
        self._episode = episode
        self._target = target
        self._converter = converter

    def act(self, observation: ObjNavObservation) -> AgentDecision:
        """Choose the next action.

        When the last MOVE_FORWARD did not move the agent, the policy's
        ``notify_blocked(observation)`` is called first, if it has one.

        Args:
            observation: The running episode's next observation.

        Returns:
            The converter's action, or the idle action when it has nothing to
            do. ``info`` says why: the converter's ``status`` and path
            geometry, whether the step was ``idle``, and the command's own
            ``info`` under ``"policy"``. It is the caller's to keep; the
            harness stores only :meth:`episode_info`.

        Raises:
            ObjNavError: If no episode has been started.
            ObservationError: If ``observation`` is not an
                :class:`ObjNavObservation`, is for another target or camera,
                or does not come after the previous one.
            TypeError: If the policy's plan is not a
                :class:`NavigationCommand`, or its ``notify_blocked`` returns
                anything but None.
            CommandError: If the command asks for a camera pitch on a
                benchmark without LOOK actions.
            ObjNavInternalError: If the chosen action is not in the action set.
        """
        check_observation(observation, self._episode, self._last_step)
        # Recorded before the policy sees the frame: once it has (and may have
        # mapped it), handing it over again is a replay even if it raises.
        self._last_step = observation.step
        if self._converter.forward_blocked(observation.pose):
            self._notify_blocked(observation)
        command = self._policy.plan(observation)
        if not isinstance(command, NavigationCommand):
            raise TypeError(
                "policy %r returned %r from plan(); a SearchPolicy returns a "
                "NavigationCommand (NavigationCommand.hold() for no opinion)"
                % (getattr(self._policy, "name", self._policy), command))
        result = self._converter.step(observation.pose, command)
        action = self._params.idle_action if result.idle else result.action
        if not self._episode.action_spec.allows(action):
            raise ObjNavInternalError(
                "the agent chose %s (converter status %r), which episode %r's "
                "action spec does not allow"
                % (action.name, result.status, self._episode.episode_id))
        self._log.count(result)
        return AgentDecision(action, step_info(result, command))

    def episode_info(self) -> Dict[str, Any]:
        """What the agent did this episode, JSON-serialisable.

        Returns:
            ``agent`` (the name), ``steps`` (decisions returned),
            ``idle_steps``, ``blocked_forward`` (decisions made after a
            MOVE_FORWARD that did not move the agent),
            ``blocked_notifications`` (calls of the policy's
            ``notify_blocked``; always 0 for a policy without one),
            ``statuses`` (decisions per converter status, every status present
            so rows line up) and ``policy`` (the policy's own
            ``episode_info()`` when it has a callable one, else ``{}``).

        Raises:
            ObjNavError: If no episode has been started.
            TypeError: If the policy's ``episode_info()`` is not a mapping.
        """
        if self._episode is None:
            raise ObjNavError("episode_info() called before reset(); there is "
                              "no episode to report on")
        return self._log.report(self._name, policy_report(self._policy))

    def _forget_episode(self) -> None:
        """Drop the episode, its target, its converter and every counter."""
        self._episode = None
        self._target = None
        self._converter = None
        self._last_step = None
        self._log = EpisodeLog()

    def _notify_blocked(self, observation: ObjNavObservation) -> None:
        """Tell the policy its last MOVE_FORWARD did not move, if it listens.

        Counted before the call, so a hook that raises is still counted.
        """
        hook = getattr(self._policy, "notify_blocked", None)
        if not callable(hook):
            return
        self._log.count_notification()
        answer = hook(observation)
        if answer is not None:
            raise TypeError(
                "policy %r returned %r from notify_blocked(); the hook returns "
                "None -- replan inside it and return the new path from the "
                "plan() call that follows"
                % (getattr(self._policy, "name", self._policy), answer))

    def __repr__(self) -> str:
        return ("HeadlessObjNavAgent(name=%r, label_mapper=%r, idle_action=%s)"
                % (self._name, self._label_mapper.name,
                   self._params.idle_action.name))


def _policy_name(policy) -> str:
    """The policy's name, to derive the agent's from, or raise."""
    name = getattr(policy, "name", None)
    if not isinstance(name, str) or not name.strip():
        raise ObjNavError(
            "policy %r has no usable name (got %r); give it a non-blank name "
            "or pass name= to the agent -- results are logged under it"
            % (policy, name))
    return name
