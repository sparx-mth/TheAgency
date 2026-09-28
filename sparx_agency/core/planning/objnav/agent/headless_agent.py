"""One legal discrete action per observation, with optional observed-map safety vetoes."""
from __future__ import annotations
from dataclasses import replace
from typing import Any, Dict, Optional
from sparx_agency.core.planning.objnav.action_converter.converter import DiscreteActionConverter
from sparx_agency.core.planning.objnav.agent.episode_checks import check_observation, map_target
from sparx_agency.core.planning.objnav.agent.episode_log import EpisodeLog, policy_report, step_info
from sparx_agency.core.planning.objnav.agent.params import HeadlessAgentParams
from sparx_agency.core.planning.objnav.errors import ObjNavError, ObjNavInternalError
from sparx_agency.core.planning.objnav.interfaces.agent import ObjNavAgent
from sparx_agency.core.planning.objnav.interfaces.label_mapper import LabelMapper
from sparx_agency.core.planning.objnav.interfaces.policy import SearchPolicy
from sparx_agency.core.planning.objnav.types.actions import DiscreteAction
from sparx_agency.core.planning.objnav.types.command import NavigationCommand
from sparx_agency.core.planning.objnav.types.decision import AgentDecision
from sparx_agency.core.planning.objnav.types.episode import ObjNavEpisode
from sparx_agency.core.planning.objnav.types.observation import ObjNavObservation
from sparx_agency.core.planning.objnav.types.target import TargetLabels


class HeadlessObjNavAgent(ObjNavAgent):
    """A policy, label mapper and converter with fresh state per episode.

    Empty commands spend the configured idle turn, never STOP. The optional
    safety hook can only replace a forward proposal with an existing turn.
    """
    def __init__(self, policy, label_mapper: LabelMapper,
                 params: Optional[HeadlessAgentParams] = None,
                 name: Optional[str] = None) -> None:
        if not isinstance(policy, SearchPolicy):
            raise TypeError("policy must be a SearchPolicy -- an object with a name, "
                            "reset(episode, target) and plan(observation) -- got %r" % (policy,))
        if not isinstance(label_mapper, LabelMapper):
            raise TypeError("label_mapper must be a LabelMapper, got %r" % (label_mapper,))
        if params is None:
            params = HeadlessAgentParams()
        if not isinstance(params, HeadlessAgentParams):
            raise TypeError("params must be a HeadlessAgentParams, got %r" % (params,))
        if name is None:
            name = "headless/%s" % (_policy_name(policy),)
        if not isinstance(name, str) or not name.strip():
            raise ObjNavError("the agent name must be a non-blank string -- every result "
                              "row is logged under it -- got %r" % (name,))
        self._policy, self._label_mapper = policy, label_mapper
        self._params, self._name = params, name
        self._forget_episode()

    @property
    def name(self) -> str:
        return self._name

    @property
    def episode(self) -> Optional[ObjNavEpisode]:
        return self._episode

    @property
    def target(self) -> Optional[TargetLabels]:
        return self._target

    def reset(self, episode: ObjNavEpisode) -> None:
        self._forget_episode()
        if not isinstance(episode, ObjNavEpisode):
            raise TypeError("reset() needs an ObjNavEpisode, got %r" % (episode,))
        target = map_target(self._label_mapper, episode)
        converter = DiscreteActionConverter(episode.action_spec, self._params.converter)
        self._policy.reset(episode, target)
        self._episode, self._target, self._converter = episode, target, converter

    def act(self, observation: ObjNavObservation) -> AgentDecision:
        check_observation(observation, self._episode, self._last_step)
        self._last_step = observation.step
        if self._converter.forward_blocked(observation.pose):
            self._notify_blocked(observation)
        command = self._policy.plan(observation)
        if not isinstance(command, NavigationCommand):
            raise TypeError("policy %r returned %r from plan(); a SearchPolicy returns a "
                            "NavigationCommand (NavigationCommand.hold() for no opinion)"
                            % (getattr(self._policy, "name", self._policy), command))
        result = self._converter.step(observation.pose, command)
        action = self._params.idle_action if result.idle else result.action
        guarded_policy = self._policy
        guard = getattr(guarded_policy, "filter_action", None)
        if not callable(guard):
            guarded_policy = getattr(self._policy, "policy", None)
            guard = getattr(guarded_policy, "filter_action", None)
        if callable(guard):
            guarded = guard(observation, action)
            if guarded != action:
                if action != DiscreteAction.MOVE_FORWARD or guarded not in (DiscreteAction.TURN_LEFT, DiscreteAction.TURN_RIGHT):
                    raise ObjNavInternalError("Safety filter may only veto forward motion into a legal turn")
                command = replace(command, info=dict(command.info, safety_veto=action.name))
                action = guarded
                result = replace(result, action=action, status="turn")
                self._converter.record_emitted_action(observation.pose, action)
        if not self._episode.action_spec.allows(action):
            raise ObjNavInternalError("the agent chose %s (converter status %r), which episode %r's "
                                      "action spec does not allow" % (action.name, result.status, self._episode.episode_id))
        self._log.count(result)
        notify = getattr(self._policy, "notify_action", None)
        if callable(notify):
            notify(observation, action)
        return AgentDecision(action, step_info(result, command))

    def episode_info(self) -> Dict[str, Any]:
        if self._episode is None:
            raise ObjNavError("episode_info() called before reset(); there is no episode to report on")
        return self._log.report(self._name, policy_report(self._policy))

    def _forget_episode(self) -> None:
        self._episode = self._target = self._converter = self._last_step = None
        self._log = EpisodeLog()

    def _notify_blocked(self, observation: ObjNavObservation) -> None:
        hook = getattr(self._policy, "notify_blocked", None)
        if not callable(hook):
            return
        self._log.count_notification()
        answer = hook(observation)
        if answer is not None:
            raise TypeError("policy %r returned %r from notify_blocked(); the hook returns "
                            "None -- replan inside it and return the new path from the "
                            "plan() call that follows" % (getattr(self._policy, "name", self._policy), answer))

    def __repr__(self) -> str:
        return "HeadlessObjNavAgent(name=%r, label_mapper=%r, idle_action=%s)" % (
            self._name, self._label_mapper.name, self._params.idle_action.name)


def _policy_name(policy) -> str:
    name = getattr(policy, "name", None)
    if not isinstance(name, str) or not name.strip():
        raise ObjNavError("policy %r has no usable name (got %r); give it a non-blank name "
                          "or pass name= to the agent -- results are logged under it" % (policy, name))
    return name
