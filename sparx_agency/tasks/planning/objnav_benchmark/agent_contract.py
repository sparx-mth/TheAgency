"""What the runner demands of an agent at every step, and what it does with an agent that fails.

An agent's bug must never reach the environment, nor be recorded as the
environment's. A decision the benchmark cannot execute -- anything but an
:class:`AgentDecision`, or an action the benchmark's spec does not allow (a
LOOK on a benchmark without tilt) -- is refused as an
:class:`AgentContractError` before the environment sees it. What follows is
the run's choice: under :data:`ON_AGENT_ERROR_RAISE`, the default, the error
propagates, because a crash is a bug to look at; under
:data:`ON_AGENT_ERROR_RECORD` the episode ends on a forced STOP and is kept as
an agent error, so one bad episode does not cost a night's sweep. The agent's
per-episode diagnostics go into the results row as they are, so they must be
a dict of strict JSON; under ``"record"`` diagnostics that are not, or that
fail to be produced at all, are kept as an error note beside a score that
stands.

What ``"record"`` never absorbs is :data:`INFRASTRUCTURE_ERRORS`: an
invariant of our own code broke -- the converter's, the headless agent's or
the harness's -- and a sweep that recorded it would charge our bug to the
method under test. Those stop the run in every mode.

Python 3.8 syntax, standard library only.
"""
from __future__ import annotations

import collections.abc
import json
from typing import Any, Dict, Optional, Tuple

from sparx_agency.core.planning.objnav.errors import (
    AgentContractError,
    ObjNavInternalError,
)
from sparx_agency.core.planning.objnav.interfaces.agent import ObjNavAgent
from sparx_agency.core.planning.objnav.types.actions import (
    DiscreteAction,
    DiscreteActionSpec,
)
from sparx_agency.core.planning.objnav.types.decision import AgentDecision
from sparx_agency.core.planning.objnav.types.observation import (
    ObjNavObservation,
)
from sparx_agency.tasks.planning.objnav_benchmark.errors import (
    HarnessError,
    ResultsError,
)
from sparx_agency.tasks.planning.objnav_benchmark.results_io import strict_json

#: Re-raise what ``agent.act`` raised: a crash is a bug to look at. The default.
ON_AGENT_ERROR_RAISE = "raise"
#: Record the crash as a failed episode and go on to the next.
ON_AGENT_ERROR_RECORD = "record"
#: Every accepted ``on_agent_error``.
ON_AGENT_ERROR_MODES = (ON_AGENT_ERROR_RAISE, ON_AGENT_ERROR_RECORD)

#: The ``agent_info`` key under which failing diagnostics are recorded.
EPISODE_INFO_ERROR_KEY = "episode_info_error"

#: Raised in every mode, never recorded: our own invariant broke, not the
#: method's code, and recording it would charge our bug to the method.
INFRASTRUCTURE_ERRORS = (ObjNavInternalError, HarnessError)


def _recordable(exc: BaseException, mode: str) -> bool:
    """Whether ``exc`` is kept as the agent's failure rather than re-raised."""
    return (mode == ON_AGENT_ERROR_RECORD
            and not isinstance(exc, INFRASTRUCTURE_ERRORS))


def _described(exc: BaseException) -> str:
    return "%s: %s" % (type(exc).__name__, exc)


def accepted_action(decision: Any, spec: DiscreteActionSpec) -> DiscreteAction:
    """The decision's action, once it is known to be one the benchmark accepts.

    Args:
        decision: What ``agent.act`` returned.
        spec: The episode's action spec.

    Returns:
        ``decision.action``.

    Raises:
        AgentContractError: If ``decision`` is not an :class:`AgentDecision`
            (a bare action included), or its action is one ``spec`` does not
            allow.
    """
    if not isinstance(decision, AgentDecision):
        raise AgentContractError(
            "agent.act must return an AgentDecision, got %s"
            % type(decision).__name__)
    if not spec.allows(decision.action):
        raise AgentContractError(
            "the agent chose %s, which this benchmark does not accept (it "
            "accepts %s)" % (decision.action.name,
                             ", ".join(a.name for a in spec.actions)))
    return decision.action


def decide(agent: ObjNavAgent, observation: ObjNavObservation,
           spec: DiscreteActionSpec,
           mode: str) -> Tuple[DiscreteAction, Optional[str]]:
    """The agent's next action; under ``"record"`` a failure becomes a forced STOP and its description.

    Args:
        agent: The agent.
        observation: The observation it acts on.
        spec: The episode's action spec.
        mode: :data:`ON_AGENT_ERROR_RAISE` or :data:`ON_AGENT_ERROR_RECORD`.

    Returns:
        ``(action, None)`` for an accepted decision. Under
        :data:`ON_AGENT_ERROR_RECORD`, ``(STOP, "ExceptionType: message")``
        when ``agent.act`` raised or broke the contract.

    Raises:
        AgentContractError: Under :data:`ON_AGENT_ERROR_RAISE`, for a decision
            the benchmark does not accept.
        Exception: Under :data:`ON_AGENT_ERROR_RAISE`, whatever ``agent.act``
            raised; in every mode, one of :data:`INFRASTRUCTURE_ERRORS`.
    """
    try:
        return accepted_action(agent.act(observation), spec), None
    except Exception as exc:  # any bug of the agent's; Ctrl-C still stops the run
        if not _recordable(exc, mode):
            raise
        return DiscreteAction.STOP, _described(exc)


def _strict_info(agent: ObjNavAgent, info: Any) -> Dict[str, Any]:
    """``info`` exactly as a results file will hold it, or a ResultsError saying why it cannot."""
    if not isinstance(info, collections.abc.Mapping):
        raise ResultsError("%s.episode_info must return a dict, got %s"
                           % (type(agent).__name__, type(info).__name__))
    # The strict-JSON round trip makes the record hold exactly what a resume
    # will read back (tuples as lists, keys as strings).
    return json.loads(strict_json(dict(info), "%s.episode_info()"
                                  % type(agent).__name__))


def agent_info(agent: ObjNavAgent, mode: str) -> Dict[str, Any]:
    """The agent's diagnostics for the episode just ended, exactly as a results file will hold them.

    Args:
        agent: The agent.
        mode: Under :data:`ON_AGENT_ERROR_RECORD`, diagnostics that cannot be
            stored -- ``episode_info`` raised, returned something that is not
            a dict of strict JSON, or returned a mapping that raises as it is
            read -- are stored as ``{EPISODE_INFO_ERROR_KEY: "Type:
            message"}``, beside a score that stands.

    Returns:
        ``agent.episode_info()`` after a strict-JSON round trip.

    Raises:
        ResultsError: Under :data:`ON_AGENT_ERROR_RAISE`, if ``episode_info``
            returns anything but a dict of strict JSON.
        Exception: Under :data:`ON_AGENT_ERROR_RAISE`, whatever
            ``episode_info`` raised, or its mapping raised as it was read; in
            every mode, one of :data:`INFRASTRUCTURE_ERRORS` other than our
            own strict-JSON refusal of the agent's data.
    """
    try:
        info = agent.episode_info()
    except Exception as exc:  # diagnostics only: under "record" the score stands
        if not _recordable(exc, mode):
            raise
        return {EPISODE_INFO_ERROR_KEY: _described(exc)}
    try:
        return _strict_info(agent, info)
    except Exception as exc:
        # The agent's own report failed to become strict JSON -- a NaN, a
        # numpy scalar, a mapping that raises as it is read, nesting too deep
        # to encode: its data, not our invariant -- so "record" keeps it like
        # a raising episode_info. Our ResultsError refusing that data is the
        # one infrastructure error that is the agent's.
        if mode == ON_AGENT_ERROR_RAISE or not (
                isinstance(exc, ResultsError) or _recordable(exc, mode)):
            raise
        return {EPISODE_INFO_ERROR_KEY: _described(exc)}
