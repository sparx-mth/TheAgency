"""Settle everything a run depends on before it runs any episode: its options, its participants, its episodes, what a resume reuses.

A typo found after a hundred episodes costs the hundred, so the whole plan is
checked up front: every option; the environment and the agent, which must
have a name to log under; the episode ids, which the environment must serve
once each and the caller request once each; and a resumed results directory,
which may hold only this run's agent and only episodes this run requests.
Otherwise the summary at the end would describe a different experiment than
the file beside it -- fewer episodes than it holds, or another agent's under
this one's name.

Python 3.8 syntax, standard library only.
"""
from __future__ import annotations

import collections
import collections.abc
from typing import Dict, Optional, Sequence, Tuple

from sparx_agency.core.planning.objnav.errors import EnvContractError
from sparx_agency.core.planning.objnav.interfaces.agent import ObjNavAgent
from sparx_agency.core.planning.objnav.interfaces.env import ObjNavEnv
from sparx_agency.tasks.planning.objnav_benchmark.agent_contract import (
    ON_AGENT_ERROR_MODES,
)
from sparx_agency.tasks.planning.objnav_benchmark.checks import (
    is_length,
    is_name,
)
from sparx_agency.tasks.planning.objnav_benchmark.errors import (
    HarnessError,
    ResultsError,
)
from sparx_agency.tasks.planning.objnav_benchmark.kinematics import (
    KinematicTolerance,
)
from sparx_agency.tasks.planning.objnav_benchmark.logger import MetricsLogger
from sparx_agency.tasks.planning.objnav_benchmark.records import EpisodeRecord


def check_options(on_agent_error, require_stop_for_success, path_tolerance_m,
                  kinematics, expected=None) -> None:
    """Refuse a malformed run option before the first step.

    The two scoring options repeat checks :func:`score_episode` makes, on
    purpose: there they fire after the first episode has run; here, before it
    starts.

    Raises:
        TypeError: If ``kinematics`` is neither a
            :class:`KinematicTolerance` nor None.
        HarnessError: On an unknown ``on_agent_error``, a
            ``require_stop_for_success`` that is not a bool, a negative or
            non-finite ``path_tolerance_m``, or an ``expected`` that is not a
            ``(benchmark, split)`` pair of names.
    """
    if on_agent_error not in ON_AGENT_ERROR_MODES:
        raise HarnessError("on_agent_error must be one of %r, got %r"
                           % (ON_AGENT_ERROR_MODES, on_agent_error))
    if not isinstance(require_stop_for_success, bool):
        raise HarnessError(
            "require_stop_for_success must be a bool, got %r (a string such "
            "as 'false' would read as True)" % (require_stop_for_success,))
    if path_tolerance_m is not None and not is_length(path_tolerance_m):
        raise HarnessError(
            "path_tolerance_m must be None (no check) or a finite "
            "non-negative number of metres, got %r" % (path_tolerance_m,))
    if kinematics is not None and not isinstance(kinematics, KinematicTolerance):
        raise TypeError(
            "kinematics must be a KinematicTolerance, or None only for a "
            "simulator whose motion has been verified otherwise; got %r"
            % (kinematics,))
    if expected is not None and not (
            isinstance(expected, tuple) and len(expected) == 2
            and all(is_name(item) for item in expected)):
        raise HarnessError(
            "expected must be None or a (benchmark, split) pair of non-blank "
            "strings, got %r" % (expected,))


def check_participants(env, agent) -> None:
    """Refuse an environment or an agent of the wrong type, or an agent with no name to log under.

    Raises:
        TypeError: If ``env`` is not an :class:`ObjNavEnv` or ``agent`` not an
            :class:`ObjNavAgent`.
        HarnessError: If the agent's name is not a non-blank string: every row
            is logged under it, and a blank one would be refused only by the
            summary, after every episode had run.
    """
    if not isinstance(env, ObjNavEnv):
        raise TypeError("env must be an ObjNavEnv, got %s" % type(env).__name__)
    if not isinstance(agent, ObjNavAgent):
        raise TypeError("agent must be an ObjNavAgent, got %s"
                        % type(agent).__name__)
    if not is_name(agent.name):
        raise HarnessError(
            "%s has no name (got %r); every result row is logged under the "
            "agent's name, so set one" % (type(agent).__name__, agent.name))


def _served_ids(env: ObjNavEnv) -> Tuple[str, ...]:
    """``env.episode_ids()``, refused when it serves an id twice."""
    served = tuple(env.episode_ids())
    repeats = sorted(item for item, n in collections.Counter(served).items()
                     if n > 1)
    if repeats:
        raise EnvContractError(
            "%s.episode_ids() serves %r more than once; episode ids must be "
            "unique within (benchmark, split), or a request cannot say which "
            "episode it means -- scene-qualify per-scene ids (Habitat "
            "ObjectNav renumbers episodes from 0 in every scene file)"
            % (type(env).__name__, repeats[:5]))
    return served


def requested_episode_ids(env: ObjNavEnv,
                          episode_ids: Optional[Sequence[str]]) -> Tuple[str, ...]:
    """The episodes to run, in order, refused before the first one if any cannot run.

    Args:
        env: The environment.
        episode_ids: The requested episodes; None for ``env.episode_ids()``.

    Returns:
        The ids to run.

    Raises:
        EnvContractError: If ``env.episode_ids()`` serves an id more than
            once; checked before the request.
        HarnessError: If the list is a single string, is empty, holds a blank
            or non-string id, repeats one, or names one ``env`` does not serve.
    """
    served = _served_ids(env)
    if episode_ids is None:
        ids = served
    elif (isinstance(episode_ids, (str, bytes))
          or not isinstance(episode_ids, collections.abc.Sequence)):
        raise HarnessError(
            "episode_ids must be a sequence of episode ids, got %r (a single "
            "string would run one episode per character)" % (episode_ids,))
    else:
        ids = tuple(episode_ids)
    if not ids:
        raise HarnessError("no episodes to run: %s"
                           % ("episode_ids is empty" if episode_ids is not None
                              else "%s serves none" % type(env).__name__))
    bad = [item for item in ids if not isinstance(item, str) or not item]
    if bad:
        raise HarnessError("episode ids must be non-empty strings, got %r"
                           % (bad[:5],))
    repeats = sorted(item for item, n in collections.Counter(ids).items() if n > 1)
    if repeats:
        raise HarnessError("episodes %r are requested more than once; each "
                           "would run and count twice" % (repeats[:5],))
    known = set(served)
    unknown = [item for item in ids if item not in known]
    if unknown:
        raise HarnessError("%s does not serve episodes %r (%d in all)"
                           % (type(env).__name__, unknown[:5], len(unknown)))
    return ids


def reusable_records(logger: Optional[MetricsLogger], ids: Sequence[str],
                     agent_name: str) -> Dict[str, EpisodeRecord]:
    """A resumed run's completed episodes by id, refused unless they belong to this run.

    Args:
        logger: The run's logger, or None for a run kept in memory.
        ids: The episodes this run requests.
        agent_name: This run's agent.

    Returns:
        The logged records by episode id; empty without a logger.

    Raises:
        ResultsError: If the logger holds an episode ``ids`` does not
            request, or another agent's episode.
    """
    if logger is None:
        return {}
    logged = {record.episode_id: record for record in logger.records}
    unrequested = sorted(set(logged) - set(ids))
    if unrequested:
        raise ResultsError(
            "%s holds %d episodes this run does not request (%r); its summary "
            "would describe fewer episodes than the file holds -- resume with "
            "the same episode list, or start a new directory"
            % (logger.episodes_path, len(unrequested), unrequested[:5]))
    others = sorted({record.agent for record in logged.values()} - {agent_name})
    if others:
        raise ResultsError(
            "%s holds episodes of agent %r, but this run's agent is %r; "
            "reusing them would summarise another agent's episodes under this "
            "one's name" % (logger.episodes_path, others[0], agent_name))
    return logged
