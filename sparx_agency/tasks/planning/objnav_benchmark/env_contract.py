"""Hold an environment to the ObjNavEnv contract at every step, so an adapter's bug stops the run instead of becoming a number.

An adapter bug does not crash a benchmark; it produces a plausible score. The
runner is the only code between an agent and a simulator, so it is where
these bugs are caught, and each function here raises
:class:`EnvContractError` on one clause of the :class:`ObjNavEnv` contract:

* the first observation belongs to the episode just reset -- its id, step 0,
  its target and camera -- and the episode is not already over (an adapter
  that forgets to clear its STOP flag ends every later episode at step 0);
* no key of the episode's metadata names privileged information. The agent
  receives the metadata at reset, and a goal position or shortest path in it
  makes SR and SPL meaningless without failing anything. This is a tripwire
  for the known leaks (:data:`PRIVILEGED_KEY_PARTS`), not a guarantee: what an
  adapter puts in the metadata stays its responsibility;
* every episode is of the run's benchmark and split;
* every observation is one step later than the last, for the same target and
  camera -- checked before the agent sees it, so an environment's bug is never
  recorded as the agent's -- and the agent moved as the episode's action spec
  says it does (``kinematics.py``);
* the episode ends on STOP or at the step budget, never earlier or later;
* the measurement counts exactly the actions sent and agrees on whether STOP
  ended the episode.

The last clause, a path length that matches the one the observed poses
trace, is checked when the episode is scored (``cross_checks.py``).

Python 3.8 syntax, standard library only.
"""
from __future__ import annotations

import collections.abc
from typing import Any, Iterator, Optional, Set, Tuple

from sparx_agency.core.planning.objnav.errors import EnvContractError
from sparx_agency.core.planning.objnav.interfaces.env import ObjNavEnv
from sparx_agency.core.planning.objnav.types.actions import DiscreteAction
from sparx_agency.core.planning.objnav.types.episode import ObjNavEpisode
from sparx_agency.core.planning.objnav.types.measurement import (
    EpisodeMeasurement,
)
from sparx_agency.core.planning.objnav.types.observation import (
    ObjNavObservation,
)
from sparx_agency.core.planning.objnav.types.pose import AgentPose
from sparx_agency.tasks.planning.objnav_benchmark.kinematics import (
    KinematicTolerance,
    check_motion,
)

_STOP = DiscreteAction.STOP

#: A metadata key holding any of these, case-insensitively and at any depth,
#: is refused at reset: the names the datasets and evaluators give privileged
#: values (RoboTHOR's ``shortest_path_length``, allenact's
#: ``distance_to_target`` and ``path_to_target``, Habitat's goal
#: ``view_points``). A tripwire for these known leaks, not a guarantee.
PRIVILEGED_KEY_PARTS = ("goal", "shortest", "geodesic", "distance_to",
                        "path_to", "view_point", "viewpoint",
                        "target_position", "object_position")


def _check_belongs(observation: ObjNavObservation, episode: ObjNavEpisode,
                   what: str) -> None:
    """Refuse an observation whose target or camera is not the episode's."""
    if observation.target_category != episode.target_category:
        raise EnvContractError(
            "%s of episode %r is for target %r, but the episode searches for "
            "%r" % (what, episode.episode_id, observation.target_category,
                    episode.target_category))
    if observation.camera != episode.camera:
        raise EnvContractError(
            "%s of episode %r was captured with another camera than the "
            "episode's: %r, not %r"
            % (what, episode.episode_id, observation.camera, episode.camera))


def _privileged_keys(value: Any, where: str, seen: Set[int]) -> Iterator[str]:
    """The path of every key in ``value``, at any depth, that names privileged information."""
    if id(value) in seen:  # a container met twice, or a cycle
        return
    if isinstance(value, collections.abc.Mapping):
        seen.add(id(value))
        for key, item in value.items():
            path = "%s[%r]" % (where, key)
            if any(part in str(key).lower() for part in PRIVILEGED_KEY_PARTS):
                yield path
            yield from _privileged_keys(item, path, seen)
    elif (isinstance(value, collections.abc.Sequence)
          and not isinstance(value, (str, bytes, bytearray))):
        seen.add(id(value))
        for index, item in enumerate(value):
            yield from _privileged_keys(item, "%s[%d]" % (where, index), seen)


def _check_metadata(episode: ObjNavEpisode) -> None:
    """Refuse metadata whose keys name privileged information."""
    leaks = list(_privileged_keys(episode.metadata, "metadata", set()))
    if leaks:
        raise EnvContractError(
            "episode %r carries %s, a key that names privileged information "
            "(any key containing one of %s); the agent receives the metadata "
            "at reset, so goal positions, shortest or geodesic paths and "
            "distances and view points stay inside the environment and reach "
            "the harness only through measure()"
            % (episode.episode_id, ", ".join(leaks[:5]),
               ", ".join(PRIVILEGED_KEY_PARTS)))


def checked_reset(env: ObjNavEnv,
                  episode_id: str) -> Tuple[ObjNavEpisode, ObjNavObservation]:
    """Reset ``env``, refusing anything but the requested episode's start.

    Args:
        env: The environment.
        episode_id: The episode to start.

    Returns:
        The episode and its first observation, once the observation is step 0
        of that episode, for its target and camera, the episode is running,
        and no metadata key names privileged information.

    Raises:
        EnvContractError: When the reset breaks any of those. The metadata
            check is a tripwire for the keys in :data:`PRIVILEGED_KEY_PARTS`,
            not a guarantee.
    """
    result = env.reset(episode_id)
    if not (isinstance(result, tuple) and len(result) == 2
            and isinstance(result[0], ObjNavEpisode)
            and isinstance(result[1], ObjNavObservation)):
        got = ("(%s)" % ", ".join(type(item).__name__ for item in result)
               if isinstance(result, tuple) else type(result).__name__)
        raise EnvContractError(
            "%s.reset must return (ObjNavEpisode, ObjNavObservation), got %s"
            % (type(env).__name__, got))
    episode, observation = result
    if episode.episode_id != episode_id:
        raise EnvContractError("reset(%r) returned episode %r"
                               % (episode_id, episode.episode_id))
    _check_metadata(episode)
    if observation.step != 0:
        raise EnvContractError(
            "the reset observation of episode %r has step %d; the "
            "observation reset returns is step 0"
            % (episode_id, observation.step))
    _check_belongs(observation, episode, "the reset observation")
    if env.episode_over:
        raise EnvContractError(
            "episode %r is already over right after reset; an adapter that "
            "does not clear its STOP flag ends every later episode at step 0"
            % (episode_id,))
    return episode, observation


def check_run_identity(episode: ObjNavEpisode,
                       expected: Optional[Tuple[str, str]]) -> None:
    """Refuse an episode of another benchmark or split than the run's.

    Args:
        episode: The episode just reset.
        expected: The run's ``(benchmark, split)``; None when the run has no
            identity yet (its first episode decides it).

    Raises:
        EnvContractError: If the episode's ``(benchmark, split)`` differs.
    """
    if expected is None or (episode.benchmark, episode.split) == tuple(expected):
        return
    raise EnvContractError(
        "episode %r is %s/%s, but this run is %s/%s; a run scores one "
        "benchmark split, so the adapter reports the wrong benchmark or split "
        "for this episode" % ((episode.episode_id, episode.benchmark,
                               episode.split) + tuple(expected)))


def check_step(env: ObjNavEnv, episode: ObjNavEpisode, observation: Any,
               action: DiscreteAction, sent: int, *, before: AgentPose,
               kinematics: Optional[KinematicTolerance]) -> None:
    """Refuse an observation, or an episode end, that breaks the contract after action ``sent``.

    Args:
        env: The environment, just stepped.
        episode: Its episode.
        observation: What ``env.step`` returned.
        action: The action it was sent.
        sent: Actions sent so far, this one included.
        before: The agent's pose before the action: the previous
            observation's.
        kinematics: How far the realised motion may stray from the episode's
            action spec (:func:`check_motion`); None skips that check.

    Raises:
        EnvContractError: If the observation is not an
            :class:`ObjNavObservation` of this episode at step ``sent``, the
            agent did not move as the action spec says, or the episode is
            running after STOP or past its budget, or over before either.
    """
    if not isinstance(observation, ObjNavObservation):
        raise EnvContractError("%s.step must return an ObjNavObservation, got %s"
                               % (type(env).__name__, type(observation).__name__))
    if observation.step != sent:
        raise EnvContractError(
            "after %d actions episode %r reports step %d; the step count must "
            "equal the actions sent" % (sent, episode.episode_id,
                                        observation.step))
    _check_belongs(observation, episode, "the observation after action %d" % sent)
    if kinematics is not None:
        try:
            check_motion(action, before, observation.pose, episode.action_spec,
                         kinematics)
        except EnvContractError as exc:
            raise EnvContractError("episode %r, action %d: %s"
                                   % (episode.episode_id, sent, exc)) from exc
    over = env.episode_over
    if action == _STOP and not over:
        raise EnvContractError("episode %r is still running after STOP; STOP "
                               "ends an episode" % (episode.episode_id,))
    if action != _STOP and sent >= episode.max_steps and not over:
        raise EnvContractError(
            "episode %r is still running after %d actions, its step budget "
            "(max_steps=%d)" % (episode.episode_id, sent, episode.max_steps))
    if action != _STOP and sent < episode.max_steps and over:
        raise EnvContractError(
            "episode %r ended after %d actions without STOP, before its budget "
            "of %d; an episode ends on STOP or at the budget, nothing else"
            % (episode.episode_id, sent, episode.max_steps))


def checked_measurement(env: ObjNavEnv, episode: ObjNavEpisode,
                        actions_sent: int,
                        last_action: DiscreteAction) -> EpisodeMeasurement:
    """The environment's measurement, once it agrees with what the runner counted.

    Args:
        env: The environment, its episode over.
        episode: That episode.
        actions_sent: Actions the runner sent, STOP included.
        last_action: The last action it sent.

    Returns:
        ``env.measure()``.

    Raises:
        EnvContractError: If the measurement is not an
            :class:`EpisodeMeasurement`, counts another number of steps, or
            disagrees on whether STOP ended the episode.
    """
    measurement = env.measure()
    if not isinstance(measurement, EpisodeMeasurement):
        raise EnvContractError(
            "%s.measure must return an EpisodeMeasurement, got %s"
            % (type(env).__name__, type(measurement).__name__))
    if measurement.steps != actions_sent:
        raise EnvContractError(
            "the measurement of episode %r counts %d steps but the runner sent "
            "%d actions; every action counts, STOP included"
            % (episode.episode_id, measurement.steps, actions_sent))
    if measurement.stop_called != (last_action == _STOP):
        raise EnvContractError(
            "the measurement of episode %r says stop_called=%r, but its last "
            "action was %s" % (episode.episode_id, measurement.stop_called,
                               last_action.name))
    return measurement
