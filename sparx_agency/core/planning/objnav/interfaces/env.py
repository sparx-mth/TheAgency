"""The :class:`ObjNavEnv` contract: a simulator behind one simulator-agnostic face.

Each benchmark branch implements this once -- Habitat for HM3D, MP3D and
Gibson, AI2-THOR for RoboTHOR -- and everything above it (the runner, the
agent, the scoring) stays the same code. The adapter's whole job is the
conversions nothing downstream should know about:

* its frames into world ENU, its yaw into CCW-from-``+x``, its camera pitch
  into REP-103 (positive looks down);
* its depth into metres of optical-frame ``z``, with ``NaN`` for no reading
  and ``+inf`` for no surface in range;
* its action names into :class:`~sparx_agency.core.planning.objnav.types.DiscreteAction`,
  by name;
* its metric keys into the canonical ``NATIVE_*`` keys.

Lifecycle::

    for episode_id in env.episode_ids():
        episode, observation = env.reset(episode_id)
        while not env.episode_over:
            observation = env.step(action)
        measurement = env.measure()

The environment owns the step budget and the STOP semantics: the episode is
over after STOP, or after ``episode.max_steps`` actions, whichever comes first.

Python 3.8 syntax, standard library only.
"""
from __future__ import annotations

import abc
from typing import Tuple


class ObjNavEnv(abc.ABC):
    """A source of ObjectNav episodes.

    Attributes:
        name: Identifier of the environment (``"habitat"``, ``"fake"``).
    """

    name = ""

    @abc.abstractmethod
    def episode_ids(self) -> Tuple[str, ...]:
        """Every episode this environment serves, in a stable order.

        Ids must be unique within ``(benchmark, split)``: results, resumes and
        paired comparisons are keyed on them, and ``reset(id)`` must mean one
        episode. Scene-qualify per-scene ids (``"<scene>/<id>"``): Habitat
        ObjectNav renumbers its episodes from 0 in every scene's content file,
        so its raw ids repeat across the scenes of a split.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def reset(self, episode_id: str):
        """Start an episode.

        Args:
            episode_id: One of :meth:`episode_ids`.

        Returns:
            ``(episode, observation)``: the public
            :class:`~sparx_agency.core.planning.objnav.types.ObjNavEpisode` and
            its first observation, whose ``step`` is 0.

        Raises:
            KeyError: ``episode_id`` is not served by this environment.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def step(self, action):
        """Execute one action.

        Args:
            action: A :class:`~sparx_agency.core.planning.objnav.types.DiscreteAction`
                the episode's action spec allows.

        Returns:
            The observation after the action, one ``step`` later. STOP returns
            one too, and ends the episode.

        Raises:
            EnvContractError: The episode is already over, or the action is
                not allowed.
        """
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def episode_over(self) -> bool:
        """True once STOP was executed or the step budget is spent."""
        raise NotImplementedError

    @abc.abstractmethod
    def measure(self):
        """The ground truth of the finished episode.

        Returns:
            An :class:`~sparx_agency.core.planning.objnav.types.EpisodeMeasurement`.

        Raises:
            EnvContractError: The episode is still running.
        """
        raise NotImplementedError

    def close(self) -> None:
        """Release the simulator. The default holds nothing."""
