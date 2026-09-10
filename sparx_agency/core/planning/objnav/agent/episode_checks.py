"""The headless agent's boundary checks: the target a mapper answers with, and each frame against its episode.

Both run before the policy sees anything, so a contract broken upstream fails
where it was broken instead of steering the search. The quiet failures they
guard against:

* **A search for the wrong object.** A category the mapper does not cover, or
  a mapper that answers with another category's target, is refused at reset,
  before the first step -- not discovered as a run of failed episodes.
* **A frame from somewhere else.** Every observation is checked against its
  episode: the target, the camera, and a step index that only moves forward.
  A stale or foreign frame fails loudly instead of steering the policy.

Python 3.8 syntax; numpy arrives only through the observation type.
"""
from __future__ import annotations

from typing import Optional

from sparx_agency.core.planning.objnav.errors import (
    ObjNavError,
    ObservationError,
    UnknownCategoryError,
)
from sparx_agency.core.planning.objnav.interfaces.label_mapper import LabelMapper
from sparx_agency.core.planning.objnav.types.episode import ObjNavEpisode
from sparx_agency.core.planning.objnav.types.observation import ObjNavObservation
from sparx_agency.core.planning.objnav.types.target import TargetLabels


def map_target(mapper: LabelMapper, episode: ObjNavEpisode) -> TargetLabels:
    """The episode's category in our vision vocabulary, or raise.

    Args:
        mapper: The benchmark's label mapper.
        episode: The episode about to run.

    Returns:
        The mapper's target for the episode's category.

    Raises:
        UnknownCategoryError: If the mapper does not cover the category; the
            message names the mapper and its categories.
        TypeError: If the mapper answers with something other than
            :class:`TargetLabels`.
        ObjNavError: If the mapper answers with another category's target.
    """
    category = episode.target_category
    if not mapper.covers(category):
        raise UnknownCategoryError(
            "label mapper %r does not cover category %r of episode %r "
            "(%s/%s); it covers %s. Add the category to that dataset's "
            "table, or run this benchmark with its own mapper"
            % (mapper.name, category, episode.episode_id,
               episode.benchmark, episode.split,
               ", ".join(repr(c) for c in mapper.categories())))
    target = mapper.target_labels(category)
    if not isinstance(target, TargetLabels):
        raise TypeError(
            "label mapper %r answered category %r with %r; a LabelMapper "
            "returns TargetLabels" % (mapper.name, category, target))
    if target.category != category:
        raise ObjNavError(
            "label mapper %r answered category %r with the target of %r; "
            "the policy would search for the wrong object"
            % (mapper.name, category, target.category))
    return target


def check_observation(observation, episode: Optional[ObjNavEpisode],
                      last_step: Optional[int]) -> None:
    """Raise unless ``observation`` is ``episode``'s next frame.

    Args:
        observation: What the environment handed over.
        episode: The running episode, or None before a reset.
        last_step: The previous observation's step, or None before the first.

    Raises:
        ObjNavError: If there is no episode.
        ObservationError: If ``observation`` is not an
            :class:`ObjNavObservation`, is for another target or camera, or
            does not come after ``last_step``.
    """
    if episode is None:
        raise ObjNavError("act() called before reset(); start an episode "
                          "first")
    if not isinstance(observation, ObjNavObservation):
        raise ObservationError(
            "act() needs an ObjNavObservation, got %r" % (observation,))
    if observation.target_category != episode.target_category:
        raise ObservationError(
            "the observation's target %r is not episode %r's target %r; "
            "the environment handed over another episode's frame"
            % (observation.target_category, episode.episode_id,
               episode.target_category))
    if observation.camera != episode.camera:
        raise ObservationError(
            "the observation's camera %r is not episode %r's camera %r; "
            "every depth pixel would be read with the wrong geometry"
            % (observation.camera, episode.episode_id, episode.camera))
    if last_step is not None and observation.step <= last_step:
        raise ObservationError(
            "observation step %r does not come after the previous step "
            "%r; the environment handed over a stale or repeated frame"
            % (observation.step, last_step))
