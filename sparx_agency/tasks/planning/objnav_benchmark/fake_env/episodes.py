"""The fake's episodes: what each one is, the ground truth it is scored by, and the ones the fake refuses.

A **test rig, not a benchmark.** Every benchmark episode is solvable -- its
start somewhere an agent can stand, its goal reachable from there -- so a fake
episode that is not is refused when the environment is built, where it costs
nothing, rather than run to the step budget, where it would read as a
method's miss. What is refused, and why each would be quiet otherwise:

* **An object the building lacks, or one with no floor near it.** Such an
  episode can only fail, and its failure says nothing about the method.
* **A start the agent could not stand at.** Off the fake's one floor, in the
  wall band, or with the camera pitched past the action spec's limits: the
  first frame would come from somewhere no simulator would put the agent.
* **A goal the start cannot reach.** A walled-off start scores 0 for a reason
  no method controls.
* **Two episodes under one id.** Ids name results rows; a repeat would put
  two episodes' numbers under one name.

The same pass computes each looked-for category's ground truth once: its goal
region -- the fake's stand-in for Habitat's goal view points -- and the
geodesic field to it. A finished episode is measured against it in Habitat's
shape: success is STOP with the agent in the region, a geodesic distance to
goal of exactly 0, and ``l`` and ``d0`` are both the geodesic from the start.

Python 3.8 syntax, numpy.
"""
from __future__ import annotations

import collections.abc
import math
from dataclasses import dataclass
from typing import Dict, Sequence, Tuple

import numpy as np

from sparx_agency.core.planning.objnav.action_converter.action_choice import (
    pitch_within_limits,
)
from sparx_agency.core.planning.objnav.errors import ObjNavError
from sparx_agency.core.planning.objnav.types.actions import DiscreteActionSpec
from sparx_agency.core.planning.objnav.types.measurement import (
    NATIVE_DISTANCE_TO_GOAL,
    NATIVE_SUCCESS,
    TERMINATION_STEP_LIMIT,
    TERMINATION_STOP,
    EpisodeMeasurement,
)
from sparx_agency.core.planning.objnav.types.pose import AgentPose
from sparx_agency.tasks.planning.objnav_benchmark.fake_env.geodesics import (
    geodesic_field,
)
from sparx_agency.tasks.planning.objnav_benchmark.fake_env.raster import goal_mask
from sparx_agency.tasks.planning.objnav_benchmark.fake_env.world import GridWorld


@dataclass(frozen=True)
class FakeEpisodeSpec:
    """One episode of the fake: where the agent starts, and what it looks for.

    Attributes:
        episode_id: Unique within the environment.
        scene_id: The scene label results are grouped under.
        target_category: An object category of the world.
        start: The start pose, on the fake's one floor (``z == 0``).

    Raises:
        ObjNavError: On a blank identifier or a start that is not an
            :class:`AgentPose`.
    """

    episode_id: str
    scene_id: str
    target_category: str
    start: AgentPose

    def __post_init__(self) -> None:
        for name in ("episode_id", "scene_id", "target_category"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value:
                raise ObjNavError("FakeEpisodeSpec.%s must be a non-empty "
                                  "string, got %r" % (name, value))
        if not isinstance(self.start, AgentPose):
            raise ObjNavError("FakeEpisodeSpec.start must be an AgentPose, "
                              "got %r" % (self.start,))


class EpisodeTable:
    """The fake's episodes by id, each checked against its building, and the ground truth each is measured by.

    Categories are prepared in the order the episodes name them, and each
    episode is checked right after its own, so the first bad episode is the
    one reported.

    Args:
        world: The building.
        episodes: The episodes, in order; ids unique.
        navigable: Read-only mask of the cells the agent can stand on.
        action_spec: The action geometry, whose pitch limits a start must
            respect.
        success_distance_m: The goal region's radius around an object.
        agent_radius_m: The agent's radius.

    Raises:
        TypeError: If an episode is not a :class:`FakeEpisodeSpec`.
        ObjNavError: On no episodes or a repeated id; a target category the
            world lacks, or one with no navigable cell near it; a start that
            is off the floor, not navigable, pitched past the spec's limits,
            or cut off from its goal.
    """

    def __init__(self, world: GridWorld, episodes: Sequence[FakeEpisodeSpec],
                 *, navigable: np.ndarray, action_spec: DiscreteActionSpec,
                 success_distance_m: float, agent_radius_m: float) -> None:
        self._world = world
        self._navigable = navigable
        self._action_spec = action_spec
        self._success_distance = success_distance_m
        self._agent_radius = agent_radius_m
        self._specs = _episode_table(episodes)
        self._goals: Dict[str, np.ndarray] = {}
        self._fields: Dict[str, np.ndarray] = {}
        for spec in self._specs.values():
            self._prepare_goal(spec.target_category)
            self._check_start(spec)

    def ids(self) -> Tuple[str, ...]:
        """Every episode id, in the order given."""
        return tuple(self._specs)

    def spec(self, episode_id: str) -> FakeEpisodeSpec:
        """The episode ``episode_id``.

        Raises:
            KeyError: If there is no such episode.
        """
        spec = (self._specs.get(episode_id) if isinstance(episode_id, str)
                else None)
        if spec is None:
            raise KeyError("the fake env serves no episode %r; it serves %s"
                           % (episode_id, ", ".join(map(repr, self._specs))))
        return spec

    def goal_region(self, category: str) -> np.ndarray:
        """Read-only mask of ``category``'s goal region.

        Raises:
            ObjNavError: If no episode looks for ``category``.
        """
        goal = self._goals.get(category) if isinstance(category, str) else None
        if goal is None:
            raise ObjNavError("no episode looks for %r; the goal regions are "
                              "of %s" % (category, ", ".join(map(repr, self._goals))))
        return goal

    def measurement(self, spec: FakeEpisodeSpec, final_pose: AgentPose, *,
                    stop_called: bool, steps: int,
                    path_length_m: float) -> EpisodeMeasurement:
        """What a finished episode measured, read off its category's geodesic field.

        Args:
            spec: The finished episode, one of this table's.
            final_pose: Where the agent ended.
            stop_called: Whether STOP ended it.
            steps: Actions taken, STOP included.
            path_length_m: Distance travelled.

        Returns:
            Success only for STOP inside the goal region (``dT == 0``);
            ``l == d0``, the geodesic from the start; native success and
            distance to goal; the start and final cells in ``info``.
        """
        field = self._fields[spec.target_category]
        start_cell = self._world.cell_of(spec.start.x, spec.start.y)
        final_cell = self._world.cell_of(final_pose.x, final_pose.y)
        start = float(field[start_cell])
        final = float(field[final_cell])
        success = stop_called and final == 0.0
        return EpisodeMeasurement(
            success=success, stop_called=stop_called,
            termination=(TERMINATION_STOP if stop_called
                         else TERMINATION_STEP_LIMIT),
            steps=steps, shortest_path_m=start,
            start_distance_to_goal_m=start, final_distance_to_goal_m=final,
            path_length_m=path_length_m,
            native_metrics={NATIVE_SUCCESS: 1.0 if success else 0.0,
                            NATIVE_DISTANCE_TO_GOAL: final},
            info={"start_cell": list(start_cell),
                  "final_cell": list(final_cell)})

    def __len__(self) -> int:
        return len(self._specs)

    def _prepare_goal(self, category: str) -> None:
        """The goal region of ``category`` and its geodesic field, once per category."""
        if category in self._goals:
            return
        if category not in self._world.categories:
            raise ObjNavError(
                "an episode looks for %r but the world has only %s"
                % (category, ", ".join(map(repr, self._world.categories))))
        goal = goal_mask(self._world, category, self._success_distance,
                         self._agent_radius)
        if not goal.any():
            raise ObjNavError(
                "no navigable cell lies within %.2f m of any %r, so no "
                "episode looking for it can succeed; move the object away "
                "from the walls" % (self._success_distance, category))
        goal.setflags(write=False)
        field = geodesic_field(self._world, goal, self._navigable)
        field.setflags(write=False)
        self._goals[category] = goal
        self._fields[category] = field

    def _check_start(self, spec: FakeEpisodeSpec) -> None:
        """Refuse a start the agent could not stand at, or from which the goal is unreachable."""
        start = spec.start
        if start.z != 0.0:
            raise ObjNavError("episode %r starts at z=%r; the fake has one "
                              "floor, at z=0" % (spec.episode_id, start.z))
        row, col = self._world.cell_of(start.x, start.y)
        if not (self._world.in_bounds(row, col) and self._navigable[row, col]):
            raise ObjNavError(
                "episode %r starts at (%.3f, %.3f), cell %r, which is not "
                "navigable for an agent of radius %.2f m"
                % (spec.episode_id, start.x, start.y, (row, col),
                   self._agent_radius))
        limits = self._action_spec
        if not pitch_within_limits(start.camera_pitch, limits.min_pitch_rad,
                                   limits.max_pitch_rad):
            raise ObjNavError(
                "episode %r starts with camera pitch %r rad, outside the "
                "action spec's limits" % (spec.episode_id, start.camera_pitch))
        if math.isinf(self._fields[spec.target_category][row, col]):
            raise ObjNavError(
                "episode %r cannot reach any %r from its start: every "
                "benchmark episode is solvable, so a fake one must be too"
                % (spec.episode_id, spec.target_category))


def _episode_table(episodes) -> Dict[str, FakeEpisodeSpec]:
    """The episodes by id, in order, once they are known to be unique."""
    if isinstance(episodes, (str, bytes)) or not isinstance(
            episodes, collections.abc.Sequence):
        raise ObjNavError("episodes must be a sequence of FakeEpisodeSpec, "
                          "got %r" % (episodes,))
    table: Dict[str, FakeEpisodeSpec] = {}
    for spec in episodes:
        if not isinstance(spec, FakeEpisodeSpec):
            raise TypeError("every episode must be a FakeEpisodeSpec, got %r"
                            % (spec,))
        if spec.episode_id in table:
            raise ObjNavError("episode id %r appears twice; ids name results "
                              "rows, so each must be unique" % (spec.episode_id,))
        table[spec.episode_id] = spec
    if not table:
        raise ObjNavError("the fake env needs at least one episode")
    return table
