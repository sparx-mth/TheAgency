"""A fake :class:`ObjNavEnv` on a :class:`GridWorld`: the whole ObjectNav loop, without a simulator.

A **test rig, not a benchmark.** It honours the environment contract the
benchmark adapters implement -- the step budget, STOP semantics, every action
counted, quantities measured rather than scores -- so the runner, the agent
and the scoring can be exercised end to end in seconds, and an adapter author
has a reference for what "correct" looks like. Its numbers describe a toy
building and a privileged oracle, never a method.

What it models, and what it deliberately does not, each beside its own code:

* **Depth is geometry, not appearance** (:mod:`.rendering`). Every frame is
  ray cast, to the pixel; the RGB is a constant grey that nothing reads.
* **Blocked means blocked** (:mod:`.motion`). A MOVE_FORWARD off the
  navigable floor does not move the agent at all -- AI2-THOR's atomic
  failure, no sliding -- and the step still counts.
* **Success is Habitat's shape** (:mod:`.episodes`). STOP with the agent in
  the goal region, the fake's stand-in for Habitat's goal view points; ``l``
  and ``d0`` are the geodesic from the start. An episode that could never be
  scored honestly is refused at construction.
* **Nothing privileged leaks.** Episodes carry no metadata; the ground truth
  a privileged oracle needs is reachable only through accessors named
  ``privileged_*``.

Its errors are the core ObjectNav family (``ObjNavError``, ``EnvContractError``),
not the harness's: it plays the part of a simulator adapter.

Python 3.8 syntax, numpy.
"""
from __future__ import annotations

import math
from typing import Optional, Sequence, Tuple

import numpy as np

from sparx_agency.core.planning.objnav.errors import (
    EnvContractError,
    ObjNavError,
)
from sparx_agency.core.planning.objnav.interfaces.env import ObjNavEnv
from sparx_agency.core.planning.objnav.types.actions import (
    DiscreteAction,
    DiscreteActionSpec,
)
from sparx_agency.core.planning.objnav.types.camera import CameraSpec
from sparx_agency.core.planning.objnav.types.episode import ObjNavEpisode
from sparx_agency.core.planning.objnav.types.measurement import EpisodeMeasurement
from sparx_agency.core.planning.objnav.types.observation import ObjNavObservation
from sparx_agency.core.planning.objnav.types.pose import AgentPose
from sparx_agency.tasks.planning.objnav_benchmark.checks import (
    is_count,
    is_length,
)
# FakeEpisodeSpec is this environment's input type, and this module its public
# home (smoke.py and the tests import it from here); it is defined beside the
# table that validates it.
from sparx_agency.tasks.planning.objnav_benchmark.fake_env.episodes import (
    EpisodeTable,
    FakeEpisodeSpec,
)
from sparx_agency.tasks.planning.objnav_benchmark.fake_env.motion import move
from sparx_agency.tasks.planning.objnav_benchmark.fake_env.raster import (
    navigable_mask,
)
from sparx_agency.tasks.planning.objnav_benchmark.fake_env.rendering import (
    FrameRenderer,
)
from sparx_agency.tasks.planning.objnav_benchmark.fake_env.world import GridWorld

#: How near an object counts as found, metres: the radius of Habitat
#: ObjectNav's goal view points around an object.
DEFAULT_SUCCESS_DISTANCE_M = 1.0
#: The agent's radius, metres: Habitat's ObjectNav agent.
DEFAULT_AGENT_RADIUS_M = 0.18
#: The step budget: Habitat ObjectNav's.
DEFAULT_MAX_STEPS = 500


def _require(value, kind, name: str) -> None:
    if not isinstance(value, kind):
        raise TypeError("%s must be a %s, got %r" % (name, kind.__name__, value))


class FakeObjNavEnv(ObjNavEnv):
    """Episodes on one grid building, served through the :class:`ObjNavEnv` contract.

    Args:
        world: The building.
        episodes: The episodes it serves, in order; ids unique.
        camera: The agent's camera; ``max_depth_m`` must be finite (the ray
            caster samples up to it) and the mount below the walls' top.
        action_spec: The action geometry and set.
        max_steps: The step budget.
        success_distance_m: The goal region's radius around an object.
        agent_radius_m: The agent's radius, which decides the navigable cells.
        benchmark: The benchmark key every episode reports.
        split: The split every episode reports.

    Raises:
        TypeError: On a wrong type for ``world``, ``camera``, ``action_spec``
            or an episode.
        ObjNavError: On a bad budget, radius, camera or name, or an episode
            the fake could not score honestly (see
            :class:`~.episodes.EpisodeTable`).
    """

    name = "fake"

    def __init__(self, world: GridWorld, episodes: Sequence[FakeEpisodeSpec],
                 *, camera: CameraSpec,
                 action_spec: DiscreteActionSpec = DiscreteActionSpec(),
                 max_steps: int = DEFAULT_MAX_STEPS,
                 success_distance_m: float = DEFAULT_SUCCESS_DISTANCE_M,
                 agent_radius_m: float = DEFAULT_AGENT_RADIUS_M,
                 benchmark: str = "fake", split: str = "test") -> None:
        _require(world, GridWorld, "world")
        _require(camera, CameraSpec, "camera")
        _require(action_spec, DiscreteActionSpec, "action_spec")
        _check_settings(max_steps, success_distance_m, benchmark, split)
        self._renderer = FrameRenderer(world, camera)
        navigable = navigable_mask(world, agent_radius_m)
        navigable.setflags(write=False)
        self._world = world
        self._camera = camera
        self._action_spec = action_spec
        self._max_steps = int(max_steps)
        self._success_distance = float(success_distance_m)
        self._agent_radius = float(agent_radius_m)
        self._benchmark = benchmark
        self._split = split
        self._navigable = navigable
        self._episodes = EpisodeTable(
            world, episodes, navigable=navigable, action_spec=action_spec,
            success_distance_m=self._success_distance,
            agent_radius_m=self._agent_radius)
        self._current: Optional[FakeEpisodeSpec] = None
        self._pose: Optional[AgentPose] = None
        self._steps = 0
        self._stop_called = False
        self._path_length_m = 0.0

    # -- the ObjNavEnv contract -----------------------------------------------

    def episode_ids(self) -> Tuple[str, ...]:
        """Every episode, in the order given."""
        return self._episodes.ids()

    def reset(self, episode_id: str) -> Tuple[ObjNavEpisode, ObjNavObservation]:
        """Start an episode: the agent at its start, no steps, no path, no STOP.

        Raises:
            KeyError: If the fake serves no such episode.
        """
        spec = self._episodes.spec(episode_id)
        self._current = spec
        self._pose = spec.start
        self._steps = 0
        self._stop_called = False
        self._path_length_m = 0.0
        episode = ObjNavEpisode(
            episode_id=spec.episode_id, scene_id=spec.scene_id,
            benchmark=self._benchmark, split=self._split,
            target_category=spec.target_category, camera=self._camera,
            action_spec=self._action_spec, max_steps=self._max_steps)
        return episode, self._observe()

    def step(self, action: DiscreteAction) -> ObjNavObservation:
        """Execute one action; every action counts, STOP included.

        Raises:
            EnvContractError: Before the first reset, after the episode ended,
                or on an action that is not a :class:`DiscreteAction` the
                spec allows.
        """
        if self._current is None:
            raise EnvContractError("step() before reset(); start an episode")
        if self.episode_over:
            raise EnvContractError(
                "episode %r is over (%s after %d steps); reset before stepping"
                % (self._current.episode_id,
                   "STOP" if self._stop_called else "budget spent",
                   self._steps))
        if not isinstance(action, DiscreteAction):
            raise EnvContractError(
                "step() needs a DiscreteAction, got %r -- a bare integer means "
                "a different action to each simulator" % (action,))
        if not self._action_spec.allows(action):
            raise EnvContractError(
                "%s is not in this episode's action set (%s)"
                % (action.name, ", ".join(a.name for a in
                                          self._action_spec.actions)))
        before = self._pose
        after = move(self._world, self._navigable, before, action,
                     self._action_spec)
        if action == DiscreteAction.STOP:
            self._stop_called = True
        self._path_length_m += math.dist(before.position(), after.position())
        self._pose = after
        self._steps += 1
        return self._observe()

    @property
    def episode_over(self) -> bool:
        """True after STOP or once the budget is spent -- and before any reset, when nothing runs."""
        if self._current is None:
            return True
        return self._stop_called or self._steps >= self._max_steps

    def measure(self) -> EpisodeMeasurement:
        """The finished episode's ground truth: geodesic distances, path, success.

        Raises:
            EnvContractError: Before the first reset, or while the episode
                runs.
        """
        if self._current is None or not self.episode_over:
            raise EnvContractError(
                "measure() needs a finished episode; %s"
                % ("none was started" if self._current is None
                   else "episode %r is still running" % self._current.episode_id))
        return self._episodes.measurement(
            self._current, self._pose, stop_called=self._stop_called,
            steps=self._steps, path_length_m=self._path_length_m)

    # -- public, non-privileged -----------------------------------------------

    @property
    def benchmark(self) -> str:
        """The benchmark key every episode reports."""
        return self._benchmark

    @property
    def split(self) -> str:
        """The split every episode reports."""
        return self._split

    @property
    def camera(self) -> CameraSpec:
        """The agent's camera."""
        return self._camera

    @property
    def action_spec(self) -> DiscreteActionSpec:
        """The action geometry and set."""
        return self._action_spec

    @property
    def max_steps(self) -> int:
        """The step budget."""
        return self._max_steps

    # -- privileged: for the oracle and for tests, never for a method ----------

    def privileged_world(self) -> GridWorld:
        """The ground-truth building. Privileged: no real method can see it."""
        return self._world

    def privileged_navigable_mask(self) -> np.ndarray:
        """Read-only mask of the cells the agent can stand on. Privileged."""
        return self._navigable

    def privileged_goal_mask(self, category: str) -> np.ndarray:
        """Read-only mask of ``category``'s goal region. Privileged.

        Raises:
            ObjNavError: If no episode looks for ``category``.
        """
        return self._episodes.goal_region(category)

    @property
    def agent_radius_m(self) -> float:
        """The agent's radius, metres. Privileged (a method knows only its own)."""
        return self._agent_radius

    @property
    def success_distance_m(self) -> float:
        """The goal region's radius around an object, metres. Privileged."""
        return self._success_distance

    # -- internals ------------------------------------------------------------

    def _observe(self) -> ObjNavObservation:
        return self._renderer.observe(self._pose, self._current.target_category,
                                      self._steps)

    def __repr__(self) -> str:
        return ("FakeObjNavEnv(benchmark=%r, split=%r, episodes=%d, world=%r)"
                % (self._benchmark, self._split, len(self._episodes),
                   self._world))


def _check_settings(max_steps, success_distance_m, benchmark, split) -> None:
    """Refuse a budget, a goal radius or a name the fake cannot run with."""
    if not (is_count(max_steps) and max_steps > 0):
        raise ObjNavError("max_steps must be a positive integer, got %r"
                          % (max_steps,))
    if not (is_length(success_distance_m) and success_distance_m > 0):
        raise ObjNavError("success_distance_m must be a positive finite "
                          "number, got %r" % (success_distance_m,))
    for name, value in (("benchmark", benchmark), ("split", split)):
        if not isinstance(value, str) or not value:
            raise ObjNavError("%s must be a non-empty string, got %r"
                              % (name, value))


__all__ = [
    "DEFAULT_AGENT_RADIUS_M",
    "DEFAULT_MAX_STEPS",
    "DEFAULT_SUCCESS_DISTANCE_M",
    "FakeEpisodeSpec",
    "FakeObjNavEnv",
]
