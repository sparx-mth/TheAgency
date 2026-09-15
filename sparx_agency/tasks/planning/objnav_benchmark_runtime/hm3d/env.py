"""HM3D episodes and habitat-lab's own ObjectNav metrics behind ObjNavEnv.

The measurements here are habitat-lab's, reimplemented against habitat-sim
because habitat-lab itself is not installed (see ``geodesic.py`` for the
distance, which is the one quantity all four metrics are built from):

===========  =========================================================
``l``        ``DistanceToGoal`` at reset, measured live -- habitat-lab's
             SPL uses exactly this as ``_start_end_episode_distance`` and
             never reads the row's ``info.geodesic_distance``. That
             published number is the *same* quantity though, so the two
             agreeing is a check on our navmesh and their disagreeing is
             a defect; see ``geodesic_agreement()``.
``d0``       the same number; in Habitat ObjectNav ``d0 == l``.
``dT``       ``DistanceToGoal`` after the last action.
``p``        the sum of 3-D displacements between successive agent
             positions from reset on, as ``SPL._agent_episode_distance``
             accumulates it.
success      ``is_stop_called and dT < success_distance`` -- strictly
             less, and STOP is required.
===========  =========================================================

``native_metrics`` carries habitat-lab's own arithmetic for those quantities so
the harness can cross-check our scoring against it. Where habitat-lab's formula
is undefined (it divides by ``d0`` without guarding it), the key is omitted
rather than filled with a convention habitat-lab does not have.

Nothing privileged reaches the agent: ``ObjNavEpisode.metadata`` stays empty.
"""
from __future__ import annotations

import hashlib
import math

from sparx_agency.core.planning.objnav.errors import EnvContractError
from sparx_agency.core.planning.objnav.interfaces.env import ObjNavEnv
from sparx_agency.core.planning.objnav.types.actions import DiscreteAction
from sparx_agency.core.planning.objnav.types.episode import ObjNavEpisode
from sparx_agency.core.planning.objnav.types.measurement import (
    EpisodeMeasurement, TERMINATION_STEP_LIMIT, TERMINATION_STOP)
from sparx_agency.core.planning.objnav.types.observation import ObjNavObservation
from sparx_agency.tasks.planning.objnav_benchmark_runtime.habitat.simulator import (
    HabitatRGBDSimulator)
from sparx_agency.tasks.planning.objnav_benchmark_runtime.hm3d.geodesic import (
    ViewPointDistance, view_point_positions)


class HM3DEnv(ObjNavEnv):
    """One HM3D ObjectNav split, executed and measured under one protocol.

    Args:
        dataset: A loaded :class:`~...hm3d.dataset.HM3DDataset`.
        protocol: The :class:`~...hm3d.protocol.HM3DProtocol` it was loaded for.
        seed: Run seed. Each episode's simulator seed is derived from it and
            the episode identity, so an episode is reproducible on its own.
        gpu_device: Rendering device index.
        simulator: An injected RGB-D bridge, for tests. Defaults to the shared
            :class:`HabitatRGBDSimulator` built from the protocol.
        excluded_episode_ids: Episodes deliberately dropped because no goal
            view point is reachable from their start. Passing them is an
            explicit, recorded decision; ``reset()`` on one still raises.
        distance_factory: ``(pathfinder, view_points) -> distance provider``.
            Defaults to :class:`ViewPointDistance`. The seam exists so the
            success rule, the path accounting and the measurement arithmetic
            can be tested without habitat-sim, on geometry chosen by the test
            rather than by a 3 GB download.
    """

    #: Identifier of the environment. The version is part of it: rows from the
    #: two HM3D datasets are never mixed, and a results directory should say
    #: which one produced it without opening ``run.json``.
    name = "habitat/hm3d-objectnav"

    def __init__(self, dataset, protocol, seed=0, gpu_device=0, simulator=None,
                 excluded_episode_ids=(), distance_factory=None):
        self.name = "habitat/hm3d-objectnav-%s" % protocol.dataset_version
        self._distance_factory = distance_factory or ViewPointDistance
        self._dataset = dataset
        self._protocol = protocol
        self._seed = seed
        self._camera = protocol.camera()
        self._actions = protocol.actions()
        self._simulator = simulator if simulator is not None else HabitatRGBDSimulator(
            self._camera, self._actions, protocol.agent_radius_m,
            protocol.allow_sliding, gpu_device, height_m=protocol.agent_height_m,
            navmesh=protocol.navmesh)
        self._distances = {}
        self._episode = None
        self._observation = None
        self._steps = 0
        self._stopped = False
        self.excluded_episode_ids = tuple(excluded_episode_ids)
        self.unreachable_episode_ids = ()
        self.navmesh_by_scene = {}
        self.published_geodesic_agreement = []

    def episode_ids(self):
        """Every published identity except the explicitly excluded ones."""
        excluded = set(self.excluded_episode_ids)
        return tuple(i for i in self._dataset.episode_ids() if i not in excluded)

    def _distance_field(self, episode):
        """The (scene, category) distance, built once per goal set per scene."""
        key = episode.goals_key
        if key not in self._distances:
            points = view_point_positions(self._dataset.goals_for(episode))
            self._distances[key] = self._distance_factory(
                self._simulator.pathfinder, points)
        return self._distances[key]

    def validate_starts(self, episode_ids=None, *, progress=None):
        """Measure ``l`` for every selected episode, up front, one scene at a time.

        The harness refuses a non-finite shortest path, and a run would stop at
        that episode on every resume, so an unreachable start must be found
        before the run rather than during it. This loads each scene once, which
        is the expensive part of preflight and the only way to know.

        Args:
            episode_ids: The exact selection to check; all of them by default.
            progress: Optional ``callable(scene_key, done, total)``.

        Returns:
            ``{episode_id: l}`` for every checked episode, ``inf`` included.
            ``self.unreachable_episode_ids`` holds the non-finite ones.
        """
        ids = self.episode_ids() if episode_ids is None else tuple(episode_ids)
        episodes = [self._dataset.episodes[i] for i in ids]
        by_scene = {}
        for episode in episodes:
            by_scene.setdefault(episode.scene_key, []).append(episode)
        distances, unreachable = {}, []
        self.navmesh_by_scene = {}
        self.published_geodesic_agreement = []
        for done, (scene_key, rows) in enumerate(sorted(by_scene.items()), start=1):
            glb, navmesh = self._dataset.scene_assets(rows[0])
            self._simulator.reset(glb, navmesh, rows[0].start_position,
                                  rows[0].start_rotation_wxyz, self._seed)
            self._distances = {}
            describe = getattr(self._simulator, "navmesh_provenance", None)
            if describe is not None:
                self.navmesh_by_scene[scene_key] = describe()
            for episode in rows:
                value = self._distance_field(episode).distance(episode.start_position)
                distances[episode.episode_id] = value
                if not math.isfinite(value):
                    unreachable.append(episode.episode_id)
                elif episode.published_geodesic_m is not None:
                    self.published_geodesic_agreement.append(
                        abs(value - episode.published_geodesic_m))
            if progress is not None:
                progress(scene_key, done, len(by_scene))
        self.unreachable_episode_ids = tuple(unreachable)
        self._episode = None
        self._observation = None
        return distances

    def geodesic_agreement(self):
        """How closely our measured ``l`` reproduces the publisher's own number.

        The publisher recorded ``info.geodesic_distance`` for every episode when
        it generated them, and it is the same quantity as ``l``: the geodesic
        from the start to the nearest goal view point. So if our navmesh, our
        frame handling and our view-point extraction are all right, our ``l``
        reproduces it -- and a systematic disagreement means one of the three is
        wrong, most likely the navmesh, since the shipped mesh and the
        agent-recomputed mesh differ by several percent. This is the single
        most informative preflight number there is. It is returned rather than
        asserted here because only the caller knows whether a deviation was
        deliberate; ``run.py`` turns a bad agreement into a preflight issue.

        Returns:
            ``None`` before :meth:`validate_starts`, otherwise the count, the
            maximum and the median absolute difference in metres.
        """
        deltas = sorted(getattr(self, "published_geodesic_agreement", []))
        if not deltas:
            return None
        return {"episodes": len(deltas), "max_abs_m": deltas[-1],
                "median_abs_m": deltas[len(deltas) // 2],
                "within_1mm": sum(d <= 1e-3 for d in deltas)}

    def reset(self, episode_id):
        if episode_id in set(self.excluded_episode_ids):
            raise EnvContractError(
                "Episode %r was excluded from this run; it must not be served"
                % (episode_id,))
        row = self._dataset.episodes[episode_id]
        glb, navmesh = self._dataset.scene_assets(row)
        tag = "%d/%s" % (self._seed, episode_id)
        seed = int.from_bytes(hashlib.sha256(tag.encode()).digest()[:4], "big")
        reloading = self._simulator_scene() != str(glb)
        frame = self._simulator.reset(glb, navmesh, row.start_position,
                                      row.start_rotation_wxyz, seed)
        if reloading:
            self._distances = {}
        self._row = row
        self._steps, self._stopped = 0, False
        self._path_m = self._protocol.path_length_epsilon_m
        self._episode = ObjNavEpisode(
            episode_id, row.scene_key, self._protocol.benchmark, self._protocol.split,
            row.category, self._camera, self._actions, self._protocol.max_steps)
        self._observation = self._observation_from(frame)
        if any(abs(a - b) > 1e-4 for a, b in
               zip(self._habitat_position(), row.start_position)):
            raise EnvContractError(
                "Simulator changed the published start of %s; the navmesh does "
                "not match the episodes" % episode_id)
        self._distance = self._distance_field(row)
        self._start_dtg = self._distance.distance(row.start_position)
        if not math.isfinite(self._start_dtg):
            raise EnvContractError(
                "No goal view point is reachable from the start of %s, so this "
                "benchmark cannot measure its shortest path. Run the start "
                "validation and exclude it explicitly." % episode_id)
        return self._episode, self._observation

    def _simulator_scene(self):
        return getattr(self._simulator, "_scene", None)

    def _observation_from(self, frame):
        rgb, depth, pose = frame
        return ObjNavObservation(rgb, depth, pose, self._camera,
                                 self._episode.target_category, self._steps)

    def _habitat_position(self):
        """Our ENU pose back in Habitat's +Y-up frame, for the pathfinder."""
        pose = self._observation.pose
        return -pose.y, pose.z, -pose.x

    def step(self, action):
        if self._episode is None or self.episode_over or not self._actions.allows(action):
            raise EnvContractError("Episode not running, or action not allowed")
        before = self._observation.pose
        frame = self._simulator.step(action)
        self._steps += 1
        self._stopped = action == DiscreteAction.STOP
        self._observation = self._observation_from(frame)
        after = self._observation.pose
        self._path_m += math.sqrt((after.x - before.x) ** 2 + (after.y - before.y) ** 2
                                  + (after.z - before.z) ** 2)
        return self._observation

    @property
    def episode_over(self):
        return self._episode is not None and (
            self._stopped or self._steps >= self._protocol.max_steps)

    def evaluation_diagnostics(self):
        """Recorder/evaluator-only values. Never part of an observation."""
        if self._episode is None:
            raise EnvContractError("No episode has been reset")
        return {"distance_to_goal_m": self._distance.distance(self._habitat_position()),
                "path_length_m": self._path_m,
                "shortest_path_m": self._start_dtg,
                "success_distance_m": self._protocol.success_distance_m,
                "published_geodesic_m": self._row.published_geodesic_m,
                "view_points": self._distance.view_point_count}

    def measure(self):
        if not self.episode_over:
            raise EnvContractError("measure() is only available after episode end")
        dtg = self._distance.distance(self._habitat_position())
        success = bool(self._stopped and dtg < self._protocol.success_distance_m)
        l, p, d0 = self._start_dtg, self._path_m, self._start_dtg
        native = {"success": float(success)}
        if max(l, p) > 0:
            ratio = l / max(l, p)
            native["spl"] = float(success) * ratio
            if d0 > 0 and math.isfinite(dtg):
                native["soft_spl"] = max(0.0, 1.0 - dtg / d0) * ratio
        native["distance_to_goal"] = dtg
        return EpisodeMeasurement(
            success=success, stop_called=self._stopped,
            termination=TERMINATION_STOP if self._stopped else TERMINATION_STEP_LIMIT,
            steps=self._steps, shortest_path_m=l, start_distance_to_goal_m=d0,
            final_distance_to_goal_m=dtg, path_length_m=p, native_metrics=native,
            info={"published_geodesic_m": self._row.published_geodesic_m,
                  "source_episode_id": self._row.source_episode_id,
                  "success_distance_m": self._protocol.success_distance_m,
                  "view_points": self._distance.view_point_count})

    def close(self):
        self._distances = {}
        self._simulator.close()
