"""MP3D episodes and habitat-lab's ObjectNav metrics behind the shared contract."""
from __future__ import annotations

from collections import OrderedDict
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
    HabitatRGBDSimulator, habitat_position)
from sparx_agency.tasks.planning.objnav_benchmark_runtime.mp3d.distance import ViewPointDistance
from sparx_agency.tasks.planning.objnav_benchmark_runtime.mp3d.protocol import PROTOCOL

#: How far our recomputed start distance may differ from the publisher's own
#: ``info.geodesic_distance`` before it is worth a person's attention. The
#: released value was computed on the same navmesh by the dataset generator;
#: a large gap means a different navmesh, a different frame, or the wrong
#: goals, so it is reported per episode rather than silently averaged away.
PUBLISHED_GEODESIC_TOLERANCE_M = 0.05


class MP3DEnv(ObjNavEnv):
    """Habitat ObjectNav MP3D v1, scored the way habitat-lab scores it.

    Success is STOP within :attr:`MP3DProtocol.success_distance_m` of a
    published goal view point; ``l`` and ``d0`` are the geodesic distance to
    the nearest view point at reset; ``p`` is the 3-D path actually travelled
    from the reset pose, with a zero initial accumulator. Goal positions,
    view points and distances reach the scorer and the recorder only.
    """

    name = "habitat/mp3d-objectnav-v1"

    def __init__(self, dataset, seed=0, gpu_device=0, simulator=None):
        self._dataset, self._seed = dataset, seed
        self._camera, self._actions = PROTOCOL.camera(), PROTOCOL.actions()
        self._simulator = simulator if simulator is not None else HabitatRGBDSimulator(
            self._camera, self._actions, PROTOCOL.agent_radius_m, PROTOCOL.allow_sliding,
            gpu_device, height_m=PROTOCOL.agent_height_m,
            navmesh=PROTOCOL.navmesh_source)
        self._episode = None
        self._distance = None
        self._steps, self._stopped = 0, False
        self.published_geodesic_gaps = {}
        self.unreachable_start_ids = []

    def episode_ids(self):
        return tuple(self._dataset.episodes)

    def validate_starts(self, episode_ids=None):
        """Preflight the exact selection on the navmesh the run will score on.

        Catches before the run what would otherwise be a mid-run crash or a
        quietly wrong score: a start from which no published view point is
        reachable (an infinite ``l`` leaves SPL undefined), and a start
        distance that disagrees with the publisher's own
        ``info.geodesic_distance``.

        It loads each selected scene through the simulator, exactly once, so
        the distances come from the same navmesh the episodes are scored on --
        a standalone ``PathFinder`` would read the published file and answer
        for a navmesh the run never uses. That means this needs the rendering
        GPU, and it is why the answer is worth having.
        """
        ids = self.episode_ids() if episode_ids is None else tuple(episode_ids)
        self.published_geodesic_gaps, self.unreachable_start_ids = {}, []
        by_scene = OrderedDict()
        for episode_id in ids:
            row = self._dataset.episodes[episode_id]
            by_scene.setdefault(row.scene, []).append(row)
        provenance = {}
        for scene, rows in by_scene.items():
            mesh, navmesh = self._dataset.scene_paths(rows[0])
            self._simulator.reset(mesh, navmesh, rows[0].start_position,
                                  rows[0].start_rotation_wxyz(), 0)
            provenance[scene] = self._simulator.navmesh_provenance()
            fields = {}
            for row in rows:
                if row.goals_key not in fields:
                    fields[row.goals_key] = ViewPointDistance(
                        self._simulator.pathfinder, self._dataset.view_points(row))
                start = fields[row.goals_key].distance(row.start_position)
                if not math.isfinite(start):
                    self.unreachable_start_ids.append(row.episode_id)
                    continue
                published = row.published_geodesic_m
                if (math.isfinite(published)
                        and abs(published - start) > PUBLISHED_GEODESIC_TOLERANCE_M):
                    self.published_geodesic_gaps[row.episode_id] = round(start - published, 4)
        return {"checked": len(ids), "navmesh": provenance,
                "unreachable_starts": list(self.unreachable_start_ids),
                "published_geodesic_gaps": dict(self.published_geodesic_gaps)}

    def reset(self, episode_id):
        row = self._dataset.episodes[episode_id]
        mesh, navmesh = self._dataset.scene_paths(row)
        tag = "%d/%s" % (self._seed, row.episode_id)
        seed = int.from_bytes(hashlib.sha256(tag.encode()).digest()[:4], "big")
        frame = self._simulator.reset(mesh, navmesh, row.start_position,
                                      row.start_rotation_wxyz(), seed)
        self._distance = ViewPointDistance(self._simulator.pathfinder,
                                           self._dataset.view_points(row))
        self._steps, self._stopped = 0, False
        self._path = PROTOCOL.path_length_epsilon_m
        self._episode = ObjNavEpisode(
            row.episode_id, row.scene, PROTOCOL.benchmark, PROTOCOL.split,
            row.category, self._camera, self._actions, PROTOCOL.max_steps,
            metadata={"published_episode_id": row.published_episode_id})
        self._observation = self._observation_from(frame)
        position = habitat_position(self._observation.pose)
        if any(abs(a - b) > 1e-4 for a, b in zip(position, row.start_position)):
            raise EnvContractError("Simulator changed the published episode start")
        self._start_dtg = self._distance.distance(position)
        if not math.isfinite(self._start_dtg):
            raise EnvContractError(
                "No published view point of %r is reachable from the start of "
                "%s; SPL is undefined for it. Audit the selection with "
                "validate_starts() rather than scoring it."
                % (row.category, episode_id))
        self._shortest = self._start_dtg
        self._published_gap = (self._start_dtg - row.published_geodesic_m
                               if math.isfinite(row.published_geodesic_m) else None)
        return self._episode, self._observation

    def _observation_from(self, frame):
        rgb, depth, pose = frame
        return ObjNavObservation(rgb, depth, pose, self._camera,
                                 self._episode.target_category, self._steps)

    def step(self, action):
        if self._episode is None or self.episode_over or not self._actions.allows(action):
            raise EnvContractError("Episode not running, or action not allowed")
        before = self._observation.pose
        frame = self._simulator.step(action)
        self._steps += 1
        self._stopped = action == DiscreteAction.STOP
        self._observation = self._observation_from(frame)
        after = self._observation.pose
        # habitat-lab's SPL accumulates the 3-D chord between successive agent
        # positions, from the reset pose, for every action including STOP.
        self._path += math.sqrt((after.x - before.x) ** 2 + (after.y - before.y) ** 2
                                + (after.z - before.z) ** 2)
        return self._observation

    @property
    def episode_over(self):
        return self._episode is not None and (self._stopped or self._steps >= PROTOCOL.max_steps)

    def _distance_to_goal(self):
        return self._distance.distance(habitat_position(self._observation.pose))

    def evaluation_diagnostics(self):
        """Recorder/evaluator-only values, never policy observations."""
        if self._episode is None:
            raise EnvContractError("No episode has been reset")
        return {"distance_to_goal_m": self._distance_to_goal(),
                "path_length_m": self._path, "shortest_path_m": self._shortest,
                "view_points": self._distance.view_point_count}

    def measure(self):
        if not self.episode_over:
            raise EnvContractError("measure() is only available after episode end")
        dtg = self._distance_to_goal()
        success = bool(self._stopped and dtg < PROTOCOL.success_distance_m)
        ratio = self._shortest / max(self._shortest, self._path) if max(self._shortest, self._path) else 1.0
        if math.isinf(dtg):
            soft_success = 0.0
        elif self._start_dtg == 0.0:
            # A start that already is a view point: the shared scoring reads
            # this corner as full progress, and the native metric must agree
            # or the cross-check refuses the episode.
            soft_success = 1.0 if dtg == 0.0 else 0.0
        else:
            soft_success = max(0.0, 1.0 - dtg / self._start_dtg)
        info = {"published_geodesic_m": self._dataset.episodes[self._episode.episode_id].published_geodesic_m,
                "published_geodesic_gap_m": self._published_gap,
                "view_points": self._distance.view_point_count}
        return EpisodeMeasurement(
            success=success, stop_called=self._stopped,
            termination=TERMINATION_STOP if self._stopped else TERMINATION_STEP_LIMIT,
            steps=self._steps, shortest_path_m=self._shortest,
            start_distance_to_goal_m=self._start_dtg, final_distance_to_goal_m=dtg,
            path_length_m=self._path,
            native_metrics={"success": float(success), "spl": float(success) * ratio,
                            "soft_spl": soft_success * ratio, "distance_to_goal": dtg},
            info=info)

    def close(self):
        self._simulator.close()
