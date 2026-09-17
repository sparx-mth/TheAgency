"""Gibson episodes and SemExp metrics behind the shared environment contract."""
from __future__ import annotations

import hashlib
import math
from functools import lru_cache

from sparx_agency.core.planning.objnav.errors import EnvContractError
from sparx_agency.core.planning.objnav.interfaces.env import ObjNavEnv
from sparx_agency.core.planning.objnav.types.actions import DiscreteAction
from sparx_agency.core.planning.objnav.types.episode import ObjNavEpisode
from sparx_agency.core.planning.objnav.types.measurement import EpisodeMeasurement, TERMINATION_STOP, TERMINATION_STEP_LIMIT
from sparx_agency.core.planning.objnav.types.observation import ObjNavObservation
from sparx_agency.tasks.planning.objnav_benchmark_runtime.gibson.distance import GibsonDistanceField
from sparx_agency.tasks.planning.objnav_benchmark_runtime.gibson.protocol import PROTOCOL
from sparx_agency.tasks.planning.objnav_benchmark_runtime.habitat.simulator import HabitatRGBDSimulator


class GibsonEnv(ObjNavEnv):
    name = "habitat/gibson-semexp-v1.1"

    def __init__(self, dataset, seed=0, gpu_device=0, simulator=None, protocol=PROTOCOL):
        self._dataset, self._seed = dataset, seed
        self.protocol = protocol
        self._camera, self._actions = protocol.camera(), protocol.actions()
        self._simulator = simulator if simulator is not None else HabitatRGBDSimulator(
            self._camera, self._actions, protocol.agent_radius_m, protocol.allow_sliding, gpu_device)
        self._episode = None
        self._steps, self._stopped = 0, False
        self.sentinel_start_ids = []

    def episode_ids(self):
        return tuple(self._dataset.episodes)

    @lru_cache(maxsize=4)
    def _field(self, scene, floor_id, category_index):
        episode = next(e for e in self._dataset.episodes.values()
                       if (e.scene, e.floor_id, e.category_index) == (scene, floor_id, category_index))
        semantic, origin = self._dataset.floor(episode)
        return GibsonDistanceField(semantic, origin, category_index)

    def validate_starts(self, episode_ids=None):
        """Audit exact selected starts; never remove upstream finite-sentinel rows."""
        ids = self.episode_ids() if episode_ids is None else tuple(episode_ids)
        episodes = [self._dataset.episodes[episode_id] for episode_id in ids]
        self.sentinel_start_ids = []
        checked = set()
        for episode in episodes:
            key = (episode.scene, episode.floor_id, episode.category_index)
            if key in checked:
                continue
            field = self._field(*key)
            for other in episodes:
                if (other.scene, other.floor_id, other.category_index) == key:
                    field.distance(other.start_position, start=True)
                    if field.start_uses_sentinel(other.start_position):
                        self.sentinel_start_ids.append(other.episode_id)
            checked.add(key)

    def reset(self, episode_id):
        row = self._dataset.episodes[episode_id]
        self._distance = self._field(row.scene, row.floor_id, row.category_index)
        self._start_dtg = self._distance.distance(row.start_position, start=True)
        self._start_sentinel = self._distance.start_uses_sentinel(row.start_position)
        self._shortest = self._start_dtg + (0.0 if self.protocol.path_length_dimension == "3d" else self.protocol.success_radius_m)
        self._steps, self._stopped = 0, False
        self._path = self.protocol.path_length_epsilon_m
        self._collisions, self._collision_available = 0, True
        self._first_success_action = 0 if self._start_dtg == 0.0 else None
        tag = "%d/%s" % (self._seed, episode_id)
        seed = int.from_bytes(hashlib.sha256(tag.encode()).digest()[:4], "big")
        self._episode = ObjNavEpisode(episode_id, row.scene, self.protocol.benchmark, self.protocol.split,
                                     row.category, self._camera, self._actions, self.protocol.max_steps)
        frame = self._simulator.reset(self._dataset.scenes_dir / (row.scene + ".glb"),
                                      self._dataset.scenes_dir / (row.scene + ".navmesh"),
                                      row.start_position, row.start_rotation, seed)
        self._observation = self._observation_from(frame)
        if any(abs(a - b) > 1e-4 for a, b in zip(self._habitat_position(), row.start_position)):
            raise EnvContractError("Simulator changed the published episode start")
        return self._episode, self._observation

    def _observation_from(self, frame):
        rgb, depth, pose = frame
        return ObjNavObservation(rgb, depth, pose, self._camera, self._episode.target_category, self._steps)

    def _habitat_position(self):
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
        if self.protocol.path_length_dimension == "3d":
            self._path += math.dist((after.x, after.y, after.z), (before.x, before.y, before.z))
        else:
            self._path += math.hypot(after.x - before.x, after.y - before.y)
        collision = getattr(self._simulator, "last_collision", None)
        if collision is None:
            self._collision_available = False
        else:
            self._collisions += int(collision)
        if self._first_success_action is None and self._distance.distance(self._habitat_position()) == 0.0:
            self._first_success_action = self._steps
        return self._observation

    @property
    def episode_over(self):
        return self._episode is not None and (self._stopped or self._steps >= self.protocol.max_steps)

    def evaluation_diagnostics(self):
        """Recorder/evaluator-only values, never policy observations."""
        if self._episode is None:
            raise EnvContractError("No episode has been reset")
        return {"distance_to_goal_m": self._distance.distance(self._habitat_position()),
                "path_length_m": self._path, "shortest_path_m": self._shortest,
                "start_uses_reference_sentinel": self._start_sentinel,
                "collisions": self._collisions if self._collision_available else None,
                "first_success_action": self._first_success_action}

    def measure(self):
        if not self.episode_over:
            raise EnvContractError("measure() is only available after episode end")
        dtg = self._distance.distance(self._habitat_position())
        success = dtg == 0.0
        native_spl = min(float(success) * self._shortest / self._path, 1.0)
        return EpisodeMeasurement(success=success, stop_called=self._stopped,
            termination=TERMINATION_STOP if self._stopped else TERMINATION_STEP_LIMIT,
            steps=self._steps, shortest_path_m=self._shortest, start_distance_to_goal_m=self._start_dtg,
            final_distance_to_goal_m=dtg, path_length_m=self._path,
            native_metrics={"success": float(success), "spl": native_spl, "distance_to_goal": dtg},
            info={"start_uses_reference_sentinel": self._start_sentinel})

    def close(self):
        self._field.cache_clear()
        self._simulator.close()
