"""Explicit cross-floor development protocol, separate from published Gibson val."""
from __future__ import annotations

from dataclasses import dataclass, replace
from functools import lru_cache
import numpy as np

from sparx_agency.core.planning.objnav.types.actions import ALL_ACTIONS, DiscreteActionSpec
from sparx_agency.tasks.planning.objnav_benchmark_runtime.gibson.development_dataset import DevelopmentDataset, DevelopmentEnv
from sparx_agency.tasks.planning.objnav_benchmark_runtime.gibson.multifloor_distance import MultiFloorDistance
from sparx_agency.tasks.planning.objnav_benchmark_runtime.gibson.protocol import GibsonProtocol

MULTIFLOOR_SCHEMA = "sparx-gibson-multistory-development/2"
MULTIFLOOR_SPLIT = "multistory-train-development"


@dataclass(frozen=True)
class MultiFloorProtocol(GibsonProtocol):
    protocol_id: str = MULTIFLOOR_SCHEMA
    split: str = MULTIFLOOR_SPLIT
    path_length_dimension: str = "3d"
    goal_definition: str = "annotated reference-floor category dilation; incomplete full-building semantics"
    distance_definition: str = "3D navmesh geodesic to 0.20m-spaced success-region samples; exact height-gated region membership"
    camera_tilt_enabled: bool = True
    max_native_vertical_m: float = 0.60

    def actions(self):
        return DiscreteActionSpec(forward_step_m=self.forward_step_m, turn_angle_deg=self.turn_angle_deg,
                                  tilt_angle_deg=30.0, min_pitch_deg=-30.0, max_pitch_deg=60.0, actions=ALL_ACTIONS)

    def kinematics(self):
        """Keep all checks; bound native stair geometry independently of policy.

        Original-navmesh audit: 240,000 moves, maximum 0.51932 m vertical
        (Coffeen). This is a simulator contract, NOT a traversability threshold.
        The policy's observed maximum tread step remains 0.24 m.
        """
        return replace(super().kinematics(), climb_m=self.max_native_vertical_m)


MULTIFLOOR_PROTOCOL = MultiFloorProtocol()


class MultiFloorDataset(DevelopmentDataset):
    schema, split = MULTIFLOOR_SCHEMA, MULTIFLOOR_SPLIT

    def __init__(self, manifest, scene=None):
        super().__init__(manifest, scene)
        generation = self.definition["generation"]
        if generation.get("policy_feedback_used") is not False:
            raise ValueError("Cross-floor generation must disclose policy independence")
        for episode in self.episodes.values():
            key = str(episode.category_index)
            region = self.definition["goal_regions"][episode.scene][key]
            points = np.asarray(region["points"], dtype=float)
            if points.ndim != 2 or points.shape[1] != 3 or not len(points) or not np.isfinite(points).all():
                raise ValueError("Invalid frozen 3D goal region")
            if abs(float(episode.start_position[1]) - region["height_m"]) < 1.5:
                raise ValueError("A cross-floor episode must start on a distinct storey")
            if int(region["floor_id"]) != episode.floor_id:
                raise ValueError("Goal annotation floor does not match episode")

    def manifest(self):
        result = super().manifest()
        result.update(semantic_coverage="reference floor only; not complete full-building ObjectNav ground truth",
                      cross_floor_required=True, published_benchmark_comparable=False,
                      scene_audits=self.definition["generation"]["scene_audits"])
        return result


class MultiFloorEnv(DevelopmentEnv):
    name = "habitat/gibson-multistory-development"

    def __init__(self, dataset, seed=0, gpu_device=0, simulator=None):
        super().__init__(dataset, seed, gpu_device, simulator, protocol=MULTIFLOOR_PROTOCOL)
        self._pathfinders = {}

    @lru_cache(maxsize=12)
    def _field(self, scene, floor_id, category_index):
        import habitat_sim
        if scene not in self._pathfinders:
            pathfinder = habitat_sim.PathFinder()
            if not pathfinder.load_nav_mesh(str(self._dataset.scenes_dir / (scene + ".navmesh"))):
                raise ValueError("Cannot load original evaluator navmesh")
            self._pathfinders[scene] = pathfinder
        row = next(e for e in self._dataset.episodes.values()
                   if (e.scene, e.floor_id, e.category_index) == (scene, floor_id, category_index))
        semantic, origin = self._dataset.floor(row)
        region = self._dataset.definition["goal_regions"][scene][str(category_index)]
        return MultiFloorDistance(self._pathfinders[scene], region["points"], semantic, origin,
                                  category_index, region["height_m"])

    def reset(self, episode_id):
        result = super().reset(episode_id)
        self._initial_height = result[1].pose.z
        self._minimum_height = self._maximum_height = self._initial_height
        self._cross_floor_action = None
        return result

    def step(self, action):
        obs = super().step(action)
        self._minimum_height = min(self._minimum_height, obs.pose.z)
        self._maximum_height = max(self._maximum_height, obs.pose.z)
        if self._cross_floor_action is None and abs(obs.pose.z - self._initial_height) >= 1.5:
            self._cross_floor_action = obs.step
        return obs

    def evaluation_diagnostics(self):
        result = super().evaluation_diagnostics()
        result.update(vertical_range_m=self._maximum_height - self._minimum_height,
                      first_cross_floor_action=self._cross_floor_action,
                      final_height_m=self._observation.pose.z,
                      protocol_id=self.protocol.protocol_id)
        return result

    def close(self):
        super().close()
        self._pathfinders.clear()

