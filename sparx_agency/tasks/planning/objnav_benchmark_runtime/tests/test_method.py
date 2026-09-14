"""Headless method regressions; fake services, real geometry and core algorithms."""
from __future__ import annotations

import re

import numpy as np
import pytest

from sparx_agency.core.planning.objnav.labels.datasets.gibson import gibson_label_mapper
from sparx_agency.core.planning.objnav.types.episode import ObjNavEpisode
from sparx_agency.core.planning.objnav.types.observation import ObjNavObservation
from sparx_agency.core.planning.objnav.types.pose import AgentPose
from sparx_agency.tasks.mapping.scene_graph.serve.contract import DetectionWire
from sparx_agency.tasks.planning.objnav_benchmark_runtime.gibson.protocol import PROTOCOL
from sparx_agency.tasks.planning.objnav_benchmark_runtime.methods.rpt_policy import RPTSearchPolicy, RPTSettings


class FakeLLM:
    def chat_json(self, system, user):
        if "Room observed objects" in user:
            return {"label": "living_room", "confidence": 0.8, "reasoning": "objects"}
        return {"rooms": [{"id": int(pid), "score": 50, "why": "unknown room"}
                          for pid in re.findall(r"id=(\d+)", user)]}


class FakeDetector:
    def __init__(self, label="chair", duplicate=False):
        self.label, self.duplicate = label, duplicate

    def detect(self, rgb):
        row = DetectionWire(self.label, 0.9, (280, 200, 360, 280))
        return [row, row] if self.duplicate else [row]


def setup_policy(label="chair", duplicate=False):
    camera = PROTOCOL.camera()
    episode = ObjNavEpisode("synthetic/0", "synthetic", "gibson", "val", "chair",
                            camera, PROTOCOL.actions(), 500)
    policy = RPTSearchPolicy(FakeDetector(label, duplicate), FakeLLM(),
                             RPTSettings(map_size_m=20.0))
    policy.reset(episode, gibson_label_mapper().target_labels("chair"))
    return policy, episode


def observation(episode, step, depth=0.7):
    k = episode.camera.intrinsics
    return ObjNavObservation(np.zeros((k.height, k.width, 3), np.uint8),
                             np.full((k.height, k.width), depth, np.float32),
                             AgentPose(0, 0, 0, 0), episode.camera, "chair", step)


def test_target_confirmation_uses_distinct_frames_and_reset_clears_memory():
    policy, episode = setup_policy(duplicate=True)
    first = policy.plan(observation(episode, 0))
    assert not first.stop
    assert len(policy.landmarks) == 1
    assert policy.landmarks.all_landmarks()[0].count == 1
    assert policy.plan(observation(episode, 1)).stop
    policy.reset(episode, policy.target)
    assert len(policy.landmarks) == 0
    assert not policy.plan(observation(episode, 0)).stop


def test_context_class_cannot_trigger_stop():
    policy, episode = setup_policy("office chair")
    # Exact vocabulary membership, not fuzzy substring/LLM target acceptance.
    policy.plan(observation(episode, 0))
    assert not policy.plan(observation(episode, 1)).stop
    assert policy._target_xy is None


def test_missing_llm_is_not_silently_run_as_uniform():
    from sparx_agency.core.planning.objnav.errors import ObjNavInternalError
    from sparx_agency.core.planning.environment import OccupancyGrid2D, OccupancyGrid2DParams, OccupancyValues
    policy, _ = setup_policy()
    policy.graph.oracle.probabilities = lambda *a: type("Failed", (), {"source": "uniform_fallback"})()
    data = np.full((100, 100), -1, np.int8)
    data[20:80, 20:80] = 0
    world = OccupancyGrid2D(data, OccupancyGrid2DParams(0.1, -5, -5),
                            values=OccupancyValues(free=0, occupied=100, unknown=-1))
    with pytest.raises(ObjNavInternalError, match="uniform"):
        policy.graph.update(world, [], policy.target)


@pytest.mark.parametrize("kwargs", [{"graph_period_steps": 0}, {"depth_stride": -1},
                                    {"map_size_m": float("nan")}, {"stop_distance_m": 0}])
def test_invalid_method_settings_refused(kwargs):
    with pytest.raises(ValueError):
        RPTSettings(**kwargs)


def test_approach_arrival_band_is_inside_stop_radius(monkeypatch):
    from types import SimpleNamespace
    policy, episode = setup_policy()
    policy._target_xy = (2.0, 0.0)
    goals = []

    def plan_to(obs, world, goal):
        goals.append(goal)
        return [goal]

    monkeypatch.setattr(policy, "_plan_to", plan_to)
    command = policy._approach(observation(episode, 0), SimpleNamespace(resolution=0.1))
    assert not command.stop
    remaining = np.linalg.norm(np.asarray(goals[0]) - policy._target_xy)
    assert remaining + policy.converter_params.goal_tolerance_m < policy.settings.stop_distance_m


def test_nearby_frontier_is_retired_not_idled_forever(monkeypatch):
    from types import SimpleNamespace
    from sparx_agency.tasks.planning.objnav_benchmark_runtime.methods import rpt_policy
    policy, episode = setup_policy()
    policy._goal = (0.25, 0.0)
    monkeypatch.setattr(rpt_policy, "in_room_frontier_goals", lambda *args: [(0.25, 0.0)])
    goal = policy._frontier(observation(episode, 1), SimpleNamespace(resolution=0.1), None, None)
    assert goal is None and (0.25, 0.0) in policy._visited_frontiers


def test_observed_mapping_reaches_llm_rpt_and_astar():
    policy, episode = setup_policy(label="bed")
    command = policy.plan(observation(episode, 0, depth=3.0))
    assert not command.stop
    assert policy.graph.queries == 1
    assert policy.solver.calls == 1
    assert policy.supervisor.stats["selections"] == 1


