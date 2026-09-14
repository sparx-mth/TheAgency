"""Refinement regressions: commitment, physical clearance, duplicate evidence."""
from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace
import math

import numpy as np

from sparx_agency.core.common.types import Pose2D
from sparx_agency.core.mapping.objects.landmarks import ObjectLandmarkMap
from sparx_agency.core.planning.objnav.action_converter.params import ActionConverterParams
from sparx_agency.core.planning.objnav.types.pose import AgentPose
from sparx_agency.tasks.mapping.scene_graph.serve.contract import DetectionWire
from sparx_agency.tasks.planning.objnav_benchmark_runtime.tests.fixtures import actions
from sparx_agency.tasks.planning.objnav_benchmark_runtime.methods.route_memory import CommittedRoute
from sparx_agency.tasks.planning.objnav_benchmark_runtime.methods.object_evidence import deduplicate_detections, TargetEvidence
from sparx_agency.tasks.planning.objnav_benchmark_runtime.tests.test_method import setup_policy, observation


def test_safe_route_survives_turns_and_goal_jitter_without_readoption():
    route = CommittedRoute(actions(), ActionConverterParams())
    path = SimpleNamespace(points=[Pose2D(0, 0), Pose2D(2, 0)])
    planner = SimpleNamespace(path_collides=lambda *args, **kwargs: False)
    route.adopt(path, (2, 0), "frontier", SimpleNamespace(step=0))
    for step in range(1, 20):
        obs = SimpleNamespace(step=step, pose=AgentPose(0, 0, 0, (-1) ** step * 0.5))
        assert route.reusable(obs, None, planner, (2, 0.02 * (-1) ** step), "frontier")
        assert route.path is path
    assert route.stats["adoptions"] == 1


def test_real_obstacle_overrides_commitment_immediately():
    route = CommittedRoute(actions(), ActionConverterParams())
    route.adopt(SimpleNamespace(points=[Pose2D(0, 0), Pose2D(2, 0)]), (2, 0), "frontier", SimpleNamespace(step=0))
    assert not route.reusable(SimpleNamespace(step=1, pose=AgentPose(0, 0, 0, 0)), None,
                              SimpleNamespace(path_collides=lambda *args, **kwargs: True), (2, 0), "frontier")
    assert route.path is None and route.reason == "route_obstructed"


def test_alias_boxes_and_same_frame_cannot_inflate_landmark_support():
    box = DetectionWire("couch", 0.9, (10, 10, 100, 100))
    kept = deduplicate_detections([box, replace(box, cls="sofa", conf=0.8)])
    assert len(kept) == 1 and kept[0].cls == "sofa"
    landmarks = ObjectLandmarkMap(nearest_match=True)
    first = landmarks.observe("sofa", (1, 0), frame_id=0)
    assert landmarks.observe("sofa", (1.1, 0), frame_id=0).count == 1
    assert landmarks.observe("sofa", (1.1, 0), frame_id=1).id == first.id
    assert first.count == 2


def test_nearest_association_not_first_inserted():
    landmarks = ObjectLandmarkMap(dedupe_radius_m=0.7, nearest_match=True)
    left = landmarks.observe("chair", (0, 0))
    right = landmarks.observe("chair", (1, 0))
    assert landmarks.observe("chair", (0.6, 0)).id == right.id
    assert left.count == 1


def test_target_support_requires_fresh_separated_views():
    evidence = TargetEvidence()
    landmark = SimpleNamespace(id=1)
    p = AgentPose(0, 0, 0, 0)
    assert not evidence.observe(landmark, p, 0)
    assert not evidence.observe(landmark, p, 0)
    assert not evidence.observe(landmark, p, 1)
    assert evidence.observe(landmark, AgentPose(0.25, 0, 0, 0), 2)
    assert not evidence.observe(landmark, p, 20)  # stale support does not accumulate forever


def test_goal_and_inflation_floor_are_recorded_without_initial_sweep():
    policy, episode = setup_policy(label="bed")
    policy.plan(observation(episode, 0, depth=3))
    config = policy.configuration()
    assert config["falcon_running"] is False
    assert config["planner"]["inflate_floor_m"] == config["adaptation"]["body_radius_m"]
    assert policy._plan_calls >= 1  # starts navigating, not a forced 360-degree routine

