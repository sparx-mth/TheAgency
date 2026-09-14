"""Synthetic regressions for route lifetime and target verification; no Gibson data."""
from dataclasses import replace
from types import SimpleNamespace

import pytest

from sparx_agency.core.common.types import Pose2D
from sparx_agency.core.planning.objnav.action_converter.converter import DiscreteActionConverter
from sparx_agency.core.planning.objnav.action_converter.params import ActionConverterParams
from sparx_agency.core.planning.objnav.types.command import NavigationCommand
from sparx_agency.core.planning.objnav.types.pose import AgentPose
from sparx_agency.tasks.planning.objnav_benchmark_runtime.gibson.protocol import PROTOCOL
from sparx_agency.tasks.planning.objnav_benchmark_runtime.methods.object_evidence import TargetEvidence
from sparx_agency.tasks.planning.objnav_benchmark_runtime.methods.route_memory import CommittedRoute
from sparx_agency.tasks.planning.objnav_benchmark_runtime.tests.test_method import observation, setup_policy


def test_snapped_target_endpoint_starts_verification_without_replanning(monkeypatch):
    policy, episode = setup_policy()
    obs = observation(episode, 0)
    policy._target_id, policy._target_xy = 7, (1.3, 0)
    policy.target_evidence.observe(SimpleNamespace(id=7), obs.pose, 0)
    path = SimpleNamespace(points=[Pose2D(0, 0), Pose2D(0.25, 0)])
    policy.route_memory.adopt(path, (0.55, 0), "target", obs)
    monkeypatch.setattr(policy, "_navigate", lambda *a, **k: NavigationCommand.hold())
    command = policy._approach(observation(episode, 1), SimpleNamespace(resolution=0.1))
    assert command.info.get("reason") == "bounded target verification"
    assert policy.target_evidence.verification_started(7)


def test_nearby_endpoint_does_not_finish_unwalked_return_leg(monkeypatch):
    policy, episode = setup_policy()
    obs = observation(episode, 0)
    policy._target_id, policy._target_xy = 7, (1.3, 0)
    policy.target_evidence.observe(SimpleNamespace(id=7), obs.pose, 0)
    path = SimpleNamespace(points=[Pose2D(0, 0), Pose2D(2, 0), Pose2D(2, 1), Pose2D(0.25, 0)])
    policy.route_memory.adopt(path, (0.25, 0), "target", obs)
    follow = NavigationCommand.follow(path)
    monkeypatch.setattr(policy, "_navigate", lambda *a, **k: follow)
    assert policy._approach(observation(episode, 1), SimpleNamespace(resolution=0.1)) is follow
    assert not policy.target_evidence.verification_started(7)


@pytest.mark.parametrize("endpoint,tolerance", [(0.25, 0.3), (0.12, 0.05), (1.0, 0.3)])
def test_route_arrival_matches_actual_converter(endpoint, tolerance):
    spec = PROTOCOL.actions()
    params = replace(ActionConverterParams(), goal_tolerance_m=tolerance)
    route = CommittedRoute(spec, params)
    obs = SimpleNamespace(step=0, pose=AgentPose(0, 0, 0, 0))
    path = SimpleNamespace(points=[Pose2D(0, 0), Pose2D(endpoint, 0)])
    route.adopt(path, (endpoint + 0.3, 0), "target", obs)
    result = DiscreteActionConverter(spec, params).step(obs.pose, NavigationCommand.follow(path))
    assert route.arrived(obs) == (result.status == "arrived")


def test_verification_is_bound_to_selected_landmark_not_first_observed(monkeypatch):
    policy, episode = setup_policy()
    obs = observation(episode, 0)
    policy.target_evidence.observe(SimpleNamespace(id=1), obs.pose, 0)
    policy.target_evidence.observe(SimpleNamespace(id=2), obs.pose, 0)
    policy._target_id, policy._target_xy = 2, (1.3, 0)
    policy.route_memory.adopt(
        SimpleNamespace(points=[Pose2D(0, 0), Pose2D(0.2, 0)]), (0.2, 0), "target", obs)
    monkeypatch.setattr(policy, "_navigate", lambda *a, **k: NavigationCommand.hold())
    world = SimpleNamespace(resolution=0.1)
    for step in (1, 2):
        command = policy._approach(observation(episode, step), world)
        assert command.info.get("reason") == "bounded target verification"
    assert policy.target_evidence.verification_started(2)
    assert not policy.target_evidence.verification_started(1)
    assert policy._approach(observation(episode, 3), world) is None
    assert policy.target_evidence.is_suppressed(2, 4)
    assert not policy.target_evidence.is_suppressed(1, 4)


def test_rejected_hypothesis_requires_new_evidence_after_cooldown():
    evidence = TargetEvidence()
    landmark = SimpleNamespace(id=1)
    for step in range(3):
        evidence.observe(landmark, AgentPose(step * 0.25, 0, 0, 0), step)
    evidence.reject(landmark.id, 3)
    assert not evidence.verification_started(landmark.id)
    assert not evidence.observe(landmark, AgentPose(1, 0, 0, 0), 4)
    assert not evidence.observe(landmark, AgentPose(1, 0, 0, 0), 23)
    assert evidence.diagnostics()["1"]["count"] == 1


def test_obstruction_replans_cannot_reset_stationary_timeout_forever():
    route = CommittedRoute(PROTOCOL.actions(), ActionConverterParams())
    path = SimpleNamespace(points=[Pose2D(0, 0), Pose2D(3, 0)])
    planner = SimpleNamespace(path_collides=lambda *a, **k: True)
    obs = SimpleNamespace(step=0, pose=AgentPose(0, 0, 0, 0))
    route.adopt(path, (3, 0), "frontier", obs)
    for step in range(1, route.settings.progress_timeout_steps + 3):
        obs = SimpleNamespace(step=step, pose=AgentPose(0, 0, 0, (-1) ** step * 0.5))
        assert not route.reusable(obs, None, planner, (3, 0), "frontier")
        if route.reason == "no_progress":
            break
        route.adopt(path, (3, 0), "frontier", obs)
    assert route.reason == "no_progress"


def test_real_progress_and_new_goal_rearm_the_route_timeout():
    route = CommittedRoute(PROTOCOL.actions(), ActionConverterParams())
    planner = SimpleNamespace(path_collides=lambda *a, **k: False)
    obs = SimpleNamespace(step=0, pose=AgentPose(0, 0, 0, 0))
    route.adopt([Pose2D(0, 0), Pose2D(20, 0)], (20, 0), "frontier", obs)
    for step in range(1, 40):
        obs = SimpleNamespace(step=step, pose=AgentPose(step * 0.25, 0, 0, 0))
        assert route.reusable(obs, None, planner, (20, 0), "frontier")
        route.clear("forward_blocked")
        route.adopt([Pose2D(obs.pose.x, 0), Pose2D(20, 0)], (20, 0), "frontier", obs)
    route.clear("no_progress")
    later = SimpleNamespace(step=100, pose=obs.pose)
    route.adopt([Pose2D(obs.pose.x, 0), Pose2D(10, 5)], (10, 5), "frontier", later)
    assert route.reusable(later, None, planner, (10, 5), "frontier")


def test_stalled_target_is_cooled_down_instead_of_immediately_readopted(monkeypatch):
    policy, episode = setup_policy()
    obs = observation(episode, 0)
    policy._target_id, policy._target_xy = 7, (3, 0)
    policy.target_evidence.observe(SimpleNamespace(id=7), obs.pose, 0)

    def stalled(*args, **kwargs):
        policy.route_memory.clear("no_progress")
        return None

    monkeypatch.setattr(policy, "_navigate", stalled)
    assert policy._approach(observation(episode, 1), SimpleNamespace(resolution=0.1)) is None
    assert policy._target_id is None and policy._target_xy is None
    assert policy.target_evidence.is_suppressed(7, 2)

