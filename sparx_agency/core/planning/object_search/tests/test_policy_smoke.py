"""The method runs end to end on synthetic frames with fake services, and keeps its service contract.

These are the package's own smoke and contract tests. The behavioural
regression suite -- thousands of assertions on the room loop, the closing,
the fallback ladder -- lives with the benchmark runtime
(``tasks/planning/objnav_benchmark_runtime/tests/``), where the Gibson
protocol, the dashboard renderers and the recorded traces it reasons about
are.
"""
from __future__ import annotations

import json
import math

import pytest

from sparx_agency.core.planning.object_search.interfaces import (
    Detection, DetectorFrameError, RECOVERABLE_DETECTOR_CAUSES, detection_from_wire, ground_truth_semantics)
from sparx_agency.core.planning.object_search.tests.fixtures import (
    FakeDetector, FakeLLM, body_actions, frame, make_policy)
from sparx_agency.core.planning.objnav.errors import ObjNavInternalError
from sparx_agency.core.planning.objnav.types.actions import DiscreteAction
from sparx_agency.core.planning.objnav.types.pose import AgentPose


def test_every_warmup_frame_updates_the_frontiers_but_routing_waits_for_the_rotation():
    detector, llm = FakeDetector("bed"), FakeLLM()
    policy, ep = make_policy(detector, llm, warmed_up=False)
    for step in range(policy.warmup_turns + 1):
        obs = frame(ep, step, depth=2.0, pose=AgentPose(0, 0, 0, math.radians(30 * step)))
        command = policy.plan(obs)
        if step < policy.warmup_turns:
            assert command.info["kind"] == "warmup" and policy.solver.calls == 0
            policy.notify_action(obs, DiscreteAction.TURN_LEFT)
    assert llm.calls == policy.frontiers.calls > 0 and not hasattr(policy, "graph")
    assert detector.calls == policy.warmup_turns + 1 and not hasattr(policy, "loop")
    assert command.info["frontier_id"] in policy.frontiers.scores and command.waypoints


def test_a_target_seen_in_two_distinct_frames_locks_the_closing_and_stops():
    policy, ep = make_policy(FakeDetector("chair", duplicate=True))
    first = policy.plan(frame(ep, 0))
    assert not first.stop and policy.closing.count == 1
    assert policy.plan(frame(ep, 0)) is first, "the same frame is decided once"
    assert policy.plan(frame(ep, 1)).stop and policy.closing.locked


def test_the_configuration_and_the_episode_info_serialise():
    policy, ep = make_policy(actions=body_actions(tilt=True))
    policy.plan(frame(ep, 0, depth=3.0))
    config, info = policy.configuration(), policy.episode_info()
    assert config["method"] == policy.name and config["training_free"] is True
    json.dumps(config)
    json.dumps(info, default=str)


def test_the_planner_is_the_settings_own_so_a_replay_of_a_recorded_action_builds_the_same_one():
    """``RPTSettings.planner_params`` is the one definition: the policy's planner and its record are built from it."""
    from dataclasses import asdict
    policy, _ = make_policy(min_clearance_m=0.22, preferred_clearance_m=0.34, detour_factor=2.0)
    params = policy.settings.planner_params()
    assert policy.planner_params == params and policy.planner.params == params
    assert (params.inflate_radius_m, params.inflate_floor_m, params.detour_factor) == (0.34, 0.22, 2.0)
    assert params.unknown_blocked is True and params.search_margin_m == policy.settings.map_size_m
    assert (params.centreline_weight, params.channel_half_width_m) == (
        policy.settings.centreline_weight, policy.settings.channel_half_width_m)
    assert policy.configuration()["planner"] == asdict(params)


def test_the_run_record_names_the_visual_prompt_and_not_the_room_oracle():
    policy, _ = make_policy()
    assert policy.configuration()["frontier_search"]["prompt"] == "boxed-frontier-v2"
    assert "node_oracle" not in policy.configuration()


class _Annotated(FakeDetector):
    """A detector whose labels are an environment's annotations, and which says so."""

    ground_truth_semantics = True


class _AnswersAnything:
    """A test double that answers every attribute -- which declares nothing."""

    def __getattr__(self, name):
        return "yes"


def test_the_run_record_says_what_perceived_as_the_services_declare_it():
    """Until 2026-11-16 the record read False whatever perceived, a ground-truth run included."""
    inferred, _ = make_policy(FakeDetector("bed"))
    annotated, _ = make_policy(_Annotated("bed"))
    assert inferred.configuration()["ground_truth_semantics"] is False
    assert annotated.configuration()["ground_truth_semantics"] is True
    assert annotated.configuration()["ground_truth_stairs"] is False, "the other declaration is untouched"
    assert ground_truth_semantics(FakeDetector(), _Annotated()) is True, "either service's declaration is enough"
    assert ground_truth_semantics(FakeDetector(), None) is False and ground_truth_semantics() is False
    assert ground_truth_semantics(_AnswersAnything()) is False, "only the literal True is a declaration"


class _Failing(FakeDetector):
    def __init__(self, error):
        super().__init__()
        self.error = error

    def detect(self, rgb):
        self.calls += 1
        raise self.error


def _chained(cause):
    try:
        raise cause
    except Exception as exc:
        error = ObjNavInternalError("Detector service failed: %s" % exc)
        error.__cause__ = exc
        return error


@pytest.mark.parametrize("cause", [OSError("connection refused"), DetectorFrameError("bad box"),
                                   ValueError("no json"), KeyError("detections")])
def test_a_transport_or_frame_failure_backs_the_detector_off_and_the_search_goes_on(cause):
    """An ``ObjNavInternalError`` chained from a recoverable cause is a lost frame, never a dead episode."""
    assert isinstance(cause, RECOVERABLE_DETECTOR_CAUSES)
    detector = _Failing(_chained(cause))
    policy, ep = make_policy(detector)
    command = policy.plan(frame(ep, 0, depth=3.0))
    assert not command.stop
    assert policy.perception.counts["detector_failures"] == 1
    assert "failed" in policy.perception.detector_evidence


def test_any_other_exception_from_the_detector_is_a_frame_failure_too():
    detector = _Failing(RuntimeError("fake detector exploded"))
    policy, ep = make_policy(detector)
    policy.plan(frame(ep, 0, depth=3.0))
    assert policy.perception.counts["detector_failures"] == 1


def test_identity_drift_ends_the_run():
    """A bare ``ObjNavInternalError`` -- another vocabulary or model answering -- must not be backed off."""
    detector = _Failing(ObjNavInternalError("Detector was reconfigured during inference"))
    policy, ep = make_policy(detector)
    with pytest.raises(ObjNavInternalError, match="reconfigured"):
        policy.plan(frame(ep, 0, depth=3.0))


def test_a_wire_record_becomes_a_detection_without_rounding():
    class Wire:
        cls, conf, xyxy = "tv", 0.65, (12.5, 7.25, 300.75, 240.0)
    detection = detection_from_wire(Wire())
    assert detection == Detection("tv", 0.65, (12.5, 7.25, 300.75, 240.0))
    assert hash(detection) == hash(Detection("tv", 0.65, (12.5, 7.25, 300.75, 240.0)))
