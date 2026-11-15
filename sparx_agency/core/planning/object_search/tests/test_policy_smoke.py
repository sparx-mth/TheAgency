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

import pytest

from sparx_agency.core.planning.object_search.interfaces import (
    Detection, DetectorFrameError, RECOVERABLE_DETECTOR_CAUSES, detection_from_wire)
from sparx_agency.core.planning.object_search.tests.fixtures import (
    FakeDetector, FakeLLM, body_actions, frame, make_policy)
from sparx_agency.core.planning.objnav.errors import ObjNavInternalError


def test_the_policy_plans_from_a_frame_through_mapping_room_reasoning_and_the_solver():
    detector, llm = FakeDetector("bed"), FakeLLM()
    policy, ep = make_policy(detector, llm)
    command = policy.plan(frame(ep, 0, depth=3.0))
    assert not command.stop
    assert detector.calls == 1 and llm.calls >= 1
    assert policy.graph.queries == 1 and policy.solver.calls == 1
    assert command.info["phase"] and command.info["floor_id"] == policy.mapping.floor_id


def test_an_episode_starts_with_the_warm_up_rotation_before_any_room_is_valued():
    detector, llm = FakeDetector("bed"), FakeLLM()
    policy, ep = make_policy(detector, llm, warmed_up=False)
    command = policy.plan(frame(ep, 0, depth=3.0))
    assert not command.stop and command.final_yaw is not None, "the warm-up is a turn in place"
    assert detector.calls == 1 and llm.calls == 0 and policy.graph.queries == 0


def test_a_target_seen_in_two_distinct_frames_locks_the_closing_and_stops():
    policy, ep = make_policy(FakeDetector("chair", duplicate=True),
                             target_closing={"stop_overshoot_steps": 0, "stop_overshoot_max_steps": 0})
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
