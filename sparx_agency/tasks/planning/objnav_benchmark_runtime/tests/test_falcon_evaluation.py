"""Additional evaluation telemetry must not change scores or reach the policy."""
import pytest

from sparx_agency.core.planning.objnav.errors import EnvContractError
from sparx_agency.core.planning.objnav.types.actions import DiscreteAction as A
from sparx_agency.tasks.planning.objnav_benchmark_runtime.gibson.dataset import GibsonDataset
from sparx_agency.tasks.planning.objnav_benchmark_runtime.gibson.env import GibsonEnv
from sparx_agency.tasks.planning.objnav_benchmark_runtime.gibson import demo
from sparx_agency.tasks.planning.objnav_benchmark_runtime.tests.test_demo import semantic_map
from sparx_agency.tasks.planning.objnav_benchmark_runtime.tests.test_gibson import LineSimulator, write_dataset


class ContactSimulator(LineSimulator):
    def step(self, action):
        self.last_collision = action == A.MOVE_FORWARD
        return super().step(action)


def test_collision_and_first_entry_are_evaluator_only(tmp_path):
    paths = write_dataset(tmp_path, semantic_map())
    env = GibsonEnv(GibsonDataset(*paths, full=False), simulator=ContactSimulator())
    try:
        episode, observation = env.reset(env.episode_ids()[0])
        assert episode.max_steps == 500
        with pytest.raises(EnvContractError):
            env.measure()
        for _ in range(4):
            observation = env.step(A.MOVE_FORWARD)
        assert not hasattr(observation, "collisions")
        assert not hasattr(observation, "first_success_action")
        assert "first_success_action" not in episode.metadata
        env.step(A.STOP)
        measured = env.measure()
        diagnostics = env.evaluation_diagnostics()
        assert measured.success
        assert diagnostics["collisions"] == 4
        assert diagnostics["first_success_action"] == 4
        assert set(measured.native_metrics) == {"success", "spl", "distance_to_goal"}
    finally:
        env.close()


def test_gibson_new_demo_keeps_completed_detector_default(monkeypatch, tmp_path):
    monkeypatch.setattr(demo, "SETTINGS_PATH", tmp_path / "not-configured.json")
    assert demo.defaults()["checkpoint"].endswith("yolov8x-worldv2.pt")


def test_burst_reasoning_classifies_accumulated_evidence_without_reset():
    from types import SimpleNamespace
    from sparx_agency.tasks.planning.objnav_benchmark_runtime.tests.test_falcon import open_fixture
    from sparx_agency.tasks.planning.objnav_benchmark_runtime.tests.test_method import setup_policy
    policy, _ = setup_policy("bed")
    world, _, _, _ = open_fixture()
    objects = [SimpleNamespace(class_name=name, xy=(3.0, 3.0)) for name in ("chair", "sofa", "television")]
    for step in (0, 10):
        policy.graph.update(world, objects, policy.target, step=step, reason=False)
    assert policy.graph.label_tracker.queries == policy.graph.queries == 0
    policy.graph.reason(world, policy.target, 10)
    assert policy.graph.label_tracker.queries > 0
    assert policy.graph.queries == 1


def test_one_frame_cannot_satisfy_deferred_classification_gate():
    from sparx_agency.tasks.planning.objnav_benchmark_runtime.methods.room_labels import RevisableRoomLabels
    from sparx_agency.tasks.planning.objnav_benchmark_runtime.tests.test_method import FakeLLM
    labels = RevisableRoomLabels(FakeLLM())
    objects = {1: ["chair", "sofa", "television"]}
    labels.update(objects, 0, allow_query=False)
    labels.update(objects, 0)
    assert labels.queries == 0
    labels.update(objects, 1)
    assert labels.queries == 1


