"""Bounded recovery for actual arrival/facing loops and malformed oracle replies."""
from __future__ import annotations

from types import SimpleNamespace

from sparx_agency.core.common.types import Pose2D
from sparx_agency.core.mapping.topology.search_oracle import OracleRoom
from sparx_agency.tasks.planning.objnav_benchmark_runtime.methods.oracle_retry import RepairingSearchOracle
from sparx_agency.tasks.planning.objnav_benchmark_runtime.tests.test_method import setup_policy, observation


def test_arrival_enters_bounded_verification_not_idle_face_ping_pong():
    policy, episode = setup_policy()
    obs = observation(episode, 0)
    policy._target_id, policy._target_xy = 7, (1.3, 0)
    policy.target_evidence.observe(SimpleNamespace(id=7), obs.pose, 0)
    policy.route_memory.adopt(SimpleNamespace(points=[Pose2D(0, 0), Pose2D(0.2, 0)]), (0.2, 0), "target", obs)
    world = SimpleNamespace(resolution=0.1)
    first = policy._approach(observation(episode, 1), world)
    second = policy._approach(observation(episode, 2), world)
    assert first.info["reason"] == second.info["reason"] == "bounded target verification"
    assert policy._approach(observation(episode, 3), world) is None
    assert policy._target_xy is None
    assert policy.target_evidence.is_suppressed(7, 4)
    assert policy._plan_calls == 0


class Replies:
    def __init__(self, items):
        self.items = iter(items)
        self.calls = 0

    def chat_json(self, system, user):
        self.calls += 1
        return next(self.items)


def test_one_schema_repair_uses_existing_oracle_scoring():
    client = Replies([{}, {"rooms": [{"id": 4, "score": 60, "why": "observed context"}]}])
    oracle = RepairingSearchOracle(client)
    result = oracle.probabilities("chair", [OracleRoom(4, "living_room")])
    assert result.source == "llm" and result.probs == {4: 1.0}
    assert client.calls == 2 and oracle.repair_successes == 1


def test_oracle_repair_is_bounded_and_does_not_invent_a_valid_answer():
    client = Replies([{}, {}])
    oracle = RepairingSearchOracle(client)
    assert oracle.probabilities("chair", [OracleRoom(4, "unknown")]).source != "llm"
    assert client.calls == 2 and oracle.repair_successes == 0



