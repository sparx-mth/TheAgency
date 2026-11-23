"""Paired end-to-end ObjectNav episodes: real factory, detector HTTP server, planners and action converter."""
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
import json
import threading

import numpy as np
import pytest

from sparx_agency.core.common.types.perception import Detection2D
from sparx_agency.core.mapping.topology.llm_client import LLMClient
from sparx_agency.core.planning.object_search.perception.target_prompts import target_stair_prompts
from sparx_agency.core.planning.objnav.labels.datasets.gibson import gibson_label_mapper
from sparx_agency.tasks.mapping.scene_graph.serve.detection_server import _ServerContext, _make_server
from sparx_agency.tasks.planning.objnav_benchmark_runtime.evaluation import run_evaluation
from sparx_agency.tasks.planning.objnav_benchmark_runtime.policy_factory import build_policy, detector_vocabulary
from sparx_agency.tasks.planning.objnav_benchmark_runtime.provenance import model_provenance, model_report_rows
from sparx_agency.tasks.planning.objnav_benchmark_runtime.services.http_detector import HttpDetector
from sparx_agency.tasks.planning.objnav_benchmark_runtime.stub_llm import stub_reply
from sparx_agency.tasks.planning.objnav_benchmark_runtime.tests.test_frame_log_run import building
from sparx_agency.tasks.planning.objnav_benchmark_runtime.tests.test_ground_truth_perception import arguments
from sparx_agency.tasks.planning.objnav_benchmark_runtime.tests.test_runtime_boundaries import Response


class PromptedDetector:
    """Record the vocabulary at *inference*, not merely the labels sent back to a client."""

    def __init__(self, prompts):
        self.set_prompts(prompts)
        self.forwards = []

    def set_prompts(self, prompts):
        self.prompts = tuple(prompts)

    def detect(self, rgb):
        self.forwards.append(self.prompts)
        if "sink" not in self.prompts:
            return []
        h, w = rgb.shape[:2]
        return [Detection2D("sink", 0.9, (w//3, h//3, 2*w//3, 2*h//3), w, h)]


@contextmanager
def detector_server(vocabulary):
    detector = PromptedDetector(vocabulary)
    metadata = {"backend": "yolo_world", "checkpoint_sha256": "fixture", "detector_config": {"conf_thresh": 0.05}}
    server = _make_server(_ServerContext(detector, "fixture", "cpu", vocabulary, metadata), "127.0.0.1", 0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield detector, "http://127.0.0.1:%d" % server.server_port
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


class ModelSession:
    def __init__(self, visual):
        self.visual = visual
        self.calls = []

    def get(self, url, **kwargs):
        assert url.endswith("/api/tags")
        return Response({"models": [{"name": n, "digest": n} for n in ("vision", "labels", "oracle")]})

    def post(self, url, data=None, **kwargs):
        if url.endswith("/api/show"):
            assert self.visual
            return Response({"capabilities": ["vision"]})
        payload = json.loads(data)
        system, user = payload["messages"]
        model = payload["model"]
        self.calls.append(model)
        if self.visual:
            assert model == "vision" and user["images"] and "Target object:" in user["content"]
            reply = {"probability": 0.55}
        else:
            assert "images" not in user and model in ("oracle", "labels")
            reply = stub_reply(system["content"], user["content"])
        return Response({"message": {"content": json.dumps(reply)}})


@pytest.mark.parametrize("agent", ["rpt", "rpt_room_llm"])
def test_both_factory_variants_finish_the_same_episodes(agent, monkeypatch, tmp_path):
    visual = agent == "rpt"
    session = ModelSession(visual)
    init = LLMClient.__init__

    def bind_transport(client, *args, **kwargs):
        init(client, *args, **kwargs)
        client.sess = session

    monkeypatch.setattr(LLMClient, "__init__", bind_transport)
    monkeypatch.setenv("LLM_BACKEND", "ollama")
    monkeypatch.setenv("LLM_MODEL", "vision" if visual else "labels")
    monkeypatch.setenv("LLM_REASONING_MODEL", "oracle")
    for name in ("LLM_ANSWERS_RECORD", "LLM_ANSWERS_REPLAY", "LLM_EXCLUSIVE", "OBJNAV_FRAME_LOG_DIR"):
        monkeypatch.delenv(name, raising=False)
    mapper, env = gibson_label_mapper(), building(episodes=2)
    vocab = detector_vocabulary(agent, mapper)
    with detector_server(vocab) as (engine, url):
        p, method = build_policy(arguments(agent=agent, detector_url=url, perception="service"), mapper,
                                 defaults={"map_size_m": 20.0})
        assert method["pipeline_variant"] == agent
        report = "\n".join(model_report_rows(model_provenance(method)))
        assert (("frontier VLM" in report) == visual) and (("LLM, room labels" in report) != visual)
        session.calls.clear()  # the assertions below are on episode decisions, not preflight
        summary = run_evaluation(env, p, mapper, output=tmp_path, config={"method": method})
    rows = [json.loads(line) for line in (tmp_path / "episodes.jsonl").read_text().splitlines()]
    assert len(rows) == summary.overall.n_episodes == 2 and all(row["agent_error"] is None for row in rows)
    assert json.loads((tmp_path / "run.json").read_text())["config"]["method"]["pipeline_variant"] == agent
    for row in rows:
        info = row["agent_info"]["policy"]
        assert info["pipeline_variant"] == agent and info["plan_calls"] > 0
        assert not info["exploration_fallback"]["failures"]
    if visual:
        assert set(session.calls) == {"vision"} and p.frontiers.scores and p.solver.calls
        assert not any(hasattr(p, name) for name in ("graph", "landmarks", "doors", "peek", "loop", "supervisor", "pursuit", "_target_xy"))
        expected = [target_stair_prompts(mapper.target_labels(row["target_category"])) for row in rows for _ in range(row["steps"])]
        assert engine.forwards == expected and expected[0] != expected[-1]
    else:
        assert set(session.calls) == {"labels", "oracle"} and p.graph.queries and p.solver.calls
        assert p.graph.labels is not None and any(lm.class_name == "sink" for lm in p.landmarks.all_landmarks())
        assert set(engine.forwards) == {vocab} and not hasattr(p, "frontiers")


def test_two_workers_prompt_atomically_and_fixed_client_still_uses_all_classes():
    mapper = gibson_label_mapper()
    vocab = detector_vocabulary("rpt", mapper)
    with detector_server(vocab) as (engine, url):
        clients = [HttpDetector(url, vocab, expected_backend="yolo_world", episode_scoped=True) for _ in range(2)]
        scopes = [target_stair_prompts(mapper.target_labels(c)) for c in ("chair", "bed")]
        img = np.zeros((16, 16, 3), np.uint8)
        for client, prompts in zip(clients, scopes):
            client.set_episode_prompts(prompts)
        with ThreadPoolExecutor(2) as pool:
            list(pool.map(lambda n: clients[n].detect(img), [0, 1]))
        assert set(engine.forwards) == set(scopes)
        assert HttpDetector(url, vocab).detect(img) == [] and engine.forwards[-1] == vocab
        assert clients[0].health()["classes"] == list(vocab)
