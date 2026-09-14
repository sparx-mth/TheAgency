"""Provenance, service drift, report and merge guards without external services."""
from __future__ import annotations

from dataclasses import asdict
import json
from types import SimpleNamespace

import numpy as np
import pytest

from sparx_agency.core.planning.objnav.errors import ObjNavInternalError
from sparx_agency.tasks.planning.objnav_benchmark.aggregate import summarise
from sparx_agency.tasks.planning.objnav_benchmark.logger import MetricsLogger
from sparx_agency.tasks.planning.objnav_benchmark.records import EpisodeRecord
from sparx_agency.tasks.planning.objnav_benchmark_runtime.gibson.protocol import PROTOCOL, SCENES
from sparx_agency.tasks.planning.objnav_benchmark_runtime.gibson.report import merge_runs, write_report
from sparx_agency.tasks.planning.objnav_benchmark_runtime.methods.perception import HttpDetector
from sparx_agency.tasks.planning.objnav_benchmark_runtime.methods.services import VerifiedLLMClient, public_service_url


def record(episode_id):
    from sparx_agency.tasks.planning.objnav_benchmark.records import ACTION_NAMES
    return EpisodeRecord(
        benchmark="gibson", split="val", episode_id=episode_id,
        scene_id=episode_id.split("/")[0], target_category="chair", agent="test-agent",
        success=False, spl=0.0, soft_spl=0.0, distance_to_goal_m=2.0,
        path_length_m=1e-5, observed_path_length_m=1e-5, shortest_path_m=3.0,
        start_distance_to_goal_m=2.0, steps=1, stop_called=True, termination="stop",
        action_counts={key: int(key == "STOP") for key in ACTION_NAMES},
        wall_s=0.01)


def shard(directory, index, *, partial=False, **overrides):
    ids = ["%s/%06d" % (s, i) for s in SCENES for i in range(200)][index::2]
    config = dict(protocol=asdict(PROTOCOL), runtime={"habitat-sim": "test"},
                  source_sha256="synthetic", method={"method": "test-agent"},
                  seed=0, reference_sim_version_match=False, dataset={"synthetic": True},
                  shards=2, shard_index=index, limit=None, full_split=False,
                  selected_episode_ids=ids, on_agent_error="record")
    config.update(overrides)
    rows = [record(i) for i in (ids[:1] if partial else ids)]
    with MetricsLogger(directory, config) as logger:
        for row in rows:
            logger.log(row)
        logger.finish(summarise(rows))


def test_complete_shards_merge_without_fabricating_missing_episodes(tmp_path):
    left, right, merged = [tmp_path / p for p in ("left", "right", "merged")]
    shard(left, 0)
    shard(right, 1)
    assert not write_report(left)["full_split_completed"]
    audit = merge_runs([left, right], merged)
    assert audit["full_split_completed"] and audit["episodes"] == 1000
    assert not audit["exact_osg_protocol_reproduction"]
    assert len((merged / "episodes.jsonl").read_text().splitlines()) == 1000


@pytest.mark.parametrize("change", [{"partial": True}, {"seed": 1}, {"limit": 1}])
def test_merge_refuses_incomplete_or_changed_shards(tmp_path, change):
    left, right = tmp_path / "left", tmp_path / "right"
    shard(left, 0)
    shard(right, 1, **change)
    with pytest.raises(ValueError):
        merge_runs([left, right], tmp_path / "out")
    assert not (tmp_path / "out" / "episodes.jsonl").exists()


def test_duplicate_shards_and_changed_record_identity_are_refused(tmp_path):
    directory = tmp_path / "shard"
    shard(directory, 0)
    with pytest.raises(ValueError, match="duplicate"):
        merge_runs([directory, directory], tmp_path / "out")
    path = directory / "episodes.jsonl"
    lines = path.read_text().splitlines()
    row = json.loads(lines[0])
    row["agent"] = "another-agent"
    lines[0] = json.dumps(row)
    path.write_text("\n".join(lines) + "\n")
    with pytest.raises(ValueError, match="identity"):
        write_report(directory)


class Response:
    def __init__(self, payload):
        self.payload = payload

    def raise_for_status(self):
        pass

    def json(self):
        return self.payload


def test_detector_checks_atomic_vocabulary_and_frozen_model():
    detector = HttpDetector("http://localhost:8092", ("chair",))
    metadata = {"checkpoint_sha256": "test-checkpoint", "detector_config": {"conf_thresh": 0.25}}
    health = {"ok": True, "model": "test", "device": "cpu", "classes": ["chair"], "metadata": metadata}
    reply = {"h": 8, "w": 8, "classes": ["chair"], "metadata": metadata, "detections": []}
    detector.session = SimpleNamespace(get=lambda *a, **k: Response(health),
                                       post=lambda *a, **k: Response(reply))
    assert detector.detect(np.zeros((8, 8, 3), np.uint8)) == []
    reply["classes"] = ["bed"]
    with pytest.raises(ObjNavInternalError, match="reconfigured"):
        detector.detect(np.zeros((8, 8, 3), np.uint8))
    health["metadata"] = dict(metadata, checkpoint_sha256="different-checkpoint")
    with pytest.raises(ObjNavInternalError, match="changed"):
        detector.health()


def test_llm_port_health_is_not_model_readiness():
    cfg = SimpleNamespace(backend="ollama", base_url="http://localhost:11434",
                          timeout_s=3.0, model="test-model")
    payload = {"models": []}
    raw = SimpleNamespace(cfg=cfg, sess=SimpleNamespace(get=lambda *a, **k: Response(payload)),
                          _auth_header=lambda: {}, chat_json=lambda *a, **k: {"ok": True})
    client = VerifiedLLMClient(raw)
    with pytest.raises(ObjNavInternalError, match="not provisioned"):
        client.health()
    payload["models"] = [{"name": "test-model:latest", "digest": "version-one"}]
    assert client.health()["immutable_revision_exposed"]
    assert client.chat_json("system", "user") == {"ok": True}
    payload["models"] = [{"name": "test-model:latest", "digest": "version-two"}]
    with pytest.raises(ObjNavInternalError, match="changed"):
        client.chat_json("system", "user")


@pytest.mark.parametrize("url", ["http://user:secret@host", "http://host?key=secret", "file:///tmp/model"])
def test_service_urls_cannot_leak_credentials(url):
    with pytest.raises(ValueError):
        public_service_url(url)


def test_real_http_service_preserves_provenance_and_rgb_order():
    import threading
    from sparx_agency.core.common.types.perception import Detection2D
    from sparx_agency.tasks.mapping.scene_graph.serve.detection_server import _ServerContext, _make_server

    class Detector:
        def detect(self, rgb):
            assert float(rgb[..., 0].mean()) > 240  # JPEG red, not BGR blue
            assert float(rgb[..., 2].mean()) < 15
            return [Detection2D(label="chair", score=0.9, bbox_xyxy=(1, 1, 6, 6),
                                 frame_w=8, frame_h=8)]

        def set_prompts(self, names):
            pass

    metadata = {"checkpoint_sha256": "synthetic", "detector_config": {"conf_thresh": 0.25}}
    context = _ServerContext(Detector(), "test", "cpu", ["chair"], metadata)
    server = _make_server(context, "127.0.0.1", 0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    client = HttpDetector("http://127.0.0.1:%d" % server.server_port, ["chair"])
    try:
        image = np.zeros((8, 8, 3), np.uint8)
        image[..., 0] = 255
        rows = client.detect(image)
        assert rows[0].cls == "chair" and context.frames_served == 1
        assert client.health()["metadata"] == metadata
        response = client.session.post(client.url + "/set_classes", json={"classes": ["bed"]})
        response.raise_for_status()
        with pytest.raises(ObjNavInternalError, match="vocabulary"):
            client.detect(image)
    finally:
        client.session.close()
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)

