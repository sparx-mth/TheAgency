"""Service/config regressions for the selectable grounded perception pipeline."""
import hashlib
import json
import threading
from types import SimpleNamespace

import pytest

from sparx_agency.core.mapping.detection.snapshots import snapshot_files
from sparx_agency.core.mapping.detection.tests.test_grounded_vlm import IMAGE, box, pipeline
from sparx_agency.tasks.mapping.scene_graph.serve.backends import selected_checkpoint_identity
from sparx_agency.tasks.mapping.scene_graph.serve.detector_args import parse_args
from sparx_agency.tasks.mapping.scene_graph.serve.detection_server import _make_server, _ServerContext
from sparx_agency.tasks.planning.objnav_benchmark_runtime.methods.perception import HttpDetector
from sparx_agency.tasks.planning.objnav_benchmark_runtime.gibson.detector_options import detector_flags


def config_file(tmp_path, **updates):
    values = {"backend": "hybrid", "grounding_model": "dino", "blip2_model": "blip2",
              "yolo_model": "yolo.pt", "clip_model": "clip.pt", "device": "cpu", "conf": .05}
    values.update(updates)
    path = tmp_path / "detector.json"
    path.write_text(json.dumps(values))
    return path


@pytest.mark.parametrize("mode", ["yolo_world", "grounded_vlm", "hybrid", "grounding_dino"])
def test_one_cli_switch_selects_mode_and_correct_model(tmp_path, mode):
    path = config_file(tmp_path)
    args = parse_args(["--config", str(path), "--backend", mode])
    assert args.backend == mode
    assert args.model == str(tmp_path / ("yolo.pt" if mode == "yolo_world" else "dino"))
    assert args.device == "cpu" and args.conf == .05
    assert args.allow_shared_gpu is False


@pytest.mark.parametrize("values", [{"backend": "not-a-model"}, {"unknown": 1},
                                   {"device": []}, {"yolo_model": " "},
                                   {"allow_shared_gpu": "false"}, {"allow_shared_gpu": 1}])
def test_invalid_config_fails_early(tmp_path, values):
    path = config_file(tmp_path, **values)
    with pytest.raises(SystemExit):
        parse_args(["--config", str(path)])


@pytest.mark.parametrize("shared", [False, True])
def test_gpu_authorization_is_explicit_and_cli_can_revoke(tmp_path, shared):
    path = config_file(tmp_path, allow_shared_gpu=shared)
    flags = ["--config", str(path), "--backend", "grounded_vlm"]
    assert parse_args(flags).allow_shared_gpu is shared
    override = "--no-allow-shared-gpu" if shared else "--allow-shared-gpu"
    assert parse_args(flags + [override]).allow_shared_gpu is (not shared)


def test_grounded_requires_explicit_weights_and_threshold():
    for flags in (["--backend", "grounded_vlm"],
                  ["--backend", "grounded_vlm", "--model", "dino", "--conf", ".05"],
                  ["--backend", "hybrid", "--model", "dino", "--blip2-model", "blip", "--conf", ".05"]):
        with pytest.raises(SystemExit):
            parse_args(flags)


def snapshot(tmp_path, name):
    path = tmp_path / name
    path.mkdir()
    (path / "model.safetensors").write_bytes(b"fixture, not a real model")
    (path / "config.json").write_text("{}")
    return path


def test_grounded_identity_never_requires_unused_yolo(tmp_path):
    dino, blip = snapshot(tmp_path, "dino"), snapshot(tmp_path, "blip")
    args = SimpleNamespace(backend="grounded_vlm", model=dino, blip2_model=blip,
                           yolo_model=tmp_path / "missing.pt", clip_model=tmp_path / "missing-clip.pt")
    before = selected_checkpoint_identity(args)
    assert set(before["components"]) == {"grounding_dino", "blip2"}
    (blip / "config.json").write_text('{"changed":true}')
    assert before["checkpoint_sha256"] != selected_checkpoint_identity(args)["checkpoint_sha256"]


def test_yolo_identity_never_reads_dino_or_blip(tmp_path):
    model = tmp_path / "yolo.pt"
    model.write_bytes(b"fixture")
    identity = selected_checkpoint_identity(SimpleNamespace(backend="yolo_world", model=model))
    assert identity["checkpoint_sha256"] == hashlib.sha256(b"fixture").hexdigest()


def test_hybrid_identity_includes_yolo_and_clip(tmp_path):
    args = SimpleNamespace(backend="hybrid", model=snapshot(tmp_path, "dino"),
                           blip2_model=snapshot(tmp_path, "blip"),
                           yolo_model=tmp_path / "yolo.pt", clip_model=tmp_path / "clip.pt")
    args.yolo_model.write_bytes(b"yolo")
    args.clip_model.write_bytes(b"clip")
    before = selected_checkpoint_identity(args)
    assert set(before["components"]) == {"grounding_dino", "blip2", "yolo_world", "clip"}
    args.clip_model.write_bytes(b"changed")
    assert before["checkpoint_sha256"] != selected_checkpoint_identity(args)["checkpoint_sha256"]


def test_shards_must_exist_and_stay_inside_snapshot(tmp_path):
    index = tmp_path / "model.safetensors.index.json"
    index.write_text(json.dumps({"weight_map": {"parameter": "part.safetensors"}}))
    with pytest.raises(FileNotFoundError, match="shard"):
        snapshot_files(tmp_path)
    (tmp_path / "part.safetensors").write_bytes(b"fixture")
    assert len(snapshot_files(tmp_path)) == 2
    index.write_text(json.dumps({"weight_map": {"parameter": "../part.safetensors"}}))
    with pytest.raises(ValueError, match="Unsafe"):
        snapshot_files(tmp_path)


def test_incomplete_download_never_loads(tmp_path):
    (tmp_path / "model.safetensors").write_bytes(b"fixture")
    (tmp_path / "download-pending.json").write_text("{}")
    with pytest.raises(FileNotFoundError, match="incomplete"):
        snapshot_files(tmp_path)


def test_http_records_verification_without_changing_health_identity():
    detector = pipeline([box()], yolo=[box("bed")])
    metadata = {"backend": "hybrid", "checkpoint_sha256": "fixture",
                "detector_config": {"conf_thresh": .05}}
    server = _make_server(_ServerContext(detector, "fixture", "cpu", detector.prompts, metadata), "127.0.0.1", 0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    client = HttpDetector("http://127.0.0.1:%d" % server.server_port, detector.prompts, expected_backend="hybrid")
    try:
        identity = client.health()
        assert client.detect(IMAGE) == []  # contradictory same-region labels
        first = client.last_diagnostics
        assert first["raw_detections"][0]["status"] == "conflicting_labels"
        assert len(first["request_sha256"]) == 64
        detector.verifier.verdict = "no"
        client.detect(IMAGE + 30)
        assert client.last_diagnostics["request_sha256"] != first["request_sha256"]
        assert client.health() == identity
    finally:
        client.session.close()
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def test_client_rejects_evidence_from_a_different_frame():
    metadata = {"backend": "grounded_vlm", "checkpoint_sha256": "fixture", "detector_config": {"conf_thresh": .05}}
    health = {"ok": True, "model": "fixture", "device": "cpu", "classes": ["stairs"], "metadata": metadata}
    def response(data):
        return SimpleNamespace(raise_for_status=lambda: None, json=lambda: data)
    client = HttpDetector("http://unused", ["stairs"])
    client.session = SimpleNamespace(get=lambda *a, **k: response(health),
        post=lambda *a, **k: response(dict(health, h=40, w=80, detections=[],
            request_sha256="wrong-frame", diagnostics={"mode": "grounded_vlm"})))
    with pytest.raises(Exception, match="different submitted frame"):
        client.detect(IMAGE)
    assert client.last_diagnostics == {} and client.last_detections == ()


@pytest.mark.parametrize("timeout", [0, -1, float("inf"), float("nan")])
def test_invalid_timeout(timeout):
    with pytest.raises(ValueError):
        HttpDetector("http://unused", ["stairs"], timeout_s=timeout)


def test_gibson_modes_and_timeout_forwarding():
    from sparx_agency.tasks.planning.objnav_benchmark_runtime.gibson.run import parser
    from sparx_agency.tasks.planning.objnav_benchmark_runtime.gibson.run_development import parser as development_parser
    flags = detector_flags(SimpleNamespace(detector_url="http://localhost:18100", detector_backend="hybrid", detector_timeout_s=300))
    assert parser().parse_args(flags).detector_timeout_s == 300
    args = development_parser().parse_args(flags + ["--manifest", "m.json", "--scene", "scene", "--output", "out", "--explorer", "frontier"])
    assert args.detector_backend == "hybrid" and args.detector_timeout_s == 300
    assert args.allow_shared_gpu is False

