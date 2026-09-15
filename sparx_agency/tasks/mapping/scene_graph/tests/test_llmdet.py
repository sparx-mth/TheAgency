"""Lightweight LLMDet/HTTP regressions: no torch, checkpoint or simulator needed."""
from contextlib import nullcontext
from dataclasses import replace
import subprocess
import sys
import threading
from types import SimpleNamespace

import numpy as np
import pytest

from sparx_agency.core.common.types.perception import Detection2D
from sparx_agency.core.mapping.detection.llmdet import LlmDetConfig, LlmDetDetector
from sparx_agency.core.mapping.detection.llmdet_postprocess import (
    caption_and_spans, decode_token_detections, tokens_for_spans, validate_prompts)
from sparx_agency.core.mapping.detection.registry import default_detection_registry
from sparx_agency.tasks.mapping.scene_graph.serve import backends
from sparx_agency.tasks.mapping.scene_graph.serve.detection_server import (
    _make_server, _ServerContext, parse_args)
from sparx_agency.tasks.planning.objnav_benchmark_runtime.methods.perception import HttpDetector


@pytest.mark.parametrize("kwargs", [{"conf_thresh": float("nan")}, {"max_det": 0},
                                    {"chunk_size": -1}, {"dtype": "int8"},
                                    {"device": "cpu", "dtype": "float16"},
                                    {"shortest_edge": 2000}])
def test_invalid_config(kwargs):
    with pytest.raises(ValueError):
        LlmDetConfig(**kwargs)


@pytest.mark.parametrize("prompts", [[], "chair", [""], [None], ["chair", "Chair"], ["a.b"]])
def test_ambiguous_vocabulary_rejected(prompts):
    with pytest.raises(ValueError):
        validate_prompts(prompts)


def test_tokens_preserve_complete_label_and_original_coordinates():
    labels = ["Potted Plant", "tv"]
    caption, spans = caption_and_spans(labels)
    assert caption == "potted plant . tv . "
    offsets = [(0, 0), (0, 3), (3, 6), (7, 12), (13, 14), (15, 17), (0, 0)]
    groups = tokens_for_spans(offsets, spans)
    assert groups == [[1, 2, 3], [5]]
    probs = np.array([[0, .9, .3, .6, 1, .1, 0]])
    result = decode_token_detections(probs, [[.5, .5, .5, .5]], groups, labels,
                                     (100, 200), .5, 100)
    assert len(result) == 1 and result[0].label == "Potted Plant"
    assert result[0].score == pytest.approx(.6)
    assert result[0].bbox_xyxy == (50, 25, 150, 75)
    assert (result[0].frame_w, result[0].frame_h) == (200, 100)
    with pytest.raises(ValueError, match="truncated"):
        tokens_for_spans(offsets[:3], spans)


def test_clipping_empty_boxes_and_invalid_output():
    result = decode_token_detections([[.9], [.8]], [[0, .5, 1, 2], [.5, .5, 0, 0]],
                                     [[0]], ["chair"], (40, 80), .1, 10)
    assert [d.bbox_xyxy for d in result] == [(0, 0, 40, 40)]
    with pytest.raises(ValueError, match="Non-finite"):
        decode_token_detections([[np.nan]], [[.5]*4], [[0]], ["chair"], (40, 80), .1, 1)


def test_chunking_never_silently_loses_labels():
    def tokenizer(caption, **kwargs):
        import re
        offsets = [(m.start(), m.end()) for m in re.finditer(r"\S+", caption)]
        return {"input_ids": [1] * (len(offsets) + 2),
                "offset_mapping": [(0, 0)] + offsets + [(0, 0)]}

    detector = LlmDetDetector()
    prompts = ["potted plant", "chair", "door frame", "tv"]
    queries = detector._prepare_queries(prompts, tokenizer, max_tokens=7)
    assert [label for _, labels, _ in queries for label in labels] == prompts
    assert len(queries) > 1
    with pytest.raises(ValueError, match="single category"):
        detector._prepare_queries(["a " * 100], tokenizer, 7)


def test_backend_preprocessor_receives_rgb_not_bgr(monkeypatch):
    detector = LlmDetDetector()
    detector.set_prompts(["chair"])
    detector._model = object()
    detector._queries = [("chair . ", ["chair"], [[1]])]
    seen = []

    def processor(**kwargs):
        seen.append(kwargs)
        raise RuntimeError("captured RGB")

    detector._processor = processor
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(inference_mode=nullcontext))
    rgb = np.zeros((5, 10, 3), np.uint8)
    rgb[..., 0] = 250
    with pytest.raises(RuntimeError, match="captured RGB"):
        detector.detect(rgb[:, ::-1])
    assert np.array_equal(seen[0]["images"], rgb[:, ::-1])
    assert seen[0]["size"] == {"shortest_edge": 800, "longest_edge": 1333}


def test_explicit_backend_threshold_and_registry():
    assert parse_args(["--model", "baseline.pt"]).backend == "yolo_world"
    assert parse_args(["--model", "baseline.pt"]).conf == .25
    with pytest.raises(SystemExit):
        parse_args(["--model", "snapshot", "--backend", "llmdet"])
    with pytest.raises(SystemExit):
        parse_args(["--model", "snapshot", "--backend", "unknown"])
    cfg = LlmDetConfig(conf_thresh=.47)
    detector = default_detection_registry(llmdet_config=cfg).create("llmdet")
    assert detector.cfg == cfg and detector._model is None
    with pytest.raises(KeyError):
        default_detection_registry().create("missing")


def test_missing_weights_and_dependencies_never_substitute(tmp_path, monkeypatch):
    detector = LlmDetDetector(LlmDetConfig(model_path=str(tmp_path)))
    detector.set_prompts(["chair"])
    with pytest.raises(FileNotFoundError):
        detector.detect(np.zeros((4, 4, 3), np.uint8))
    (tmp_path / "model.safetensors").write_text("fixture, not weights")
    monkeypatch.setitem(sys.modules, "transformers", None)
    with pytest.raises(ImportError):
        detector.detect(np.zeros((4, 4, 3), np.uint8))
    assert detector._model is None


def test_checkpoint_identity_covers_weights_tokenizer_and_preprocessing(tmp_path):
    (tmp_path / "model.safetensors").write_bytes(b"test fixture")
    (tmp_path / "config.json").write_text("{}")
    path = tmp_path / "preprocessor_config.json"
    path.write_text('{"size":800}')
    before = backends.checkpoint_identity(tmp_path)
    path.write_text('{"size":1000}')
    after = backends.checkpoint_identity(tmp_path)
    assert before["checkpoint_sha256"] != after["checkpoint_sha256"]
    assert before["checkpoint_files"]["model.safetensors"] == after["checkpoint_files"]["model.safetensors"]
    (tmp_path / "vocab.txt").write_text("chair")
    assert after != backends.checkpoint_identity(tmp_path)


def test_occupied_gpu_refused_before_torch(monkeypatch):
    monkeypatch.setattr(backends.subprocess, "run", lambda *a, **kw: SimpleNamespace(stdout="7000\n"))
    monkeypatch.setitem(sys.modules, "torch", None)
    with pytest.raises(RuntimeError, match="occupied"):
        backends._configure_runtime(SimpleNamespace(device="cuda:0"))


def test_http_rgb_coordinates_and_provenance_drift():
    seen = []

    class Stub:
        def detect(self, rgb):
            seen.append(rgb.copy())
            return [Detection2D("chair", .9, (2, 3, 12, 13), 20, 16)]

    metadata = {"checkpoint_sha256": "fixture", "detector_config": {"backend": "fixture"}}
    ctx = _ServerContext(Stub(), "test", "cpu", ["chair"], metadata)
    server = _make_server(ctx, "127.0.0.1", 0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    client = HttpDetector("http://127.0.0.1:%d" % server.server_port, ["chair"])
    try:
        rgb = np.zeros((16, 20, 3), np.uint8)
        rgb[..., 0] = 230
        detections = client.detect(rgb)
        assert seen[0][..., 0].mean() > 220 and seen[0][..., 2].mean() < 5
        assert detections[0].xyxy == (2, 3, 12, 13)
        ctx.metadata = dict(metadata, backend="changed")
        with pytest.raises(Exception, match="changed"):
            client.detect(rgb)
    finally:
        client.session.close()
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def test_lightweight_imports_do_not_load_model_packages():
    code = ("import sys; from sparx_agency.core.mapping.detection.registry import default_detection_registry; "
            "r=default_detection_registry(); r.create('llmdet'); r.create('yolo_world'); "
            "assert not {'torch','transformers','ultralytics'} & set(sys.modules)")
    subprocess.run([sys.executable, "-c", code], check=True)


def test_accuracy_counts_duplicates_misses_confusion_and_negatives():
    from sparx_agency.tasks.mapping.scene_graph.serve.compare_detectors import assess
    frames = [{"annotations": [{"label": "mug", "xyxy": [0, 0, 16, 16]},
                                {"label": "apple", "xyxy": [20, 20, 30, 30]}]},
              {"annotations": []}]
    box = lambda label, score, xyxy: {"cls": label, "conf": score, "xyxy": xyxy}
    predictions = [[box("mug", .9, [0, 0, 16, 16]), box("mug", .8, [0, 0, 16, 16]),
                    box("bowl", .7, [20, 20, 30, 30])], [box("apple", .6, [0, 0, 5, 5])]]
    result = assess(frames, predictions, ["mug", "apple", "bowl"], .5)
    assert (result["tp"], result["fp"], result["fn"], result["duplicates"]) == (1, 3, 1, 1)
    assert result["precision"] == .25 and result["small_recall"] == .5
    assert result["confusion"] == {"bowl -> apple": 1}
    with pytest.raises(ValueError, match="Every annotated frame"):
        assess(frames, predictions[:1], ["mug"], .5)


def test_persistent_false_hypothesis_can_still_pass_existing_stop_gate():
    # The RGB fixture contains no actual chair. Repeated false boxes are NOT
    # made truthful by viewpoint confirmation; keep this limitation explicit.
    from sparx_agency.core.planning.objnav.types.pose import AgentPose
    from sparx_agency.tasks.planning.objnav_benchmark_runtime.tests.test_method import setup_policy, observation
    policy, episode = setup_policy(label="chair", duplicate=True)
    assert not policy.plan(observation(episode, 0)).stop
    assert not policy.plan(observation(episode, 1)).stop
    obs = replace(observation(episode, 2), pose=AgentPose(0, .25, 0, 0))
    assert policy.plan(obs).stop
    assert policy.landmarks.all_landmarks()[0].count == 3


def test_loader_never_ties_different_pretrained_decoder_stages():
    from sparx_agency.core.mapping.detection.llmdet_loading import checkpoint_model_class
    original = {r"bbox_embed.(?![0])\d+": "bbox_embed.0"}
    base = type("Upstream", (), {"_tied_weights_keys": original})
    cls = checkpoint_model_class(base)
    assert cls._tied_weights_keys == {"model.decoder.bbox_embed": "bbox_embed",
                                     "model.decoder.class_embed": "class_embed"}
    assert base._tied_weights_keys == original  # no global upstream monkeypatch


def test_silent_pretrained_tensor_substitution_fails(monkeypatch):
    from sparx_agency.core.mapping.detection.llmdet_loading import verify_decoder_weights

    class Tensor:
        dtype = "float32"

        def __init__(self, value):
            self.value = value

        def detach(self):
            return self

        def cpu(self):
            return self

        def to(self, **kwargs):
            return self

    key = "model.decoder.bbox_embed.1.layers.2.bias"
    checkpoint = SimpleNamespace(keys=lambda: [key], get_tensor=lambda name: Tensor(2))
    monkeypatch.setitem(sys.modules, "safetensors", SimpleNamespace(
        safe_open=lambda *a, **kw: nullcontext(checkpoint)))
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(equal=lambda a, b: a.value == b.value))
    model = SimpleNamespace(state_dict=lambda: {key: Tensor(1)})
    with pytest.raises(RuntimeError, match="substituted"):
        verify_decoder_weights(model, "fixture.safetensors")
    model.state_dict = lambda: {key: Tensor(2)}
    assert verify_decoder_weights(model, "fixture.safetensors") == 1


def test_yolo_prompt_feature_provenance_changes_with_reprompting():
    values = np.ones((1, 2, 4), np.float32)
    features = SimpleNamespace(numpy=lambda: values)
    features.detach = features.float = features.cpu = lambda: features
    detector = SimpleNamespace(cfg=SimpleNamespace(imgsz=640),
                               _model=SimpleNamespace(model=SimpleNamespace(txt_feats=features)))
    before = backends.yolo_preprocessing(detector)
    values[0, 0, 0] = 0
    after = backends.refresh_vocabulary_metadata(detector, {"backend": "yolo_world"})
    assert before["prompt_features_sha256"] != after["preprocessing"]["prompt_features_sha256"]


def test_capture_refuses_nontraining_scenes_before_gpu_access(tmp_path, monkeypatch):
    from sparx_agency.tasks.mapping.scene_graph.serve.capture_detector_dev import capture
    monkeypatch.setattr(subprocess, "check_output", lambda *a, **kw: pytest.fail("GPU was queried"))
    args = SimpleNamespace(output=tmp_path / "new", scenes=["FloorPlan_Train1_1",
                                                           "FloorPlan_Train2_1", "FloorPlan_Val1_1"])
    with pytest.raises(ValueError, match="training scenes"):
        capture(args)

