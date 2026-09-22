"""Synthetic contract tests, not detector or staircase-accuracy measurements."""
from contextlib import nullcontext
from dataclasses import replace
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from sparx_agency.core.common.types.perception import Detection2D
from sparx_agency.core.mapping.detection.blip2 import Blip2Config, Blip2Verifier, answer_verdict
from sparx_agency.core.mapping.detection.grounded_vlm import GroundedVlmConfig, GroundedVlmDetector
from sparx_agency.core.mapping.detection.grounding_dino_loading import require_materialized, shared_bbox_model_class
from sparx_agency.core.mapping.detection.registry import default_detection_registry
from sparx_agency.core.mapping.detection.yolo_world import YoloWorldConfig


IMAGE = np.zeros((40, 80, 3), np.uint8)


def box(label="stairs", score=.8, xyxy=(10, 5, 30, 35)):
    return Detection2D(label, score, xyxy, 80, 40)


class ProposalStub:
    def __init__(self, detections):
        self.detections, self.calls = detections, 0

    def set_prompts(self, prompts):
        self.prompts = list(prompts)

    def detect(self, rgb):
        self.calls += 1
        return self.detections


class VerifierStub:
    def __init__(self, verdict="yes"):
        self.verdict, self.calls = verdict, []

    def load(self):
        pass

    def verify(self, rgb, detection):
        self.calls.append((rgb.copy(), detection))
        return {"verdict": self.verdict, "answers": [self.verdict] * 2}


def pipeline(dino=(), yolo=None, **kwargs):
    config = GroundedVlmConfig(yolo=YoloWorldConfig(device="cpu") if yolo is not None else None, **kwargs)
    detector = GroundedVlmDetector(config)
    detector.grounding = ProposalStub(dino)
    detector.yolo = ProposalStub(yolo) if yolo is not None else None
    detector.verifier = VerifierStub()
    detector.set_prompts(["stairs", "staircase", "bed", "chair", "sofa"])
    return detector


def test_dino_can_find_stairs_when_yolo_misses_everything():
    detector = pipeline([box()], yolo=[])
    assert detector.detect(IMAGE) == [box()]
    assert detector.yolo.calls == detector.grounding.calls == 1
    assert detector.diagnostics()["raw_detections"][0]["source"] == "grounding_dino"


def test_grounded_only_does_not_construct_yolo(monkeypatch):
    import sparx_agency.core.mapping.detection.grounded_vlm as module
    monkeypatch.setitem(module.__dict__, "YoloWorldDetector", lambda *a: pytest.fail("YOLO constructed"))
    assert default_detection_registry().create("grounded_vlm").yolo is None


def test_all_modes_remain_lazy_on_import_and_construction():
    code = """import sys
from sparx_agency.core.mapping.detection.registry import default_detection_registry
registry = default_detection_registry()
for name in registry.names():
    registry.create(name).set_prompts(['stairs', 'bed'])
assert not {'torch', 'transformers', 'ultralytics'} & set(sys.modules)
"""
    subprocess.run([sys.executable, "-c", code], check=True)


def test_yolo_mode_never_constructs_other_models(monkeypatch):
    import sparx_agency.core.mapping.detection.grounded_vlm as module
    monkeypatch.setattr(module, "GroundedVlmDetector", lambda *a: pytest.fail("VLM constructed"))
    detector = default_detection_registry().create("yolo_world")
    assert detector._model is None


def test_duplicate_scores_are_not_averaged_or_counted_twice():
    detector = pipeline([box(score=.99)], yolo=[box(score=.65)])
    result = detector.detect(IMAGE)
    assert len(result) == 1 and result[0].score == .65
    assert len(detector.verifier.calls) == 1
    rows = detector.diagnostics()["raw_detections"]
    assert [row["status"] for row in rows] == ["verified", "duplicate"]
    assert rows[1]["conf"] == .99


@pytest.mark.parametrize("verdict", ["no", "uncertain"])
def test_rejected_or_uncertain_candidates_never_pass(verdict):
    detector = pipeline([box()])
    detector.verifier.verdict = verdict
    assert detector.detect(IMAGE) == []
    assert detector.diagnostics()["raw_detections"][0]["status"] == verdict


def test_budget_overflow_abstains_and_stairs_have_priority():
    detector = pipeline([box("bed", xyxy=(40, 5, 70, 35)), box()], max_verifications=1)
    assert detector.detect(IMAGE) == [box()]
    assert detector.diagnostics()["raw_detections"][0]["status"] == "budget_exhausted"


def test_verification_scope_and_all_classes_option():
    detector = pipeline([box("chair")], verify_labels=("stairs", "bed"))
    assert detector.detect(IMAGE) and not detector.verifier.calls
    assert detector.diagnostics()["raw_detections"][0]["status"] == "not_requested"
    detector = pipeline([box("chair")])
    detector.verifier.verdict = "uncertain"
    assert not detector.detect(IMAGE) and detector.verifier.calls


def test_conflicting_stair_and_bed_confirmations_both_abstain():
    detector = pipeline([box()], yolo=[box("bed")])
    assert not detector.detect(IMAGE)
    assert all(row["status"] == "conflicting_labels" for row in detector.diagnostics()["raw_detections"])


def test_a_real_bed_beside_stairs_is_not_blacklisted():
    detector = pipeline([box()], yolo=[box("bed", xyxy=(40, 5, 70, 35))])
    assert {item.label for item in detector.detect(IMAGE)} == {"stairs", "bed"}


def test_no_answer_cache_between_frames_and_no_half_reprompt():
    detector = pipeline([box()])
    assert detector.detect(IMAGE)
    detector.verifier.verdict = "no"
    assert not detector.detect(IMAGE + 1)
    assert len(detector.verifier.calls) == 2
    before = detector.prompts
    with pytest.raises(ValueError, match="Restart"):
        detector.set_prompts(["toilet"])
    assert detector.prompts == detector.grounding.prompts == before


def test_model_failures_propagate_without_yolo_fallback(monkeypatch):
    detector = pipeline([box()], yolo=[box()])
    detector.detect(IMAGE)
    def fail(*args):
        raise RuntimeError("model failure")
    monkeypatch.setitem(detector.verifier.__dict__, "verify", fail)
    with pytest.raises(RuntimeError, match="model failure"):
        detector.detect(IMAGE)
    assert detector.diagnostics() == {}


@pytest.mark.parametrize("answer,expected", [("Yes.", "yes"), (" no ", "no"),
    ("uncertain", "uncertain"), ("yes and no", "uncertain"), ("not sure", "uncertain")])
def test_answer_parser_does_not_guess(answer, expected):
    assert answer_verdict(answer) == expected


def test_blip_receives_context_and_crop_without_mutating_rgb(monkeypatch):
    verifier = Blip2Verifier()
    seen = []
    monkeypatch.setattr(verifier, "load", lambda: None)
    def answer(images, questions):
        seen.extend(images)
        assert "red rectangle" in questions[0]
        return ["yes", "uncertain"]
    monkeypatch.setattr(verifier, "_answer", answer)
    assert verifier.verify(IMAGE, box())["verdict"] == "uncertain"
    assert [image.size for image in seen] == [(80, 40), (20, 30)]
    assert np.all(IMAGE == 0)


@pytest.mark.parametrize("kwargs", [{"max_verifications": 0}, {"duplicate_iou": float("nan")},
                                     {"verify_labels": "stairs"}])
def test_invalid_pipeline_config(kwargs):
    with pytest.raises(ValueError):
        GroundedVlmConfig(**kwargs)


def test_invalid_blip_config_and_missing_checkpoint(tmp_path):
    with pytest.raises(ValueError):
        Blip2Config(dtype="float16")
    with pytest.raises(FileNotFoundError):
        Blip2Verifier(Blip2Config(model_path=str(tmp_path))).load()


def test_invalid_frame_box_and_model_dimensions():
    detector = pipeline([replace(box(), frame_w=100)])
    with pytest.raises(ValueError, match="frame"):
        detector.detect(IMAGE)


def test_grounding_aliases_point_directly_to_the_saved_head(monkeypatch):
    canonical = "model.decoder.bbox_embed.0"
    keys = [canonical + ".layers.%d.%s" % (i, suffix) for i in range(3) for suffix in ("weight", "bias")]
    monkeypatch.setitem(sys.modules, "safetensors", SimpleNamespace(
        safe_open=lambda *a, **kw: nullcontext(SimpleNamespace(keys=lambda: keys))))
    config = SimpleNamespace(decoder_bbox_embed_share=True, two_stage_bbox_embed_share=False, decoder_layers=6)
    original = {"bad": "alias"}
    base = type("Base", (), {"_tied_weights_keys": original})
    model_class = shared_bbox_model_class(base, config, "fixture")
    assert len(model_class._tied_weights_keys) == 11
    assert set(model_class._tied_weights_keys.values()) == {canonical}
    assert canonical not in model_class._tied_weights_keys
    assert base._tied_weights_keys == original
    keys.append("model.decoder.bbox_embed.1.layers.0.weight")
    with pytest.raises(ValueError, match="layout"):
        shared_bbox_model_class(base, config, "fixture")


def test_unmaterialized_tensors_never_pass():
    model = SimpleNamespace(named_parameters=lambda: [("head", SimpleNamespace(is_meta=True))],
                            named_buffers=lambda: [])
    with pytest.raises(RuntimeError, match="Unmaterialized"):
        require_materialized(model)


