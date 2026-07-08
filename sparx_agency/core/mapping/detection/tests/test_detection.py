"""Unit tests for the open-vocabulary detection core.

These tests exercise the config validation, the ``YoloWorldDetector`` prompt /
detect contract, and the ``DetectionRegistry`` factory idiom **without** loading
a real YOLO-World model. ``ultralytics``/``torch`` may be absent (GPU-free CI /
Python-3.8 Noetic side): construction and prompt staging are lazy, and the only
place a heavy import happens (``_ensure_model``) is forced to fail deterministically
by shadowing ``sys.modules['ultralytics']`` so the test does not depend on whether
ultralytics is installed.
"""
from __future__ import annotations

import sys
from typing import List, Sequence

import numpy as np
import pytest

from sparx_agency.core.common.types.perception import Detection2D
from sparx_agency.core.mapping.interfaces.detection_model import DetectionModel
from sparx_agency.core.mapping.detection.yolo_world import (
    YoloWorldConfig,
    YoloWorldDetector,
)
from sparx_agency.core.mapping.detection.registry import (
    DetectionRegistry,
    DetectorFactory,
    default_detection_registry,
)


# ── YoloWorldConfig validation ───────────────────────────────────────────
def test_config_defaults_ok():
    cfg = YoloWorldConfig()
    assert cfg.model_path == "yolov8s-world.pt"
    assert 0.0 <= cfg.conf_thresh <= 1.0
    assert 0.0 <= cfg.iou_thresh <= 1.0
    assert cfg.imgsz > 0


def test_config_rejects_empty_model_path():
    with pytest.raises(ValueError):
        YoloWorldConfig(model_path="")
    with pytest.raises(ValueError):
        YoloWorldConfig(model_path="   ")


@pytest.mark.parametrize("conf", [-0.01, 1.5])
def test_config_rejects_conf_out_of_range(conf):
    with pytest.raises(ValueError):
        YoloWorldConfig(conf_thresh=conf)


@pytest.mark.parametrize("iou", [-0.1, 2.0])
def test_config_rejects_iou_out_of_range(iou):
    with pytest.raises(ValueError):
        YoloWorldConfig(iou_thresh=iou)


@pytest.mark.parametrize("imgsz", [0, -640])
def test_config_rejects_nonpositive_imgsz(imgsz):
    with pytest.raises(ValueError):
        YoloWorldConfig(imgsz=imgsz)


def test_config_accepts_boundary_thresholds():
    cfg = YoloWorldConfig(conf_thresh=0.0, iou_thresh=1.0, imgsz=1)
    assert cfg.conf_thresh == 0.0
    assert cfg.iou_thresh == 1.0
    assert cfg.imgsz == 1


# ── YoloWorldDetector construction & prompts ─────────────────────────────
def test_detector_constructs_without_ultralytics():
    # No config -> default config; no heavy import should happen here.
    det = YoloWorldDetector()
    assert isinstance(det, DetectionModel)
    assert det.prompts == []
    # Custom config is stored.
    det2 = YoloWorldDetector(YoloWorldConfig(device="cpu"))
    assert det2.cfg.device == "cpu"


def test_set_prompts_empty_raises():
    det = YoloWorldDetector(YoloWorldConfig(device="cpu"))
    with pytest.raises(ValueError):
        det.set_prompts([])
    # All-whitespace prompts are cleaned away -> also empty -> raises.
    with pytest.raises(ValueError):
        det.set_prompts(["", "   "])


def test_set_prompts_stores_and_cleans():
    det = YoloWorldDetector(YoloWorldConfig(device="cpu"))
    det.set_prompts([" chair ", "refrigerator", ""])
    # Whitespace stripped, empty dropped, order preserved.
    assert det.prompts == ["chair", "refrigerator"]
    # `prompts` returns a copy: mutating it must not affect internal state.
    got = det.prompts
    got.append("mutated")
    assert det.prompts == ["chair", "refrigerator"]


def test_detect_before_set_prompts_raises_runtimeerror():
    det = YoloWorldDetector(YoloWorldConfig(device="cpu"))
    rng = np.random.default_rng(0)
    rgb = rng.integers(0, 256, size=(8, 8, 3), dtype=np.uint8)
    with pytest.raises(RuntimeError):
        det.detect(rgb)


def test_detect_with_prompts_but_no_ultralytics_raises(monkeypatch):
    # Force the lazy ultralytics import to fail deterministically regardless of
    # whether ultralytics is actually installed: a None entry in sys.modules
    # makes `from ultralytics import ...` raise ImportError.
    monkeypatch.setitem(sys.modules, "ultralytics", None)
    det = YoloWorldDetector(YoloWorldConfig(device="cpu"))
    det.set_prompts(["chair"])
    rng = np.random.default_rng(42)
    rgb = rng.integers(0, 256, size=(8, 8, 3), dtype=np.uint8)
    # _ensure_model re-raises as ImportError; assert it raises (type-tolerant).
    with pytest.raises(Exception):
        det.detect(rgb)


def test_detect_bad_shape_raises_valueerror():
    # Shape check happens after the prompt guard but before the heavy model load.
    det = YoloWorldDetector(YoloWorldConfig(device="cpu"))
    det.set_prompts(["chair"])
    with pytest.raises(ValueError):
        det.detect(np.zeros((8, 8), dtype=np.uint8))  # missing channel dim
    with pytest.raises(ValueError):
        det.detect(np.zeros((8, 8, 4), dtype=np.uint8))  # wrong channel count


# ── Registry / factory idiom ─────────────────────────────────────────────
def test_default_registry_lists_yolo_world():
    reg = default_detection_registry()
    assert "yolo_world" in reg.names()


def test_registry_create_returns_detection_model_lazily():
    reg = default_detection_registry()
    det = reg.create("yolo_world")
    assert isinstance(det, DetectionModel)
    assert isinstance(det, YoloWorldDetector)
    # Construction is lazy: no model loaded, no prompts yet.
    assert det.prompts == []


def test_registry_create_unknown_raises_keyerror():
    reg = default_detection_registry()
    with pytest.raises(KeyError):
        reg.create("bogus")


def test_registry_duplicate_register_raises_valueerror():
    reg = default_detection_registry()
    with pytest.raises(ValueError):
        reg.register(
            DetectorFactory(name="yolo_world", create=lambda: YoloWorldDetector())
        )


def test_registry_register_and_create_custom_factory():
    reg = DetectionRegistry()
    reg.register(DetectorFactory(name="fake", create=lambda: _FakeDetector()))
    assert reg.names() == ["fake"]
    det = reg.create("fake")
    assert isinstance(det, _FakeDetector)


# ── Minimal ABC satisfiability proof ─────────────────────────────────────
class _FakeDetector(DetectionModel):
    """Tiny concrete ``DetectionModel`` proving the ABC is satisfiable."""

    def __init__(self) -> None:
        self._prompts: List[str] = []

    def set_prompts(self, prompts: Sequence[str]) -> None:
        self._prompts = [str(p) for p in prompts]

    def detect(self, rgb: np.ndarray) -> List[Detection2D]:
        h, w = int(rgb.shape[0]), int(rgb.shape[1])
        return [
            Detection2D(
                label="fake",
                score=0.9,
                bbox_xyxy=(0, 0, w, h),
                frame_w=w,
                frame_h=h,
            )
        ]


def test_fake_detector_satisfies_abc():
    det = _FakeDetector()
    assert isinstance(det, DetectionModel)
    det.set_prompts(["anything"])
    rng = np.random.default_rng(7)
    rgb = rng.integers(0, 256, size=(6, 10, 3), dtype=np.uint8)
    dets = det.detect(rgb)
    assert len(dets) == 1
    d = dets[0]
    assert isinstance(d, Detection2D)
    assert d.label == "fake"
    assert d.bbox_xyxy == (0, 0, 10, 6)
    assert d.frame_w == 10 and d.frame_h == 6
