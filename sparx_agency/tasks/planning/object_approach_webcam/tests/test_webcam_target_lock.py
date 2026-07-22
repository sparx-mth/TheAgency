"""Headless tests for the webcam object-approach demo.

No camera and no display: synthetic frames stand in for the webcam, so the colour
detector, the folder frame source, the frame geometry, and — most importantly — the
full detect -> track -> servo -> FSM lifecycle (through the real
``TargetLockPipeline``) are all exercised end-to-end. Uses a *textured* red object,
because the robust Median-Flow tracker honestly refuses a flat, textureless blob
(a real webcam object always has surface texture).
"""
from __future__ import annotations

import os
from collections import Counter

import numpy as np
import cv2
import pytest

from sparx_agency.core.common.types import Intrinsics
from sparx_agency.core.planning.visual_servo import VisualServoParams, ConfirmationGateConfig
from sparx_agency.tasks.planning.object_approach_offline.pipeline import TargetLockPipeline
from sparx_agency.tasks.planning.object_approach_offline import overlay
from sparx_agency.tasks.planning.object_approach_webcam.color_detector import (
    ColorDetectorConfig,
    MockColorDetector,
)
from sparx_agency.tasks.planning.object_approach_webcam.detector_factory import make_webcam_detector
from sparx_agency.tasks.planning.object_approach_webcam.webcam_frame_publisher import center_crop_resize
from sparx_agency.tasks.planning.object_approach_webcam.run_webcam_target_lock import _FolderSource

W, H = 240, 160
BOX = (90, 45, 170, 125)          # the red object, centred-ish
TARGET = "cup"


def _frame(present: bool, seed: int = 0) -> np.ndarray:
    """A dim grey scene; if ``present``, a textured RED object in BOX (BGR)."""
    rng = np.random.default_rng(seed)
    bgr = np.full((H, W, 3), 90, np.uint8)           # low-saturation grey background
    if present:
        x1, y1, x2, y2 = BOX
        for _ in range(50):                          # varying-brightness red patches
            rx, ry = int(rng.integers(x1, x2 - 6)), int(rng.integers(y1, y2 - 6))
            r = int(rng.integers(140, 256))
            cv2.rectangle(bgr, (rx, ry),
                          (min(rx + int(rng.integers(4, 12)), x2),
                           min(ry + int(rng.integers(4, 12)), y2)),
                          (20, 20, r), -1)            # BGR: strong red, low B/G
    return bgr


def _make_pipeline(lock_mode: str) -> TargetLockPipeline:
    intr = Intrinsics(width=W, height=H, fx=float(W), fy=float(W), cx=W / 2, cy=H / 2)
    return TargetLockPipeline(
        target=TARGET, intrinsics=intr, lock_mode=lock_mode,
        servo_params=VisualServoParams(use_depth=False),
        gate_config=ConfirmationGateConfig(n_confirm=3, min_score=0.30))


# ── colour detector ──────────────────────────────────────────────────────
def test_color_detector_finds_red_object_labelled_target():
    det = MockColorDetector(ColorDetectorConfig(color="red"))
    det.set_prompts([TARGET])
    rgb = cv2.cvtColor(_frame(present=True), cv2.COLOR_BGR2RGB)
    dets = det.detect(rgb)
    assert len(dets) == 1
    assert dets[0].label == TARGET
    # The detected box overlaps the drawn object box.
    x1, y1, x2, y2 = dets[0].bbox_xyxy
    assert x1 < BOX[2] and x2 > BOX[0] and y1 < BOX[3] and y2 > BOX[1]


def test_color_detector_empty_scene_returns_nothing():
    det = MockColorDetector(ColorDetectorConfig(color="red"))
    det.set_prompts([TARGET])
    rgb = cv2.cvtColor(_frame(present=False), cv2.COLOR_BGR2RGB)
    assert det.detect(rgb) == []


# ── full lifecycle through the real pipeline ─────────────────────────────
@pytest.mark.parametrize("lock_mode", ["detector_tracker", "detector"])
def test_lifecycle_acquire_then_lose_then_recover(lock_mode):
    det = make_webcam_detector("color", target=TARGET, color="red")
    pipe = _make_pipeline(lock_mode)

    states = []
    captions = []
    # 20 frames object present (acquire + approach), then 40 with it gone.
    for i in range(60):
        present = i < 20
        bgr = _frame(present=present, seed=i if present else 0)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        res = pipe.step(bgr, i / 15.0, det.detect(rgb))
        states.append(res.fsm_mode)
        captions.append(overlay.classify_lock(res)[0])
        overlay.render(bgr, res)                    # must not raise

    seen = Counter(states)
    # It acquires (leaves SEARCH) while the object is present...
    assert "APPROACH" in seen or "HOVER_LOCK" in seen, (lock_mode, seen)
    # ...and once the object is gone it recovers, then eventually re-searches.
    assert "RECOVER" in seen, (lock_mode, seen)
    # A green (DETECTED) frame happened while the object was visible.
    assert "DETECTED" in captions[:20]


# ── frame source (rolling folder the publisher writes) ───────────────────
def test_folder_source_returns_newest_once(tmp_path):
    src = _FolderSource(str(tmp_path))
    assert src.read() is None                        # empty folder
    cv2.imwrite(str(tmp_path / "frame_00000000.jpg"), _frame(True))
    a = src.read()
    assert a is not None and a.shape[:2] == (H, W)
    assert src.read() is None                        # no new frame yet
    p1 = tmp_path / "frame_00000001.jpg"
    cv2.imwrite(str(p1), _frame(False))
    os.utime(p1, (2e9, 2e9))                         # ensure a strictly newer mtime
    assert src.read() is not None                    # a newer frame appeared


def test_folder_source_picks_newest_by_mtime_not_name(tmp_path):
    # A leftover from a previous run: HIGH number but OLD mtime; a fresh frame with
    # a LOW number but NEW mtime. The reader must pick the fresh one.
    stale = tmp_path / "frame_00000500.jpg"          # old run, high number, grey
    cv2.imwrite(str(stale), _frame(present=False))
    os.utime(stale, (1000, 1000))
    fresh = tmp_path / "frame_00000001.jpg"          # this run, low number, red object
    cv2.imwrite(str(fresh), _frame(present=True))
    os.utime(fresh, (2000, 2000))

    img = _FolderSource(str(tmp_path)).read()
    assert img is not None
    assert int(cv2.split(img)[2].max()) > 130        # the RED (fresh) frame, not the grey leftover


def test_clear_stale_frames_removes_leftovers(tmp_path):
    from sparx_agency.tasks.planning.object_approach_webcam.webcam_frame_publisher import (
        clear_stale_frames,
    )
    for i in (0, 5, 500):
        cv2.imwrite(str(tmp_path / ("frame_%08d.jpg" % i)), _frame(present=False))
    (tmp_path / "frame_00000006.jpg.part").write_bytes(b"x")   # a temp write in flight
    assert clear_stale_frames(tmp_path) == 4
    assert list(tmp_path.glob("frame_*")) == []


# ── frame geometry (drone resolution) ────────────────────────────────────
def test_center_crop_resize_hits_target_resolution():
    cam = np.zeros((480, 640, 3), np.uint8)
    out = center_crop_resize(cam, 504, 294)
    assert out.shape[:2] == (294, 504)


# ── detector factory ─────────────────────────────────────────────────────
def test_factory_unknown_kind_raises():
    with pytest.raises(ValueError):
        make_webcam_detector("magic", target=TARGET)


def test_factory_builds_yoloworld_and_color():
    from sparx_agency.core.mapping.detection import YoloWorldDetector
    # The real detector is open-vocab YOLO-World (never plain COCO); 'yolo' aliases it.
    assert isinstance(make_webcam_detector("yoloworld", target=TARGET), YoloWorldDetector)
    assert isinstance(make_webcam_detector("yolo", target=TARGET), YoloWorldDetector)
    assert isinstance(make_webcam_detector("color", target=TARGET), MockColorDetector)


def test_yoloworld_defers_ultralytics_to_detect():
    # Building + prompting a YOLO-World detector needs no ultralytics (lazy load);
    # only detect() does. Where ultralytics is absent, that surfaces a clear hint.
    det = make_webcam_detector("yoloworld", target=TARGET)   # must not raise here
    try:
        import ultralytics  # noqa: F401
    except ImportError:
        with pytest.raises(ImportError, match="ultralytics"):
            det.detect(np.zeros((16, 16, 3), np.uint8))
    else:
        pytest.skip("ultralytics is installed; the missing-dep path cannot be tested")
