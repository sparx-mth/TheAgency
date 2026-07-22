"""Unit tests for :class:`MedianFlowBoxTracker`.

Beyond the basic seed/track/lose contract these pin the *robustness* the tracker
exists for: when the object it locked onto is replaced by background, it declares
loss (the appearance model + forward-backward check refuse the box) instead of
plain LK's failure mode of confidently reporting a box on the background. All
frames are built from a fixed RNG so the corners the flow locks onto are
deterministic.
"""
from __future__ import annotations

from typing import Tuple

import numpy as np
import cv2
import pytest

from sparx_agency.core.mapping.tracking.interface import BoxObservation
from sparx_agency.core.mapping.tracking.median_flow_box_tracker import (
    MedianFlowBoxTracker,
    MedianFlowConfig,
)

SEED = 20260712
FRAME_SIZE = (200, 200)  # (H, W)
BOX = (60.0, 60.0, 140.0, 140.0)  # (x1, y1, x2, y2)


def _background(rng: np.random.Generator, size=FRAME_SIZE) -> np.ndarray:
    """A dim, mildly textured field covering the whole frame."""
    h, w = size
    img = rng.integers(20, 55, size=(h, w), dtype=np.uint8)
    for _ in range(40):
        x, y = int(rng.integers(0, w - 8)), int(rng.integers(0, h - 8))
        cv2.rectangle(img, (x, y), (x + int(rng.integers(3, 8)), y + int(rng.integers(3, 8))),
                      int(rng.integers(55, 100)), -1)
    return img


def _paint_object(img: np.ndarray, box, rng: np.random.Generator) -> np.ndarray:
    """Overwrite the box with a distinct, bright, richly textured 'object'."""
    out = img.copy()
    x1, y1, x2, y2 = (int(v) for v in box)
    for _ in range(50):
        rx, ry = int(rng.integers(x1, x2 - 6)), int(rng.integers(y1, y2 - 6))
        cv2.rectangle(out, (rx, ry),
                      (min(rx + int(rng.integers(3, 11)), x2), min(ry + int(rng.integers(3, 11)), y2)),
                      int(rng.integers(180, 256)), -1)
    return out


def _center(bbox) -> Tuple[float, float]:
    x1, y1, x2, y2 = bbox
    return 0.5 * (x1 + x2), 0.5 * (y1 + y2)


@pytest.fixture()
def rng() -> np.random.Generator:
    return np.random.default_rng(SEED)


@pytest.fixture()
def object_frame(rng: np.random.Generator) -> np.ndarray:
    return _paint_object(_background(rng), BOX, rng)


# ── config validation ────────────────────────────────────────────────────
def test_config_rejects_bad_values() -> None:
    with pytest.raises(ValueError):
        MedianFlowConfig(inner_frac=0.0)
    with pytest.raises(ValueError):
        MedianFlowConfig(min_survival_frac=1.5)
    with pytest.raises(ValueError):
        MedianFlowConfig(fb_max_error=0.0)


# ── basic contract ───────────────────────────────────────────────────────
def test_seed_on_textured_box_succeeds(object_frame: np.ndarray) -> None:
    tracker = MedianFlowBoxTracker()
    assert tracker.seed(object_frame, BOX) is True
    assert tracker.is_valid is True


def test_seed_on_tiny_flat_region_fails() -> None:
    tracker = MedianFlowBoxTracker()
    flat = np.full(FRAME_SIZE, 127, dtype=np.uint8)
    assert tracker.seed(flat, (10.0, 10.0, 12.0, 12.0)) is False
    assert tracker.is_valid is False


def test_seed_on_large_flat_region_fails() -> None:
    """A big but textureless box has no appearance to lock onto -> refuse to seed."""
    tracker = MedianFlowBoxTracker()
    flat = np.full(FRAME_SIZE, 127, dtype=np.uint8)
    assert tracker.seed(flat, (40.0, 40.0, 160.0, 160.0)) is False


def test_update_before_seed_is_invalid(object_frame: np.ndarray) -> None:
    tracker = MedianFlowBoxTracker()
    obs = tracker.update(object_frame)
    assert obs.valid is False
    assert obs.bbox_xyxy is None
    assert obs.n_matches == 0


def test_update_same_frame_holds_box(object_frame: np.ndarray) -> None:
    tracker = MedianFlowBoxTracker()
    assert tracker.seed(object_frame, BOX) is True
    obs = tracker.update(object_frame)
    assert isinstance(obs, BoxObservation)
    assert obs.valid is True
    mcx, mcy = _center(obs.bbox_xyxy)
    scx, scy = _center(BOX)
    assert abs(mcx - scx) <= 6.0 and abs(mcy - scy) <= 6.0


def test_tracks_small_translation(object_frame: np.ndarray) -> None:
    """Rolling the whole image +x moves the box centre the same way and amount."""
    shift = 4
    tracker = MedianFlowBoxTracker()
    assert tracker.seed(object_frame, BOX) is True
    base = tracker.update(object_frame)
    base_cx, base_cy = _center(base.bbox_xyxy)

    moved = tracker.update(np.roll(object_frame, shift, axis=1))
    assert moved.valid is True
    mcx, mcy = _center(moved.bbox_xyxy)
    assert (mcx - base_cx) > 0.0
    assert abs((mcx - base_cx) - shift) <= 2.0
    assert abs(mcy - base_cy) <= 2.0


def test_reset_invalidates_tracker(object_frame: np.ndarray) -> None:
    tracker = MedianFlowBoxTracker()
    assert tracker.seed(object_frame, BOX) is True
    tracker.reset()
    assert tracker.is_valid is False
    assert tracker.update(object_frame).valid is False


# ── the robustness property this tracker exists for ──────────────────────
def test_declares_loss_when_object_replaced_by_background(rng: np.random.Generator) -> None:
    """Seed on the object; next frame the object is GONE (box region is now plain
    background). The tracker must report loss, not a confident box on the
    background — the exact plain-LK failure Median-Flow's appearance + FB checks
    are here to prevent."""
    bg = _background(rng)
    with_obj = _paint_object(bg, BOX, rng)   # object present
    without_obj = bg                         # object gone, background remains in box

    tracker = MedianFlowBoxTracker()
    assert tracker.seed(with_obj, BOX) is True
    obs = tracker.update(without_obj)
    assert obs.valid is False
    assert obs.bbox_xyxy is None
    assert tracker.is_valid is False


def test_disabling_appearance_still_loses_on_blank(object_frame: np.ndarray) -> None:
    """Even with the appearance template off, a blank frame kills lock via FB /
    survival: no honest correspondence exists, so the track is not kept."""
    tracker = MedianFlowBoxTracker(MedianFlowConfig(template_size=0))
    assert tracker.seed(object_frame, BOX) is True
    blank = np.full(FRAME_SIZE, 127, dtype=np.uint8)
    assert tracker.update(blank).valid is False
