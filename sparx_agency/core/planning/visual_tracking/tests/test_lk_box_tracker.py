"""Unit tests for :class:`LucasKanadeBoxTracker`.

These exercise the sparse Lucas-Kanade box tracker end-to-end on *synthetic*
grayscale frames: a richly-textured square (many small high-contrast rectangles)
on a faint noise background, so Shi-Tomasi (``goodFeaturesToTrack``) reliably
finds well-localised corners inside the seed box and pyramidal LK tracks them
under small translations. Everything is seeded from a fixed ``numpy`` RNG so the
frames — and therefore the corners LK locks onto — are deterministic.
"""
from __future__ import annotations

from typing import Tuple

import numpy as np
import cv2
import pytest

from sparx_agency.core.planning.visual_tracking.interface import BoxObservation
from sparx_agency.core.planning.visual_tracking.lk_box_tracker import (
    LKBoxTrackerConfig,
    LucasKanadeBoxTracker,
)

SEED = 1234
FRAME_SIZE = (200, 200)  # (H, W)
BOX = (50.0, 50.0, 150.0, 150.0)  # (x1, y1, x2, y2)


def _make_textured_frame(
    rng: np.random.Generator,
    size: Tuple[int, int] = FRAME_SIZE,
    box: Tuple[float, float, float, float] = BOX,
) -> np.ndarray:
    """Build a HxW uint8 frame: a strongly textured box on a faint noise field.

    The box is filled with many small, randomly placed high-contrast rectangles.
    This yields plenty of sharp corners spread across the box (good for both
    Shi-Tomasi seeding and stable LK tracking), while the dim background carries
    almost no gradient so corners concentrate inside the box.
    """
    h, w = size
    img = np.full((h, w), 30, dtype=np.uint8)
    img = (img + rng.integers(0, 15, size=(h, w), dtype=np.uint8)).astype(np.uint8)
    x1, y1, x2, y2 = (int(v) for v in box)
    for _ in range(50):
        rx1 = int(rng.integers(x1, x2 - 6))
        ry1 = int(rng.integers(y1, y2 - 6))
        rw = int(rng.integers(3, 12))
        rh = int(rng.integers(3, 12))
        val = int(rng.integers(120, 256))
        cv2.rectangle(img, (rx1, ry1), (min(rx1 + rw, x2), min(ry1 + rh, y2)), val, -1)
    return img


def _center(bbox: Tuple[float, float, float, float]) -> Tuple[float, float]:
    x1, y1, x2, y2 = bbox
    return 0.5 * (x1 + x2), 0.5 * (y1 + y2)


@pytest.fixture()
def rng() -> np.random.Generator:
    return np.random.default_rng(SEED)


@pytest.fixture()
def frame(rng: np.random.Generator) -> np.ndarray:
    return _make_textured_frame(rng)


def test_seed_on_textured_box_succeeds(frame: np.ndarray) -> None:
    """Seeding a well-textured box finds enough corners and validates the tracker."""
    tracker = LucasKanadeBoxTracker()
    assert tracker.seed(frame, BOX) is True
    assert tracker.is_valid is True


def test_seed_on_tiny_flat_region_fails() -> None:
    """A tiny (2px) region on a flat image yields no corners -> seed returns False."""
    tracker = LucasKanadeBoxTracker()
    flat = np.full(FRAME_SIZE, 127, dtype=np.uint8)
    assert tracker.seed(flat, (10.0, 10.0, 12.0, 12.0)) is False
    assert tracker.is_valid is False


def test_update_same_frame_returns_valid_box_near_seed(frame: np.ndarray) -> None:
    """update() on the SAME frame holds every corner: valid, enough matches,
    and a box that closely matches the seed box."""
    tracker = LucasKanadeBoxTracker()
    assert tracker.seed(frame, BOX) is True

    obs = tracker.update(frame)
    assert isinstance(obs, BoxObservation)
    assert obs.valid is True
    assert obs.bbox_xyxy is not None
    assert obs.n_matches >= tracker.cfg.min_matches

    # The measured box is the extent of the tracked corners, so it sits just
    # inside the seed box; every edge should be within a few px of the seed.
    for measured, seeded in zip(obs.bbox_xyxy, BOX):
        assert abs(measured - seeded) <= 15.0
    mcx, mcy = _center(obs.bbox_xyxy)
    scx, scy = _center(BOX)
    assert abs(mcx - scx) <= 15.0
    assert abs(mcy - scy) <= 15.0


def test_update_tracks_small_translation(frame: np.ndarray) -> None:
    """Rolling the whole image a few px to the right moves the box centre the same
    way by ~the same amount (LK holds for small motion)."""
    shift = 3
    tracker = LucasKanadeBoxTracker()
    assert tracker.seed(frame, BOX) is True

    # Baseline: corner-cloud centre on the unshifted frame.
    base = tracker.update(frame)
    assert base.valid is True
    base_cx, base_cy = _center(base.bbox_xyxy)

    shifted = np.roll(frame, shift, axis=1)  # content moves +x by `shift`
    moved = tracker.update(shifted)
    assert moved.valid is True
    assert moved.n_matches >= tracker.cfg.min_matches
    moved_cx, moved_cy = _center(moved.bbox_xyxy)

    dx = moved_cx - base_cx
    dy = moved_cy - base_cy
    assert dx > 0.0  # same direction as the shift
    assert abs(dx - shift) <= 2.0  # ~same magnitude
    assert abs(dy) <= 2.0  # no spurious vertical drift


def test_update_before_seed_is_invalid(frame: np.ndarray) -> None:
    """Calling update() with no prior seed yields an invalid, empty observation."""
    tracker = LucasKanadeBoxTracker()
    obs = tracker.update(frame)
    assert obs.valid is False
    assert obs.bbox_xyxy is None
    assert obs.n_matches == 0


def test_reset_invalidates_tracker(frame: np.ndarray) -> None:
    """reset() drops lock; a subsequent update() is invalid until re-seeded."""
    tracker = LucasKanadeBoxTracker()
    assert tracker.seed(frame, BOX) is True
    assert tracker.is_valid is True

    tracker.reset()
    assert tracker.is_valid is False

    obs = tracker.update(frame)
    assert obs.valid is False
    assert obs.bbox_xyxy is None


def test_custom_config_is_respected(frame: np.ndarray) -> None:
    """A custom config is stored and its min_matches gate is honoured."""
    cfg = LKBoxTrackerConfig(min_matches=4, max_corners=40)
    tracker = LucasKanadeBoxTracker(cfg)
    assert tracker.cfg.min_matches == 4
    assert tracker.seed(frame, BOX) is True
    obs = tracker.update(frame)
    assert obs.valid is True
    assert obs.n_matches >= 4
