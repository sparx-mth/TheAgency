"""Unit tests for :class:`TargetTracker` (LK + motion model + detector re-seed).

These exercise the *composition* the mission node drives every frame:

  * a fresh :class:`Detection2D` seeds the tracker on a synthetic, richly-textured
    frame (so real Shi-Tomasi / Lucas-Kanade lock reliably), then a slightly
    shifted frame produces a measured :class:`Track2D` with a populated
    image-plane velocity;
  * a dropout — LK losing lock — is covered two ways: a *controllable* injected
    :class:`BoxTracker` gives a crisp, deterministic ``measured -> predicted ->
    invalid`` ordering at timestamps we own (the constructor's ``box_tracker``
    injection point exists exactly for this), and a real-LK run fed blank frames
    confirms lock is genuinely lost and the track goes invalid past
    ``max_predict_s``.

Everything is seeded from a fixed ``numpy`` RNG and driven with explicit
``stamp_s`` values, so the frames, the corners LK locks onto, and the
prediction-window timing are all deterministic.
"""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import cv2
import pytest

from sparx_agency.core.common.types.perception import Detection2D, Track2D
from sparx_agency.core.mapping.tracking.interface import (
    BBox,
    BoxObservation,
    BoxTracker,
)
from sparx_agency.core.mapping.tracking.target_tracker import (
    TargetTracker,
    TargetTrackerConfig,
)

SEED = 4321
FRAME_SIZE = (200, 200)  # (H, W)
BOX = (50, 50, 150, 150)  # (x1, y1, x2, y2) — centred ~100x100 box
LABEL = "drone"
SCORE = 0.87


# ── synthetic-frame helpers ──────────────────────────────────────────────
def _make_textured_bgr(
    rng: np.random.Generator,
    size: Tuple[int, int] = FRAME_SIZE,
    box: Tuple[int, int, int, int] = BOX,
) -> np.ndarray:
    """Build a HxWx3 BGR uint8 frame: a strongly textured box on faint noise.

    The box is filled with many small high-contrast rectangles so Shi-Tomasi
    finds well-localised corners inside it and pyramidal LK tracks them under
    small translations. Returned as 3-channel BGR to exercise the tracker's
    default BGR->gray path.
    """
    h, w = size
    gray = np.full((h, w), 30, dtype=np.uint8)
    gray = (gray + rng.integers(0, 15, size=(h, w), dtype=np.uint8)).astype(np.uint8)
    x1, y1, x2, y2 = box
    for _ in range(60):
        rx1 = int(rng.integers(x1, x2 - 6))
        ry1 = int(rng.integers(y1, y2 - 6))
        rw = int(rng.integers(3, 12))
        rh = int(rng.integers(3, 12))
        val = int(rng.integers(120, 256))
        cv2.rectangle(gray, (rx1, ry1), (min(rx1 + rw, x2), min(ry1 + rh, y2)), val, -1)
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)


def _shift(frame: np.ndarray, dx: int, dy: int) -> np.ndarray:
    """Translate the whole frame by (dx, dy) px (content moves +x right / +y down)."""
    return np.roll(np.roll(frame, dy, axis=0), dx, axis=1)


def _detection(box: Tuple[int, int, int, int] = BOX) -> Detection2D:
    h, w = FRAME_SIZE
    return Detection2D(label=LABEL, score=SCORE, bbox_xyxy=box, frame_w=w, frame_h=h)


@pytest.fixture()
def rng() -> np.random.Generator:
    return np.random.default_rng(SEED)


@pytest.fixture()
def frame(rng: np.random.Generator) -> np.ndarray:
    return _make_textured_bgr(rng)


# ── controllable fake tracker (deterministic dropout) ────────────────────
class _ScriptedBoxTracker(BoxTracker):
    """A ``BoxTracker`` whose lock and per-frame motion the test fully controls.

    On :meth:`seed` it locks onto the given box. Each valid :meth:`update` shifts
    the box by ``step`` (so the motion model builds a non-zero velocity). Setting
    ``lost = True`` makes the next :meth:`update` report the track as lost, which
    is exactly the LK-dropout the composing :class:`TargetTracker` must coast
    through and then abandon.
    """

    name = "scripted"

    def __init__(self, step: Tuple[float, float] = (6.0, 4.0), n_matches: int = 24) -> None:
        self._step = step
        self._n_matches = n_matches
        self.lost = False
        self.reset()

    def reset(self) -> None:
        self._box: Optional[BBox] = None
        self._valid = False

    @property
    def is_valid(self) -> bool:
        return self._valid

    def seed(self, gray: np.ndarray, bbox_xyxy: BBox) -> bool:
        self._box = tuple(float(v) for v in bbox_xyxy)
        self._valid = True
        return True

    def update(self, gray: np.ndarray) -> BoxObservation:
        if self.lost or not self._valid or self._box is None:
            self._valid = False
            return BoxObservation(bbox_xyxy=None, n_matches=0, valid=False)
        x1, y1, x2, y2 = self._box
        dx, dy = self._step
        self._box = (x1 + dx, y1 + dy, x2 + dx, y2 + dy)
        self._valid = True
        return BoxObservation(bbox_xyxy=self._box, n_matches=self._n_matches, valid=True)


# ── tests: pre-seed state ────────────────────────────────────────────────
def test_has_target_false_before_seed() -> None:
    """A fresh tracker has no target, no lock, no track, and no valid history."""
    tt = TargetTracker()
    assert tt.has_target is False
    assert tt.is_locked is False
    assert tt.last_track is None
    assert tt.label == ""
    assert tt.time_since_valid(0.0) is None


# ── tests: seeding from a detection ──────────────────────────────────────
def test_on_detection_seeds_locks_and_carries_label(frame: np.ndarray) -> None:
    """A detection on a textured box seeds LK: locked, has_target, label carried."""
    tt = TargetTracker()
    ok = tt.on_detection(frame, _detection(), stamp_s=0.0)

    assert ok is True
    assert tt.has_target is True
    assert tt.is_locked is True
    assert tt.label == LABEL

    seed_track = tt.last_track
    assert isinstance(seed_track, Track2D)
    assert seed_track.valid is True
    assert seed_track.predicted is False  # a seed is a measurement, not a prediction
    assert seed_track.label == LABEL
    assert seed_track.score == pytest.approx(SCORE)
    # Same-stamp query: no time has elapsed since the (valid) seed.
    assert tt.time_since_valid(0.0) == pytest.approx(0.0)


# ── tests: measured tracking on a shifted frame ──────────────────────────
def test_on_frame_shifted_returns_valid_track_with_velocity(frame: np.ndarray) -> None:
    """A slightly shifted frame yields a valid, measured (non-predicted) track
    whose image-plane velocity is populated in the direction of the shift."""
    dx, dy = 3, 2
    tt = TargetTracker()
    assert tt.on_detection(frame, _detection(), stamp_s=0.0) is True

    track = tt.on_frame(_shift(frame, dx, dy), stamp_s=0.1)

    assert isinstance(track, Track2D)
    assert track.valid is True
    assert track.predicted is False
    assert track.label == LABEL
    assert track.n_matches > 0  # a real LK measurement, not a coast
    # Velocity is populated (non-zero) and points the same way as the shift.
    vx, vy = track.velocity_px
    assert track.velocity_px != (0.0, 0.0)
    assert vx > 0.5 and vy > 0.5


# ── tests: deterministic dropout ordering (injected tracker) ─────────────
def test_dropout_states_ordered_measured_predicted_invalid() -> None:
    """With a controllable tracker and stamps we own, a lost lock produces the
    ordered states measured (valid, not predicted) -> predicted (valid, coasting)
    -> invalid, and ``time_since_valid`` grows monotonically across the dropout."""
    fake = _ScriptedBoxTracker(step=(6.0, 4.0))
    cfg = TargetTrackerConfig(max_predict_s=0.4)
    tt = TargetTracker(config=cfg, box_tracker=fake)

    blank = np.zeros((*FRAME_SIZE, 3), dtype=np.uint8)
    assert tt.on_detection(blank, _detection(), stamp_s=0.0) is True

    # 1) MEASURED: fake still locked -> valid box, velocity built, not predicted.
    measured = tt.on_frame(blank, stamp_s=0.1)
    assert measured.valid is True
    assert measured.predicted is False
    assert measured.velocity_px != (0.0, 0.0)
    assert tt.is_locked is True

    # Trigger the dropout: every subsequent update reports the track lost.
    fake.lost = True

    # 2) PREDICTED (early in the window): coasting on the motion model.
    pred_a = tt.on_frame(blank, stamp_s=0.2)  # lost for 0.1s <= 0.4s
    assert pred_a.valid is True
    assert pred_a.predicted is True
    assert tt.is_locked is False  # LK lock is gone; only the model carries it

    # 3) PREDICTED (still within the window).
    pred_b = tt.on_frame(blank, stamp_s=0.3)  # lost for 0.2s <= 0.4s
    assert pred_b.valid is True
    assert pred_b.predicted is True

    # 4) INVALID: past max_predict_s the track is abandoned.
    invalid = tt.on_frame(blank, stamp_s=0.6)  # lost for 0.5s > 0.4s
    assert invalid.valid is False
    assert invalid.predicted is False

    # time_since_valid grows monotonically across the dropout.
    tsv = [
        tt.time_since_valid(0.2),
        tt.time_since_valid(0.3),
        tt.time_since_valid(0.6),
    ]
    assert all(v is not None for v in tsv)
    assert tsv[0] < tsv[1] < tsv[2]
    assert tsv[0] == pytest.approx(0.1)
    assert tsv[2] == pytest.approx(0.5)

    # The invalid track still carries the last known label + a box (for recovery).
    assert invalid.label == LABEL
    assert len(invalid.bbox_xyxy) == 4


def test_predicted_box_moves_along_velocity_during_coast() -> None:
    """While coasting, the predicted box centre keeps moving along the estimated
    velocity (dead reckoning), so the servo gets a sensible moving target."""
    fake = _ScriptedBoxTracker(step=(6.0, 0.0))
    tt = TargetTracker(config=TargetTrackerConfig(max_predict_s=1.0), box_tracker=fake)
    blank = np.zeros((*FRAME_SIZE, 3), dtype=np.uint8)

    tt.on_detection(blank, _detection(), stamp_s=0.0)
    tt.on_frame(blank, stamp_s=0.1)   # measured, builds +x velocity
    fake.lost = True
    p1 = tt.on_frame(blank, stamp_s=0.2)  # predicted
    p2 = tt.on_frame(blank, stamp_s=0.3)  # predicted, later

    assert p1.predicted is True and p2.predicted is True
    assert p1.valid is True and p2.valid is True
    assert p2.cx > p1.cx  # centre keeps drifting in +x while coasting


# ── tests: real-LK dropout on blank frames ───────────────────────────────
def test_blank_frames_lose_lock_then_track_goes_invalid(frame: np.ndarray) -> None:
    """Feeding blank/uniform frames makes real LK lose lock; the track coasts
    (predicted) within max_predict_s and becomes invalid once the window passes."""
    tt = TargetTracker()
    assert tt.on_detection(frame, _detection(), stamp_s=0.0) is True
    # One genuine measurement so the motion model has a state to coast on.
    tt.on_frame(_shift(frame, 2, 1), stamp_s=0.1)

    blank = np.full((*FRAME_SIZE, 3), 127, dtype=np.uint8)

    # A couple of consecutive blank frames reliably kill LK lock (the previous,
    # gradient-free frame yields no trackable eigen-structure). Scan a few so the
    # exact loss frame need not be pinned across OpenCV versions.
    lost_track: Optional[Track2D] = None
    lost_t = 0.0
    for stamp in (0.2, 0.3, 0.4):
        tr = tt.on_frame(blank, stamp_s=stamp)
        if not tt.is_locked:
            lost_track, lost_t = tr, stamp
            break

    assert lost_track is not None, "LK never lost lock on blank frames"
    assert tt.is_locked is False

    tsv_at_loss = tt.time_since_valid(lost_t)
    assert tsv_at_loss is not None
    # Immediately after loss we are within the predict window: valid but coasting.
    if tsv_at_loss <= tt.cfg.max_predict_s:
        assert lost_track.valid is True
        assert lost_track.predicted is True

    # Advance well past the predict window -> the track is abandoned.
    far_t = lost_t + tt.cfg.max_predict_s + 1.0
    far_track = tt.on_frame(blank, stamp_s=far_t)
    assert far_track.valid is False
    assert far_track.predicted is False
    assert tt.time_since_valid(far_t) > tsv_at_loss  # grew across the dropout


# ── tests: reset ─────────────────────────────────────────────────────────
def test_reset_clears_all_state(frame: np.ndarray) -> None:
    """reset() forgets the target entirely: no lock, no target, no track/history."""
    tt = TargetTracker()
    assert tt.on_detection(frame, _detection(), stamp_s=0.0) is True
    tt.on_frame(_shift(frame, 2, 2), stamp_s=0.1)
    assert tt.has_target is True

    tt.reset()

    assert tt.has_target is False
    assert tt.is_locked is False
    assert tt.last_track is None
    assert tt.label == ""
    assert tt.time_since_valid(1.0) is None
