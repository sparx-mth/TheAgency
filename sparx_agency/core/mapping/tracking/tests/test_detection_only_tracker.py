"""Unit tests for :class:`DetectionOnlyTracker` (detector's box, no propagation).

The image argument is irrelevant to this tracker, so these drive it purely with
detections and timestamps: a detection makes the track valid, and it stays valid
only while fresh (within ``max_det_age_s``), then goes invalid — the detector-only
analogue of the tracked path's coast-then-lose behaviour.
"""
from __future__ import annotations

import numpy as np
import pytest

from sparx_agency.core.common.types.perception import Detection2D, Track2D
from sparx_agency.core.mapping.tracking.detection_only_tracker import (
    DetectionOnlyTracker,
    DetectionOnlyConfig,
)

FRAME_W, FRAME_H = 200, 120
LABEL, SCORE = "bottle", 0.71
BOX = (40, 30, 120, 100)
_IMG = np.zeros((FRAME_H, FRAME_W, 3), dtype=np.uint8)  # ignored by the tracker


def _det(box=BOX, label=LABEL, score=SCORE) -> Detection2D:
    return Detection2D(label=label, score=score, bbox_xyxy=box,
                       frame_w=FRAME_W, frame_h=FRAME_H)


def test_pre_seed_state() -> None:
    t = DetectionOnlyTracker()
    assert t.has_target is False
    assert t.is_locked is False
    assert t.last_track is None
    assert t.label == ""
    assert t.time_since_valid(0.0) is None


def test_detection_makes_track_valid_and_carries_fields() -> None:
    t = DetectionOnlyTracker()
    assert t.on_detection(_IMG, _det(), stamp_s=1.0) is True
    assert t.has_target is True
    assert t.is_locked is True
    assert t.label == LABEL

    tr = t.last_track
    assert isinstance(tr, Track2D)
    assert tr.valid is True
    assert tr.predicted is False               # a detection is a measurement
    assert tr.velocity_px == (0.0, 0.0)        # no motion model in this mode
    assert tr.score == pytest.approx(SCORE)
    assert tr.bbox_xyxy == tuple(float(v) for v in BOX)


def test_degenerate_detection_rejected() -> None:
    t = DetectionOnlyTracker()
    assert t.on_detection(_IMG, _det(box=(50, 50, 50, 80)), stamp_s=0.0) is False
    assert t.has_target is False


def test_track_stays_valid_while_fresh_then_goes_invalid() -> None:
    t = DetectionOnlyTracker(DetectionOnlyConfig(max_det_age_s=0.5))
    assert t.on_detection(_IMG, _det(), stamp_s=0.0) is True

    fresh = t.on_frame(_IMG, stamp_s=0.4)      # within the window: still valid
    assert fresh.valid is True
    assert t.is_locked is True

    stale = t.on_frame(_IMG, stamp_s=0.9)      # beyond the window: lost
    assert stale.valid is False
    assert t.is_locked is False
    # The invalid track still carries the last known box for recovery direction.
    assert stale.bbox_xyxy == tuple(float(v) for v in BOX)


def test_fresh_detection_refreshes_the_window() -> None:
    t = DetectionOnlyTracker(DetectionOnlyConfig(max_det_age_s=0.5))
    t.on_detection(_IMG, _det(), stamp_s=0.0)
    assert t.on_frame(_IMG, stamp_s=0.9).valid is False   # went stale
    t.on_detection(_IMG, _det(box=(45, 35, 125, 105)), stamp_s=1.0)  # new detection
    back = t.on_frame(_IMG, stamp_s=1.2)                   # fresh again
    assert back.valid is True
    assert back.bbox_xyxy == (45.0, 35.0, 125.0, 105.0)


def test_time_since_valid_grows_after_last_detection() -> None:
    t = DetectionOnlyTracker(DetectionOnlyConfig(max_det_age_s=0.3))
    t.on_detection(_IMG, _det(), stamp_s=1.0)
    t.on_frame(_IMG, stamp_s=1.2)              # still fresh -> updates last-valid
    assert t.time_since_valid(1.2) == pytest.approx(0.0)
    t.on_frame(_IMG, stamp_s=1.8)              # stale -> last-valid frozen at 1.2
    assert t.time_since_valid(1.9) == pytest.approx(0.7)


def test_reset_clears_state() -> None:
    t = DetectionOnlyTracker()
    t.on_detection(_IMG, _det(), stamp_s=0.0)
    t.reset()
    assert t.has_target is False
    assert t.is_locked is False
    assert t.last_track is None
    assert t.label == ""
