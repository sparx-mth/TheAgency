"""Set-flicker defence: hysteresis and ROI rescue keep the used tag set stable.

The pose jumps this guards against come from the visible tag SET changing, not
from bad solves — so what these tests pin is exactly which tags survive which
frame, for detections hovering around the margin threshold and for tags the
detector misses outright.
"""
from dataclasses import dataclass

import numpy as np
import pytest

from sparx_agency.core.localization.providers.tag_persistence import (
    TagPersistence,
    TagPersistenceParams,
)


@dataclass
class FakeDet:
    tag_id: int
    corners: np.ndarray
    decision_margin: float


def det(tid, margin, x0=100.0, y0=100.0, size=40.0):
    c = np.array([[x0, y0 + size], [x0 + size, y0 + size],
                  [x0 + size, y0], [x0, y0]], dtype=np.float64)
    return FakeDet(tid, c, margin)


GRAY = np.zeros((294, 504), dtype=np.uint8)
KNOWN = [1, 2, 3]


def make(detector=None, **kw):
    kw.setdefault("rescue", detector is not None)
    return TagPersistence(detector, TagPersistenceParams(**kw))


def ids(out):
    return sorted(d.tag_id for d in out)


def test_new_tag_needs_enter_margin():
    p = make(enter_margin=10.0, keep_margin=5.0)
    assert ids(p.filter(GRAY, [det(1, 7.0)], KNOWN)) == []      # 7 < 10: out
    assert ids(p.filter(GRAY, [det(1, 12.0)], KNOWN)) == [1]    # enters at 12


def test_active_tag_survives_down_to_keep_margin():
    """The flicker case: margin oscillating around the old hard threshold."""
    p = make(enter_margin=10.0, keep_margin=5.0)
    p.filter(GRAY, [det(1, 12.0)], KNOWN)                       # enters
    assert ids(p.filter(GRAY, [det(1, 7.0)], KNOWN)) == [1]     # stays at 7
    assert ids(p.filter(GRAY, [det(1, 6.0)], KNOWN)) == [1]     # stays at 6
    assert ids(p.filter(GRAY, [det(1, 4.0)], KNOWN)) == []      # 4 < 5: gone
    assert ids(p.filter(GRAY, [det(1, 7.0)], KNOWN)) == []      # must RE-ENTER at 10


def test_equal_rails_reproduce_the_old_hard_threshold():
    p = make(enter_margin=10.0, keep_margin=10.0)
    p.filter(GRAY, [det(1, 12.0)], KNOWN)
    assert ids(p.filter(GRAY, [det(1, 9.0)], KNOWN)) == []


def test_unknown_tags_never_participate():
    p = make()
    assert ids(p.filter(GRAY, [det(99, 50.0)], KNOWN)) == []
    # ...and cannot become "active" either
    assert ids(p.filter(GRAY, [det(99, 50.0)], KNOWN)) == []


def test_stale_corners_are_never_reissued():
    """A missing tag is MISSING unless a rescue re-detects it. Reusing its old
    corners while the camera moves would inject a wrong constraint — the one
    cure worse than the flicker."""
    p = make(enter_margin=10.0, keep_margin=5.0)
    p.filter(GRAY, [det(1, 20.0)], KNOWN)
    assert ids(p.filter(GRAY, [], KNOWN)) == []


def test_rescue_recovers_a_dropout_and_offsets_corners():
    calls = {}

    class FakeRescueDetector:
        def detect(self, crop):
            calls["shape"] = crop.shape
            # Found near the middle of the (upscaled) crop.
            return [det(1, 9.0, x0=60.0, y0=60.0, size=80.0)]

    p = make(FakeRescueDetector(), rescue_frames=2)
    p.filter(GRAY, [det(1, 20.0, x0=200.0, y0=120.0)], KNOWN)
    out = p.filter(GRAY, [], KNOWN)                    # detector missed it
    assert ids(out) == [1]
    assert out[0].rescued
    # Crop was upscaled 2x, so the found corners must be scaled back down and
    # offset by the crop origin — i.e. land inside the original neighbourhood.
    assert calls["shape"][0] > 0
    assert 150.0 < out[0].corners[:, 0].mean() < 300.0
    assert 60.0 < out[0].corners[:, 1].mean() < 220.0


def test_rescue_gives_up_after_rescue_frames():
    class NeverFinds:
        def detect(self, crop):
            return []

    p = make(NeverFinds(), rescue_frames=2)
    p.filter(GRAY, [det(1, 20.0)], KNOWN)
    for _ in range(3):
        p.filter(GRAY, [], KNOWN)
    # Tag forgotten: a later rescue-eligible frame no longer even tries.
    class MustNotRun:
        def detect(self, crop):
            raise AssertionError("rescue ran for a forgotten tag")

    p._rescue_detector = MustNotRun()
    assert ids(p.filter(GRAY, [], KNOWN)) == []


def test_rescued_tag_still_needs_keep_margin():
    class WeakFind:
        def detect(self, crop):
            return [det(1, 2.0)]                       # below keep_margin 5

    p = make(WeakFind())
    p.filter(GRAY, [det(1, 20.0)], KNOWN)
    assert ids(p.filter(GRAY, [], KNOWN)) == []


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
