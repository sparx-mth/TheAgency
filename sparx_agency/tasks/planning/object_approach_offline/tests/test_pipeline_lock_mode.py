"""Pipeline-level regression tests for the ``--lock-mode`` closure switch.

Guards the detector-only freeze bug: a non-propagating tracker (the detector's box
only, no optical-flow propagation) must be re-fed on every matching detection
regardless of ``reseed_on_detection`` — otherwise, once the first detection ages
out of ``max_det_age_s``, a continuously-visible target is abandoned and the
mission sweeps blindly in RECOVER forever.
"""
from __future__ import annotations

from collections import Counter

import numpy as np
import cv2
import pytest

from sparx_agency.core.common.types import Intrinsics
from sparx_agency.core.common.types.perception import Detection2D
from sparx_agency.core.mapping.tracking import make_lock_tracker
from sparx_agency.core.planning.visual_servo import VisualServoParams
from sparx_agency.tasks.planning.object_approach_offline.pipeline import TargetLockPipeline

W, H = 200, 120
BOX = (80, 40, 140, 100)


def _intr() -> Intrinsics:
    return Intrinsics(width=W, height=H, fx=150.0, fy=150.0, cx=W / 2, cy=H / 2)


def _det() -> Detection2D:
    return Detection2D(label="bottle", score=0.95, bbox_xyxy=BOX, frame_w=W, frame_h=H)


def _textured_frame() -> np.ndarray:
    """A richly-textured target box (so the optical-flow tracker can lock onto it)."""
    rng = np.random.default_rng(11)
    img = rng.integers(20, 55, size=(H, W), dtype=np.uint8)
    x1, y1, x2, y2 = BOX
    for _ in range(60):
        rx, ry = int(rng.integers(x1, x2 - 6)), int(rng.integers(y1, y2 - 6))
        cv2.rectangle(img, (rx, ry),
                      (min(rx + int(rng.integers(3, 10)), x2), min(ry + int(rng.integers(3, 10)), y2)),
                      int(rng.integers(180, 256)), -1)
    return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)


def _run_modes(lock_mode: str, reseed: bool, n: int = 120) -> list:
    p = TargetLockPipeline(target="bottle", intrinsics=_intr(), lock_mode=lock_mode,
                           reseed_on_detection=reseed,
                           servo_params=VisualServoParams(use_depth=False))
    bgr = _textured_frame()   # a real object the tracker can hold, detected every frame
    return [p.step(bgr, i / 15.0, [_det()]).fsm_mode for i in range(n)]


def test_propagation_flags() -> None:
    assert make_lock_tracker("detector_tracker").propagates is True
    assert make_lock_tracker("detector").propagates is False


@pytest.mark.parametrize("reseed", [True, False])
def test_detector_only_holds_lock_when_target_always_visible(reseed: bool) -> None:
    # 120 frames @ 15 fps = 8 s, far beyond the 0.5 s detection-freshness window: a
    # detector-only lock that only seeded once would have gone stale ~7.5 s ago.
    modes = _run_modes("detector", reseed, n=120)
    recover = Counter(modes)["RECOVER"]
    assert recover < 5, \
        "detector-only abandoned a continuously-visible target (%d RECOVER frames)" % recover
    assert modes[-1] in ("APPROACH", "HOVER_LOCK")


def test_detector_tracker_also_holds_lock() -> None:
    modes = _run_modes("detector_tracker", reseed=True, n=120)
    assert modes[-1] in ("APPROACH", "HOVER_LOCK")
