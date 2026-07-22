"""Unit tests for :func:`make_lock_tracker` (the closure-strategy seam)."""
from __future__ import annotations

import pytest

from sparx_agency.core.mapping.tracking.factory import (
    DETECTOR,
    DETECTOR_TRACKER,
    LOCK_MODES,
    make_lock_tracker,
)
from sparx_agency.core.mapping.tracking.target_tracker import (
    TargetTracker,
    TargetTrackerConfig,
)
from sparx_agency.core.mapping.tracking.detection_only_tracker import (
    DetectionOnlyConfig,
    DetectionOnlyTracker,
)


def test_default_is_detector_tracker() -> None:
    t = make_lock_tracker()
    assert isinstance(t, TargetTracker)


def test_modes_build_the_right_tracker() -> None:
    assert isinstance(make_lock_tracker(DETECTOR_TRACKER), TargetTracker)
    assert isinstance(make_lock_tracker(DETECTOR), DetectionOnlyTracker)
    assert set(LOCK_MODES) == {DETECTOR, DETECTOR_TRACKER}


def test_case_insensitive() -> None:
    assert isinstance(make_lock_tracker(" Detector "), DetectionOnlyTracker)


def test_bad_mode_raises() -> None:
    with pytest.raises(ValueError):
        make_lock_tracker("magic")


def test_configs_are_applied() -> None:
    tt = make_lock_tracker(DETECTOR_TRACKER,
                           tracker_config=TargetTrackerConfig(backend="lucas_kanade"))
    assert tt._lk.name == "lucas_kanade"
    do = make_lock_tracker(DETECTOR,
                           detection_config=DetectionOnlyConfig(max_det_age_s=1.5))
    assert do.cfg.max_det_age_s == 1.5
