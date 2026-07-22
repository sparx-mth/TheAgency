"""Build the object-lock tracker for a chosen closure strategy.

One seam to pick *how the mission keeps its box on the target*, shared by the
offline pipeline and the live ROS node so they never drift apart:

  * ``"detector_tracker"`` (default) — a detector seeds an optical-flow tracker
    (:class:`TargetTracker`) that propagates the box every camera frame between
    detections; robust to a slow / intermittent detector.
  * ``"detector"`` — the detector alone drives closure
    (:class:`DetectionOnlyTracker`): the box is whatever it last reported, held
    only while fresh. Use it when the detector already keeps up with the RGB
    stream, so tracking adds nothing but a way to drift onto the background.

Both implement :class:`ObjectLockTracker`, so everything downstream (servo, FSM,
recovery, HUD) is identical either way.
"""
from __future__ import annotations

from typing import Optional

from sparx_agency.core.mapping.tracking.object_lock_tracker import ObjectLockTracker
from sparx_agency.core.mapping.tracking.target_tracker import (
    TargetTracker,
    TargetTrackerConfig,
)
from sparx_agency.core.mapping.tracking.detection_only_tracker import (
    DetectionOnlyTracker,
    DetectionOnlyConfig,
)

DETECTOR = "detector"                    # detector only, no optical-flow tracking
DETECTOR_TRACKER = "detector_tracker"    # detector + optical-flow tracking (default)
LOCK_MODES = (DETECTOR, DETECTOR_TRACKER)


def make_lock_tracker(mode: str = DETECTOR_TRACKER,
                      tracker_config: Optional[TargetTrackerConfig] = None,
                      detection_config: Optional[DetectionOnlyConfig] = None
                      ) -> ObjectLockTracker:
    """Construct the :class:`ObjectLockTracker` for ``mode``.

    Args:
        mode: One of :data:`LOCK_MODES` (case-insensitive).
        tracker_config: Config for the ``detector_tracker`` path (ignored otherwise).
        detection_config: Config for the ``detector`` path (ignored otherwise).

    Raises:
        ValueError: If ``mode`` is not a known lock mode.
    """
    m = str(mode).strip().lower()
    if m == DETECTOR_TRACKER:
        return TargetTracker(tracker_config or TargetTrackerConfig())
    if m == DETECTOR:
        return DetectionOnlyTracker(detection_config or DetectionOnlyConfig())
    raise ValueError(
        "lock mode must be one of %s, got %r" % (list(LOCK_MODES), mode))
