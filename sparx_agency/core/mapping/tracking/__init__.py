"""Real-time visual target tracking (ROS-free).

Detect-once / track-many tracking of a single target in the image stream, used by
the visual-approach mission to keep a lock on a detected object at camera rate
between the (slow) detector fires. Lives under ``core/mapping`` because tracking is
a perception concern; the control law that consumes its tracks lives in
:mod:`sparx_agency.core.planning.visual_servo`.

Layers:
  * :class:`BoxTracker` — the classic single-object box-propagation contract, with
    two implementations: :class:`MedianFlowBoxTracker` (the robust default —
    forward-backward consistency + median consensus + appearance validation, so it
    fails honestly instead of tracking the background) and
    :class:`LucasKanadeBoxTracker` (the leaner, faster sparse-LK tracker);
  * :class:`ConstantVelocityBoxModel` — smoothing + velocity + short-horizon
    prediction through dropouts;
  * :class:`ObjectLockTracker` — the high-level per-target contract with two
    strategies: :class:`TargetTracker` (detector + box tracker) and
    :class:`DetectionOnlyTracker` (detector alone). Pick one with
    :func:`make_lock_tracker`; both emit a
    :class:`~sparx_agency.core.common.types.Track2D` per frame.
"""
from __future__ import annotations

from sparx_agency.core.mapping.tracking.interface import (
    BoxTracker,
    BoxObservation,
)
from sparx_agency.core.mapping.tracking.object_lock_tracker import ObjectLockTracker
from sparx_agency.core.mapping.tracking.lk_box_tracker import (
    LucasKanadeBoxTracker,
    LKBoxTrackerConfig,
)
from sparx_agency.core.mapping.tracking.median_flow_box_tracker import (
    MedianFlowBoxTracker,
    MedianFlowConfig,
)
from sparx_agency.core.mapping.tracking.motion_model import (
    ConstantVelocityBoxModel,
    MotionModelConfig,
)
from sparx_agency.core.mapping.tracking.target_tracker import (
    TargetTracker,
    TargetTrackerConfig,
)
from sparx_agency.core.mapping.tracking.detection_only_tracker import (
    DetectionOnlyTracker,
    DetectionOnlyConfig,
)
from sparx_agency.core.mapping.tracking.factory import (
    make_lock_tracker,
    DETECTOR,
    DETECTOR_TRACKER,
    LOCK_MODES,
)
from sparx_agency.core.mapping.tracking.registry import (
    BoxTrackerFactory,
    BoxTrackerRegistry,
    default_box_tracker_registry,
)

__all__ = [
    "BoxTracker",
    "BoxObservation",
    "ObjectLockTracker",
    "LucasKanadeBoxTracker",
    "LKBoxTrackerConfig",
    "MedianFlowBoxTracker",
    "MedianFlowConfig",
    "ConstantVelocityBoxModel",
    "MotionModelConfig",
    "TargetTracker",
    "TargetTrackerConfig",
    "DetectionOnlyTracker",
    "DetectionOnlyConfig",
    "make_lock_tracker",
    "DETECTOR",
    "DETECTOR_TRACKER",
    "LOCK_MODES",
    "BoxTrackerFactory",
    "BoxTrackerRegistry",
    "default_box_tracker_registry",
]
