"""Real-time visual target tracking (ROS-free).

Detect-once / track-many tracking of a single target in the image stream, used by
the visual-approach mission to keep a lock on a detected object at camera rate
between the (slow) detector fires.

Layers:
  * :class:`BoxTracker` / :class:`LucasKanadeBoxTracker` — classic sparse-LK box
    propagation (fast, GPU-free);
  * :class:`ConstantVelocityBoxModel` — smoothing + velocity + short-horizon
    prediction through dropouts;
  * :class:`TargetTracker` — composes the above with detector re-seeds and emits a
    :class:`~sparx_agency.core.common.types.Track2D` per frame.

The bbox->control law that consumes these tracks lives in
:mod:`sparx_agency.core.planning.visual_servo`.
"""
from __future__ import annotations

from sparx_agency.core.planning.visual_tracking.interface import (
    BoxTracker,
    BoxObservation,
)
from sparx_agency.core.planning.visual_tracking.lk_box_tracker import (
    LucasKanadeBoxTracker,
    LKBoxTrackerConfig,
)
from sparx_agency.core.planning.visual_tracking.motion_model import (
    ConstantVelocityBoxModel,
    MotionModelConfig,
)
from sparx_agency.core.planning.visual_tracking.target_tracker import (
    TargetTracker,
    TargetTrackerConfig,
)
from sparx_agency.core.planning.visual_tracking.registry import (
    BoxTrackerFactory,
    BoxTrackerRegistry,
    default_box_tracker_registry,
)

__all__ = [
    "BoxTracker",
    "BoxObservation",
    "LucasKanadeBoxTracker",
    "LKBoxTrackerConfig",
    "ConstantVelocityBoxModel",
    "MotionModelConfig",
    "TargetTracker",
    "TargetTrackerConfig",
    "BoxTrackerFactory",
    "BoxTrackerRegistry",
    "default_box_tracker_registry",
]
