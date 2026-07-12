"""Detector-only object lock: the detector's own box, no optical-flow tracking.

The lean half of the closure switch. Where :class:`TargetTracker` seeds an
optical-flow tracker and propagates the box every camera frame between detector
fires, this tracker just *holds the detector's most recent box* and reports it as
the track for as long as it is fresh (``max_det_age_s``). Use it when the detector
already runs at (or above) the camera frame rate — as on the target edge hardware,
where YOLO is faster than the RGB stream — so frame-to-frame propagation buys
nothing and only adds a failure mode (drift onto the background).

No optical flow, no motion model, no image processing at all: the ``image``
argument is accepted (to satisfy the :class:`ObjectLockTracker` contract) and
ignored. The emitted :class:`~sparx_agency.core.common.types.Track2D` is always a
measurement (``predicted`` is False — a detector box is never a dead-reckon) and
carries no velocity, so the servo / FSM / recovery stack behaves identically to
the tracked path, just with lock loss the instant detections go stale rather than
after an optical-flow coast.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

from sparx_agency.core.common.types.perception import Detection2D, Track2D
from sparx_agency.core.mapping.tracking.object_lock_tracker import ObjectLockTracker

BBox = Tuple[float, float, float, float]


@dataclass(frozen=True)
class DetectionOnlyConfig:
    """Tuning for :class:`DetectionOnlyTracker`.

    Attributes:
        max_det_age_s: Hold the last detection as a valid track for this long after
            it arrived; once no matching detection has landed within the window the
            track goes invalid (loss). This is the detector-only analogue of the
            tracked path's ``max_predict_s`` coast: it bridges the gap between
            detector fires, so the detector must run at least ~``1/max_det_age_s``
            Hz to avoid flicker.
    """

    max_det_age_s: float = 0.5

    def __post_init__(self) -> None:
        if self.max_det_age_s < 0.0:
            raise ValueError("max_det_age_s must be >= 0.")


class DetectionOnlyTracker(ObjectLockTracker):
    """Report the detector's most recent box as the track (no propagation)."""

    def __init__(self, config: Optional[DetectionOnlyConfig] = None) -> None:
        self.cfg = config or DetectionOnlyConfig()
        self.reset()

    def reset(self) -> None:
        self._bbox: Optional[BBox] = None
        self._det_t: Optional[float] = None
        self._label: str = ""
        self._score: float = 0.0
        self._frame_wh: Tuple[int, int] = (0, 0)
        self._seeded_t: Optional[float] = None
        self._last_valid_t: Optional[float] = None
        self._locked: bool = False
        self._last_track: Optional[Track2D] = None

    # ── properties ───────────────────────────────────────────────────
    @property
    def label(self) -> str:
        return self._label

    @property
    def has_target(self) -> bool:
        return self._seeded_t is not None

    @property
    def propagates(self) -> bool:
        return False  # no state to advance between detections; must be re-fed each

    @property
    def is_locked(self) -> bool:
        return self._locked

    @property
    def last_track(self) -> Optional[Track2D]:
        return self._last_track

    def time_since_valid(self, stamp_s: float) -> Optional[float]:
        if self._last_valid_t is None:
            return None
        return float(stamp_s) - self._last_valid_t

    # ── inputs ───────────────────────────────────────────────────────
    def on_detection(self, image: np.ndarray, detection: Detection2D,
                     stamp_s: float) -> bool:
        """Store a fresh detection as the current box; True unless degenerate."""
        x1, y1, x2, y2 = (float(v) for v in detection.bbox_xyxy)
        if (x2 - x1) < 1.0 or (y2 - y1) < 1.0:
            return False
        stamp = float(stamp_s)
        self._bbox = (x1, y1, x2, y2)
        self._det_t = stamp
        self._label = detection.label
        self._score = float(detection.score)
        self._frame_wh = (int(detection.frame_w), int(detection.frame_h))
        if self._seeded_t is None:
            self._seeded_t = stamp
        self._last_valid_t = stamp
        self._locked = True
        self._last_track = self._track(valid=True, stamp=stamp)
        return True

    def on_frame(self, image: np.ndarray, stamp_s: float) -> Track2D:
        """Return the held detection as a track while fresh, else an invalid one."""
        stamp = float(stamp_s)
        if self._bbox is None or self._det_t is None:
            self._locked = False
            self._last_track = self._track(valid=False, stamp=stamp)
            return self._last_track
        if (stamp - self._det_t) <= self.cfg.max_det_age_s:
            self._locked = True
            self._last_valid_t = stamp
            self._last_track = self._track(valid=True, stamp=stamp)
            return self._last_track
        self._locked = False
        self._last_track = self._track(valid=False, stamp=stamp)
        return self._last_track

    # ── internals ────────────────────────────────────────────────────
    def _track(self, valid: bool, stamp: float) -> Track2D:
        bbox = self._bbox if self._bbox is not None else (0.0, 0.0, 0.0, 0.0)
        w, h = self._frame_wh
        age = 0.0 if self._seeded_t is None else max(0.0, stamp - self._seeded_t)
        return Track2D(
            label=self._label,
            bbox_xyxy=(float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])),
            frame_w=int(w), frame_h=int(h),
            valid=bool(valid), n_matches=0, score=self._score,
            velocity_px=(0.0, 0.0), predicted=False, age_s=age,
        )
