"""Real-time single-target tracker: LK box tracker + motion model + detector re-seed.

This is the object a task node drives every frame. It composes the three tracking
pieces into the detect-once/track-many loop the mission needs:

  * :class:`LucasKanadeBoxTracker` propagates the box at camera rate (the classic,
    fast, GPU-free core of the tracker);
  * :class:`ConstantVelocityBoxModel` smooths the box centre, supplies an
    image-plane velocity, and — for a short ``max_predict_s`` window — dead-reckons
    the box through a brief LK dropout so the servo does not stall on one bad frame;
  * fresh detections (fed in by the node whenever the detector re-fires) re-seed
    the LK tracker, bounding drift to the detector's inter-arrival time.

Output is a :class:`~sparx_agency.core.common.types.Track2D` per frame carrying
validity, velocity, and a ``predicted`` flag so the servo and the re-search policy
can tell a measured box from a dead-reckoned one. ROS-free: the caller passes a
monotonic ``stamp_s`` (wall clock or ROS stamp); this module owns no I/O.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import cv2

from sparx_agency.core.common.types.perception import Detection2D, Track2D
from sparx_agency.core.common.math.bbox import (
    xyxy_to_cxcywh,
    cxcywh_to_xyxy,
    clip_xyxy,
)
from sparx_agency.core.planning.visual_tracking.interface import BoxTracker
from sparx_agency.core.planning.visual_tracking.lk_box_tracker import (
    LucasKanadeBoxTracker,
    LKBoxTrackerConfig,
)
from sparx_agency.core.planning.visual_tracking.motion_model import (
    ConstantVelocityBoxModel,
    MotionModelConfig,
)


@dataclass(frozen=True)
class TargetTrackerConfig:
    """Tuning for :class:`TargetTracker`.

    Attributes:
        lk: Lucas-Kanade box-tracker config.
        motion: Constant-velocity motion-model config.
        max_predict_s: How long (s) to dead-reckon the box after LK loses lock
            before declaring the track invalid. 0 disables prediction.
        input_is_bgr: Channel order of 3-channel input frames (for gray conversion).
    """

    lk: LKBoxTrackerConfig = field(default_factory=LKBoxTrackerConfig)
    motion: MotionModelConfig = field(default_factory=MotionModelConfig)
    max_predict_s: float = 0.4
    input_is_bgr: bool = True

    def __post_init__(self) -> None:
        if self.max_predict_s < 0.0:
            raise ValueError("max_predict_s must be >= 0.")


class TargetTracker:
    """Compose LK tracking, motion prediction, and detector re-seeds for one target."""

    def __init__(self, config: Optional[TargetTrackerConfig] = None,
                 box_tracker: Optional[BoxTracker] = None) -> None:
        """Args:
            config: Tuning; defaults are sized for a ~640x360 forward camera.
            box_tracker: Classic box tracker to propagate the box (default the
                Lucas-Kanade tracker). Inject a different :class:`BoxTracker`
                (e.g. a future DNN tracker) here without touching the servo/FSM.
        """
        self.cfg = config or TargetTrackerConfig()
        self._lk: BoxTracker = box_tracker or LucasKanadeBoxTracker(self.cfg.lk)
        self._motion = ConstantVelocityBoxModel(self.cfg.motion)
        self.reset()

    def reset(self) -> None:
        """Forget the target entirely."""
        self._lk.reset()
        self._motion.reset()
        self._label: str = ""
        self._score: float = 0.0
        self._seeded_t: Optional[float] = None
        self._last_frame_t: Optional[float] = None
        self._last_valid_t: Optional[float] = None
        self._last_track: Optional[Track2D] = None

    # ── properties ───────────────────────────────────────────────────
    @property
    def label(self) -> str:
        return self._label

    @property
    def has_target(self) -> bool:
        """True once a detection has seeded the tracker (even if now lost)."""
        return self._seeded_t is not None

    @property
    def is_locked(self) -> bool:
        """True while LK holds direct lock (not merely predicting)."""
        return self._lk.is_valid

    @property
    def last_track(self) -> Optional[Track2D]:
        return self._last_track

    def time_since_valid(self, stamp_s: float) -> Optional[float]:
        """Seconds since the last measured (non-predicted) box, or None."""
        if self._last_valid_t is None:
            return None
        return float(stamp_s) - self._last_valid_t

    # ── inputs ───────────────────────────────────────────────────────
    def on_detection(self, image: np.ndarray, detection: Detection2D,
                      stamp_s: float) -> bool:
        """(Re)seed the tracker from a fresh detection on ``image``.

        Args:
            image: The RGB/BGR (or gray) frame the detection was made on.
            detection: The detector output whose box seeds the LK tracker.
            stamp_s: Monotonic timestamp (s) of ``image``.

        Returns:
            True if the LK tracker seeded successfully.
        """
        gray = self._to_gray(image)
        bbox = tuple(float(v) for v in detection.bbox_xyxy)
        ok = self._lk.seed(gray, bbox)
        if not ok:
            return False
        h, w = gray.shape[:2]
        self._label = detection.label
        self._score = float(detection.score)
        if self._seeded_t is None:
            self._seeded_t = float(stamp_s)
        # Anchor the motion model on the seed box (re-seed = fresh anchor).
        self._motion.reset()
        self._motion.update(xyxy_to_cxcywh(clip_xyxy(bbox, w, h)), dt=0.0)
        self._last_frame_t = float(stamp_s)
        self._last_valid_t = float(stamp_s)
        self._last_track = self._make_track(bbox, w, h, n_matches=0,
                                            predicted=False, stamp_s=stamp_s)
        return True

    def on_frame(self, image: np.ndarray, stamp_s: float) -> Track2D:
        """Advance the tracker by one frame and return the current track.

        The track is always returned (never None); check ``.valid``. A valid track
        may be ``.predicted`` when LK has briefly lost lock but the motion model is
        still within ``max_predict_s`` of the last measurement.
        """
        gray = self._to_gray(image)
        h, w = gray.shape[:2]
        stamp = float(stamp_s)
        dt = 0.0 if self._last_frame_t is None else max(0.0, stamp - self._last_frame_t)
        self._last_frame_t = stamp

        if not self._lk.is_valid:
            return self._predict_or_invalid(w, h, dt, stamp)

        obs = self._lk.update(gray)
        if obs.valid and obs.bbox_xyxy is not None:
            filt = self._motion.update(xyxy_to_cxcywh(obs.bbox_xyxy), dt)
            bbox = clip_xyxy(cxcywh_to_xyxy(filt), w, h)
            self._last_valid_t = stamp
            self._last_track = self._make_track(bbox, w, h, n_matches=obs.n_matches,
                                                predicted=False, stamp_s=stamp)
            return self._last_track

        # LK just lost lock this frame -> try to coast.
        return self._predict_or_invalid(w, h, dt, stamp)

    # ── internals ────────────────────────────────────────────────────
    def _predict_or_invalid(self, w: int, h: int, dt: float, stamp: float) -> Track2D:
        lost_for = self.time_since_valid(stamp)
        can_predict = (
            self.cfg.max_predict_s > 0.0
            and self._motion.has_state
            and lost_for is not None
            and lost_for <= self.cfg.max_predict_s
        )
        if can_predict:
            pred = self._motion.predict(dt)
            if pred is not None:
                bbox = clip_xyxy(cxcywh_to_xyxy(pred), w, h)
                self._last_track = self._make_track(bbox, w, h, n_matches=0,
                                                    predicted=True, stamp_s=stamp)
                return self._last_track
        # Truly lost: emit an invalid track that still carries the last known
        # box + velocity so the recovery policy can read a re-search direction.
        self._last_track = self._make_invalid_track(w, h, stamp)
        return self._last_track

    def _make_track(self, bbox, w, h, n_matches, predicted, stamp_s) -> Track2D:
        age = 0.0 if self._seeded_t is None else max(0.0, float(stamp_s) - self._seeded_t)
        return Track2D(
            label=self._label,
            bbox_xyxy=(float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])),
            frame_w=int(w),
            frame_h=int(h),
            valid=True,
            n_matches=int(n_matches),
            score=self._score,
            velocity_px=self._motion.velocity_px,
            predicted=bool(predicted),
            age_s=age,
        )

    def _make_invalid_track(self, w, h, stamp_s) -> Track2D:
        prev = self._last_track
        bbox = prev.bbox_xyxy if prev is not None else (0.0, 0.0, 0.0, 0.0)
        age = 0.0 if self._seeded_t is None else max(0.0, float(stamp_s) - self._seeded_t)
        return Track2D(
            label=self._label,
            bbox_xyxy=bbox,
            frame_w=int(w),
            frame_h=int(h),
            valid=False,
            n_matches=0,
            score=self._score,
            velocity_px=self._motion.velocity_px,
            predicted=False,
            age_s=age,
        )

    def _to_gray(self, image: np.ndarray) -> np.ndarray:
        img = np.asarray(image)
        if img.ndim == 2:
            return img if img.dtype == np.uint8 else img.astype(np.uint8)
        if img.ndim == 3 and img.shape[2] == 3:
            code = cv2.COLOR_BGR2GRAY if self.cfg.input_is_bgr else cv2.COLOR_RGB2GRAY
            return cv2.cvtColor(img, code)
        raise ValueError("TargetTracker expects HxW or HxWx3, got %r" % (img.shape,))
