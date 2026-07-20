"""Keep marginal AprilTags from flickering in and out of the detected set.

The dominant localization jump on this stack is not a bad solve — it is the
*visible tag set changing*. Each subset of tags carries its own systematic error
(measured on the deployed map: half a metre between two single-tag answers at
the same hover point), so a tag that is detected on one frame and missed on the
next teleports the reported pose even though nothing moved. Two mechanisms here
attack exactly that boundary flicker, and both are deliberately cheap:

1. **Margin hysteresis.** The detector's ``decision_margin`` threshold becomes a
   Schmitt trigger: a NEW tag must clear ``enter_margin``, but a tag that was in
   the set last frame stays down to ``keep_margin``. A tag oscillating around
   one hard threshold is the textbook flicker source; two rails remove it for
   free.

2. **ROI rescue.** A tag that was seen a frame or two ago but is missing now is
   looked for AGAIN — only inside a small padded crop around where it just was,
   upscaled 2x. The upscale is the whole mechanism: it doubles the effective
   resolution exactly where the marginal small tag is, which recovers
   decode-stage dropouts, and running it on a ~100-300 px crop costs well under
   a millisecond — it does not slow the frame down, because it only runs for
   tags that just vanished. The SAME detector instance as the full-frame pass
   is reused on purpose: pupil_apriltags segfaults at interpreter teardown when
   two used Detector objects are destroyed, so this module never builds one.

Stale corners are never reused as detections: a tag the rescue cannot re-find is
genuinely gone from the frame, and pretending otherwise would inject wrong
constraints while the camera moves. Rescue re-DETECTS or gives up.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np


@dataclass
class RawDet:
    """Detector-agnostic detection: what the pose pipeline needs, nothing more."""

    tag_id: int
    corners: np.ndarray        # (4, 2) float64, image pixels
    decision_margin: float
    rescued: bool = False


@dataclass(frozen=True)
class TagPersistenceParams:
    """Tuning for :class:`TagPersistence`.

    Attributes:
        enter_margin: ``decision_margin`` a tag needs to ENTER the used set.
        keep_margin: Margin a tag already in the set needs to STAY. Setting it
            equal to ``enter_margin`` disables the hysteresis.
        rescue: Run the ROI re-detection pass for recently-seen missing tags.
        rescue_frames: How many consecutive frames a tag may be missing and
            still be looked for. Past this it is treated as genuinely gone.
        rescue_max_per_frame: Cost bound — at most this many crops per frame.
        rescue_upscale: Crop upscale factor before re-detection. 2.0 doubles
            the effective resolution exactly where the marginal tag is.
        rescue_pad_frac: Crop padding as a fraction of the tag's last apparent
            size, absorbing inter-frame motion at cruise speed.
    """

    enter_margin: float = 10.0
    keep_margin: float = 5.0
    rescue: bool = True
    rescue_frames: int = 2
    rescue_max_per_frame: int = 3
    rescue_upscale: float = 2.0
    rescue_pad_frac: float = 0.6


class TagPersistence:
    """Stateful per-tag filter: hysteresis on margin + ROI rescue of dropouts.

    Args:
        detector: The pupil_apriltags ``Detector`` (or anything with the same
            ``detect(gray)`` shape) shared with the full-frame pass. May be
            ``None`` when ``rescue`` is off.
        params: Tuning; see :class:`TagPersistenceParams`.
    """

    def __init__(self, detector,
                 params: Optional[TagPersistenceParams] = None) -> None:
        self.p = params or TagPersistenceParams()
        self._rescue_detector = detector
        self._active: set = set()          # tag ids used last frame
        #: id -> (last corners, consecutive frames missing)
        self._last: Dict[int, Tuple[np.ndarray, int]] = {}

    # ── main entry ───────────────────────────────────────────────────
    def filter(self, gray: np.ndarray, dets: Sequence,
               known_ids: Iterable[int]) -> List[RawDet]:
        """One frame: apply hysteresis, rescue dropouts, update tracking state.

        Args:
            gray: The full grayscale frame (needed only if a rescue runs).
            dets: Raw pupil_apriltags detections for this frame.
            known_ids: Tag ids present in the map; others are ignored entirely.

        Returns:
            The accepted detections, rescued ones included.
        """
        known = set(int(i) for i in known_ids)
        accepted: Dict[int, RawDet] = {}

        for d in dets:
            tid = int(d.tag_id)
            if tid not in known:
                continue
            margin = float(d.decision_margin)
            gate = self.p.keep_margin if tid in self._active else self.p.enter_margin
            if margin < gate:
                continue
            corners = np.array(d.corners, dtype=np.float64).reshape(4, 2)
            prev = accepted.get(tid)
            if prev is None or margin > prev.decision_margin:
                accepted[tid] = RawDet(tid, corners, margin)

        if self.p.rescue:
            self._rescue_missing(gray, accepted)

        # Track state for the next frame.
        for tid, det in accepted.items():
            self._last[tid] = (det.corners, 0)
        for tid in list(self._last):
            if tid not in accepted:
                corners, missing = self._last[tid]
                if missing + 1 > self.p.rescue_frames:
                    del self._last[tid]
                else:
                    self._last[tid] = (corners, missing + 1)
        self._active = set(accepted)
        return list(accepted.values())

    # ── rescue pass ──────────────────────────────────────────────────
    def _rescue_missing(self, gray: np.ndarray, accepted: Dict[int, RawDet]) -> None:
        """Second-chance detection in an upscaled crop around each lost tag."""
        candidates = [(tid, corners) for tid, (corners, missing) in self._last.items()
                      if tid not in accepted and missing < self.p.rescue_frames]
        if not candidates or gray is None or gray.size == 0 \
                or self._rescue_detector is None:
            return

        h, w = gray.shape[:2]
        for tid, corners in candidates[: self.p.rescue_max_per_frame]:
            x0, y0, x1, y1 = self._crop_box(corners, w, h)
            if x1 - x0 < 16 or y1 - y0 < 16:
                continue
            crop = np.ascontiguousarray(gray[y0:y1, x0:x1])
            up = self.p.rescue_upscale
            if up > 1.0:
                crop = cv2.resize(crop, None, fx=up, fy=up,
                                  interpolation=cv2.INTER_CUBIC)
            for d in self._rescue_detector.detect(crop):
                if int(d.tag_id) != tid:
                    continue
                if float(d.decision_margin) < self.p.keep_margin:
                    continue
                found = np.array(d.corners, dtype=np.float64).reshape(4, 2)
                found = found / up + np.array([x0, y0], dtype=np.float64)
                accepted[tid] = RawDet(tid, found, float(d.decision_margin),
                                       rescued=True)
                break

    def _crop_box(self, corners: np.ndarray, w: int, h: int) -> Tuple[int, int, int, int]:
        """Padded, clamped bounding box around the tag's last known corners."""
        x_min, y_min = corners.min(axis=0)
        x_max, y_max = corners.max(axis=0)
        pad = self.p.rescue_pad_frac * max(x_max - x_min, y_max - y_min, 24.0)
        return (int(max(0.0, x_min - pad)), int(max(0.0, y_min - pad)),
                int(min(float(w), x_max + pad)), int(min(float(h), y_max + pad)))

    def reset(self) -> None:
        self._active = set()
        self._last = {}
