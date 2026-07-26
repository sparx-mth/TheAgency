"""Sparse Lucas-Kanade bounding-box tracker (detect-once / track-many).

The lightest robust single-object tracker we have: seed Shi-Tomasi corners inside
a detection's box, then propagate them frame-to-frame with pyramidal Lucas-Kanade
optical flow. The bounding rectangle of the surviving corners *is* the new box, so
scale is implicit — the box grows as the drone closes in on the target, which the
visual servo reads directly as a proximity signal. No neural inference, no
``opencv-contrib`` modules, no CUDA: ~2-3 ms per 640x360 frame on a Jetson AGX
Orin CPU, leaving the SoC budget for FALCON's voxel mapping.

Improvements over the reference orchestrator's inline tracker:
  * the output box is a **MAD-outlier-rejected** bounding rect (``outlier_k_mad``),
    so a single corner that jumps onto the background cannot balloon the box, and
    (unlike percentile trimming) a clean corner cloud is not systematically shrunk;
  * corners that drift outside the image or fail LK are dropped every frame, and
    lock is declared lost below ``min_matches`` survivors.

Grayscale in, :class:`BoxObservation` out. Re-seeding on fresh detections (to
bound drift) is orchestrated by the composing ``TargetTracker``, not here.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import cv2

from sparx_agency.core.common.math.bbox import bounds_rect, clip_xyxy
from sparx_agency.core.mapping.tracking.interface import (
    BBox,
    BoxObservation,
    BoxTracker,
)


@dataclass(frozen=True)
class LKBoxTrackerConfig:
    """Tuning for :class:`LucasKanadeBoxTracker`.

    Attributes:
        max_corners: Cap on Shi-Tomasi corners seeded per box.
        quality_level: Shi-Tomasi quality threshold (lower => more, weaker corners).
        min_distance: Minimum separation between seeded corners (px).
        block_size: Shi-Tomasi neighbourhood size (px).
        lk_win: Lucas-Kanade window size (px); bigger handles larger motion.
        lk_levels: LK pyramid levels; handles motion up to ~win * 2^levels per frame.
        lk_iters: Max iterations of the LK termination criteria.
        lk_eps: Epsilon of the LK termination criteria.
        min_matches: Below this many surviving corners the track is lost.
        outlier_k_mad: Robust-sigma multiplier for rejecting a jumped corner when
            forming the output box (see :func:`...common.math.bbox.bounds_rect`);
            ``0`` => exact min/max of survivors.
    """

    max_corners: int = 80
    quality_level: float = 0.05
    min_distance: float = 5.0
    block_size: int = 7
    lk_win: int = 21
    lk_levels: int = 3
    lk_iters: int = 20
    lk_eps: float = 0.03
    min_matches: int = 8
    outlier_k_mad: float = 3.0

    def __post_init__(self) -> None:
        if self.max_corners < 1:
            raise ValueError("max_corners must be >= 1.")
        if self.min_matches < 2:
            raise ValueError("min_matches must be >= 2.")
        if self.outlier_k_mad < 0.0:
            raise ValueError("outlier_k_mad must be >= 0.")


class LucasKanadeBoxTracker(BoxTracker):
    """Sparse-LK single-object box tracker; see module docstring."""

    name = "lucas_kanade"

    def __init__(self, config: Optional[LKBoxTrackerConfig] = None) -> None:
        self.cfg = config or LKBoxTrackerConfig()
        self._lk_params = dict(
            winSize=(int(self.cfg.lk_win), int(self.cfg.lk_win)),
            maxLevel=int(self.cfg.lk_levels),
            criteria=(
                cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
                int(self.cfg.lk_iters),
                float(self.cfg.lk_eps),
            ),
        )
        self.reset()

    # ── BoxTracker API ───────────────────────────────────────────────
    def reset(self) -> None:
        self._prev_gray: Optional[np.ndarray] = None
        self._prev_pts: Optional[np.ndarray] = None  # (N, 1, 2) float32
        self._bbox: Optional[BBox] = None
        self._valid = False

    @property
    def is_valid(self) -> bool:
        return self._valid

    def seed(self, gray: np.ndarray, bbox_xyxy: BBox) -> bool:
        gray = self._as_gray(gray)
        h, w = gray.shape[:2]
        x1, y1, x2, y2 = clip_xyxy(bbox_xyxy, w, h)
        bw, bh = x2 - x1, y2 - y1
        if bw < 2.0 or bh < 2.0:
            self._valid = False
            return False

        # Corner mask = the box itself; the (padded) ROI only bounds where we look.
        mask = np.zeros((h, w), dtype=np.uint8)
        mask[int(y1):int(np.ceil(y2)), int(x1):int(np.ceil(x2))] = 255

        pts = cv2.goodFeaturesToTrack(
            gray,
            maxCorners=int(self.cfg.max_corners),
            qualityLevel=float(self.cfg.quality_level),
            minDistance=float(self.cfg.min_distance),
            mask=mask,
            blockSize=int(self.cfg.block_size),
        )
        if pts is None or len(pts) < self.cfg.min_matches:
            self._valid = False
            return False

        self._prev_gray = gray
        self._prev_pts = pts.astype(np.float32)
        self._bbox = (float(x1), float(y1), float(x2), float(y2))
        self._valid = True
        return True

    def update(self, gray: np.ndarray) -> BoxObservation:
        if not self._valid or self._prev_gray is None or self._prev_pts is None:
            return BoxObservation(bbox_xyxy=None, n_matches=0, valid=False)

        gray = self._as_gray(gray)
        new_pts, status, _err = cv2.calcOpticalFlowPyrLK(
            self._prev_gray, gray, self._prev_pts, None, **self._lk_params
        )
        if new_pts is None or status is None:
            self._valid = False
            return BoxObservation(bbox_xyxy=None, n_matches=0, valid=False)

        h, w = gray.shape[:2]
        xy = new_pts.reshape(-1, 2)
        alive = (status.flatten() == 1)
        in_bounds = (
            (xy[:, 0] >= 0) & (xy[:, 0] < w) & (xy[:, 1] >= 0) & (xy[:, 1] < h)
        )
        ok = alive & in_bounds
        n = int(ok.sum())
        if n < self.cfg.min_matches:
            self._valid = False
            return BoxObservation(bbox_xyxy=None, n_matches=n, valid=False)

        survivors = xy[ok].astype(np.float32)
        bbox = bounds_rect(survivors, k_mad=float(self.cfg.outlier_k_mad))
        bbox = clip_xyxy(bbox, w, h)

        # Roll state forward.
        self._prev_gray = gray
        self._prev_pts = survivors.reshape(-1, 1, 2)
        self._bbox = bbox
        self._valid = True
        return BoxObservation(bbox_xyxy=bbox, n_matches=n, valid=True)

    # ── helpers ──────────────────────────────────────────────────────
    @staticmethod
    def _as_gray(image: np.ndarray) -> np.ndarray:
        """Accept HxW gray or HxWx3 (assumed already the caller's channel order)."""
        img = np.asarray(image)
        if img.ndim == 2:
            return img if img.dtype == np.uint8 else img.astype(np.uint8)
        if img.ndim == 3 and img.shape[2] == 3:
            return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        raise ValueError("LucasKanadeBoxTracker expects HxW or HxWx3, got %r" % (img.shape,))
