"""Median-Flow bounding-box tracker: the *robust* detect-once / track-many core.

A hardened replacement for the plain sparse-LK tracker
(:mod:`.lk_box_tracker`), aimed squarely at the failure that makes plain LK
"lie": when the target is occluded or leaves the frame, LK's surviving corners
latch onto whatever background texture happens to sit inside the box and the
tracker keeps *confidently* reporting a box on the wrong thing (the background),
never admitting it lost the object.

Median-Flow (Kalal, Mikolajczyk & Matas, 2010 — *Forward-Backward Error:
Automatic Detection of Tracking Failures*) fixes this on base OpenCV + numpy
(no ``opencv-contrib``, no CUDA), with four ideas:

  1. **Forward-backward (FB) consistency.** Each point is tracked forward to the
     next frame and back again; a point whose round trip does not return near its
     origin was not tracked reliably (an occluder, a repeated texture, the frame
     edge) and is dropped. The *median* FB error over all points is the single
     number that tells an honest track from a lie — it spikes the instant the
     object is occluded or exits, because most points then fail their round trip.
  2. **Median consensus, not extremes.** The box translation is the *median* of
     the surviving point displacements and its scale the *median* of pairwise
     inter-point distance ratios. A few background points that move differently
     from the object are out-voted instead of dragging or ballooning the box
     (plain LK used the bounding rect of survivors, so one stuck background corner
     moved the whole box).
  3. **Appearance model.** The seed box is remembered as a small normalised
     template; each frame the candidate box must still correlate (NCC) with it.
     This catches the slow, insidious drift the FB check alone misses — the box
     sliding off a static object onto equally static background, where every
     one-frame round trip looks fine but the box no longer *looks like* the
     target. Re-seeding on a fresh detection refreshes the template, so it tracks
     the target's changing scale/lighting as the drone closes in.
  4. **Honest loss.** Lock is declared lost — never silently kept — when the
     median FB error is too large, too few points survive (absolute *and* as a
     fraction of the seed grid), the appearance correlation drops, or the box
     degenerates. That honest loss is exactly what lets the mission FSM re-search
     (or the detector re-acquire) instead of servoing onto the background.

Seed points are a regular grid over the *inner* part of the box (``inner_frac``),
because a detector's rectangle usually frames a non-rectangular object: its
corners are background, so biasing the grid inward keeps more points on the
object itself.

Grayscale in, :class:`BoxObservation` out — the same contract as
:class:`LucasKanadeBoxTracker`, so it drops straight into
:class:`~...target_tracker.TargetTracker` via the box-tracker registry.
ROS-free, cv2 + numpy only, Python-3.8-safe.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import cv2

from sparx_agency.core.common.math.bbox import clip_xyxy
from sparx_agency.core.mapping.tracking.interface import (
    BBox,
    BoxObservation,
    BoxTracker,
)


@dataclass(frozen=True)
class MedianFlowConfig:
    """Tuning for :class:`MedianFlowBoxTracker`.

    Attributes:
        grid: Seed a ``grid x grid`` regular grid of points inside the box.
        inner_frac: Fraction of the box (centred) the grid spans; ``< 1`` biases
            the points toward the object interior and away from the background
            that a rectangle traps at the corners of a non-rectangular object.
        lk_win: Lucas-Kanade window size (px); bigger handles larger motion.
        lk_levels: LK pyramid levels; handles motion up to ~win * 2^levels/frame.
        lk_iters: Max iterations of the LK termination criteria.
        lk_eps: Epsilon of the LK termination criteria.
        fb_max_error: Forward-backward round-trip tolerance (px). Points above it
            are dropped, and the track is declared lost if the *median* FB error
            over all points exceeds it (the occlusion / left-frame signal).
        min_points: Absolute floor on surviving points below which lock is lost.
        min_survival_frac: Relative floor: lock is lost below this fraction of the
            seeded grid surviving (catches "most points failed" independent of
            grid size).
        max_scale_step: Clamp the per-frame scale change to
            ``[1/(1+s), 1+s]`` — a robust box does not jump size in one frame.
        template_size: Side (px) of the square appearance template; ``0`` disables
            the appearance check entirely (FB + consensus only).
        template_min_ncc: Minimum normalised cross-correlation between the current
            box and the seed template to keep lock; below it the box no longer
            looks like the target and lock is lost.
        min_box_size: A box narrower/shorter than this (px) is treated as lost.
    """

    grid: int = 10
    inner_frac: float = 0.8
    lk_win: int = 21
    lk_levels: int = 3
    lk_iters: int = 20
    lk_eps: float = 0.03
    fb_max_error: float = 3.0
    min_points: int = 10
    min_survival_frac: float = 0.2
    max_scale_step: float = 0.15
    template_size: int = 16
    template_min_ncc: float = 0.5
    min_box_size: float = 6.0

    def __post_init__(self) -> None:
        if self.grid < 2:
            raise ValueError("grid must be >= 2.")
        if not (0.0 < self.inner_frac <= 1.0):
            raise ValueError("inner_frac must be in (0, 1].")
        if self.fb_max_error <= 0.0:
            raise ValueError("fb_max_error must be > 0.")
        if self.min_points < 2:
            raise ValueError("min_points must be >= 2.")
        if not (0.0 < self.min_survival_frac <= 1.0):
            raise ValueError("min_survival_frac must be in (0, 1].")
        if self.max_scale_step < 0.0:
            raise ValueError("max_scale_step must be >= 0.")
        if self.template_size < 0:
            raise ValueError("template_size must be >= 0.")


class MedianFlowBoxTracker(BoxTracker):
    """Robust Median-Flow single-object box tracker; see module docstring."""

    name = "median_flow"

    def __init__(self, config: Optional[MedianFlowConfig] = None) -> None:
        self.cfg = config or MedianFlowConfig()
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
        self._bbox: Optional[BBox] = None
        self._template: Optional[np.ndarray] = None
        self._valid = False

    @property
    def is_valid(self) -> bool:
        return self._valid

    def seed(self, gray: np.ndarray, bbox_xyxy: BBox) -> bool:
        gray = self._as_gray(gray)
        h, w = gray.shape[:2]
        x1, y1, x2, y2 = clip_xyxy(bbox_xyxy, w, h)
        if (x2 - x1) < self.cfg.min_box_size or (y2 - y1) < self.cfg.min_box_size:
            self._valid = False
            return False

        template = None
        if self.cfg.template_size > 0:
            template = self._make_template(gray, (x1, y1, x2, y2))
            if template is None:
                # A flat/degenerate box carries no appearance to lock onto — refuse
                # to seed, the same honest failure LK makes when it finds no corners.
                self._valid = False
                return False

        self._prev_gray = gray
        self._bbox = (float(x1), float(y1), float(x2), float(y2))
        self._template = template
        self._valid = True
        return True

    def update(self, gray: np.ndarray) -> BoxObservation:
        if not self._valid or self._prev_gray is None or self._bbox is None:
            return BoxObservation(bbox_xyxy=None, n_matches=0, valid=False)

        gray = self._as_gray(gray)
        h, w = gray.shape[:2]
        grid = self._grid_points(self._bbox)
        n_seed = grid.shape[0]

        p1, keep, median_fb = self._forward_backward(self._prev_gray, gray, grid, w, h)
        n = int(keep.sum())
        if (median_fb is None or median_fb > self.cfg.fb_max_error
                or n < self.cfg.min_points
                or n < self.cfg.min_survival_frac * n_seed):
            return self._lose(n)

        p0k = grid.reshape(-1, 2)[keep]
        p1k = p1.reshape(-1, 2)[keep]
        dx, dy, scale = self._median_shift_scale(p0k, p1k)
        bbox = self._apply_motion(self._bbox, dx, dy, scale, w, h)
        if (bbox[2] - bbox[0]) < self.cfg.min_box_size or \
                (bbox[3] - bbox[1]) < self.cfg.min_box_size:
            return self._lose(n)
        if self._template is not None and not self._appearance_ok(gray, bbox):
            return self._lose(n)

        self._prev_gray = gray
        self._bbox = bbox
        self._valid = True
        return BoxObservation(bbox_xyxy=bbox, n_matches=n, valid=True)

    # ── tracking internals ───────────────────────────────────────────
    def _grid_points(self, bbox: BBox) -> np.ndarray:
        """Regular ``grid x grid`` point cloud over the inner part of ``bbox``."""
        x1, y1, x2, y2 = bbox
        mx = 0.5 * (1.0 - self.cfg.inner_frac) * (x2 - x1)
        my = 0.5 * (1.0 - self.cfg.inner_frac) * (y2 - y1)
        g = int(self.cfg.grid)
        xs = np.linspace(x1 + mx, x2 - mx, g, dtype=np.float32)
        ys = np.linspace(y1 + my, y2 - my, g, dtype=np.float32)
        gx, gy = np.meshgrid(xs, ys)
        pts = np.stack([gx.ravel(), gy.ravel()], axis=1).astype(np.float32)
        return pts.reshape(-1, 1, 2)

    def _forward_backward(self, prev: np.ndarray, cur: np.ndarray, grid: np.ndarray,
                          w: int, h: int) -> Tuple[Optional[np.ndarray], np.ndarray,
                                                    Optional[float]]:
        """Track the grid forward then back; return (fwd points, keep mask, median FB).

        ``keep`` is True for points that survive LK both ways, stay in bounds, and
        round-trip within ``fb_max_error``. ``median FB`` is over all LK-valid
        points (None if none survived LK) — the occlusion/left-frame health metric.
        """
        empty = np.zeros(grid.shape[0], dtype=bool)
        p1, st1, _ = cv2.calcOpticalFlowPyrLK(prev, cur, grid, None, **self._lk_params)
        if p1 is None or st1 is None:
            return None, empty, None
        p0, st2, _ = cv2.calcOpticalFlowPyrLK(cur, prev, p1, None, **self._lk_params)
        if p0 is None or st2 is None:
            return p1, empty, None

        g = grid.reshape(-1, 2)
        fwd = p1.reshape(-1, 2)
        back = p0.reshape(-1, 2)
        alive = (st1.flatten() == 1) & (st2.flatten() == 1)
        in_bounds = (fwd[:, 0] >= 0) & (fwd[:, 0] < w) & (fwd[:, 1] >= 0) & (fwd[:, 1] < h)
        alive &= in_bounds
        if not alive.any():
            return p1, empty, None
        fb = np.linalg.norm(g - back, axis=1)
        median_fb = float(np.median(fb[alive]))
        keep = alive & (fb <= self.cfg.fb_max_error)
        return p1, keep, median_fb

    def _median_shift_scale(self, p0: np.ndarray, p1: np.ndarray) -> Tuple[float, float, float]:
        """Median translation and (clamped) scale between two point sets."""
        d = p1 - p0
        dx = float(np.median(d[:, 0]))
        dy = float(np.median(d[:, 1]))
        return dx, dy, self._median_scale(p0, p1)

    def _median_scale(self, p0: np.ndarray, p1: np.ndarray) -> float:
        k = p0.shape[0]
        if k < 2:
            return 1.0
        idx = np.arange(k)
        if k > 50:  # bound the O(k^2) pairwise computation
            idx = np.unique(np.linspace(0, k - 1, 50).astype(int))
        a, b = p0[idx], p1[idx]
        d0 = np.linalg.norm(a[:, None, :] - a[None, :, :], axis=2)
        d1 = np.linalg.norm(b[:, None, :] - b[None, :, :], axis=2)
        iu = np.triu_indices(len(idx), k=1)
        r0, r1 = d0[iu], d1[iu]
        ok = r0 > 1e-3
        if not ok.any():
            return 1.0
        s = float(np.median(r1[ok] / r0[ok]))
        lo, hi = 1.0 / (1.0 + self.cfg.max_scale_step), 1.0 + self.cfg.max_scale_step
        return min(max(s, lo), hi)

    def _apply_motion(self, bbox: BBox, dx: float, dy: float, scale: float,
                      w: int, h: int) -> BBox:
        cx = 0.5 * (bbox[0] + bbox[2]) + dx
        cy = 0.5 * (bbox[1] + bbox[3]) + dy
        bw = (bbox[2] - bbox[0]) * scale
        bh = (bbox[3] - bbox[1]) * scale
        moved = (cx - 0.5 * bw, cy - 0.5 * bh, cx + 0.5 * bw, cy + 0.5 * bh)
        return clip_xyxy(moved, w, h)

    def _lose(self, n: int) -> BoxObservation:
        self._valid = False
        return BoxObservation(bbox_xyxy=None, n_matches=n, valid=False)

    # ── appearance model ─────────────────────────────────────────────
    def _make_template(self, gray: np.ndarray, bbox: BBox) -> Optional[np.ndarray]:
        """Resize the box content to a fixed square, or None for a flat/tiny box."""
        x1 = max(0, int(round(bbox[0])))
        y1 = max(0, int(round(bbox[1])))
        x2 = min(gray.shape[1], int(round(bbox[2])))
        y2 = min(gray.shape[0], int(round(bbox[3])))
        if x2 - x1 < 2 or y2 - y1 < 2:
            return None
        t = int(self.cfg.template_size)
        patch = cv2.resize(gray[y1:y2, x1:x2], (t, t)).astype(np.float32)
        if float(patch.std()) < 1e-3:  # flat: no appearance to correlate against
            return None
        return patch

    def _appearance_ok(self, gray: np.ndarray, bbox: BBox) -> bool:
        patch = self._make_template(gray, bbox)
        if patch is None:
            return False
        return self._ncc(patch, self._template) >= self.cfg.template_min_ncc

    @staticmethod
    def _ncc(a: np.ndarray, b: np.ndarray) -> float:
        """Zero-mean normalised cross-correlation of two equal-size patches."""
        a = a.astype(np.float32) - float(a.mean())
        b = b.astype(np.float32) - float(b.mean())
        na, nb = float(np.linalg.norm(a)), float(np.linalg.norm(b))
        if na < 1e-6 or nb < 1e-6:
            return 0.0
        return float((a * b).sum() / (na * nb))

    @staticmethod
    def _as_gray(image: np.ndarray) -> np.ndarray:
        """Accept HxW gray or HxWx3 (assumed already the caller's channel order)."""
        img = np.asarray(image)
        if img.ndim == 2:
            return img if img.dtype == np.uint8 else img.astype(np.uint8)
        if img.ndim == 3 and img.shape[2] == 3:
            return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        raise ValueError(
            "MedianFlowBoxTracker expects HxW or HxWx3, got %r" % (img.shape,))
