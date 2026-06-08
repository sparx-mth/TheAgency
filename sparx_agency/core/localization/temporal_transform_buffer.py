"""
Timestamp-exact pairing of an asynchronous data stream (e.g. depth frames)
with pose estimates, by buffering poses and matching on capture time.

Pure numpy + :mod:`se3`; no ROS. This is the algorithmic heart of the FALCON
``mapping_sync`` adapter: keep a short, time-sorted history of SE(3) transforms
per localization source and, for a query stamp, return the co-temporal
transform (exact / nearest within a tolerance, or SLERP-interpolated across a
small bracket). The ROS node owns the message types, threading and the
non-blocking "wait for a late pose" gate; the matching maths live here.

Conventions:
  - stamps are seconds (float); transforms are 4x4 homogeneous matrices.
  - a single source is a :class:`TemporalTransformBuffer`; several prioritized
    sources are a :class:`MultiSourceTemporalMatcher` (first with a co-temporal
    pose wins).
"""
from __future__ import annotations

from bisect import bisect_left
from typing import List, Optional, Tuple

import numpy as np

from ..common.math import se3

# lookup() result kinds
EXACT, NEAREST, INTERP = "exact", "nearest", "interp"
EMPTY, NO_MATCH = "empty", "no_match"

LookupResult = Tuple[Optional[np.ndarray], str, Optional[float]]


class TemporalTransformBuffer:
    """Time-sorted SE(3) history for ONE source, with nearest/interp lookup."""

    def __init__(self, buffer_sec: float = 5.0):
        if buffer_sec <= 0.0:
            raise ValueError(f"buffer_sec must be > 0, got {buffer_sec}")
        self.buffer_sec = float(buffer_sec)
        self._stamps: List[float] = []
        self._tfs: List[np.ndarray] = []

    def __len__(self) -> int:
        return len(self._stamps)

    def insert(self, stamp: float, transform: np.ndarray) -> None:
        """Insert (stamp, 4x4) keeping order; drop history older than buffer_sec."""
        t = float(stamp)
        i = bisect_left(self._stamps, t)
        if i < len(self._stamps) and self._stamps[i] == t:
            self._tfs[i] = transform            # same stamp -> overwrite
        else:
            self._stamps.insert(i, t)
            self._tfs.insert(i, transform)
        self._prune()

    def _prune(self) -> None:
        if not self._stamps:
            return
        cutoff = self._stamps[-1] - self.buffer_sec
        d = 0
        while d < len(self._stamps) and self._stamps[d] < cutoff:
            d += 1
        if d:
            del self._stamps[:d]
            del self._tfs[:d]

    def nearest_dt(self, t: float) -> Optional[float]:
        """Time gap (s) to the closest stamp, or None if the buffer is empty."""
        n = len(self._stamps)
        if n == 0:
            return None
        i = bisect_left(self._stamps, t)
        best = None
        for j in (i - 1, i):
            if 0 <= j < n:
                d = abs(self._stamps[j] - t)
                best = d if best is None else min(best, d)
        return best

    def lookup(self, t: float, tol: float, gap: float = 0.0) -> LookupResult:
        """Co-temporal transform for stamp ``t``.

        Returns ``(T, kind, dt)``: the nearest pose within ``tol`` (kind
        EXACT/NEAREST), else a SLERP interpolation if ``t`` is bracketed by two
        poses no more than ``gap`` apart (kind INTERP, ``gap<=0`` disables it),
        else ``(None, EMPTY|NO_MATCH, None)``.
        """
        n = len(self._stamps)
        if n == 0:
            return None, EMPTY, None
        i = bisect_left(self._stamps, t)
        bj, bd = None, None
        for j in (i - 1, i):
            if 0 <= j < n:
                d = abs(self._stamps[j] - t)
                if bd is None or d < bd:
                    bd, bj = d, j
        if bj is not None and bd <= tol:
            return self._tfs[bj].copy(), (EXACT if bd == 0.0 else NEAREST), bd
        if gap > 0.0 and 0 < i < n:
            t0, t1 = self._stamps[i - 1], self._stamps[i]
            if (t1 - t0) <= gap and t0 <= t <= t1:
                return self._interp(i - 1, i, t0, t1, t), INTERP, 0.0
        return None, NO_MATCH, None

    def _interp(self, i0: int, i1: int, t0: float, t1: float,
                t: float) -> np.ndarray:
        T0, T1 = self._tfs[i0], self._tfs[i1]
        r = (t - t0) / (t1 - t0) if t1 > t0 else 0.0
        q = se3.quaternion_slerp(se3.quaternion_from_matrix(T0),
                                 se3.quaternion_from_matrix(T1), r)
        T = se3.quaternion_matrix(q)
        T[:3, 3] = (1.0 - r) * T0[:3, 3] + r * T1[:3, 3]
        return T


class MultiSourceTemporalMatcher:
    """Several prioritized buffers; the first with a co-temporal pose wins."""

    def __init__(self, n_sources: int, buffer_sec: float = 5.0):
        if n_sources < 1:
            raise ValueError("n_sources must be >= 1")
        self.n_sources = n_sources
        self.buffers = [TemporalTransformBuffer(buffer_sec)
                        for _ in range(n_sources)]

    def insert(self, src: int, stamp: float, transform: np.ndarray) -> None:
        self.buffers[src].insert(stamp, transform)

    def lookup(self, t: float, tol: float,
               gap: float = 0.0) -> Tuple[Optional[np.ndarray], str, int]:
        """Try sources in priority order; return (T, kind, src) or (None, NO_MATCH, -1)."""
        for s, buf in enumerate(self.buffers):
            T, kind, _ = buf.lookup(t, tol, gap)
            if T is not None:
                return T, kind, s
        return None, NO_MATCH, -1

    def disagreement(self, src: int, t: float, transform: np.ndarray,
                     tol: float) -> Optional[float]:
        """Max position gap (m) between ``transform`` and other sources co-temporal
        at ``t`` (a loud sign two sources live in different world frames)."""
        worst = None
        for s, buf in enumerate(self.buffers):
            if s == src:
                continue
            To, _, _ = buf.lookup(t, tol, 0.0)
            if To is None:
                continue
            d = float(np.linalg.norm(To[:3, 3] - transform[:3, 3]))
            worst = d if worst is None else max(worst, d)
        return worst

    def nearest_dt(self, t: float) -> Optional[float]:
        """Smallest time gap to any source's nearest stamp (drop classification)."""
        best = None
        for buf in self.buffers:
            dt = buf.nearest_dt(t)
            if dt is None:
                continue
            best = dt if best is None else min(best, dt)
        return best

    def sizes(self) -> List[int]:
        return [len(b) for b in self.buffers]
