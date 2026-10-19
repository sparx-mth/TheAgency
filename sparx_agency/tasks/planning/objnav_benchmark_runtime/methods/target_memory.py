"""The closing's 3-D memory of the target: one fused XYZ centroid, refined by every associated frame.

Saved the moment a takeover starts and updated on every frame that projects
onto the candidate, the memory is what the rest of the closing reads: the
elevation the dynamic pitch controller tilts toward, the range the approach
and the terminal test measure against, the spot the support-surface test and
the approach-history STOP are anchored to. The first centroid stays the
association anchor (:attr:`anchor`); the estimate moves.

Fusion is precision-weighted. Depth noise and the visible-part centroid shift
of a big object both grow with range, so a frame from one metre says more
about where the object is than one from four: each observation enters with
weight ``1 / sigma(range)^2`` where ``sigma = noise_floor_m + noise_per_m *
range``. The estimate's standard error ``1 / sqrt(sum of weights)`` shrinks
with every frame -- the precision the diagnostics and the HUD report -- and
the weighted RMS deviation of the observations (:attr:`spread_m`) says how
far apart the frames put the object.
"""
from __future__ import annotations

import math
from typing import Dict, Optional, Sequence, Tuple


class TargetMemory3D:
    """Precision-weighted running estimate of the target's XYZ centroid (West's weighted algorithm)."""

    def __init__(self, xyz: Sequence[float], range_m: float, step: int,
                 noise_floor_m: float = 0.03, noise_per_m: float = 0.03):
        if len(xyz) != 3 or not all(math.isfinite(float(v)) for v in xyz):
            raise ValueError("Target memory needs a finite xyz centroid")
        if not (math.isfinite(noise_floor_m) and noise_floor_m > 0 and math.isfinite(noise_per_m) and noise_per_m >= 0):
            raise ValueError("Depth noise model must be positive floor, non-negative slope")
        self.anchor: Tuple[float, float, float] = tuple(float(v) for v in xyz)
        self.noise_floor_m, self.noise_per_m = float(noise_floor_m), float(noise_per_m)
        self._mean = [0.0, 0.0, 0.0]
        self._m2 = [0.0, 0.0, 0.0]         # weighted sum of squared deviations per axis
        self.weight = 0.0
        self.n = 0
        self.first_step = self.last_step = int(step)
        self.update(xyz, range_m, step)

    def update(self, xyz: Sequence[float], range_m: Optional[float], step: int) -> None:
        """Fuse one associated observation; ``range_m`` sets its weight (None reads as the noise floor alone)."""
        values = [float(v) for v in xyz]
        if len(values) != 3 or not all(math.isfinite(v) for v in values):
            raise ValueError("Target memory update needs a finite xyz centroid")
        distance = 0.0 if range_m is None or not math.isfinite(float(range_m)) else max(0.0, float(range_m))
        sigma = self.noise_floor_m + self.noise_per_m * distance
        weight = 1.0 / (sigma * sigma)
        total = self.weight + weight
        for axis, value in enumerate(values):
            delta = value - self._mean[axis]
            self._mean[axis] += delta * weight / total
            self._m2[axis] += weight * delta * (value - self._mean[axis])
        self.weight = total
        self.n += 1
        self.last_step = int(step)

    @property
    def xyz(self) -> Tuple[float, float, float]:
        return tuple(self._mean)

    @property
    def sigma_m(self) -> float:
        """Standard error of the fused centroid: the precision, shrinking with every frame."""
        return 1.0 / math.sqrt(self.weight)

    @property
    def spread_m(self) -> float:
        """Weighted RMS deviation of the observations from the fused centroid, over the three axes."""
        return math.sqrt(max(0.0, sum(self._m2)) / self.weight) if self.n > 1 else 0.0

    def planar_range(self, xy: Sequence[float]) -> float:
        return math.dist((float(xy[0]), float(xy[1])), self._mean[:2])

    def shift_from_anchor_m(self) -> float:
        """How far the fused estimate has moved from the first centroid, in the plane."""
        return math.dist(self.anchor[:2], self._mean[:2])

    def diagnostics(self) -> Dict[str, object]:
        return {"xyz": [round(v, 3) for v in self._mean], "anchor": [round(v, 3) for v in self.anchor],
                "frames": self.n, "sigma_m": round(self.sigma_m, 4), "spread_m": round(self.spread_m, 3),
                "shift_from_anchor_m": round(self.shift_from_anchor_m(), 3),
                "first_step": self.first_step, "last_step": self.last_step}
