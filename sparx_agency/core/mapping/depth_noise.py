"""Depth-image sensor noise model (pure numpy, ROS-free).

Adds additive and/or multiplicative Gaussian noise to a metric depth image,
for testing how robust the mapping pipeline is to a noisy depth sensor. Only
valid pixels (finite and ``> 0``) are perturbed; invalid pixels (``0``, ``inf``,
``nan``) are left untouched, and the result is clamped to be non-negative.

This is unrelated to localization noise (see
:mod:`sparx_agency.core.localization.dead_reckoning_noise`) and is applied
independently.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass(frozen=True)
class DepthNoiseParams:
    """Parameters for :func:`add_depth_noise`.

    Attributes:
        std: Additive Gaussian noise std, in metres. ``0`` disables it.
        proportional: Multiplicative noise — each valid pixel is scaled by a
            sample from ``N(1, proportional)``. ``0`` disables it. Models a
            range-dependent error (a percentage of the measured depth).
    """

    std: float = 0.0
    proportional: float = 0.0

    def enabled(self) -> bool:
        """True if either noise source would change the image."""
        return self.std > 0.0 or self.proportional > 0.0


def add_depth_noise(depth: np.ndarray, params: DepthNoiseParams,
                    rng: Optional[np.random.RandomState] = None) -> np.ndarray:
    """Return a noisy copy of a metric depth image.

    Args:
        depth: HxW float depth in metres. Not modified in place.
        params: Additive/multiplicative noise configuration.
        rng: Seeded ``numpy.random.RandomState`` (or compatible) for
            reproducibility. A fresh one is used if omitted.

    Returns:
        A new ``float32`` HxW array with noise applied to the valid pixels and
        clamped to ``>= 0``. If ``params`` is disabled the input is returned
        unchanged (as ``float32``).
    """
    if rng is None:
        rng = np.random.RandomState()
    arr = np.asarray(depth, dtype=np.float32).copy()
    if not params.enabled():
        return arr
    valid = np.isfinite(arr) & (arr > 0.0)
    if params.std > 0.0:
        noise = rng.normal(0.0, params.std, arr.shape).astype(np.float32)
        arr[valid] += noise[valid]
    if params.proportional > 0.0:
        scale = rng.normal(1.0, params.proportional, arr.shape).astype(np.float32)
        arr[valid] *= scale[valid]
    np.maximum(arr, 0.0, out=arr)
    return arr
