"""Viewpoint / pitch augmentation to close the drone-height domain gap.

The pretrained policies saw a ~0.3 m ground-robot camera; the drone flies at
~1.0 m. The dominant visual consequences (report-verified) are a **shifted horizon
line** (the drone's principal point sits high, ``cy=90`` on 294 rows) and a
**receding, foreshortened ground plane**. A camera-pitch rotation reproduces both:
tilting the virtual camera down brings the near ground back into frame, emulating a
lower effective viewpoint.

A pure-rotation homography ``H = K R K^{-1}`` is *exact* for a camera that only
rotates, and is the cheapest, artifact-free way to manufacture the missing
viewpoints. Height translation (which also warps the ground plane) is left as an
extension because it needs the ground-plane geometry; the pitch rotation plus depth
range jitter covers the first-order gap. Applied identically to RGB and depth so
they stay registered.

numpy + cv2 only (runs in the plain ``.venv``).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

try:
    import cv2
except Exception as exc:  # pragma: no cover
    cv2 = None
    _cv2_err = exc

from sparx_agency.core.common.types import Intrinsics


@dataclass(frozen=True)
class ViewpointAugmentConfig:
    """Ranges for random viewpoint / photometric jitter.

    Attributes:
        pitch_deg_range: Uniform range of synthetic pitch rotation (deg,
            nose-down positive). Emulates the height/horizon gap.
        depth_scale_range: Multiplicative jitter on metric depth (robustness to
            the K-vs-P ~1.27 metric ambiguity and range shift). NavDP only.
        depth_offset_m: Additive depth jitter (meters). NavDP only.
        brightness_range: Multiplicative RGB brightness jitter.
        enabled: Master switch (off -> identity, for eval).
    """

    pitch_deg_range: Tuple[float, float] = (-8.0, 8.0)
    depth_scale_range: Tuple[float, float] = (0.85, 1.15)
    depth_offset_m: float = 0.15
    brightness_range: Tuple[float, float] = (0.8, 1.2)
    enabled: bool = True


def pitch_homography(intrinsics: Intrinsics, pitch_deg: float) -> np.ndarray:
    """``3x3`` homography for a pure pitch rotation about the optical x-axis.

    ``H = K R_x(pitch) K^{-1}``. Exact for rotation-only viewpoint change.
    """
    k = np.array(
        [[intrinsics.fx, 0.0, intrinsics.cx],
         [0.0, intrinsics.fy, intrinsics.cy],
         [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    a = np.deg2rad(pitch_deg)
    rx = np.array(
        [[1.0, 0.0, 0.0],
         [0.0, np.cos(a), -np.sin(a)],
         [0.0, np.sin(a), np.cos(a)]],
        dtype=np.float64,
    )
    return (k @ rx @ np.linalg.inv(k)).astype(np.float64)


def warp_rgb(rgb: np.ndarray, homography: np.ndarray) -> np.ndarray:
    """Warp an RGB image by a homography (bilinear, replicate border)."""
    if cv2 is None:  # pragma: no cover
        raise RuntimeError(f"cv2 required: {_cv2_err}")
    h, w = rgb.shape[:2]
    return cv2.warpPerspective(
        rgb, homography, (w, h),
        flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE,
    )


def warp_depth(depth_m: np.ndarray, homography: np.ndarray) -> np.ndarray:
    """Warp a metric depth image by a homography (nearest, so depths aren't blended)."""
    if cv2 is None:  # pragma: no cover
        raise RuntimeError(f"cv2 required: {_cv2_err}")
    h, w = depth_m.shape[:2]
    return cv2.warpPerspective(
        depth_m.astype(np.float32), homography, (w, h),
        flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0.0,
    )


@dataclass(frozen=True)
class AugmentedFrame:
    """A jittered observation and the pitch actually applied (label geometry uses it)."""

    rgb: Optional[np.ndarray]
    depth_m: Optional[np.ndarray]
    pitch_deg: float
    depth_scale: float


def apply_viewpoint_augment(
    intrinsics: Intrinsics,
    config: ViewpointAugmentConfig,
    rng: np.random.Generator,
    rgb: Optional[np.ndarray] = None,
    depth_m: Optional[np.ndarray] = None,
) -> AugmentedFrame:
    """Sample and apply a random viewpoint + photometric jitter to a frame.

    The applied ``pitch_deg`` is returned so the label geometry (which back-projects
    the same depth) can use the matching ``LocalMapConfig.pitch_deg`` and stay
    consistent with the warped observation.

    Args:
        intrinsics: Intrinsics for ``rgb`` / ``depth_m``.
        config: Jitter ranges.
        rng: A seeded ``np.random.Generator`` (for reproducibility).
        rgb: Optional ``(H, W, 3)`` image.
        depth_m: Optional ``(H, W)`` metric depth.

    Returns:
        :class:`AugmentedFrame`.
    """
    if not config.enabled:
        return AugmentedFrame(rgb=rgb, depth_m=depth_m, pitch_deg=0.0, depth_scale=1.0)

    pitch = float(rng.uniform(*config.pitch_deg_range))
    h = pitch_homography(intrinsics, pitch)

    out_rgb = None
    if rgb is not None:
        out_rgb = warp_rgb(rgb, h)
        b = float(rng.uniform(*config.brightness_range))
        out_rgb = np.clip(out_rgb.astype(np.float32) * b, 0, 255).astype(rgb.dtype)

    depth_scale = float(rng.uniform(*config.depth_scale_range))
    out_depth = None
    if depth_m is not None:
        out_depth = warp_depth(depth_m, h)
        offset = float(rng.uniform(-config.depth_offset_m, config.depth_offset_m))
        valid = out_depth > 0
        out_depth[valid] = out_depth[valid] * depth_scale + offset

    return AugmentedFrame(rgb=out_rgb, depth_m=out_depth, pitch_deg=pitch, depth_scale=depth_scale)
