# core/mapping/costmap/occupancy.py
"""
Occupancy grid normalization utilities.

This module standardizes different occupancy representations into a binary
occupancy grid suitable for planning and distance-field generation.

Convention:
- occupancy == 0 -> free
- occupancy != 0 -> occupied

All outputs are np.uint8 with values in {0, 1}.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass(frozen=True, slots=True)
class OccupancyThresholds:
    """
    Threshold configuration for converting a grayscale map image (0..255) into occupancy.

    Typical ROS map images:
- 0   = occupied (black)
- 254/255 = free (white)
    But you may have different conventions.

    Args:
        occupied_if_below: Pixel values <= this are considered occupied.
        free_if_above: Pixel values >= this are considered free.
        unknown_value: Optional pixel value treated as unknown. If provided, unknowns can be
                       mapped to occupied or free by `unknown_as_occupied`.
        unknown_as_occupied: If True, unknowns become occupied, else free.
    """
    occupied_if_below: int = 50
    free_if_above: int = 250
    unknown_value: Optional[int] = None
    unknown_as_occupied: bool = True

    def __post_init__(self) -> None:
        if not (0 <= self.occupied_if_below <= 255):
            raise ValueError("occupied_if_below must be in [0,255]")
        if not (0 <= self.free_if_above <= 255):
            raise ValueError("free_if_above must be in [0,255]")
        if self.occupied_if_below >= self.free_if_above:
            raise ValueError("occupied_if_below must be < free_if_above")
        if self.unknown_value is not None and not (0 <= self.unknown_value <= 255):
            raise ValueError("unknown_value must be in [0,255]")


def normalize_binary_occupancy(occupancy: np.ndarray) -> np.ndarray:
    """
    Normalize any occupancy-like grid to binary uint8 {0,1}.

    Accepts:
      - bool
      - int/float arrays where 0 means free and nonzero means occupied

    Returns:
      np.uint8 array with 0=free, 1=occupied
    """
    if occupancy.ndim != 2:
        raise ValueError(f"occupancy must be 2D, got shape {occupancy.shape}")
    occ = (occupancy != 0).astype(np.uint8)
    return occ


def occupancy_from_grayscale(
    img: np.ndarray,
    thresholds: OccupancyThresholds = OccupancyThresholds(),
    *,
    flip_y: bool = False,
) -> np.ndarray:
    """
    Convert grayscale map image (H,W) into binary occupancy.

    Args:
        img: uint8 or float array (H,W). If float, assumed in [0,1] or [0,255].
        thresholds: threshold rules.
        flip_y: if True, flips vertically (useful when image origin differs from map origin).

    Returns:
        occupancy uint8 (H,W) with 0=free, 1=occupied.
    """
    if img.ndim != 2:
        raise ValueError(f"img must be 2D grayscale, got shape {img.shape}")

    if img.dtype != np.uint8:
        # Try to map float images to 0..255
        arr = img.astype(np.float32)
        if arr.max() <= 1.0 + 1e-6:
            arr = arr * 255.0
        img_u8 = np.clip(arr, 0.0, 255.0).astype(np.uint8)
    else:
        img_u8 = img

    if flip_y:
        img_u8 = np.flipud(img_u8)

    occ = np.zeros_like(img_u8, dtype=np.uint8)

    # Occupied / free bands
    occ[img_u8 <= thresholds.occupied_if_below] = 1
    occ[img_u8 >= thresholds.free_if_above] = 0

    # Unknown band in the middle -> treat as unknown
    mid = (img_u8 > thresholds.occupied_if_below) & (img_u8 < thresholds.free_if_above)
    if thresholds.unknown_value is not None:
        mid = mid | (img_u8 == thresholds.unknown_value)

    if thresholds.unknown_as_occupied:
        occ[mid] = 1
    else:
        occ[mid] = 0

    return occ
