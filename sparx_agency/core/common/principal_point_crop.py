"""Principal-point-relocating image crop (pure numpy, ROS-free).

A camera with a *centred* principal point (``cx = W/2``, ``cy = H/2``) can be
made to match a target camera whose principal point is off-centre by cropping
asymmetrically. Cropping does not change the focal length per pixel, so the
focal length is preserved while the principal point moves to the desired
location in the cropped frame.

Concretely, to land the optical axis at ``(cx, cy)`` in the cropped image, the
crop window starts at ``(W/2 - cx, H/2 - cy)``.

The crop itself is encoding-agnostic: it slices the raw byte buffer by whole
rows and whole pixels (``step`` bytes per row, ``step / width`` bytes per
pixel), so it works for ``rgb8``, ``bgr8``, ``mono8``, ``32FC1``, ``16UC1`` —
anything — without ever decoding the image.
"""
from __future__ import annotations

from typing import Tuple

import numpy as np


def compute_crop_offsets(render_w: int, render_h: int,
                         target_w: int, target_h: int,
                         cx: float, cy: float) -> Tuple[int, int]:
    """Top-left crop offset that relocates the principal point to ``(cx, cy)``.

    Args:
        render_w: Width of the rendered (source) image.
        render_h: Height of the rendered (source) image.
        target_w: Width of the desired cropped image.
        target_h: Height of the desired cropped image.
        cx: Desired principal-point x in the cropped image.
        cy: Desired principal-point y in the cropped image.

    Returns:
        ``(crop_x, crop_y)`` — the top-left corner of the crop window in the
        source image.

    Raises:
        ValueError: If the resulting crop window does not fit inside the
            rendered image (the source render is too small, or ``cx``/``cy``
            place the window out of bounds). Failing loudly here avoids
            silently dropping every frame downstream.
    """
    crop_x = int(round(render_w / 2.0 - cx))
    crop_y = int(round(render_h / 2.0 - cy))
    if (crop_x < 0 or crop_y < 0
            or crop_x + target_w > render_w
            or crop_y + target_h > render_h):
        raise ValueError(
            "crop window out of bounds: render=%dx%d target=%dx%d "
            "crop=(%d,%d) right=%d bottom=%d — enlarge the render size "
            "or adjust cx/cy"
            % (render_w, render_h, target_w, target_h, crop_x, crop_y,
               crop_x + target_w, crop_y + target_h))
    return crop_x, crop_y


def crop_raw_image(data: bytes, height: int, width: int, step: int,
                   crop_x: int, crop_y: int,
                   target_w: int, target_h: int) -> Tuple[bytes, int]:
    """Crop a raw image byte buffer to a sub-rectangle, encoding-agnostic.

    Args:
        data: Raw image bytes, row-major, ``step`` bytes per row.
        height: Source image height in pixels.
        width: Source image width in pixels.
        step: Source row stride in bytes (``width * bytes_per_pixel``).
        crop_x: Left edge of the crop window (pixels).
        crop_y: Top edge of the crop window (pixels).
        target_w: Crop width (pixels).
        target_h: Crop height (pixels).

    Returns:
        ``(cropped_bytes, new_step)`` where ``new_step = target_w *
        bytes_per_pixel``.

    Raises:
        ValueError: If the crop window does not fit inside the source image.
    """
    bpp = step // width  # bytes per pixel
    if (crop_x < 0 or crop_y < 0
            or crop_x + target_w > width
            or crop_y + target_h > height):
        raise ValueError(
            "crop window (%d,%d)+%dx%d does not fit in %dx%d image"
            % (crop_x, crop_y, target_w, target_h, width, height))
    arr = np.frombuffer(data, dtype=np.uint8).reshape(height, step)
    crop = arr[crop_y:crop_y + target_h,
               crop_x * bpp:(crop_x + target_w) * bpp].copy()
    return crop.tobytes(), target_w * bpp
