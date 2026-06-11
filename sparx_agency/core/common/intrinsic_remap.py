"""Camera-intrinsic image resample (pure numpy, ROS-free).

Resamples an image rendered by a *source* pinhole camera so it matches a
*target* pinhole camera's intrinsics -- both the focal lengths and the
principal point. This generalises a principal-point crop (see
``principal_point_crop``): a crop can only relocate the optical axis, never
change the focal length, so a square-pixel renderer such as Gazebo
(``fx == fy`` always) can never reproduce a target camera whose ``fx != fy``.
Sampling the source pixel that lies on each target pixel's ray lifts that
restriction and reproduces an anisotropic target exactly.

The mapping is *separable*. For a target pixel ``(u, v)`` the back-projected
ray has normalised coordinates ``((u - cx_t)/fx_t, (v - cy_t)/fy_t)``; the
source pixel on that same ray is::

    u_s = (fx_s / fx_t) * (u - cx_t) + cx_s
    v_s = (fy_s / fy_t) * (v - cy_t) + cy_s

so the column map depends only on ``u`` and the row map only on ``v``. We
round to the NEAREST source pixel -- no interpolation. Nearest-neighbour is
the right choice for metric depth: blending across a depth discontinuity
would invent surfaces at intermediate ranges that do not exist.

Like the crop, the resample is encoding-agnostic: it gathers whole pixels
from the raw byte buffer (``step`` bytes per row, ``step // width`` bytes per
pixel), so it works for ``rgb8``, ``bgr8``, ``mono8``, ``32FC1``, ``16UC1`` --
anything -- without ever decoding the image.
"""
from __future__ import annotations

from typing import Tuple

import numpy as np


def build_remap(src_fx: float, src_fy: float, src_cx: float, src_cy: float,
                src_w: int, src_h: int,
                dst_fx: float, dst_fy: float, dst_cx: float, dst_cy: float,
                dst_w: int, dst_h: int) -> Tuple[np.ndarray, np.ndarray]:
    """Nearest-neighbour source-pixel indices for every target pixel.

    Args:
        src_fx: Source (rendered) focal length x, pixels.
        src_fy: Source focal length y, pixels.
        src_cx: Source principal point x, pixels.
        src_cy: Source principal point y, pixels.
        src_w: Source image width, pixels.
        src_h: Source image height, pixels.
        dst_fx: Target focal length x, pixels.
        dst_fy: Target focal length y, pixels.
        dst_cx: Target principal point x, pixels.
        dst_cy: Target principal point y, pixels.
        dst_w: Target image width, pixels.
        dst_h: Target image height, pixels.

    Returns:
        ``(row_idx, col_idx)`` integer arrays of length ``dst_h`` and
        ``dst_w``. Target pixel ``(u, v)`` samples source pixel
        ``(row_idx[v], col_idx[u])``.

    Raises:
        ValueError: If any target ray falls outside the source image -- the
            source field of view is too narrow, or its resolution too small,
            to cover the target camera. Failing loudly here avoids silently
            clamping to the edge (which would fabricate geometry). The message
            reports the offending source span so the caller can widen the
            render ``horizontal_fov`` or enlarge the render resolution.
    """
    u = np.arange(dst_w, dtype=np.float64)
    v = np.arange(dst_h, dtype=np.float64)
    col = np.rint((src_fx / dst_fx) * (u - dst_cx) + src_cx).astype(np.int64)
    row = np.rint((src_fy / dst_fy) * (v - dst_cy) + src_cy).astype(np.int64)
    if int(col.min()) < 0 or int(col.max()) >= src_w:
        raise ValueError(
            "target columns map to source x in [%d, %d], outside [0, %d): the "
            "render horizontal_fov is too narrow (or the render width too "
            "small) to cover the target camera"
            % (int(col.min()), int(col.max()), src_w))
    if int(row.min()) < 0 or int(row.max()) >= src_h:
        raise ValueError(
            "target rows map to source y in [%d, %d], outside [0, %d): the "
            "render vertical FOV is too narrow (or the render height too "
            "small) to cover the target camera"
            % (int(row.min()), int(row.max()), src_h))
    return row, col


def remap_raw_image(data: bytes, src_h: int, src_w: int, src_step: int,
                    row_idx: np.ndarray,
                    col_idx: np.ndarray) -> Tuple[bytes, int]:
    """Gather target pixels from a raw image byte buffer, encoding-agnostic.

    Args:
        data: Raw source image bytes, row-major, ``src_step`` bytes per row.
        src_h: Source image height, pixels.
        src_w: Source image width, pixels.
        src_step: Source row stride in bytes (``src_w * bytes_per_pixel``).
        row_idx: Source row indices from :func:`build_remap` (length ``dst_h``).
        col_idx: Source column indices from :func:`build_remap` (length
            ``dst_w``).

    Returns:
        ``(out_bytes, out_step)`` for the ``dst_w x dst_h`` target image, where
        ``out_step = dst_w * bytes_per_pixel``.
    """
    bpp = src_step // src_w  # bytes per pixel
    arr = np.frombuffer(data, dtype=np.uint8).reshape(src_h, src_step)
    # Drop any row padding, then view as (h, w, bpp) whole pixels.
    arr = arr[:, :src_w * bpp].reshape(src_h, src_w, bpp)
    out = arr[row_idx[:, None], col_idx[None, :], :]  # (dst_h, dst_w, bpp)
    return np.ascontiguousarray(out).tobytes(), int(col_idx.shape[0]) * bpp
