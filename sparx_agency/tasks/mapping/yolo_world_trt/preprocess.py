"""Letterbox an arbitrary RGB frame into the engine's fixed ``HxW`` input.

A TensorRT (and DLA) engine has a *static* input shape, but the camera frame
(504x294 after the XTEND resize, or 720x420 raw later) will not match it exactly.
Letterboxing scales the frame by a single factor to fit inside ``(H, W)`` and pads
the remainder with a constant colour, preserving aspect ratio -- the standard
YOLO preprocessing. The returned :class:`LetterboxTransform` carries the scale and
padding so :mod:`postprocess` can map detections back to original pixels.

Pure numpy + OpenCV; no torch / tensorrt. Importable and unit-testable anywhere.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import cv2
import numpy as np


@dataclass(frozen=True)
class LetterboxTransform:
    """Maps original-image pixels <-> letterboxed engine-input pixels.

    Attributes:
        scale: single scale factor applied to the original frame.
        pad_x: left padding added after scaling (pixels).
        pad_y: top padding added after scaling (pixels).
        orig_w: original frame width.
        orig_h: original frame height.
    """

    scale: float
    pad_x: float
    pad_y: float
    orig_w: int
    orig_h: int

    def undo_xyxy(self, boxes: np.ndarray) -> np.ndarray:
        """Map ``[N,4]`` xyxy boxes from engine-input space to original pixels."""
        out = boxes.astype(np.float32).copy()
        out[:, [0, 2]] = (out[:, [0, 2]] - self.pad_x) / self.scale
        out[:, [1, 3]] = (out[:, [1, 3]] - self.pad_y) / self.scale
        np.clip(out[:, [0, 2]], 0, self.orig_w - 1, out=out[:, [0, 2]])
        np.clip(out[:, [1, 3]], 0, self.orig_h - 1, out=out[:, [1, 3]])
        return out


def letterbox(rgb: np.ndarray, imgsz: Tuple[int, int],
              pad_value: int = 114) -> Tuple[np.ndarray, LetterboxTransform]:
    """Resize+pad ``rgb`` (HxWx3) into ``(H, W)`` keeping aspect ratio.

    Args:
        rgb: input RGB frame, ``HxWx3`` uint8.
        imgsz: target ``(H, W)`` engine input (stride-32 multiples).
        pad_value: constant padding colour (114 is the YOLO default grey).

    Returns:
        ``(padded_rgb HxWx3 uint8, transform)``.
    """
    img = np.asarray(rgb)
    if img.ndim != 3 or img.shape[2] != 3:
        raise ValueError("letterbox expects HxWx3 RGB, got shape %s" % (img.shape,))
    oh, ow = int(img.shape[0]), int(img.shape[1])
    th, tw = int(imgsz[0]), int(imgsz[1])

    scale = min(tw / ow, th / oh)
    nw, nh = int(round(ow * scale)), int(round(oh * scale))
    interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
    resized = cv2.resize(img, (nw, nh), interpolation=interp)

    pad_x = (tw - nw) / 2.0
    pad_y = (th - nh) / 2.0
    top, bottom = int(round(pad_y - 0.1)), int(round(pad_y + 0.1))
    left, right = int(round(pad_x - 0.1)), int(round(pad_x + 0.1))
    canvas = cv2.copyMakeBorder(resized, top, bottom, left, right,
                                cv2.BORDER_CONSTANT, value=(pad_value,) * 3)
    # copyMakeBorder rounds each side independently; snap to the exact target.
    canvas = canvas[:th, :tw]
    if canvas.shape[:2] != (th, tw):
        fixed = np.full((th, tw, 3), pad_value, np.uint8)
        fixed[:canvas.shape[0], :canvas.shape[1]] = canvas
        canvas = fixed
    return canvas, LetterboxTransform(scale, float(left), float(top), ow, oh)


def to_engine_tensor(padded_rgb: np.ndarray) -> np.ndarray:
    """Convert a padded ``HxWx3`` uint8 RGB frame to ``[1,3,H,W]`` float32 in [0,1].

    Matches ultralytics' default preprocessing (``/255``, no mean/std, RGB, NCHW),
    which is what the exported YOLO-World graph expects.
    """
    chw = np.transpose(padded_rgb.astype(np.float32) / 255.0, (2, 0, 1))[None]
    return np.ascontiguousarray(chw)
