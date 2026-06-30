"""RGB image -> FlowNav model tensors (parity with ``transform_images``).

FlowNav's reference ``deployment/src/utils.py::transform_images`` does, per frame:
``PIL.resize(image_size)`` -> ``ToTensor`` (``/255``, HWC->CHW) -> ``Normalize``
with ImageNet mean/std, then concatenates the ``context_size+1`` context frames
on the channel axis. This module reproduces that exactly in numpy/PIL so the host
server can build the encoder inputs the same way the model was trained.

The observation stack is ``(1, 3*(context_size+1), H, W)`` with the **oldest**
frame first and the **current** frame last (the depth-prior branch reads the last
3 channels). The goal is a single ``(1, 3, H, W)`` frame.
"""
from __future__ import annotations

import numpy as np

# ImageNet normalization (matches transforms.Normalize in transform_images).
_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def _size_wh(image_size):
    """Normalize an int or (w, h) image_size to a (width, height) tuple."""
    if isinstance(image_size, (tuple, list)):
        return int(image_size[0]), int(image_size[1])
    return int(image_size), int(image_size)


def preprocess_frame(rgb_uint8, image_size):
    """One RGB uint8 frame -> normalized ``(3, H, W)`` float32 (CHW).

    Args:
        rgb_uint8: HxWx3 uint8 image in RGB order.
        image_size: target size as an int (square) or ``(width, height)``.

    Returns:
        ``(3, H, W)`` float32, resized + ``/255`` + ImageNet-normalized.
    """
    from PIL import Image as PILImage

    w, h = _size_wh(image_size)
    pil = PILImage.fromarray(np.ascontiguousarray(rgb_uint8, dtype=np.uint8), "RGB")
    pil = pil.resize((w, h))                          # transform_images: default resample
    arr = np.asarray(pil, dtype=np.float32) / 255.0   # HWC in [0,1]
    arr = (arr - _MEAN) / _STD                        # ImageNet normalize (broadcast)
    return np.ascontiguousarray(arr.transpose(2, 0, 1), dtype=np.float32)  # CHW


def build_obs_stack(frames, image_size, context_plus_one):
    """Stack the context frames into the encoder's ``obs_img`` input.

    Args:
        frames: list of HxWx3 uint8 RGB frames, oldest first, newest last
            (length >= 1).
        image_size: target size (int or ``(w, h)``).
        context_plus_one: number of frames in the stack (``context_size + 1``).

    Returns:
        ``(1, 3*context_plus_one, H, W)`` float32. If fewer than
        ``context_plus_one`` frames are available, the oldest is repeated at the
        front (left-pad), so the current frame always stays last.
    """
    if not frames:
        raise ValueError("build_obs_stack needs at least one frame")
    fr = list(frames)
    while len(fr) < context_plus_one:
        fr.insert(0, fr[0])
    fr = fr[-context_plus_one:]
    chw = [preprocess_frame(f, image_size) for f in fr]
    stack = np.concatenate(chw, axis=0)               # (3*(ctx+1), H, W)
    return stack[None]                                # (1, ...)


def build_goal(goal_rgb_uint8, image_size):
    """One goal RGB frame -> the encoder's ``goal_img`` input ``(1, 3, H, W)``."""
    return preprocess_frame(goal_rgb_uint8, image_size)[None]
