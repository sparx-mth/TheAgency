"""Turn a recorded frame into exactly the tensors NavDP was pretrained on.

Getting this wrong is silent: the network still runs, the loss still falls, and
the policy is simply worse than it should be. So this module is a single
implementation, shared by the feature cache, the training dataset and the
evaluation inference, and it mirrors ``NavDP_Agent.process_image`` /
``process_depth`` from the upstream repo step for step:

    keep-aspect resize so the long side is 224 -> centre-pad to a square
    -> resize to exactly 224x224 -> divide by 255
    (depth additionally: zero everything outside [0.1, 5.0] m)

**Colour order.** The upstream NavDP server, every upstream baseline server, and
this repo's own TensorRT server all do ``cvtColor(RGB2BGR)`` *before*
``process_image`` -- so the array that reaches the encoder is **BGR**, and the
ImageNet mean/std inside ``EncoderWrapper`` are applied positionally to BGR
channels. That is an upstream quirk, but it is what the pretrained weights
learned and what the deployed server serves, so it is the default here.

Note this differs from ``..verify/navdp_infer.preprocess_rgb``, which converts to
RGB and is therefore one channel swap away from the deployed stack. Fine-tuning
under a different colour order than deployment would train the network to undo a
swap that inference does not apply. ``color_order`` exists so the discrepancy can
be measured rather than argued about.

**The 5 m depth ceiling** is not a detail either: NavDP zeroes anything beyond
it, so in an office corridor most of the far wall reads as "no measurement".
That is by design -- the depth image solves the next few metres, and the goal
token, which comes from the map, carries everything beyond.

numpy + cv2 only; no torch, so the feature cache and the dataset agree by
construction.
"""
from __future__ import annotations

from typing import Tuple

import numpy as np

IMAGE_SIZE = 224
DEPTH_MIN_M = 0.1
DEPTH_MAX_M = 5.0
COLOR_ORDERS = ("bgr", "rgb")


def resize_pad(array: np.ndarray, size: int = IMAGE_SIZE) -> np.ndarray:
    """Keep-aspect resize to fit ``size``, centre-pad to a square, resize to exact.

    The final resize is a no-op for most inputs and is kept because upstream
    keeps it: an odd-sized pad leaves the square one pixel short, and letting
    that through would change the patch grid.
    """
    import cv2

    proportion = size / max(array.shape[0], array.shape[1])
    resized = cv2.resize(array, (-1, -1), fx=proportion, fy=proportion)
    pad_w = max((size - resized.shape[1]) // 2, 0)
    pad_h = max((size - resized.shape[0]) // 2, 0)
    pad = (((pad_h, pad_h), (pad_w, pad_w), (0, 0)) if resized.ndim == 3
           else ((pad_h, pad_h), (pad_w, pad_w)))
    return cv2.resize(np.pad(resized, pad, mode="constant", constant_values=0),
                      (size, size))


def preprocess_rgb(rgb: np.ndarray, color_order: str = "bgr",
                   size: int = IMAGE_SIZE) -> np.ndarray:
    """``(H, W, 3)`` uint8 **RGB** -> ``(3, size, size)`` float32 in [0, 1].

    Args:
        rgb: Colour frame in RGB order, as ``FlightRecording.rgb`` returns it.
        color_order: Channel order handed to the encoder. ``"bgr"`` (default)
            matches the deployed server; ``"rgb"`` is the alternative.
        size: Square side.

    Returns:
        CHW float32, ready to stack into the 8-frame memory.
    """
    if color_order not in COLOR_ORDERS:
        raise ValueError(f"color_order must be one of {COLOR_ORDERS}, got {color_order!r}")
    frame = rgb[:, :, ::-1] if color_order == "bgr" else rgb
    scaled = resize_pad(np.ascontiguousarray(frame), size).astype(np.float32) / 255.0
    return np.ascontiguousarray(scaled.transpose(2, 0, 1))


def preprocess_depth(depth_m: np.ndarray, size: int = IMAGE_SIZE,
                     depth_min_m: float = DEPTH_MIN_M,
                     depth_max_m: float = DEPTH_MAX_M) -> np.ndarray:
    """``(H, W)`` metric depth -> ``(1, size, size)`` float32, out-of-range zeroed."""
    depth = np.asarray(depth_m, dtype=np.float32).copy()
    depth[~np.isfinite(depth)] = 0.0
    depth = resize_pad(depth, size)
    depth[(depth > depth_max_m) | (depth < depth_min_m)] = 0.0
    return depth[None, :, :]


def memory_indices(frame: int, memory_size: int = 8,
                   stride: int = 1) -> Tuple[Tuple[int, ...], Tuple[bool, ...]]:
    """The frame indices making up NavDP's memory, oldest first.

    NavDP's agent fills its queue from the left with zeros until enough frames
    have arrived, so the newest frame is always last. Reproducing that here --
    rather than tiling the current frame eight times, as the earlier pixel-goal
    dataset did -- is what lets the policy actually use motion.

    Args:
        frame: Index of the current frame (the newest).
        memory_size: Queue length (NavDP: 8).
        stride: Frames skipped between memory slots. 1 = consecutive.

    Returns:
        ``(indices, valid)`` -- indices clamped to 0, and a mask that is False
        where the slot falls before the start of the recording and must be
        zero-filled.
    """
    raw = [frame - (memory_size - 1 - i) * stride for i in range(memory_size)]
    return tuple(max(0, i) for i in raw), tuple(i >= 0 for i in raw)
