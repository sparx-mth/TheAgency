"""Turn a recorded frame into exactly the tensors NavDP was pretrained on.

The implementation lives in
:mod:`sparx_agency.core.planning.vlas.navdp.preprocess`, next to the runtime it
feeds, because the fine-tune datasets, the feature cache, the evaluation
inference and the click-to-verify tool must all agree with the deployed server
and with each other. This module is the world-goal pipeline's view of it: the
frames arrive as **RGB** (that is what :meth:`FlightRecording.rgb` returns) and
the torch stack wants **CHW**, so those two are fixed here and only
``color_order`` -- the order handed to the encoder -- stays a knob.

See the core module for the colour-order quirk (the deployed server feeds BGR),
the 5 m depth ceiling, and why both matter more than they look.

numpy + cv2 only; no torch, so the feature cache and the dataset agree by
construction.
"""
from __future__ import annotations

from typing import Tuple

import numpy as np

from sparx_agency.core.planning.vlas.navdp.preprocess import (
    COLOR_ORDERS,
    DEPTH_MAX_M,
    DEPTH_MIN_M,
    ENCODER_COLOR_ORDER,
    IMAGE_SIZE,
    resize_pad,
)
from sparx_agency.core.planning.vlas.navdp import preprocess as _core

__all__ = ["COLOR_ORDERS", "DEPTH_MAX_M", "DEPTH_MIN_M", "ENCODER_COLOR_ORDER",
           "IMAGE_SIZE", "resize_pad", "preprocess_rgb", "preprocess_depth",
           "memory_indices"]


def preprocess_rgb(rgb: np.ndarray, color_order: str = ENCODER_COLOR_ORDER,
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
    return _core.preprocess_rgb(rgb, input_order="rgb", encoder_order=color_order,
                                size=size, layout="chw")


def preprocess_depth(depth_m: np.ndarray, size: int = IMAGE_SIZE,
                     depth_min_m: float = DEPTH_MIN_M,
                     depth_max_m: float = DEPTH_MAX_M) -> np.ndarray:
    """``(H, W)`` metric depth -> ``(1, size, size)`` float32, out-of-range zeroed."""
    return _core.preprocess_depth(depth_m, size=size, depth_min_m=depth_min_m,
                                  depth_max_m=depth_max_m, layout="chw")


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
