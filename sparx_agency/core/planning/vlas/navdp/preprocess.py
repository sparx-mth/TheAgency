"""A frame -> exactly the tensors NavDP was pretrained on.

This mirrors ``NavDP_Agent.process_image`` / ``process_depth`` from the upstream
repo, step for step::

    keep-aspect resize so the long side is 224 -> centre-pad to a square
    -> resize to exactly 224x224 -> divide by 255
    (depth additionally: zero everything outside [0.1, 5.0] m)

Getting it wrong is silent. The network still runs, the loss still falls, and
the policy is simply worse than it should be -- which is why there is one
implementation, here, next to the runtime every caller drives, rather than a
copy per tool.

**Colour order.** The upstream NavDP server, every upstream baseline server, and
this repo's own TensorRT server all apply ``cvtColor(RGB2BGR)`` *before*
``process_image`` (see ``tasks/planning/vlas/navdp/serve/navdp_trt_server.py``),
so the array that reaches the encoder is **BGR** and the ImageNet mean/std inside
``EncoderWrapper`` are applied positionally to BGR channels. That is an upstream
quirk, but it is what the pretrained weights learned and what the deployed server
serves, so :data:`ENCODER_COLOR_ORDER` is the default. ``input_order`` says what
you *have*; ``encoder_order`` says what the model gets. Fine-tuning or verifying
under a different ``encoder_order`` than deployment trains or measures the network
against a channel swap inference does not apply -- the parameter exists so that
discrepancy can be measured rather than argued about.

**The 5 m depth ceiling** is not a detail either: NavDP zeroes anything beyond
it, so in an office corridor most of the far wall reads as "no measurement".
That is by design -- the depth image solves the next few metres, and the goal
token carries everything beyond.

**Layout** is the caller's, not the model's: the TensorRT policy takes the RGB
memory as ``(B, T, H, W, 3)`` and depth as ``(B, H, W, 1)``, while the torch
fine-tune stacks ``(3, H, W)`` / ``(1, H, W)``. Both come out of here rather than
out of a transpose bolted on at each call site.

``cv2`` is imported inside the functions, not at module scope: ``core`` must stay
importable in the FALCON Noetic container, which has neither this module's
callers nor a modern OpenCV. Numpy-1.17 API only, Python 3.8 clean.
"""
from __future__ import annotations

import numpy as np

from sparx_agency.core.planning.vlas.navdp.errors import NavDPError

#: NavDP's square model input side.
IMAGE_SIZE = 224
#: Depth outside this band is zeroed -- "no measurement", not "far away".
DEPTH_MIN_M = 0.1
DEPTH_MAX_M = 5.0
#: Accepted channel orders.
COLOR_ORDERS = ("bgr", "rgb")
#: Accepted memory layouts. ``hwc`` feeds the TRT policy, ``chw`` the torch stack.
LAYOUTS = ("hwc", "chw")
#: What the deployed server hands the encoder. See the module docstring.
ENCODER_COLOR_ORDER = "bgr"


def _check(value, allowed, what):
    """Raise unless ``value`` is one of ``allowed``."""
    if value not in allowed:
        raise NavDPError("%s must be one of %s, got %r" % (what, allowed, value))
    return value


def resize_pad(array, size=IMAGE_SIZE):
    """Keep-aspect resize to fit ``size``, centre-pad to a square, resize to exact.

    The final resize is a no-op for most inputs and is kept because upstream
    keeps it: an odd-sized pad leaves the square one pixel short, and letting
    that through would change the patch grid.

    Args:
        array: ``(H, W)`` or ``(H, W, C)`` image. Any dtype ``cv2.resize`` takes.
        size: Square side in pixels.

    Returns:
        ``(size, size)`` or ``(size, size, C)``, same dtype as ``array``.
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


def preprocess_rgb(image, input_order="rgb", encoder_order=ENCODER_COLOR_ORDER,
                   size=IMAGE_SIZE, layout="chw"):
    """``(H, W, 3)`` uint8 colour frame -> float32 in [0, 1], model-ready.

    Args:
        image: ``(H, W, 3)`` uint8 frame whose channels are in ``input_order``.
        input_order: Channel order of ``image`` -- ``"rgb"`` (e.g.
            ``FlightRecording.rgb``) or ``"bgr"`` (e.g. ``cv2.imread``).
        encoder_order: Channel order to hand the encoder. Defaults to
            :data:`ENCODER_COLOR_ORDER`, which is what the deployed server does;
            pass the other one only to measure the difference.
        size: Square side.
        layout: ``"chw"`` -> ``(3, size, size)``; ``"hwc"`` -> ``(size, size, 3)``.

    Returns:
        float32 in [0, 1], contiguous, in the requested layout.

    Raises:
        NavDPError: an unknown colour order or layout.
    """
    _check(input_order, COLOR_ORDERS, "input_order")
    _check(encoder_order, COLOR_ORDERS, "encoder_order")
    _check(layout, LAYOUTS, "layout")

    frame = image[:, :, ::-1] if input_order != encoder_order else image
    scaled = resize_pad(np.ascontiguousarray(frame), size).astype(np.float32) / 255.0
    if layout == "chw":
        return np.ascontiguousarray(scaled.transpose(2, 0, 1))
    return np.ascontiguousarray(scaled)


def preprocess_depth(depth_m, size=IMAGE_SIZE, depth_min_m=DEPTH_MIN_M,
                     depth_max_m=DEPTH_MAX_M, layout="chw"):
    """``(H, W)`` metric depth -> float32, out-of-range zeroed, model-ready.

    Non-finite samples are zeroed *before* the resize, so a NaN cannot bleed
    into its neighbours through the interpolation.

    Args:
        depth_m: ``(H, W)`` depth in metres.
        size: Square side.
        depth_min_m: Below this reads as no measurement.
        depth_max_m: Above this reads as no measurement (NavDP's 5 m ceiling).
        layout: ``"chw"`` -> ``(1, size, size)``; ``"hwc"`` -> ``(size, size, 1)``.

    Returns:
        float32 in the requested layout, zeroed outside the band.

    Raises:
        NavDPError: an unknown layout.
    """
    _check(layout, LAYOUTS, "layout")

    depth = np.asarray(depth_m, dtype=np.float32).copy()
    depth[~np.isfinite(depth)] = 0.0
    depth = resize_pad(depth, size)
    depth[(depth > depth_max_m) | (depth < depth_min_m)] = 0.0
    if layout == "chw":
        return np.ascontiguousarray(depth[None, :, :])
    return np.ascontiguousarray(depth[:, :, None])
