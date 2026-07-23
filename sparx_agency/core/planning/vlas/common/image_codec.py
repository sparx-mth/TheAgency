"""PNG encoding for the VLA HTTP wire format (ROS-free, numpy-only at import).

Every policy server in this stack takes its observation as multipart PNG files,
so the encode step was written once per client and then copied. It lives here
once instead.

``PIL`` is imported lazily inside the functions so importing this module stays
numpy-only -- the FALCON Noetic adapter imports ``core`` under Python 3.8 and
must not pull heavy deps at import.

Depth encoding (do NOT widen without matching the server)
---------------------------------------------------------
Depth is clipped to ``depth_max_m`` then scaled by :data:`DEPTH_SCALE` (10000)
into uint16. uint16 caps at 65535, so ``65535 / 10000 = 6.5535`` m is the hard
ceiling -- a 7 m pixel would overflow and read back as ~0.45 m, a phantom wall
right where the operator clicked the far floor. NavDP's own ``process_depth``
zeroes depth beyond 5 m anyway, so 5 m is the honest default cap. Widen only by
also widening the encoding (uint32, or a smaller scale) on BOTH sides.

Python 3.8 compatible.
"""
from __future__ import annotations

import io

import numpy as np

#: depth_m -> uint16 multiplier; the server divides by the same value.
DEPTH_SCALE = 10000.0

#: Hard ceiling implied by the uint16 encoding, in metres.
MAX_DEPTH_M = 65535.0 / DEPTH_SCALE


def check_depth_cap(depth_max_m):
    """Raise if ``depth_max_m`` would overflow the uint16 depth encoding.

    Args:
        depth_max_m: the clip applied before encoding, in metres.

    Returns:
        ``float(depth_max_m)``, unchanged, so this can wrap an assignment.

    Raises:
        ValueError: ``depth_max_m`` exceeds :data:`MAX_DEPTH_M`. Failing loud
            here beats silently wrapping a far pixel into a near phantom wall.
    """
    value = float(depth_max_m)
    if value > MAX_DEPTH_M:
        raise ValueError(
            "depth_max_m=%.3f exceeds the uint16 encoding ceiling %.4f m "
            "(DEPTH_SCALE=%g); lower it or widen the encoding on both sides."
            % (value, MAX_DEPTH_M, DEPTH_SCALE))
    return value


def rgb_to_png(rgb):
    """HxWx3 uint8 RGB array -> a seek-0 PNG :class:`io.BytesIO` buffer.

    Args:
        rgb: HxWx3 array in **RGB** channel order (not OpenCV's BGR).
    """
    from PIL import Image as PILImage

    buf = io.BytesIO()
    PILImage.fromarray(np.ascontiguousarray(rgb, dtype=np.uint8), "RGB").save(
        buf, format="PNG")
    buf.seek(0)
    return buf


def depth_to_png(depth_m, depth_max_m):
    """HxW metric depth -> a seek-0 uint16 (``I;16``) PNG buffer.

    Args:
        depth_m: HxW float array of metric depth in metres; NaN/0 allowed.
        depth_max_m: clip applied before scaling (see the module docstring).
    """
    from PIL import Image as PILImage

    scaled = (np.clip(depth_m, 0.0, float(depth_max_m)) * DEPTH_SCALE).astype(np.uint16)
    buf = io.BytesIO()
    PILImage.fromarray(scaled, mode="I;16").save(buf, format="PNG")
    buf.seek(0)
    return buf


def png_to_rgb(data):
    """PNG bytes -> HxWx3 uint8 RGB array.

    Used to read an image back off a policy server (e.g. FlowNav's ``/get_goal``,
    where the goal frame lives server-side and the node has no local copy).
    """
    from PIL import Image as PILImage

    return np.asarray(PILImage.open(io.BytesIO(data)).convert("RGB"))
