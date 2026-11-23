"""One visual question about one frontier, not about the room or the camera heading."""
from __future__ import annotations

import math
from typing import Sequence

import cv2
import numpy as np

from sparx_agency.core.common.math.bbox import BBox, clip_xyxy
from sparx_agency.core.planning.object_search.frame_log.codec import FrameCodec
from sparx_agency.core.planning.object_search.interfaces import VisualReasoner

PROMPT_VERSION = "boxed-frontier-v2"
SYSTEM = ('The pictures show the same boundary of explored floor in an indoor object search. '
          'Each red box marks that boundary, not an object detection. Use the visible surroundings to '
          'estimate the chance of finding the requested object by exploring beyond that boundary. '
          'RGB can see farther than depth: consider distant objects and visible space on the bearing '
          'through the box, not evidence behind a different opening. The box is not a distance to an object. '
          'Return only JSON: {"probability": <number between 0 and 1>}.')
MAX_VIEW_SIDE = 640
WIRE_CODEC = FrameCodec(jpeg_quality=85)


def marked_view(rgb: np.ndarray, window: BBox) -> np.ndarray:
    """Draw the projected window in source pixels before scaling. Never mark the source frame."""
    h, w = rgb.shape[:2]
    x0, y0, x1, y1 = clip_xyxy(window, w - 1, h - 1)
    marked = rgb.copy()
    cv2.rectangle(marked, (math.floor(x0), math.floor(y0)), (math.ceil(x1), math.ceil(y1)),
                  (255, 0, 0), max(2, math.ceil(max(h, w) / 250)))
    if max(h, w) > MAX_VIEW_SIDE:
        scale = MAX_VIEW_SIDE / max(h, w)
        marked = cv2.resize(marked, (max(1, round(w * scale)), max(1, round(h * scale))),
                            interpolation=cv2.INTER_AREA)
    return marked


def ask_frontier(client: VisualReasoner, target: str, marked: Sequence[np.ndarray], *, depth_limit_m=None) -> float:
    """One to three same-camera views in one JPEG; works even with single-image model servers."""
    if not 1 <= len(marked) <= 3:
        raise ValueError("a frontier request needs one to three marked views")
    sheet = marked[0] if len(marked) == 1 else np.concatenate(marked, axis=1)
    query = "Target object: %s" % target
    if depth_limit_m is not None and math.isfinite(depth_limit_m):
        query += "\nMetric depth ends at %g m; the pictures are not clipped there." % depth_limit_m
    reply = client.chat_json(SYSTEM, query, image_jpeg=WIRE_CODEC.encode(sheet))
    return frontier_probability(reply)


def frontier_probability(reply) -> float:
    """Validate the model's answer before using or caching it."""
    value = reply.get("probability") if isinstance(reply, dict) else None
    if type(value) not in (int, float) or not 0 <= value <= 1:
        raise ValueError("frontier VLM must return a numeric probability in [0, 1], got %r" % (reply,))
    return float(value)
