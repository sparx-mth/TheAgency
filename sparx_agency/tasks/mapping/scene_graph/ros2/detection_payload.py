"""Client-side handling of a ``/detect`` server reply.

Turns the detection server's HTTP reply into the ``/perception/detections``
topic payload, and renders the optional debug overlay. Split out of
``detector_client_node`` and kept **rclpy-free on purpose** so the topic
contract is unit-testable in the plain ``.venv`` without a sourced ROS
environment (see ``tests/test_detector_client_payload.py``).

The topic contract (fixed):
    ``{"stamp": <float sec of the SOURCE RGB header>, "w": int, "h": int,
    "ms": float, "detections": [{"cls", "conf", "xyxy"}]}``
"""
from __future__ import annotations

from typing import Any, Dict, Sequence

import cv2
import numpy as np

from sparx_agency.tasks.mapping.scene_graph.serve.contract import (
    detections_from_json,
    detections_to_json,
)

# Debug overlay style, matching the old SJTU ``yolo_detector.py`` viewer.
_BOX_COLOR_BGR = (0, 200, 0)


def build_detections_payload(stamp_sec: float,
                             reply: Dict[str, Any]) -> Dict[str, Any]:
    """Assemble the ``/perception/detections`` payload from a server reply.

    Args:
        stamp_sec: Header stamp of the SOURCE RGB image, in float seconds.
            The server's own timing never enters the stamp — downstream
            joins detections with depth/pose by this value.
        reply: Decoded JSON body of a successful ``POST /detect``
            (``{"w", "h", "ms", "detections": [...]}``).

    Returns:
        The topic payload dict, ready for ``json.dumps``.

    Raises:
        ValueError: The reply is not a dict, misses a field, or carries a
            malformed detection entry (loud failure per repo rule — a
            half-parsed detection frame must never reach the mapper).
    """
    if not isinstance(reply, dict):
        raise ValueError("detect reply is not a JSON object: %r" % (reply,))
    try:
        w = int(reply["w"])
        h = int(reply["h"])
        ms = float(reply["ms"])
        items = reply["detections"]
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("detect reply missing/bad field: %s (reply=%r)"
                         % (exc, reply))
    if not isinstance(items, list):
        raise ValueError("detect reply 'detections' is not a list: %r"
                         % (items,))
    # Round-trip through the wire dataclasses so every entry is validated
    # and re-serialized in canonical form (floats, 4-element boxes).
    wires = detections_from_json(items)
    return {
        "stamp": float(stamp_sec),
        "w": w,
        "h": h,
        "ms": ms,
        "detections": detections_to_json(wires),
    }


def draw_detections(bgr: np.ndarray,
                    detections: Sequence[Dict[str, Any]]) -> np.ndarray:
    """Return a copy of ``bgr`` with green boxes + ``cls conf`` labels drawn.

    Args:
        bgr: ``HxWx3`` uint8 BGR frame (not modified).
        detections: The ``detections`` list of a payload built by
            :func:`build_detections_payload`.

    Returns:
        A new ``HxWx3`` uint8 BGR frame with the overlay.
    """
    out = np.ascontiguousarray(bgr).copy()
    for det in detections:
        x1, y1, x2, y2 = (int(round(float(v))) for v in det["xyxy"])
        label = "%s %.2f" % (det["cls"], float(det["conf"]))
        cv2.rectangle(out, (x1, y1), (x2, y2), _BOX_COLOR_BGR, 2)
        cv2.putText(out, label, (x1, max(y1 - 5, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, _BOX_COLOR_BGR, 1,
                    cv2.LINE_AA)
    return out
