"""Wire contract for the scene-graph detection HTTP server.

Importable by BOTH sides of the split-runtime boundary — the conda (torch/GPU)
detection server and the ROS2 (.venv, no torch) client node — so it depends on
nothing heavier than the stdlib, numpy and cv2. Keep it that way: one heavy
import here and the ROS2 side can no longer speak the protocol.

The protocol (see ``detection_server.py`` for the routes):
  * frames travel as raw JPEG bytes (``encode_frame`` / ``decode_frame``);
  * detections travel as JSON objects ``{"cls", "conf", "xyxy"}``
    (``detections_to_json`` / ``detections_from_json`` around
    :class:`DetectionWire`).

Deliberate near-duplication note (house rule): ``cv2.imencode`` also appears in
``tasks/planning/object_approach_webcam/webcam_frame_publisher.py`` — that is an
atomic *file* writer with a caller-chosen quality, not a wire codec; sharing it
would couple the wire contract to an unrelated task, so this module keeps its own
two-line codec with the quality fixed by the protocol.

Deliberate duplication note (house rule), the important one: this is a SECOND
detections wire format. :mod:`sparx_agency.core.common.detection_message` calls
itself "the single definition of that wire format" and it is — of the
**object-approach / XTEND** format, ``{"label", "score", "bbox"}`` inside a
``std_msgs/String``, spoken by the ROS1 adapter nodes and their ROS2 sidecar.
This stack speaks ``{"cls", "conf", "xyxy"}`` over HTTP instead, on purpose and
for two reasons. First, it is a verbatim port of sjtu_project's
``yolo_detector.py`` — the same port the vocabulary above came from — so a run
recorded against the original stack stays replayable here without a translation
step nobody would maintain. Second, the reply envelope carries ``ms``, the
detector's own inference time, which the mission uses to reason about detector
latency and for which the core format has no slot; widening core's schema for
one consumer would push a field on every ROS1 adapter that cannot fill it. The
two formats therefore never meet on a wire — nothing bridges one into the other
— and neither should be "fixed" to match the other without moving both stacks
together.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Sequence, Tuple

import cv2
import numpy as np

DEFAULT_PORT = 8092
JPEG_QUALITY = 90

# Ported VERBATIM from sjtu_project
# perception_docker/semantic_mapper/semantic_mapper/yolo_detector.py
# (DEFAULT_VOCABULARY). Small, focused vocabulary: YOLO-World is more accurate
# with fewer, well-separated prompts. Add/remove per scene as needed.
DEFAULT_HOSPITAL_VOCABULARY = [
    "person",

    "chair",
    "office chair",
    "wheelchair",
    "sofa",

    "table",
    "desk",
    "bedside table",
    "cabinet",
    "drawer",
    "shelf",
    "cart",

    "toilet",
    "sink",
    "shower",

    "tv",
    "refrigerator",
    "trash can",
    "vending machine",

    "hospital bed",
    "medical trolley",
    "surgical trolley",
    "instrument cart",
    "anesthesia machine",
    "x-ray machine",
    "iv stand",
    "blood pressure monitor",
]


@dataclass(frozen=True)
class DetectionWire:
    """One detection as it travels over the wire.

    Attributes:
        cls: Class label (the open-vocabulary prompt that matched).
        conf: Detector confidence in ``[0, 1]``.
        xyxy: Pixel bounding box ``(x1, y1, x2, y2)``, origin top-left.
    """

    cls: str
    conf: float
    xyxy: Tuple[float, float, float, float]


def encode_frame(bgr: np.ndarray) -> bytes:
    """Encode a BGR uint8 frame to JPEG bytes at the protocol quality.

    Args:
        bgr: ``HxWx3`` uint8 image in OpenCV BGR channel order.

    Returns:
        JPEG-compressed bytes suitable for a ``POST /detect`` body.

    Raises:
        ValueError: If the input is not ``HxWx3`` uint8 or encoding fails.
    """
    img = np.asarray(bgr)
    if img.ndim != 3 or img.shape[2] != 3 or img.dtype != np.uint8:
        raise ValueError(
            "encode_frame expects HxWx3 uint8 BGR, got shape %s dtype %s"
            % (img.shape, img.dtype)
        )
    ok, buf = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY])
    if not ok:
        raise ValueError("cv2.imencode failed to produce a JPEG")
    return buf.tobytes()


def decode_frame(data: bytes) -> np.ndarray:
    """Decode JPEG bytes back to a BGR uint8 frame.

    Args:
        data: JPEG bytes as produced by :func:`encode_frame`.

    Returns:
        ``HxWx3`` uint8 BGR image.

    Raises:
        ValueError: If the bytes do not decode to a color image.
    """
    buf = np.frombuffer(data, dtype=np.uint8)
    img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("decode_frame: bytes are not a decodable image")
    return img


def detections_to_json(detections: Sequence[DetectionWire]) -> List[Dict[str, Any]]:
    """Convert detections to JSON-ready dicts (the ``detections`` list on the wire).

    The result embeds directly into a larger response object and round-trips
    exactly through :func:`detections_from_json`.
    """
    return [
        {
            "cls": str(d.cls),
            "conf": float(d.conf),
            "xyxy": [float(v) for v in d.xyxy],
        }
        for d in detections
    ]


def detections_from_json(items: Sequence[Dict[str, Any]]) -> List[DetectionWire]:
    """Parse the ``detections`` list of a server response back to dataclasses.

    Args:
        items: The decoded JSON list (each item ``{"cls", "conf", "xyxy"}``).

    Returns:
        One :class:`DetectionWire` per item.

    Raises:
        ValueError: If an item is missing a field or its box is not 4 numbers.
    """
    out: List[DetectionWire] = []
    for item in items:
        try:
            xyxy = tuple(float(v) for v in item["xyxy"])
            if len(xyxy) != 4:
                raise ValueError("xyxy must have 4 elements, got %d" % len(xyxy))
            out.append(
                DetectionWire(cls=str(item["cls"]), conf=float(item["conf"]), xyxy=xyxy)
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("bad detection item %r: %s" % (item, exc))
    return out
