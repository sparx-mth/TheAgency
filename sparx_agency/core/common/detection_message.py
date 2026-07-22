"""Codec for the object-approach "detections" message.

The detector and the tracker/servo are deliberately decoupled by a plain
``std_msgs/String`` carrying JSON, so the heavy TensorRT/torch dependency lives in
one process and the lean control node in another -- possibly on the other side of a
ROS1<->ROS2 bridge, or on another host entirely. This module is the single
definition of that wire format, shared by every producer and consumer::

    {"stamp": 1780843795.329, "w": 504, "h": 294,
     "detections": [{"label": "refrigerator", "score": 0.83,
                     "bbox": [x1, y1, x2, y2]}, ...]}

``stamp`` is the SOURCE FRAME's capture stamp (not the publish time), so a consumer
can seed its tracker on the matching buffered frame; ``w``/``h`` are that frame's
size, which the consumer needs to normalise the box without assuming its own
intrinsics describe the detector's input.

Labels are normalised (stripped, lower-cased) on parse, so a consumer's label
matching never depends on how a particular detector cased its prompts.

This module is deliberately ROS-free and Python 3.8 compatible so it can be shared
by the ROS1 adapter nodes (which import ``core`` under Python 3.8) and the ROS2
sidecar alike.
"""

import json
from typing import Iterable, List, NamedTuple, Optional

from sparx_agency.core.common.types.perception import Detection2D


class ParsedDetections(NamedTuple):
    """A parsed detections message.

    Attributes:
        stamp: Capture timestamp of the source frame, floating-point seconds.
        width: Source frame width in pixels.
        height: Source frame height in pixels.
        detections: The detections, labels normalised to lower-case.
    """

    stamp: float
    width: int
    height: int
    detections: List[Detection2D]


def encode_detections(detections: Iterable[Detection2D], stamp: float,
                      width: int, height: int) -> str:
    """Serialize detections to the JSON wire format.

    Args:
        detections: Detections from one frame.
        stamp: The source frame's capture stamp, in seconds.
        width: Source frame width in pixels.
        height: Source frame height in pixels.

    Returns:
        The JSON payload for a ``std_msgs/String``.
    """
    payload = {
        "stamp": float(stamp),
        "w": int(width),
        "h": int(height),
        "detections": [
            {"label": str(d.label), "score": float(d.score),
             "bbox": [int(v) for v in d.bbox_xyxy]}
            for d in detections
        ],
    }
    return json.dumps(payload)


def parse_detections_message(data: str, default_width: Optional[int] = None,
                             default_height: Optional[int] = None,
                             default_stamp: Optional[float] = None
                             ) -> ParsedDetections:
    """Parse a detections JSON message.

    Args:
        data: The raw string payload of the message.
        default_width: Frame width to assume when the payload omits ``w``.
        default_height: Frame height to assume when the payload omits ``h``.
        default_stamp: Capture stamp to assume when the payload omits ``stamp``
            (e.g. the consumer's latest frame stamp).

    Returns:
        The parsed stamp, frame size and detections.

    Raises:
        ValueError: If the payload is not a JSON object, ``detections`` is not a
            list, a detection is missing a field or carries a malformed ``bbox``,
            or a needed field is absent with no default supplied.
    """
    try:
        payload = json.loads(data)
    except ValueError as e:
        raise ValueError("detections message is not valid JSON: %s" % e)
    if not isinstance(payload, dict):
        raise ValueError("detections message must be a JSON object, got %r"
                         % type(payload).__name__)

    width = _require(payload, "w", default_width, int)
    height = _require(payload, "h", default_height, int)
    stamp = _require(payload, "stamp", default_stamp, float)

    raw = payload.get("detections", [])
    if not isinstance(raw, list):
        raise ValueError("'detections' must be a list, got %r" % type(raw).__name__)

    detections = []
    for i, d in enumerate(raw):
        if not isinstance(d, dict):
            raise ValueError("detections[%d] must be an object" % i)
        try:
            bbox = tuple(int(v) for v in d["bbox"])
            det = Detection2D(label=str(d["label"]).strip().lower(),
                              score=float(d["score"]), bbox_xyxy=bbox,
                              frame_w=width, frame_h=height)
        except KeyError as e:
            raise ValueError("detections[%d] missing field %s" % (i, e))
        except (TypeError, ValueError):
            raise ValueError("detections[%d] has a malformed bbox/score: %r" % (i, d))
        if len(bbox) != 4:
            raise ValueError("detections[%d] bbox must have 4 values, got %d"
                             % (i, len(bbox)))
        detections.append(det)

    return ParsedDetections(stamp=stamp, width=width, height=height,
                            detections=detections)


def _require(payload, key, default, cast):
    """Read ``key`` from ``payload``, falling back to ``default``; raise if neither."""
    if key in payload:
        try:
            return cast(payload[key])
        except (TypeError, ValueError):
            raise ValueError("detections message field %r is malformed: %r"
                             % (key, payload[key]))
    if default is None:
        raise ValueError("detections message is missing %r and no default was given"
                         % key)
    return cast(default)
