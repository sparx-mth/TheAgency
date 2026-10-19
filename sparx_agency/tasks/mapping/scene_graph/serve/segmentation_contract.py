"""Wire contract for the scene-graph segmentation HTTP server.

Importable by BOTH sides of the split-runtime boundary -- the torch venv that
serves Segment Anything and the Habitat / ROS2 client that has no torch -- so
it depends on nothing heavier than the stdlib, numpy and cv2. Keep it that way.

The protocol (see ``segmentation_server.py`` for the routes):
  * a request is one JSON object: the frame as the detector protocol's JPEG
    (:func:`~sparx_agency.tasks.mapping.scene_graph.serve.contract.encode_frame`),
    base64-encoded, plus the boxes to segment, ``{"image_jpeg_b64", "boxes"}``
    (:func:`segment_request` / :func:`parse_segment_request`);
  * a reply carries one mask per box IN BOX ORDER, each a lossless PNG of a
    0/255 uint8 image, base64-encoded (:func:`encode_mask` / :func:`decode_mask`),
    beside its pixel count, so the client can line masks up with its
    detections by position and never by matching boxes.

JPEG for the frame is deliberate: it is the SAME bytes the detector sees
(same codec, same quality), so the mask is drawn on the image the boxes were
drawn on. PNG for the masks is deliberate too: a mask is a decision, and a
lossy mask would move the object's edge by a pixel or two exactly where the
occluder meets it.

Deliberate near-duplication note (house rule): the JPEG frame codec is
:mod:`sparx_agency.tasks.mapping.scene_graph.serve.contract`'s, imported, not
copied. The PNG mask codec is this module's own: the detector contract carries
no masks and widening it would touch every detector client for a field none
of them can fill.
"""
from __future__ import annotations

import base64
from typing import Any, Dict, List, Sequence, Tuple

import cv2
import numpy as np

from sparx_agency.tasks.mapping.scene_graph.serve.contract import decode_frame, encode_frame

DEFAULT_SEGMENTATION_PORT = 8093
#: The one backend the service answers as; a client refuses any other on /health.
SEGMENTATION_BACKEND = "sam"

BoxXYXY = Tuple[float, float, float, float]


def encode_mask(mask: np.ndarray) -> str:
    """Encode a boolean ``HxW`` mask as a base64 PNG (0/255 uint8), losslessly.

    Raises:
        ValueError: If the mask is not two-dimensional or encoding fails.
    """
    array = np.asarray(mask)
    if array.ndim != 2:
        raise ValueError("encode_mask expects an HxW mask, got shape %s" % (array.shape,))
    image = (array.astype(bool).astype(np.uint8) * 255)
    ok, buf = cv2.imencode(".png", image, [int(cv2.IMWRITE_PNG_COMPRESSION), 3])
    if not ok:
        raise ValueError("cv2.imencode failed to produce a PNG mask")
    return base64.b64encode(buf.tobytes()).decode("ascii")


def decode_mask(data: str, shape: Tuple[int, int]) -> np.ndarray:
    """Decode a base64 PNG mask back to a boolean ``HxW`` array of ``shape``.

    Raises:
        ValueError: If the bytes do not decode to an image of ``shape``.
    """
    try:
        raw = base64.b64decode(data, validate=True)
    except (ValueError, TypeError) as exc:
        raise ValueError("decode_mask: not base64: %s" % exc)
    image = cv2.imdecode(np.frombuffer(raw, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError("decode_mask: bytes are not a decodable image")
    if tuple(image.shape) != tuple(int(v) for v in shape):
        raise ValueError("decode_mask: mask shape %s does not match the frame %s" % (image.shape, tuple(shape)))
    return image > 127


def segment_request(bgr: np.ndarray, boxes: Sequence[BoxXYXY]) -> Dict[str, Any]:
    """Build the ``POST /segment`` body for a BGR frame and its boxes."""
    return {"image_jpeg_b64": base64.b64encode(encode_frame(bgr)).decode("ascii"),
            "boxes": [[float(v) for v in box] for box in boxes]}


def parse_segment_request(body: Dict[str, Any]) -> Tuple[np.ndarray, bytes, List[List[float]]]:
    """Decode a request body to ``(bgr frame, jpeg bytes, boxes)``.

    Raises:
        ValueError: If the body is not the contract's object, the image does
            not decode, or a box is not four finite numbers inside the frame.
    """
    if not isinstance(body, dict):
        raise ValueError("segment request must be a JSON object")
    try:
        jpeg = base64.b64decode(body["image_jpeg_b64"], validate=True)
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("segment request needs base64 JPEG bytes under image_jpeg_b64: %s" % exc)
    bgr = decode_frame(jpeg)
    h, w = bgr.shape[:2]
    boxes = body.get("boxes")
    if not isinstance(boxes, list):
        raise ValueError("segment request needs a list of boxes under boxes")
    parsed: List[List[float]] = []
    for item in boxes:
        try:
            values = [float(v) for v in item]
        except (TypeError, ValueError) as exc:
            raise ValueError("bad box %r: %s" % (item, exc))
        if len(values) != 4 or not all(np.isfinite(values)):
            raise ValueError("a box must be four finite numbers, got %r" % (item,))
        x1, y1, x2, y2 = values
        if not (0 <= x1 < x2 <= w and 0 <= y1 < y2 <= h):
            raise ValueError("box %r lies outside the %dx%d frame or is empty" % (item, w, h))
        parsed.append(values)
    return bgr, jpeg, parsed


def masks_to_json(masks: Sequence[np.ndarray], boxes: Sequence[Sequence[float]]) -> List[Dict[str, Any]]:
    """The ``masks`` list of a reply: one entry per box, in box order."""
    if len(masks) != len(boxes):
        raise ValueError("%d masks for %d boxes" % (len(masks), len(boxes)))
    return [{"xyxy": [float(v) for v in box], "png_b64": encode_mask(mask), "area_px": int(np.count_nonzero(mask))}
            for mask, box in zip(masks, boxes)]


def masks_from_json(items: Sequence[Dict[str, Any]], shape: Tuple[int, int]) -> List[np.ndarray]:
    """Parse a reply's ``masks`` list back to boolean arrays, checking every pixel count.

    Raises:
        ValueError: If an item is malformed or its decoded mask disagrees
            with its own ``area_px``.
    """
    out: List[np.ndarray] = []
    for item in items:
        try:
            mask = decode_mask(item["png_b64"], shape)
            area = int(item["area_px"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("bad mask item: %s" % exc)
        if int(np.count_nonzero(mask)) != area:
            raise ValueError("mask pixel count %d disagrees with the reply's area_px %d" % (int(np.count_nonzero(mask)), area))
        out.append(mask)
    return out
