"""The segmentation service's client: pinned identity, per-frame masks for the target's boxes.

The sibling of :class:`~sparx_agency.tasks.planning.objnav_benchmark_runtime.methods.perception.HttpDetector`
for the box-prompted Segment Anything service
(``tasks/mapping/scene_graph/serve/segmentation_server.py``), with the same
two failure families: identity drift (another checkpoint or configuration
answering mid-run) ends the run as an :class:`ObjNavInternalError` with no
transport cause; a transport or frame failure (timeout, dead service, a
malformed reply for one frame) is the same error class WITH a cause, which the
perception cycle turns into a frame without masks, counted and recorded.
"""
from __future__ import annotations

import base64
import hashlib
import math

import numpy as np
import requests

from sparx_agency.core.planning.objnav.errors import ObjNavInternalError
from sparx_agency.tasks.mapping.scene_graph.serve.segmentation_contract import (
    SEGMENTATION_BACKEND, masks_from_json, segment_request)


class SegmenterFrameError(ValueError):
    """One frame's reply was unusable (shape, hash, a malformed mask): the frame fails, the run goes on."""


class HttpSegmenter:
    """Pin the segmenter's checkpoint and configuration for the run; one POST per frame with target boxes."""

    def __init__(self, url, timeout_s=30.0, expected_model=None):
        self.url = url.rstrip("/")
        if not math.isfinite(timeout_s) or timeout_s <= 0:
            raise ValueError("Segmenter timeout must be finite and positive")
        self.timeout_s = float(timeout_s)
        self.expected_model = expected_model
        self.session = requests.Session()
        self._identity = None
        self.last_masks = ()
        self.last_inference_ms = None
        self.last_peak_rss_mib = None

    def health(self):
        response = self.session.get(self.url + "/health", timeout=self.timeout_s)
        response.raise_for_status()
        data = response.json()
        metadata = data.get("metadata", {}) if isinstance(data, dict) else {}
        if not data.get("ok") or metadata.get("backend") != SEGMENTATION_BACKEND:
            raise ObjNavInternalError("Segmenter service is not the box-prompted SAM service (backend %r)" % metadata.get("backend"))
        if not metadata.get("checkpoint_sha256") or not metadata.get("segmenter_config"):
            raise ObjNavInternalError("Segmenter lacks checkpoint/config provenance; restart the service from this checkout")
        if self.expected_model is not None and data.get("model") != self.expected_model:
            raise ObjNavInternalError("Segmenter serves %r, not the requested %r" % (data.get("model"), self.expected_model))
        identity = {key: data.get(key) for key in ("model", "device", "metadata")}
        if self._identity is not None and identity != self._identity:
            raise ObjNavInternalError("Segmenter model/config changed during the evaluation")
        self._identity = identity
        return identity

    def segment(self, rgb, boxes):
        """One boolean ``HxW`` mask per box, in box order; raises ``ObjNavInternalError`` on any failure."""
        self.last_masks = ()
        boxes = [tuple(float(v) for v in box) for box in boxes]
        if not boxes:
            return []
        try:
            if self._identity is None:
                self.health()
            body = segment_request(np.ascontiguousarray(rgb[..., ::-1]), boxes)
            response = self.session.post(self.url + "/segment", json=body, timeout=self.timeout_s)
            response.raise_for_status()
            data = response.json()
            if not isinstance(data, dict):
                raise SegmenterFrameError("Segmenter reply is not a JSON object")
            if data.get("metadata") != self._identity["metadata"] or data.get("model") != self._identity["model"]:
                raise ObjNavInternalError("Segmenter service failed: Segmenter was reconfigured during inference")
            if (data.get("h"), data.get("w")) != rgb.shape[:2]:
                raise SegmenterFrameError("Segmenter masks do not refer to the submitted frame")
            if data.get("request_sha256") != hashlib.sha256(base64.b64decode(body["image_jpeg_b64"])).hexdigest():
                raise SegmenterFrameError("Segmenter reply belongs to a different submitted frame")
            items = data.get("masks")
            if not isinstance(items, list) or len(items) != len(boxes):
                raise SegmenterFrameError("Segmenter returned %r masks for %d boxes" % (None if not isinstance(items, list) else len(items), len(boxes)))
            for item, box in zip(items, boxes):
                if tuple(float(v) for v in item.get("xyxy", ())) != box:
                    raise SegmenterFrameError("Segmenter masks are not in the order of the submitted boxes")
            try:
                masks = masks_from_json(items, rgb.shape[:2])
            except ValueError as exc:
                raise SegmenterFrameError("Malformed masks in the reply: %s" % exc) from exc
            self.last_masks = tuple(masks)
            self.last_inference_ms = data.get("ms")
            self.last_peak_rss_mib = data.get("peak_rss_mib")
            return masks
        except ObjNavInternalError:
            raise
        except (requests.RequestException, SegmenterFrameError, ValueError, KeyError, TypeError) as exc:
            raise ObjNavInternalError("Segmenter service failed: %s" % exc) from exc
