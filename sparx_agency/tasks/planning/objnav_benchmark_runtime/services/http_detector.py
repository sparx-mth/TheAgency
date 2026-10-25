"""The detection service's client: pinned identity, one POST per frame, boxes as core :class:`Detection` records.

Implements the :class:`~sparx_agency.core.planning.object_search.interfaces.Detector`
contract over the scene-graph detection server
(``tasks/mapping/scene_graph/serve/detection_server.py``). Two failure
families, as the contract states: identity drift (another vocabulary, model
or configuration answering mid-run) is an
:class:`~sparx_agency.core.planning.objnav.errors.ObjNavInternalError` with
no cause and ends the run; a transport or frame failure (timeout, dead
service, a malformed reply for one frame) is the same error class raised
``from`` its cause, which the perception cycle turns into a back-off.
"""
from __future__ import annotations

import hashlib
import math

import numpy as np
import requests

from sparx_agency.core.planning.object_search.interfaces import DetectorFrameError, detection_from_wire
from sparx_agency.core.planning.objnav.errors import ObjNavInternalError
from sparx_agency.tasks.mapping.scene_graph.serve.contract import detections_from_json, encode_frame


class HttpDetector:
    """Pin vocabulary/model/configuration without reconfiguring shared services."""
    def __init__(self, url, vocabulary, timeout_s=30.0, expected_backend=None):
        self.url = url.rstrip("/")
        self.vocabulary = tuple(vocabulary)
        if not math.isfinite(timeout_s) or timeout_s <= 0:
            raise ValueError("Detector timeout must be finite and positive")
        self.timeout_s = float(timeout_s)
        self.expected_backend = expected_backend
        self.session = requests.Session()
        self._identity = None
        self.last_detections = ()
        self.last_diagnostics = {}
        self.last_inference_ms = None
        self.last_peak_rss_mib = None

    def health(self):
        response = self.session.get(self.url + "/health", timeout=self.timeout_s)
        response.raise_for_status()
        data = response.json()
        if not data.get("ok") or tuple(data.get("classes", ())) != self.vocabulary:
            raise ObjNavInternalError("Detector vocabulary mismatch; configure a dedicated service with the benchmark's --print-vocabulary output, in that exact order")
        metadata = data.get("metadata", {})
        if self.expected_backend is not None and metadata.get("backend") != self.expected_backend:
            raise ObjNavInternalError("Detector backend differs from the explicitly requested backend")
        if not metadata.get("checkpoint_sha256") or not metadata.get("detector_config"):
            raise ObjNavInternalError("Detector lacks checkpoint/config provenance; restart the dedicated service from this checkout")
        identity = {key: data.get(key) for key in ("model", "device", "classes", "metadata")}
        if self._identity is not None and identity != self._identity:
            raise ObjNavInternalError("Detector model/config changed during the evaluation")
        self._identity = identity
        return identity

    def detect(self, rgb):
        self.last_detections, self.last_diagnostics = (), {}
        try:
            if self._identity is None:
                # One identity check; every /detect reply echoes classes and metadata,
                # which are checked below, so a GET per frame bought nothing.
                self.health()
            body = encode_frame(np.ascontiguousarray(rgb[..., ::-1]))
            response = self.session.post(self.url + "/detect", data=body,
                                         headers={"Content-Type": "image/jpeg"}, timeout=self.timeout_s)
            response.raise_for_status()
            data = response.json()
            if not isinstance(data, dict):
                raise DetectorFrameError("Detector reply is not a JSON object")
            if tuple(data.get("classes", ())) != self.vocabulary or data.get("metadata") != self._identity["metadata"]:
                # Identity drift: another vocabulary, model or configuration is answering.
                raise ObjNavInternalError("Detector service failed: Detector was reconfigured during inference "
                                          "(vocabulary or model configuration changed)")
            if (data.get("h"), data.get("w")) != rgb.shape[:2]:
                raise DetectorFrameError("Detector boxes do not refer to the submitted frame")
            request_hash = data.get("request_sha256")
            verified = self._identity["metadata"].get("backend") in ("grounded_vlm", "hybrid")
            if (verified or request_hash is not None) and request_hash != hashlib.sha256(body).hexdigest():
                raise DetectorFrameError("Detector verification belongs to a different submitted frame")
            diagnostics = data.get("diagnostics", {})
            if not isinstance(diagnostics, dict) or (verified and diagnostics.get("mode") != self._identity["metadata"]["backend"]):
                raise DetectorFrameError("Invalid detector verification diagnostics")
            try:
                detections = [detection_from_wire(record) for record in detections_from_json(data["detections"])]
            except (KeyError, TypeError, ValueError) as exc:
                raise DetectorFrameError("Malformed detections in the reply: %s" % exc) from exc
            for detection in detections:
                if (detection.cls not in self.vocabulary or not math.isfinite(detection.conf)
                        or not 0 <= detection.conf <= 1 or not all(math.isfinite(v) for v in detection.xyxy)):
                    raise DetectorFrameError("Invalid detection or unexpected vocabulary")
            self.last_detections = tuple(detections)
            self.last_diagnostics = dict(diagnostics, request_sha256=request_hash) if diagnostics else {}
            self.last_inference_ms = data.get("ms")
            self.last_peak_rss_mib = data.get("peak_rss_mib")
            return detections
        except ObjNavInternalError:
            raise
        except (requests.RequestException, DetectorFrameError, ValueError, KeyError) as exc:
            raise ObjNavInternalError("Detector service failed: %s" % exc) from exc

