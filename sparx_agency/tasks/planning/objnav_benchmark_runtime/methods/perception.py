"""Detector provenance and coherent RGB-D object projection."""
from __future__ import annotations

import math
import numpy as np
import requests
from scipy import ndimage

from sparx_agency.core.mapping.objects.geometry import robust_bbox_depth
from sparx_agency.core.planning.objnav.camera_geometry import world_T_camera_optical
from sparx_agency.core.planning.objnav.errors import ObjNavInternalError
from sparx_agency.tasks.mapping.scene_graph.serve.contract import detections_from_json, encode_frame

STAIR_LABELS = frozenset(("stairs", "staircase"))


class HttpDetector:
    """Pin vocabulary/model/configuration without reconfiguring shared services."""
    def __init__(self, url, vocabulary, timeout_s=30.0, expected_backend=None):
        self.url = url.rstrip("/")
        self.vocabulary = tuple(vocabulary)
        self.timeout_s = timeout_s
        self.expected_backend = expected_backend
        self.session = requests.Session()
        self._identity = None
        self.last_detections = ()
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
        try:
            self.health()
            body = encode_frame(np.ascontiguousarray(rgb[..., ::-1]))
            response = self.session.post(self.url + "/detect", data=body,
                                         headers={"Content-Type": "image/jpeg"}, timeout=self.timeout_s)
            response.raise_for_status()
            data = response.json()
            if (data.get("h"), data.get("w")) != rgb.shape[:2]:
                raise ValueError("Detector boxes do not refer to the submitted frame")
            if tuple(data.get("classes", ())) != self.vocabulary or data.get("metadata") != self._identity["metadata"]:
                raise ValueError("Detector was reconfigured during inference")
            detections = detections_from_json(data["detections"])
            for detection in detections:
                if (detection.cls not in self.vocabulary or not math.isfinite(detection.conf)
                        or not 0 <= detection.conf <= 1 or not all(math.isfinite(v) for v in detection.xyxy)):
                    raise ValueError("Invalid detection or unexpected vocabulary")
            self.last_detections = tuple(detections)
            self.last_inference_ms = data.get("ms")
            self.last_peak_rss_mib = data.get("peak_rss_mib")
            return detections
        except (requests.RequestException, ValueError, KeyError) as exc:
            raise ObjNavInternalError("Detector service failed: %s" % exc) from exc


def observed_objects(observation, detections, confidence, diagnostics=None):
    """Yield coherent centroids, not percentile depths attached to centre rays."""
    camera = observation.camera
    k = camera.intrinsics
    transform = world_T_camera_optical(observation.pose, camera)
    for detection in detections:
        row = {"label": detection.cls, "confidence": float(detection.conf), "xyxy": list(detection.xyxy),
               "step": observation.step, "status": "low_confidence"}
        if diagnostics is not None:
            diagnostics.append(row)
        if detection.conf < confidence:
            continue
        x1, y1, x2, y2 = detection.xyxy
        row["status"] = "invalid_box"
        if not (0 <= x1 < x2 <= k.width and 0 <= y1 < y2 <= k.height):
            continue
        xyz = _supported_centroid(observation, detection.xyxy, transform, row)
        if xyz is not None:
            yield detection.cls, xyz


def _supported_centroid(observation, box, transform, row):
    """Measure connected support before downsampling; a stride can alias holes."""
    camera, k = observation.camera, observation.camera.intrinsics
    row["status"] = "insufficient_depth"
    depth = robust_bbox_depth(observation.depth_m, box, min_depth_m=camera.min_depth_m, max_depth_m=camera.max_depth_m)
    if depth is None:
        return None
    x1, y1, x2, y2 = box
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    left, right = int(cx - (x2 - x1) / 4), int(cx + (x2 - x1) / 4)
    top, bottom = int(cy - (y2 - y1) / 4), int(cy + (y2 - y1) / 4)
    stride = max(1, int(math.sqrt((right - left) * (bottom - top) / 900)))
    patch = observation.depth_m[top:bottom, left:right]
    valid = np.isfinite(patch) & (patch > camera.min_depth_m) & (patch < camera.max_depth_m)
    supported = valid & (np.abs(patch - depth) <= max(0.12, depth * 0.06))
    components, count = ndimage.label(supported)
    if not count:
        return None
    sizes = np.bincount(components.ravel())
    sizes[0] = 0
    component = components == int(np.argmax(sizes))
    fraction = float(component.sum()) / max(1, int(valid.sum()))
    row.update(supported_fraction=fraction, valid_pixels=int(valid.sum()), status="mixed_depth")
    if component.sum() < 8 or fraction < 0.35:
        return None
    v, u = np.nonzero(component[::stride, ::stride])
    if len(v) < 8:
        return None
    d = patch[v * stride, u * stride]
    optical = np.stack(((left + u * stride - k.cx) * d / k.fx, (top + v * stride - k.cy) * d / k.fy, d), axis=1)
    points = optical @ transform[:3, :3].T + transform[:3, 3]
    xyz = np.median(points, axis=0)
    row.update(status="projected", xyz=xyz.tolist(), height_range_m=[float(np.min(points[:, 2])), float(np.max(points[:, 2]))])
    return xyz
