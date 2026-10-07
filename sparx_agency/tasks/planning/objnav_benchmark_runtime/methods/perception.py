"""Detector provenance and coherent RGB-D object projection."""
from __future__ import annotations

import hashlib
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
            request_hash = data.get("request_sha256")
            verified = self._identity["metadata"].get("backend") in ("grounded_vlm", "hybrid")
            if (verified or request_hash is not None) and request_hash != hashlib.sha256(body).hexdigest():
                raise ValueError("Detector verification belongs to a different submitted frame")
            diagnostics = data.get("diagnostics", {})
            if not isinstance(diagnostics, dict) or (verified and diagnostics.get("mode") != self._identity["metadata"]["backend"]):
                raise ValueError("Invalid detector verification diagnostics")
            detections = detections_from_json(data["detections"])
            for detection in detections:
                if (detection.cls not in self.vocabulary or not math.isfinite(detection.conf)
                        or not 0 <= detection.conf <= 1 or not all(math.isfinite(v) for v in detection.xyxy)):
                    raise ValueError("Invalid detection or unexpected vocabulary")
            self.last_detections = tuple(detections)
            self.last_diagnostics = dict(diagnostics, request_sha256=request_hash) if diagnostics else {}
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
    # The range to the object's NEAR surface, not to its middle: a 10th-percentile
    # horizontal distance of the supported points, robust to a few stray pixels.
    # The benchmark's success radius is measured to the object, so the terminal
    # test of the target closing reads this beside the centroid.
    pose = observation.pose
    near = float(np.percentile(np.hypot(points[:, 0] - pose.x, points[:, 1] - pose.y), 10))
    row.update(status="projected", xyz=xyz.tolist(), height_range_m=[float(np.min(points[:, 2])), float(np.max(points[:, 2]))],
               radius_m=footprint_radius_m(box, float(depth), k.fx), range_near_m=near)
    return xyz


def clipped_box(box, intrinsics, margin_px, min_fraction=0.3, floor_cut_min_fraction=0.15):
    """Whether ``box`` is a SLIVER at an image edge: touching it within ``margin_px`` AND narrow.

    The one test behind both target gates -- the takeover's start and the
    legacy target evidence -- so a sliver the one refuses the other cannot
    pursue. A clipped box's depth centroid is the centroid of what is in
    frame, not of the object, and the clipped end of a bed reads as a sofa
    (the Ranchester bed: 84 x 210 px on the left edge of a 640 x 480 frame).
    A box that touches an edge but spans at least ``min_fraction`` of the
    frame in both dimensions is the opposite case -- a big object close up,
    filling the view (the Ranchester couch from 0.7 m: 392 x 379 px) -- and
    is not clipped in this sense: refusing it meant the takeover never
    started from beside the couch.

    **The bottom edge alone is the camera's floor cutoff, not a sliver**
    (since 2026-10-07). A level camera 0.88 m up sees the floor from about
    1.4 m out, so a LOW object -- a toilet, a chair seat -- at 1-2 m is cut
    by the bottom edge on every frame while standing whole in the frame
    laterally; its depth centroid and near edge are on the object. The
    Allensville toilet run refused twenty-three such frames of a 0.97
    toilet at 1.2-1.7 m, never counted two in a row and released the
    candidate twice. A box that touches ONLY the bottom edge, whose top lies
    in the lower half of the frame and which spans at least
    ``floor_cut_min_fraction`` of the frame in BOTH dimensions (the toilet:
    113 x 187 px of 640 x 480; a 60 px sliver is still a sliver), is not
    clipped; a box also touching a side edge, or the top, is judged as before.
    """
    m = int(margin_px)
    x1, y1, x2, y2 = box
    lateral = bool(x1 < m or x2 > intrinsics.width - m)
    top = bool(y1 < m)
    bottom = bool(y2 > intrinsics.height - m)
    if not (lateral or top or bottom):
        return False
    narrow = (x2 - x1) < min_fraction * intrinsics.width or (y2 - y1) < min_fraction * intrinsics.height
    if not narrow:
        return False
    floor_cut = (bottom and not lateral and not top and y1 >= intrinsics.cy
                 and (y2 - y1) >= floor_cut_min_fraction * intrinsics.height
                 and (x2 - x1) >= floor_cut_min_fraction * intrinsics.width)
    return not floor_cut


def footprint_radius_m(box, depth_m, fx, floor_m=0.15, ceiling_m=1.5):
    """Half the box's width at its depth, as the object's footprint half-extent.

    A disc, not a box: the detector's box is an image-plane extent and the
    agent sees the object from one side, so the plan-view shape is unknown;
    the half-width at the measured depth is the one footprint figure the
    frame supports. Clamped so a sliver and a wall-to-wall box both stay
    inside the range of indoor furniture.
    """
    x1, _, x2, _ = box
    radius = 0.5 * max(0.0, float(x2) - float(x1)) * float(depth_m) / float(fx)
    return float(min(ceiling_m, max(floor_m, radius)))

