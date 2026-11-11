"""``GroundTruthDetector`` and ``GroundTruthSegmenter``: the two perception contracts, answered from the semantic sensor.

Both implement the method's own seams
(:class:`~sparx_agency.core.planning.object_search.interfaces.Detector` and
:class:`~sparx_agency.core.planning.object_search.interfaces.Segmenter`) and
offer the same ``health()`` identity the HTTP clients do, so
``policy_factory.build_policy`` records them the way it records a service
and the perception cycle, the recorder and the dashboard read them unchanged.

What "ideal" means here, exactly:

* every annotated instance visible in the frame whose annotation maps into
  the method's vocabulary is reported, once, with its tight pixel box and
  confidence ``1.0`` -- precision and recall against the annotations are
  100 % by construction, and the box has no drift;
* an instance covering fewer than ``min_pixels`` pixels is *not* reported.
  A detector has a resolution floor and the method downstream (the
  projection's eight-pixel support, the landmark association) was built for
  boxes of that order; a three-pixel sliver glimpsed through a crack is not
  what "ideal perception" means. The floor is declared in the identity and
  the frozen configuration;
* the segmenter's mask for a box is the instance's own pixels -- the exact
  instance mask, nothing of what occludes it -- so the depth read on it is
  the object's;
* with ``on_object_m`` set, a pixel is the instance's only where the depth puts
  it ON the instance -- inside the annotation's own box, inflated by that
  much. The semantic mesh is registered to the render mesh by the publisher's
  config, but not everywhere: on MP3D's Z6MFQCViBuw the sensor painted goal
  chairs, curtains and vases across the wall of a staircase, 3-7 m from where
  those objects stand (2026-11-10, the single-storey baseline: 20 of 49
  failures STOPped at such phantoms, 8-13 m from every goal). An instance
  whose pixels mostly lie off its box is not reported; one partly off keeps
  its on-box pixels alone. ON by default (``DEFAULT_ON_OBJECT_M``, 0.25 m)
  since 2026-11-10; ``--ground-truth-on-object-m 0`` restores the raw sensor.
  The rule is the sensor's, not a dataset's: it reads what the adapter
  publishes -- the frame's depth, pose and camera, the scene's boxes -- and
  every Habitat adapter publishes them (``habitat.simulator.semantic_boxes_enu``).
  Where it has nothing to judge an instance against it reports the instance
  as rendered and counts it (``unjudged`` in the frame diagnostics), so a
  hand-off that is missing is visible in the record rather than inferred
  from a score.

What stays the method's: the alias folding and duplicate suppression
(``deduplicate_detections``), the confidence gates, the door finder, the
projection, the landmark map and everything after. The oracle changes the
input to the pipeline, not the pipeline.
"""
from __future__ import annotations

import math
import time
from typing import Dict, List, Sequence

import numpy as np

from sparx_agency.core.common.math.bbox import BBox
from sparx_agency.core.planning.object_search.interfaces import Detection
from sparx_agency.core.planning.objnav.errors import ObjNavInternalError
from sparx_agency.tasks.planning.objnav_benchmark_runtime.ground_truth.semantic_frame import (
    InstanceExtent, instance_extents, instance_mask)
from sparx_agency.tasks.planning.objnav_benchmark_runtime.ground_truth.source import GroundTruthSource

#: The ``metadata["backend"]`` both identities carry; what a run directory says its perception was.
GROUND_TRUTH_BACKEND = "ground_truth"
#: Default resolution floor: an 8x8 patch at 640x480.
DEFAULT_MIN_PIXELS = 64
#: The on-object filter's default tolerance: metres the annotation's box is inflated by before a rendered pixel's
#: depth is judged against it. ON by default since 2026-11-10 (the user's decision: the oracle is a stand-in for a real
#: detector, and a real detector does not see chairs through walls); 0 turns it off.
DEFAULT_ON_OBJECT_M = 0.25
#: Pixels with a depth reading an instance needs before its placement can be judged at all.
ON_OBJECT_MIN_READINGS = 8
#: The share of an instance's depth-read pixels that must lie on its box for the instance to be reported.
ON_OBJECT_MIN_FRACTION = 0.5


class GroundTruthDetector:
    """Every visible annotated instance in the method's vocabulary, as a confidence-1.0 detection with its tight box.

    Args:
        source: The :class:`GroundTruthSource` the environment publishes frames to.
        min_pixels: Instances covering fewer pixels are not reported (see the module docstring).
        on_object_m: The on-object filter's tolerance: a pixel counts as the instance's only where the
            frame's depth places it inside the instance's annotated box inflated by this much
            (``DEFAULT_ON_OBJECT_M``); 0 reports every pixel the sensor paints, as before 2026-11-10.
    """

    def __init__(self, source: GroundTruthSource, min_pixels: int = DEFAULT_MIN_PIXELS,
                 on_object_m: float = DEFAULT_ON_OBJECT_M):
        if type(min_pixels) is not int or min_pixels < 1:
            raise ValueError("min_pixels must be a positive integer, got %r" % (min_pixels,))
        if isinstance(on_object_m, bool) or not math.isfinite(float(on_object_m)) or float(on_object_m) < 0.0:
            raise ValueError("on_object_m must be finite and non-negative (0 disables the filter), got %r" % (on_object_m,))
        self.source = source
        self.vocabulary = tuple(source.vocabulary)
        self.min_pixels = min_pixels
        self.on_object_m = float(on_object_m)
        #: ``instance -> HxW bool`` of the last frame's on-box pixels, for the segmenter (filter on only).
        self.last_consistent: Dict[int, np.ndarray] = {}
        self.off_object_instances = 0
        self.trimmed_instances = 0
        #: Instances reported as rendered because the filter, though on, had nothing to judge them against: a frame
        #: published without depth, pose or camera, or an instance the scene began without a box for. Non-zero on
        #: every frame of a scene means the adapter is not making the hand-off (``scene_report["instances_with_boxes"]``).
        self.unjudged_instances = 0
        self.timeout_s = math.inf                   # read by the factory's record; there is no transport
        self.last_detections: Sequence[Detection] = ()
        self.last_diagnostics: Dict[str, object] = {}
        self.last_inference_ms = None
        self.last_peak_rss_mib = None
        #: ``xyxy -> instance id`` of the last frame, for the segmenter's exact box lookup.
        self.last_instances_by_box: Dict[BBox, int] = {}
        self._last_ids = None

    def health(self) -> Dict[str, object]:
        """The identity the factory pins and records, shaped like the detection service's."""
        return {"model": "habitat-sim semantic sensor (%s)" % self.source.categories.annotations,
                "device": "simulator", "classes": list(self.vocabulary),
                "metadata": {"backend": GROUND_TRUTH_BACKEND,
                             "checkpoint_sha256": self.source.categories.sha256(),
                             "detector_config": {"conf_thresh": 0.0, "min_pixels": self.min_pixels,
                                                 "on_object_m": self.on_object_m,
                                                 "label_table": self.source.label_mapper.name}}}

    def detect(self, rgb: np.ndarray) -> List[Detection]:
        """Detections for one ``HxWx3`` RGB frame -- the simulator's current one, or the run ends."""
        started = time.monotonic()
        self.last_detections, self.last_diagnostics, self.last_instances_by_box = (), {}, {}
        self.last_consistent = {}
        ids = self.source.lookup(rgb)
        extents = instance_extents(ids, min_pixels=1)
        labels = self.source.labels
        points = self._world_points(ids) if self.on_object_m > 0.0 else None
        detections, by_box, below_floor, off_object, trimmed, unjudged = [], {}, 0, 0, 0, 0
        for instance, extent in extents.items():
            label = labels.get(instance)
            if label is None:
                continue
            if extent.pixels < self.min_pixels:
                below_floor += 1
                continue
            if points is not None and instance in self.source.boxes:
                kept = self._on_object(ids, instance, extent, points)
                if kept is None:
                    off_object += 1
                    continue
                if kept is not extent:
                    trimmed += 1
                    if kept.pixels < self.min_pixels:
                        below_floor += 1
                        continue
                extent = kept
            elif self.on_object_m > 0.0:
                unjudged += 1       # the filter is on, but this frame has no depth / pose / camera or this instance no box
            detections.append(Detection(label, 1.0, extent.xyxy))
            by_box[extent.xyxy] = instance
        self._last_ids = ids
        self.last_instances_by_box = by_box
        self.last_detections = tuple(detections)
        self.off_object_instances += off_object
        self.trimmed_instances += trimmed
        self.unjudged_instances += unjudged
        self.last_inference_ms = (time.monotonic() - started) * 1000.0
        self.last_diagnostics = {"mode": GROUND_TRUTH_BACKEND, "visible_instances": len(extents),
                                 "reported": len(detections), "below_min_pixels": below_floor,
                                 "off_object": off_object, "trimmed_to_object": trimmed, "unjudged": unjudged}
        return detections

    def _world_points(self, ids):
        """The frame's pixels as world points, ``HxWx3`` (NaN where the depth is unusable), or None without a depth frame."""
        depth, pose, camera = self.source.depth, self.source.pose, self.source.camera
        if depth is None or pose is None or camera is None:
            return None
        from sparx_agency.core.planning.objnav.camera_geometry import world_T_camera_optical
        depth = np.asarray(depth, dtype=np.float64)
        if depth.shape != ids.shape:
            return None
        k = camera.intrinsics
        valid = np.isfinite(depth) & (depth > camera.min_depth_m) & (depth < camera.max_depth_m)
        v, u = np.indices(depth.shape)
        z = np.where(valid, depth, np.nan)
        optical = np.stack(((u - k.cx) * z / k.fx, (v - k.cy) * z / k.fy, z), axis=-1)
        transform = world_T_camera_optical(pose, camera)
        return optical @ transform[:3, :3].T + transform[:3, 3]

    def _on_object(self, ids, instance, extent, points):
        """The instance's extent over the pixels whose depth puts them on its box; None when most lie off it.

        Fewer than :data:`ON_OBJECT_MIN_READINGS` depth-read pixels (the
        instance beyond the depth range, or a few pixels of it) cannot be
        judged and keep the extent as rendered: the method's projection
        refuses such a box on its own.
        """
        mask = ids == instance
        pts = points[mask]
        readable = np.isfinite(pts[:, 0])
        if int(readable.sum()) < ON_OBJECT_MIN_READINGS:
            return extent
        center, half = self.source.boxes[instance]
        lo = np.asarray(center) - np.asarray(half) - self.on_object_m
        hi = np.asarray(center) + np.asarray(half) + self.on_object_m
        inside = readable & np.all((pts >= lo) & (pts <= hi), axis=1)
        fraction = float(inside.sum()) / float(readable.sum())
        if fraction < ON_OBJECT_MIN_FRACTION:
            return None
        if bool(inside.all()):
            return extent
        consistent = np.zeros(ids.shape, dtype=bool)
        rows, cols = np.nonzero(mask)
        # Pixels without a depth reading stay with the instance when most readable ones are on it.
        keep = inside | ~readable
        consistent[rows[keep], cols[keep]] = True
        self.last_consistent[instance] = consistent
        rows, cols = rows[keep], cols[keep]
        return InstanceExtent(int(instance), (float(cols.min()), float(rows.min()), float(cols.max() + 1),
                                              float(rows.max() + 1)), int(keep.sum()))


class GroundTruthSegmenter:
    """The exact instance mask for each prompted box, in box order.

    Args:
        detector: The :class:`GroundTruthDetector` whose boxes are prompted; a box
            it emitted maps straight to its instance. A box it did not emit (none
            today: the perception cycle prompts the detector's own boxes) is
            resolved to the reported instance with the most pixels inside it, or
            an empty mask when there is none.
    """

    def __init__(self, detector: GroundTruthDetector):
        self.detector = detector
        self.timeout_s = math.inf
        self.last_masks: Sequence[np.ndarray] = ()
        self.last_inference_ms = None
        self.last_peak_rss_mib = None

    def health(self) -> Dict[str, object]:
        """The identity the factory records, shaped like the segmentation service's."""
        return {"model": "habitat-sim semantic sensor (instance masks)", "device": "simulator",
                "metadata": {"backend": GROUND_TRUTH_BACKEND,
                             "checkpoint_sha256": self.detector.source.categories.sha256(),
                             "segmenter_config": {"mask": "exact instance pixels", "min_pixels": self.detector.min_pixels,
                                                  "on_object_m": self.detector.on_object_m}}}

    def segment(self, rgb: np.ndarray, boxes_xyxy: Sequence[BBox]) -> List[np.ndarray]:
        started = time.monotonic()
        self.last_masks = ()
        boxes = [tuple(float(v) for v in box) for box in boxes_xyxy]
        if not boxes:
            return []
        ids = self.detector.source.lookup(rgb)
        if self.detector._last_ids is not ids:
            raise ObjNavInternalError("Ground-truth segmenter was prompted for a frame the detector has not read")
        masks = []
        for box in boxes:
            instance = self._instance_for(ids, box)
            if instance is None:
                masks.append(np.zeros(ids.shape, dtype=bool))
            elif instance in self.detector.last_consistent:
                masks.append(self.detector.last_consistent[instance])      # the on-box pixels alone (the filter)
            else:
                masks.append(instance_mask(ids, instance))
        self.last_masks = tuple(masks)
        self.last_inference_ms = (time.monotonic() - started) * 1000.0
        return masks

    def _instance_for(self, ids, box):
        instance = self.detector.last_instances_by_box.get(box)
        if instance is not None:
            return instance
        x1, y1, x2, y2 = (int(round(v)) for v in box)
        patch = ids[max(0, y1):max(0, y2), max(0, x1):max(0, x2)]
        if patch.size == 0:
            return None
        present, counts = np.unique(patch, return_counts=True)
        best = None
        for instance, pixels in sorted(zip(present.tolist(), counts.tolist()), key=lambda item: -item[1]):
            if instance in self.detector.source.labels:
                best = int(instance)
                break
        return best
