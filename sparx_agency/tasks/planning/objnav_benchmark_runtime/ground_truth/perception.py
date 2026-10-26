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
  the object's.

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
from sparx_agency.tasks.planning.objnav_benchmark_runtime.ground_truth.hm3d_categories import table_sha256
from sparx_agency.tasks.planning.objnav_benchmark_runtime.ground_truth.semantic_frame import (
    instance_extents, instance_mask)
from sparx_agency.tasks.planning.objnav_benchmark_runtime.ground_truth.source import GroundTruthSource

#: The ``metadata["backend"]`` both identities carry; what a run directory says its perception was.
GROUND_TRUTH_BACKEND = "ground_truth"
#: Default resolution floor: an 8x8 patch at 640x480.
DEFAULT_MIN_PIXELS = 64


class GroundTruthDetector:
    """Every visible annotated instance in the method's vocabulary, as a confidence-1.0 detection with its tight box.

    Args:
        source: The :class:`GroundTruthSource` the environment publishes frames to.
        min_pixels: Instances covering fewer pixels are not reported (see the module docstring).
    """

    def __init__(self, source: GroundTruthSource, min_pixels: int = DEFAULT_MIN_PIXELS):
        if type(min_pixels) is not int or min_pixels < 1:
            raise ValueError("min_pixels must be a positive integer, got %r" % (min_pixels,))
        self.source = source
        self.vocabulary = tuple(source.vocabulary)
        self.min_pixels = min_pixels
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
        return {"model": "habitat-sim semantic sensor (HM3D-Semantics annotations)",
                "device": "simulator", "classes": list(self.vocabulary),
                "metadata": {"backend": GROUND_TRUTH_BACKEND,
                             "checkpoint_sha256": table_sha256(),
                             "detector_config": {"conf_thresh": 0.0, "min_pixels": self.min_pixels,
                                                 "label_table": self.source.label_mapper.name}}}

    def detect(self, rgb: np.ndarray) -> List[Detection]:
        """Detections for one ``HxWx3`` RGB frame -- the simulator's current one, or the run ends."""
        started = time.monotonic()
        self.last_detections, self.last_diagnostics, self.last_instances_by_box = (), {}, {}
        ids = self.source.lookup(rgb)
        extents = instance_extents(ids, min_pixels=1)
        labels = self.source.labels
        detections, by_box, below_floor = [], {}, 0
        for instance, extent in extents.items():
            label = labels.get(instance)
            if label is None:
                continue
            if extent.pixels < self.min_pixels:
                below_floor += 1
                continue
            detections.append(Detection(label, 1.0, extent.xyxy))
            by_box[extent.xyxy] = instance
        self._last_ids = ids
        self.last_instances_by_box = by_box
        self.last_detections = tuple(detections)
        self.last_inference_ms = (time.monotonic() - started) * 1000.0
        self.last_diagnostics = {"mode": GROUND_TRUTH_BACKEND, "visible_instances": len(extents),
                                 "reported": len(detections), "below_min_pixels": below_floor}
        return detections


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
                "metadata": {"backend": GROUND_TRUTH_BACKEND, "checkpoint_sha256": table_sha256(),
                             "segmenter_config": {"mask": "exact instance pixels", "min_pixels": self.detector.min_pixels}}}

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
            masks.append(instance_mask(ids, instance) if instance is not None else np.zeros(ids.shape, dtype=bool))
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
