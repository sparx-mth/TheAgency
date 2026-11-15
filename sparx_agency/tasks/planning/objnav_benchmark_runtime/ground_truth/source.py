"""The privileged hand-off between the environment and the oracle perception: scene labels and the current frame.

The simulator renders the instance-id frame beside RGB-D; the method's
perception cycle calls ``detector.detect(rgb)`` with the RGB array alone.
This object is the only thing standing between the two. The environment
:meth:`publish`\\ es each frame as it builds the observation; the detector and
the segmenter :meth:`lookup` the frame for the RGB array they were handed,
and the lookup is **strict**: the array must be the one published (the same
object, or pixel-identical). An oracle answering for the wrong frame would be
a silent perception error inside the run meant to have none, so a mismatch is
an :class:`~sparx_agency.core.planning.objnav.errors.ObjNavInternalError`
without a cause -- the identity-drift family the perception cycle does not
back off from, and the run ends.

Nothing here reaches an observation; the policy holds the detector and the
segmenter, never this object's frame or labels directly.
"""
from __future__ import annotations

from collections import Counter
from typing import Dict, FrozenSet, Mapping, Optional

import numpy as np

from sparx_agency.core.planning.objnav.errors import ObjNavInternalError
from sparx_agency.tasks.planning.objnav_benchmark_runtime.ground_truth.hm3d_categories import HM3D_TABLE


class GroundTruthSource:
    """Per-scene instance labels and the current instance-id frame, for the oracle detector and segmenter.

    Args:
        label_mapper: The benchmark's label table; its ``vocabulary()`` bounds
            what the oracle may emit.
        categories: The dataset's category table (``HM3D_TABLE``, the default,
            or ``mp3d_categories.MP3D_TABLE``): how one scene's annotations
            become the instances the oracle reports. The only dataset-specific
            part of the oracle.
    """

    def __init__(self, label_mapper, categories=HM3D_TABLE):
        self.label_mapper = label_mapper
        self.categories = categories
        self.vocabulary = tuple(label_mapper.vocabulary())
        self.labels: Dict[int, str] = {}
        self.scene_key: Optional[str] = None
        self.scene_report: Dict[str, object] = {}
        self._rgb = None
        self._ids = None
        #: The frame's depth, pose and camera (``publish``), and the scene's instance boxes (``begin_scene``): what
        #: the detector's on-object filter judges a rendered pixel against. None when the adapter publishes none.
        self.depth = None
        self.pose = None
        self.camera = None
        self.boxes: Dict[int, tuple] = {}
        self.frames = 0
        self.frames_without_instances = 0
        self.lookup_misses = 0

    def begin_scene(self, scene_key: str, semantic_objects: Mapping[int, str],
                    goal_instances: Mapping[str, FrozenSet[int]], boxes: Optional[Mapping[int, tuple]] = None) -> Dict[str, object]:
        """Build the scene's ``{instance: vocabulary word}`` table and describe it.

        ``boxes`` is ``{instance: (center xyz, half sizes xyz)}`` in the
        method's world frame (the recording's ENU), the annotation's own
        axis-aligned box per instance, for the detector's on-object filter;
        None or empty leaves the filter nothing to judge against.

        Returns:
            A JSON-safe account: how many instances the scene annotates, how
            many the oracle will report and under which words, which goal
            categories the publisher listed, and the raw names left unmapped
            (the twenty most frequent) -- so a gap in the name table is visible
            in the run directory, not discovered from a score.
        """
        self.labels = self.categories.instance_labels(semantic_objects, goal_instances, self.vocabulary)
        self.scene_key = scene_key
        deferred, unmapped = self.categories.unlabelled_account(semantic_objects, self.labels, goal_instances)
        self.scene_report = {
            "scene": scene_key,
            "annotated_instances": len(semantic_objects),
            "reported_instances": len(self.labels),
            "reported_by_label": dict(sorted(Counter(self.labels.values()).items())),
            "goal_instances": {category: len(ids) for category, ids in sorted(goal_instances.items())},
            "named_for_a_goal_category_but_not_a_goal": dict(sorted(deferred.items())),
            "unmapped_raw_names": dict(unmapped.most_common(20)),
            "category_table_sha256": self.categories.sha256(),
        }
        self._rgb = self._ids = None
        self.boxes = {int(k): (tuple(float(c) for c in v[0]), tuple(float(h) for h in v[1]))
                      for k, v in (boxes or {}).items()}
        self.scene_report["instances_with_boxes"] = len(self.boxes)
        return self.scene_report

    def publish(self, rgb: np.ndarray, ids: np.ndarray, depth=None, pose=None, camera=None) -> None:
        """Make ``ids`` the frame the oracle answers for ``rgb``; ``depth``, ``pose`` and ``camera`` ride along for the on-object filter."""
        if self.scene_key is None:
            raise ObjNavInternalError("Ground-truth source has no scene; begin_scene() must precede the first frame")
        ids = np.asarray(ids)
        if ids.ndim != 2 or ids.shape != np.asarray(rgb).shape[:2]:
            raise ObjNavInternalError("Ground-truth frame %s does not match the RGB frame %s"
                                      % (ids.shape, np.asarray(rgb).shape[:2]))
        self._rgb, self._ids = rgb, ids
        self.depth, self.pose, self.camera = depth, pose, camera
        self.frames += 1
        if not ids.any():
            self.frames_without_instances += 1

    def lookup(self, rgb: np.ndarray) -> np.ndarray:
        """The instance-id frame of ``rgb``; the run ends if it is not the frame published."""
        if self._ids is None:
            raise ObjNavInternalError("Ground-truth source holds no frame; the environment has not published one")
        if rgb is not self._rgb and not (
                np.asarray(rgb).shape == np.asarray(self._rgb).shape and np.array_equal(rgb, self._rgb)):
            self.lookup_misses += 1
            raise ObjNavInternalError("Ground-truth perception was asked about an RGB frame that is not the "
                                      "simulator's current frame; the oracle will not answer for another image")
        return self._ids

    def diagnostics(self) -> Dict[str, object]:
        """Counters for the evaluation sidecar: frames served and frames in which nothing was annotated."""
        return {"frames": self.frames, "frames_without_instances": self.frames_without_instances,
                "lookup_misses": self.lookup_misses, "scene": self.scene_key,
                "reported_instances": len(self.labels)}
