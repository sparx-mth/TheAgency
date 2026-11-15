"""Box-prompted instance segmentation interface (ROS-free).

Mirrors :class:`sparx_agency.core.mapping.interfaces.detection_model.DetectionModel`:
a tiny ABC with a numpy-in / numpy-out contract. Backends live in
:mod:`sparx_agency.core.mapping.segmentation` (``SamPromptedSegmenter`` wrapping
the ultralytics Segment Anything family), each lazy-importing its heavy
dependencies so ``core`` stays light and Python-3.8 importable.

A segmenter answers one question a detector cannot: *which pixels of this box
belong to the detected object?* A detection box overhangs its object and
admits whatever stands in front of it, so a depth read inside the box can be
the depth of an occluder -- a chair in front of a sofa reads as the sofa's
near surface. The mask restricts that read to the object itself. The
segmenter takes no intrinsics and no depth: lifting the masked pixels to 3-D
is the caller's concern
(:func:`sparx_agency.core.mapping.objects.geometry.robust_mask_depth`).
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Sequence, Tuple

import numpy as np

BoxXYXY = Tuple[float, float, float, float]


class SegmentationModel(ABC):
    """Box-prompted segmenter API (ROS-free).

    Input: RGB image ``HxWx3`` uint8 and the pixel boxes to segment.
    Output: one boolean ``HxW`` mask per box, in the order of the boxes --
    the pixels of the object each box was drawn around. A box the model
    cannot resolve yields an all-``False`` mask, never a missing entry, so
    the caller can line masks up with its detections by position.
    """

    @abstractmethod
    def segment(self, rgb: np.ndarray, boxes_xyxy: Sequence[BoxXYXY]) -> List[np.ndarray]:
        """Segment the object inside each box of one RGB frame.

        Args:
            rgb: ``HxWx3`` RGB image, uint8.
            boxes_xyxy: Pixel boxes ``(x1, y1, x2, y2)``, origin top-left,
                in the frame's own pixel coordinates.

        Returns:
            ``len(boxes_xyxy)`` boolean masks of shape ``HxW``, one per box,
            in the same order. Empty input gives an empty list.
        """
        raise NotImplementedError
