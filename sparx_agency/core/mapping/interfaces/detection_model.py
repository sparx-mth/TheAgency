"""Open-vocabulary 2D object detector interface (ROS-free).

Mirrors :class:`sparx_agency.core.mapping.interfaces.depth_model.DepthModel`: a
tiny, single-method ABC with a numpy-in / dataclass-out contract. Backends live
in :mod:`sparx_agency.core.mapping.detection` (e.g. ``YoloWorldDetector`` wrapping
ultralytics YOLO-World, and later a TensorRT runtime), each lazy-importing its
heavy dependencies so ``core`` stays light and Python-3.8 importable.

The detector takes only the image plus the open-vocabulary ``prompts`` (class
strings). It does **not** take camera intrinsics: 2D -> 3D lifting is a separate
downstream concern handled by
:func:`sparx_agency.core.mapping.depth.depth_bbox_fusion.bbox_to_xyz_cam_from_depth`.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Sequence

import numpy as np

from sparx_agency.core.common.types.perception import Detection2D


class DetectionModel(ABC):
    """Open-vocabulary 2D detector API (ROS-free).

    Input: RGB image ``HxWx3`` (uint8 or float32) and a list of prompt strings.
    Output: list of :class:`Detection2D` in pixel coordinates (``bbox_xyxy``).
    """

    @abstractmethod
    def set_prompts(self, prompts: Sequence[str]) -> None:
        """Set / replace the open-vocabulary class prompts to detect.

        Open-vocabulary detectors accept an arbitrary list of class strings with
        no retraining; changing the target object at runtime is just a new prompt
        list. Implementations should treat this as cheap and idempotent.
        """
        raise NotImplementedError

    @abstractmethod
    def detect(self, rgb: np.ndarray) -> List[Detection2D]:
        """Run detection on one RGB frame against the current prompts.

        Args:
            rgb: ``HxWx3`` RGB image, uint8 or float32.

        Returns:
            Detections in pixel coordinates. Empty list when nothing is found.
        """
        raise NotImplementedError
