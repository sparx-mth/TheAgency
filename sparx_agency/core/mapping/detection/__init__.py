"""Open-vocabulary object detection (ROS-free).

The ``DetectionModel`` ABC lives in
:mod:`sparx_agency.core.mapping.interfaces.detection_model`; this package holds the
backends. ``YoloWorldDetector`` ("OpenYOLO") is the default; a TensorRT runtime
backend can be added alongside it, mirroring the depth backends. Downstream, a
detection is lifted to 3D with
:func:`sparx_agency.core.mapping.depth.depth_bbox_fusion.bbox_to_xyz_cam_from_depth`,
and tracked in real time by :mod:`sparx_agency.core.mapping.tracking`.
"""
from __future__ import annotations

from sparx_agency.core.mapping.interfaces.detection_model import DetectionModel
from sparx_agency.core.mapping.detection.yolo_world import (
    YoloWorldConfig,
    YoloWorldDetector,
)
from sparx_agency.core.mapping.detection.registry import (
    DetectorFactory,
    DetectionRegistry,
    default_detection_registry,
)

__all__ = [
    "DetectionModel",
    "YoloWorldConfig",
    "YoloWorldDetector",
    "DetectorFactory",
    "DetectionRegistry",
    "default_detection_registry",
]
