"""Open-vocabulary object detection (ROS-free).

The ``DetectionModel`` ABC lives in
:mod:`sparx_agency.core.mapping.interfaces.detection_model`; this package holds the
backends. ``YoloWorldDetector`` remains the baseline; ``LlmDetDetector`` supplies
the selectable official pretrained LLMDet implementation. Downstream, a
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
from sparx_agency.core.mapping.detection.llmdet import LlmDetConfig, LlmDetDetector
from sparx_agency.core.mapping.detection.registry import (
    DetectorFactory,
    DetectionRegistry,
    default_detection_registry,
)

__all__ = [
    "DetectionModel",
    "YoloWorldConfig",
    "YoloWorldDetector",
    "LlmDetConfig",
    "LlmDetDetector",
    "DetectorFactory",
    "DetectionRegistry",
    "default_detection_registry",
]
