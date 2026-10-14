"""Box-prompted segmentation (ROS-free).

The ``SegmentationModel`` ABC lives in
:mod:`sparx_agency.core.mapping.interfaces.segmentation_model`; this package holds
the backends. ``SamPromptedSegmenter`` wraps the ultralytics Segment Anything
family (MobileSAM, SAM, SAM 2 / 2.1) prompted with detection boxes, so the
depth read for a detection can be restricted to the object's own pixels
(:func:`sparx_agency.core.mapping.objects.geometry.robust_mask_depth`).
"""
from __future__ import annotations

from sparx_agency.core.mapping.interfaces.segmentation_model import SegmentationModel
from sparx_agency.core.mapping.segmentation.sam_prompted import (
    SamPromptedConfig,
    SamPromptedSegmenter,
)

__all__ = ["SegmentationModel", "SamPromptedConfig", "SamPromptedSegmenter"]
