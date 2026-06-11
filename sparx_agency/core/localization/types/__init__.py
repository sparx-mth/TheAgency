"""Localization-only data types.

These dataclasses are specific to localization and are kept out of
:mod:`sparx_agency.core.common.types` (which holds the cross-cutting,
stack-wide vocabulary). Algorithmic code lives in the parent
``localization`` package; only the types live here.
"""
from .tag_azimuth import TagBearingObservation
from .tag_triangulation import TagWorldPose, TagTransformObservation, PoseEstimate
from .dead_reckoning import AXES, DeadReckoningNoiseParams

__all__ = [
    "TagBearingObservation",
    "TagWorldPose",
    "TagTransformObservation",
    "PoseEstimate",
    "AXES",
    "DeadReckoningNoiseParams",
]
