"""Shared, model-agnostic fine-tuning building blocks.

The numpy modules (``frames``, ``esdf_target``, ``label_format``, ``augment``) are
importable without torch and are re-exported here. The torch modules
(``esdf_penalty``, ``l2sp``, ``ema``) are intentionally **not** imported at package
level so this package loads in the plain ``.venv``; import them directly from a
torch environment.
"""
from .augment import (
    AugmentedFrame,
    ViewpointAugmentConfig,
    apply_viewpoint_augment,
    pitch_homography,
)
from .esdf_target import (
    EsdfTargetConfig,
    PerFrameTarget,
    generate_target,
    signed_sdf,
)
from .frames import (
    LocalMapConfig,
    OCC_VALUES,
    cloud_to_occupancy_grid,
    depth_to_body_cloud,
    occupancy_binary,
    occupancy_probability,
)
from .label_format import resample_arclength, to_flownav_label, to_navdp_label

__all__ = [
    "AugmentedFrame",
    "ViewpointAugmentConfig",
    "apply_viewpoint_augment",
    "pitch_homography",
    "EsdfTargetConfig",
    "PerFrameTarget",
    "generate_target",
    "signed_sdf",
    "LocalMapConfig",
    "OCC_VALUES",
    "cloud_to_occupancy_grid",
    "depth_to_body_cloud",
    "occupancy_binary",
    "occupancy_probability",
    "resample_arclength",
    "to_flownav_label",
    "to_navdp_label",
]
