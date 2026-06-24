"""Planner-agnostic path correction.

A :class:`PathCorrector` reshapes a planned ``Path2D`` against a live
``OccupancyGrid2D`` so it is never less safe than the input -- e.g. recentring it
off walls toward corridor centres. Strategies are pluggable and selected by name
via :func:`make_path_corrector`, so the ROS node that owns the "receive a path ->
correct -> republish" wiring is independent of both the planner that produced the
path and the strategy used to correct it.

Currently available:
    * ``"potential_field"`` -- :class:`PotentialFieldPathCorrector`, repulsive
      Gaussian field recentring (the historical A* APF post-process, extracted).
    * ``"esdf"`` -- :class:`EsdfPathCorrector`, distance-field gradient ascent
      (push each waypoint up ``+∇D`` toward the corridor / doorway centre).

ROS-free; Python 3.8 compatible (the FALCON Noetic adapter imports core under 3.8).
"""
from .base import PathCorrector, PathCorrectionResult
from .esdf_corrector import EsdfCorrectorConfig, EsdfPathCorrector
from .factory import AVAILABLE_CORRECTORS, make_path_corrector
from .grid_collision import InflatedGridCollisionChecker
from .map_safety import clip_to_clear, dampen_unknown
from .potential_field_corrector import (
    PotentialFieldCorrectorConfig,
    PotentialFieldPathCorrector,
)

__all__ = [
    "PathCorrector",
    "PathCorrectionResult",
    "PotentialFieldCorrectorConfig",
    "PotentialFieldPathCorrector",
    "EsdfCorrectorConfig",
    "EsdfPathCorrector",
    "InflatedGridCollisionChecker",
    "clip_to_clear",
    "dampen_unknown",
    "make_path_corrector",
    "AVAILABLE_CORRECTORS",
]
