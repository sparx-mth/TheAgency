"""Trajectory simplification / cleanup for waypoint paths (ROS-free, 2D).

This package is deliberately separate from ``safety/path_correction`` (the
potential field). The corrector makes a path *safe* — it pushes waypoints off
walls. This package makes an already-safe path *cleaner and easier to fly*: it
removes redundant, zig-zag and near-duplicate waypoints and enforces a sensible
spacing. It computes no repulsion and reads no map of its own; geometry-changing
steps are validated through an injected ``clear_fn`` so cleanup never degrades
the safety the corrector established.
"""
from .simplifier_2d import (
    ClearFn,
    SimplifyResult,
    TrajectorySimplifier2D,
    TrajectorySimplifierConfig,
    simplify_collinear_capped_2d,
    smooth_zigzags_2d,
    thin_by_spacing_2d,
)

__all__ = [
    "ClearFn",
    "SimplifyResult",
    "TrajectorySimplifier2D",
    "TrajectorySimplifierConfig",
    "simplify_collinear_capped_2d",
    "smooth_zigzags_2d",
    "thin_by_spacing_2d",
]
