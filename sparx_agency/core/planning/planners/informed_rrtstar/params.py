"""Informed RRT* planner configuration."""
from __future__ import annotations
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class InformedRRTStarParams:
    """
    Configuration for Informed RRT* 3D planning.

    Informed RRT* is great for 3D because it:
    - Samples in an ellipsoidal subset once initial solution found
    - Focuses search toward the goal using heuristics
    - Converges faster than standard RRT* in higher dimensions
    """
    timeout: float = 5.0
    use_clearance_objective: bool = True
    clearance_weight: float = 10.0
    min_clearance_for_keep: float = 0.3
    interpolation_spacing: float = 0.2
    frame_id: str = "map"
    collision_check_resolution: float = 0.005
    longest_valid_segment_m: float | None = None

    # Informed RRT* specific
    range_m: float | None = None  # Max edge length (None = auto)

    debug_enabled: bool = False
