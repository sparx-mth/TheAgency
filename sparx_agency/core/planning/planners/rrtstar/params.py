"""RRT* planner configuration."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class RRTStarParams:
    """
    Configuration for RRT* path planning.

    Attributes:
        timeout: Maximum planning time in seconds.
        use_clearance_objective: Optimize for obstacle clearance if available.
        clearance_weight: Weight for clearance cost (higher = prefer open space).
        min_clearance_for_keep: Minimum clearance (meters) to allow waypoint removal.
        interpolation_spacing: Target spacing between output waypoints (meters).
        frame_id: Coordinate frame identifier for output path.
    """
    timeout: float = 3.0
    use_clearance_objective: bool = True
    clearance_weight: float = 10.0
    min_clearance_for_keep: float = 0.3
    interpolation_spacing: float = 0.2
    frame_id: str = "map"