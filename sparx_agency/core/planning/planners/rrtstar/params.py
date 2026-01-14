"""RRT* planner configuration."""
from __future__ import annotations
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class RRTStarOmplParams:
    """
    Configuration for RRT* path planning (2D).

    Attributes:
        timeout: Maximum planning time in seconds.
        use_clearance_objective: Optimize for obstacle clearance if available.
        clearance_weight: Weight for clearance cost (higher = prefer open space).
        min_clearance_for_keep: Minimum clearance (meters) to allow waypoint removal.
        interpolation_spacing: Target spacing between output waypoints (meters).
        frame_id: Coordinate frame identifier for output path.
        collision_check_resolution: Fraction of space extent for validity checking.
        longest_valid_segment_m: Maximum distance (meters) between collision checks.
    """
    timeout: float = 3.0
    use_clearance_objective: bool = True
    clearance_weight: float = 10.0
    min_clearance_for_keep: float = 0.3
    interpolation_spacing: float = 0.2
    frame_id: str = "map"

    # OMPL: state validity resolution is a FRACTION of space extent.
    collision_check_resolution: float = 0.005

    # OMPL: additional discretization hint (meters) -> converted to fraction.
    longest_valid_segment_m: float | None = None

    # Debug
    debug_enabled: bool = False
    debug_every_n_validity: int = 2000
    debug_max_print_validity: int = 200


@dataclass(frozen=True, slots=True)
class RRTStarOmpl3DParams:
    """
    Configuration for RRT* path planning (3D).

    Same as 2D params - kept separate for clarity and future divergence.
    """
    timeout: float = 3.0
    use_clearance_objective: bool = True
    clearance_weight: float = 10.0
    min_clearance_for_keep: float = 0.3
    interpolation_spacing: float = 0.2
    frame_id: str = "map"
    collision_check_resolution: float = 0.005
    longest_valid_segment_m: float | None = None
    rrt_range_m: float | None = None

    # Debug
    debug_enabled: bool = False
    debug_every_n_validity: int = 2000
    debug_max_print_validity: int = 200
