"""Path planner configuration (RRT*, BIT*, Informed RRT*)."""
from __future__ import annotations
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class RRTStarOmplParams:
    """Configuration for RRT* path planning (2D)."""
    timeout: float = 3.0
    use_clearance_objective: bool = True
    clearance_weight: float = 10.0
    min_clearance_for_keep: float = 0.3
    interpolation_spacing: float = 0.2
    frame_id: str = "map"
    collision_check_resolution: float = 0.005
    longest_valid_segment_m: float | None = None
    debug_enabled: bool = False


@dataclass(frozen=True, slots=True)
class RRTStarOmpl3DParams:
    """Configuration for RRT* path planning (3D)."""
    timeout: float = 3.0
    use_clearance_objective: bool = True
    clearance_weight: float = 10.0
    min_clearance_for_keep: float = 0.3
    interpolation_spacing: float = 0.2
    frame_id: str = "map"
    collision_check_resolution: float = 0.005
    longest_valid_segment_m: float | None = None
    rrt_range_m: float | None = None
    debug_enabled: bool = False


@dataclass(frozen=True, slots=True)
class BITStarParams:
    """
    Configuration for BIT* (Batch Informed Trees) 3D planning.

    BIT* is excellent for 3D because it:
    - Uses batch processing for faster convergence
    - Focuses search in promising regions using heuristics
    - Provides asymptotic optimality with better performance than RRT*
    """
    timeout: float = 5.0
    use_clearance_objective: bool = True
    clearance_weight: float = 10.0
    min_clearance_for_keep: float = 0.3
    interpolation_spacing: float = 0.2
    frame_id: str = "map"
    collision_check_resolution: float = 0.005
    longest_valid_segment_m: float | None = None

    # BIT* specific
    samples_per_batch: int = 100  # Number of samples per batch
    use_k_nearest: bool = True    # Use k-nearest instead of r-disc
    rewire_factor: float = 1.1    # Rewiring factor (higher = more rewiring)

    debug_enabled: bool = False


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