"""
RRT* (OMPL) planner parameters.

ROS-free configuration for OMPL-based RRT* planning on a 2D costmap.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class RRTStarOmplParams:
    # Planning
    planning_timeout_s: float = 3.0

    # Clearance objective (if world provides clearance in meters or cells)
    use_clearance_objective: bool = True
    clearance_weight: float = 10.0  # cost ~ weight/(clearance+1)

    # Path post-processing (adaptive reduction similar to your C++)
    min_clearance_for_keep: float = 0.3  # meters (or "world clearance units")
    # If world clearance is in *cells* and you prefer that, set accordingly.

    # Interpolation (world meters)
    interpolation_spacing_m: float = 0.2

    # Misc
    frame_id: str = "map"
