"""Configuration for the FALCON 2D/2.5D adaptation (not the ROS/UAV binary)."""
from __future__ import annotations

from dataclasses import dataclass
import math


SOURCE = {
    "paper": "https://arxiv.org/abs/2407.00577v2",
    "doi": "10.1109/TRO.2024.3522148",
    "repository": "https://github.com/HKUST-Aerial-Robotics/FALCON",
    "branch": "ros1-noetic",
    "revision": "312eb4d32c6c7af1a482f94a0a204aa2bb150cca",
    "implementation": "sparx-falcon-planar/1",
    "native_ros_binary": False,
    "fallback_explorer": None,
    "ordering_solver": "deterministic precedence-constrained dynamic programming/beam",
}


@dataclass(frozen=True)
class FalconParams:
    """Sensor-scaled decomposition, FUEL-style samples and CP/SOP planning.

    The bounded domain substitutes for FALCON's building-sized exploration box.
    Unknown-zone paths are hypothetical guidance ONLY, never execution paths.
    Solver truncation is reported; no alternate exploration backend is loaded.
    """

    burst_actions: int = 48
    scope_radius_m: float = 4.0
    cell_size_m: float = 4.0
    cluster_radius_m: float = 1.0
    min_cluster_cells: int = 4
    candidate_radii_m: tuple = (0.4, 0.8, 1.2)
    candidate_angles: int = 16
    top_viewpoints: int = 4
    max_frontiers: int = 18
    viewpoint_z_score: float = 0.0
    view_decay: float = 0.8
    visibility_rays: int = 41
    unknown_penalty: float = 2.0
    hybrid_distance_m: float = 4.0
    exact_order_limit: int = 10
    beam_width: int = 128
    max_order_nodes: int = 64
    planning_deadline_s: float = 1.5
    max_grid_nodes: int = 18000
    low_gain_actions: int = 14
    low_gain_m2: float = 0.05
    max_failed_goals: int = 4
    recovery_actions: int = 3
    target_actions: int = 24
    transit_actions: int = 90
    entry_confirm_actions: int = 2
    revisit_cooldown_actions: int = 16
    max_region_bursts: int = 3

    def __post_init__(self):
        integers = (
            "burst_actions", "min_cluster_cells", "candidate_angles", "top_viewpoints",
            "max_frontiers", "visibility_rays", "exact_order_limit", "beam_width",
            "max_order_nodes", "max_grid_nodes", "low_gain_actions",
            "max_failed_goals", "recovery_actions", "target_actions", "transit_actions",
            "entry_confirm_actions", "revisit_cooldown_actions", "max_region_bursts")
        for key in integers:
            if type(getattr(self, key)) is not int or getattr(self, key) <= 0:
                raise ValueError("%s must be a positive integer" % key)
        for key in ("scope_radius_m", "cell_size_m", "cluster_radius_m", "unknown_penalty",
                    "hybrid_distance_m", "planning_deadline_s", "low_gain_m2"):
            value = getattr(self, key)
            if isinstance(value, bool) or not math.isfinite(value) or value <= 0:
                raise ValueError("%s must be positive and finite" % key)
        if not math.isfinite(self.viewpoint_z_score) or not 0 < self.view_decay <= 1:
            raise ValueError("Invalid viewpoint qualification/decay")
        radii = tuple(self.candidate_radii_m)
        if not radii or any(isinstance(r, bool) or not math.isfinite(r) or r <= 0 for r in radii):
            raise ValueError("candidate_radii_m must contain positive finite radii")
        object.__setattr__(self, "candidate_radii_m", radii)
        if self.unknown_penalty < 1 or self.exact_order_limit > 14:
            raise ValueError("Unknown penalty must be >= 1; exact_order_limit must be <= 14")
