"""Configuration shared by the preserved frontier baseline and FALCON adaptation."""
from __future__ import annotations

from dataclasses import dataclass, field
import math

from sparx_agency.core.planning.exploration.falcon.params import FalconParams
from sparx_agency.core.planning.exploration.floor_atlas import MultiFloorParams


@dataclass(frozen=True)
class RPTSettings:
    map_size_m: float = 80.0
    map_resolution_m: float = 0.1
    depth_stride: int = 8
    graph_period_steps: int = 10
    replan_steps: int = 5  # accepted for old configs; not a path replacement timer
    action_time_s: float = 1.0
    detection_confidence: float = 0.35
    stop_distance_m: float = 0.75
    target_memory_steps: int = 30
    seed: int = 0
    max_rooms: int = 12
    body_height_m: float = 0.88
    body_radius_m: float = 0.18
    preferred_clearance_m: float = 0.30
    local_exploration: str = "frontier"
    falcon: FalconParams = field(default_factory=FalconParams)
    multifloor: MultiFloorParams = field(default_factory=MultiFloorParams)

    def __post_init__(self):
        for key in ("map_size_m", "map_resolution_m", "action_time_s", "stop_distance_m", "body_height_m", "body_radius_m", "preferred_clearance_m"):
            value = getattr(self, key)
            if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value) or value <= 0:
                raise ValueError("%s must be positive and finite" % key)
        for key in ("depth_stride", "graph_period_steps", "replan_steps", "target_memory_steps", "max_rooms"):
            if type(getattr(self, key)) is not int or getattr(self, key) <= 0:
                raise ValueError("%s must be a positive integer" % key)
        if not 0 <= self.detection_confidence <= 1 or type(self.seed) is not int:
            raise ValueError("Invalid confidence or seed")
        if self.preferred_clearance_m < self.body_radius_m:
            raise ValueError("Preferred clearance cannot be smaller than the robot radius")
        if self.local_exploration not in ("frontier", "falcon"):
            raise ValueError("local_exploration must be frontier or falcon; no silent fallback")
        if isinstance(self.falcon, dict):
            object.__setattr__(self, "falcon", FalconParams(**self.falcon))
        if not isinstance(self.falcon, FalconParams):
            raise ValueError("falcon must be FalconParams or an object of parameter overrides")
        if isinstance(self.multifloor, dict):
            object.__setattr__(self, "multifloor", MultiFloorParams(**self.multifloor))
        if not isinstance(self.multifloor, MultiFloorParams):
            raise ValueError("multifloor must be MultiFloorParams or parameter overrides")
