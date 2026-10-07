"""Configuration shared by the preserved frontier baseline and FALCON adaptation."""
from __future__ import annotations

from dataclasses import dataclass, field
import math

from sparx_agency.core.planning.exploration.falcon.params import FalconParams
from sparx_agency.core.planning.exploration.floor_atlas import MultiFloorParams
from sparx_agency.tasks.planning.objnav_benchmark_runtime.methods.doorway_candidates import PeekSettings
from sparx_agency.tasks.planning.objnav_benchmark_runtime.methods.target_closing import TargetClosingSettings


@dataclass(frozen=True)
class RPTSettings:
    """Method configuration.

    Attributes:
        graph_period_steps: How often the scene graph's GEOMETRY is refreshed
            -- rooms, object->room association, doors, per-room search time
            and frontier counts. Every action by default: the room-search
            loop reads the room masks and frontier counts on every action,
            and a ten-action-old mask confines routes to a room that has
            since moved. This is not the LLM cadence; the room LLM runs at
            the loop's own points (see ``room_search_loop.LoopSettings``).
        warmup_steps: Actions of the warm-up rotation taken in place where
            the agent stands before any room is chosen (and again on every
            storey first entered). Twelve is one full circle at the
            benchmark's 30-degree turn; the position is then recorded as a
            completed scan, so the spawn room is finished before the search
            begins.
        doorway_peek: The one-shot doorway inspections. OFF by default since
            2026-10-04: under the scan visit a peek -- approach, a half
            turn from the threshold, the walk back -- costs as much as a
            room's own scan and finishes nothing, and the four peeks of the
            last Ranchester recording were all cancelled. ``{"enabled":
            True}`` restores them; the floor-departure gate they fed is a
            further option of their own (``gate_floor_departure``).
    """

    map_size_m: float = 80.0
    map_resolution_m: float = 0.1
    depth_stride: int = 8
    graph_period_steps: int = 1
    replan_steps: int = 5  # accepted for old configs; not a path replacement timer
    action_time_s: float = 1.0
    # The floor under which a box is not projected at all. 0.30 since 2026-10-07,
    # the takeover's own tracking threshold: a landmark needs two votes to be
    # confirmed, so a weak box costs nothing on its own, and the 0.35 floor
    # dropped a cup at 0.34 on the table in front of the Allensville spawn point
    # (the "low_confidence" rows of the recording's first frame).
    detection_confidence: float = 0.30
    # The landmark map's same-class association (``core/mapping/objects/landmarks.py``;
    # classes never merge). Two observations of one class are one instance within
    # ``landmark_dedupe_radius_m`` of each other -- 0.35 m, re-observation jitter, down
    # from the ported 0.70 m that merged two dining chairs -- or when their measured
    # footprint discs overlap by ``landmark_footprint_iou``: 0.25 is about one radius
    # apart for equal discs, so a bed re-seen from its other side (0.8 m) is one bed and
    # two chairs 0.3 m apart are two (since 2026-10-07).
    landmark_dedupe_radius_m: float = 0.35
    landmark_footprint_iou: float = 0.25
    stop_distance_m: float = 0.75
    target_memory_steps: int = 30
    seed: int = 0
    max_rooms: int = 12
    body_height_m: float = 0.88
    body_radius_m: float = 0.18
    preferred_clearance_m: float = 0.30
    local_exploration: str = "frontier"
    warmup_steps: int = 12
    target_closing: TargetClosingSettings = field(default_factory=TargetClosingSettings)
    doorway_peek: PeekSettings = field(default_factory=lambda: PeekSettings(enabled=False))
    falcon: FalconParams = field(default_factory=lambda: FalconParams(burst_actions=10))
    multifloor: MultiFloorParams = field(default_factory=MultiFloorParams)

    def __post_init__(self):
        for key in ("map_size_m", "map_resolution_m", "action_time_s", "stop_distance_m", "body_height_m", "body_radius_m",
                    "preferred_clearance_m", "landmark_dedupe_radius_m"):
            value = getattr(self, key)
            if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value) or value <= 0:
                raise ValueError("%s must be positive and finite" % key)
        if (isinstance(self.landmark_footprint_iou, bool) or not math.isfinite(self.landmark_footprint_iou)
                or not 0 < self.landmark_footprint_iou <= 1):
            raise ValueError("landmark_footprint_iou must lie in (0, 1]")
        for key in ("depth_stride", "graph_period_steps", "replan_steps", "target_memory_steps", "max_rooms", "warmup_steps"):
            if type(getattr(self, key)) is not int or getattr(self, key) <= 0:
                raise ValueError("%s must be a positive integer" % key)
        if not 0 <= self.detection_confidence <= 1 or type(self.seed) is not int:
            raise ValueError("Invalid confidence or seed")
        if self.preferred_clearance_m < self.body_radius_m:
            raise ValueError("Preferred clearance cannot be smaller than the robot radius")
        if self.local_exploration not in ("frontier", "falcon"):
            raise ValueError("local_exploration must be frontier or falcon; no silent fallback")
        if isinstance(self.target_closing, dict):
            object.__setattr__(self, "target_closing", TargetClosingSettings(**self.target_closing))
        if not isinstance(self.target_closing, TargetClosingSettings):
            raise ValueError("target_closing must be TargetClosingSettings or parameter overrides")
        if isinstance(self.doorway_peek, dict):
            object.__setattr__(self, "doorway_peek", PeekSettings(**self.doorway_peek))
        if not isinstance(self.doorway_peek, PeekSettings):
            raise ValueError("doorway_peek must be PeekSettings or parameter overrides")
        if isinstance(self.falcon, dict):
            object.__setattr__(self, "falcon", FalconParams(**dict({"burst_actions": 10}, **self.falcon)))
        if not isinstance(self.falcon, FalconParams):
            raise ValueError("falcon must be FalconParams or an object of parameter overrides")
        if isinstance(self.multifloor, dict):
            object.__setattr__(self, "multifloor", MultiFloorParams(**self.multifloor))
        if not isinstance(self.multifloor, MultiFloorParams):
            raise ValueError("multifloor must be MultiFloorParams or parameter overrides")
