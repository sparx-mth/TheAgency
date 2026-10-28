"""Configuration shared by the preserved frontier baseline and FALCON adaptation."""
from __future__ import annotations

from dataclasses import dataclass, field
import math

from sparx_agency.core.planning.exploration.falcon.params import FalconParams
from sparx_agency.core.localization.floor_atlas import MultiFloorParams
from sparx_agency.core.planning.object_search.exploration.doorway_candidates import PeekSettings
from sparx_agency.core.planning.object_search.closing.target_closing import TargetClosingSettings
from sparx_agency.core.planning.object_search.navigation.settings import (KinematicSettings, LookDownSettings,
                                                                           ScrapeSettings)


@dataclass(frozen=True)
class RPTSettings:
    """Method configuration.

    Attributes:
        detour_factor: The planner's "squeeze or detour" rule
            (``WeightedAStarParams.detour_factor``): a route at the preferred
            clearance more than this many times longer than the route the
            body radius allows is a detour, and the squeeze is walked
            instead. 1.5 since 2026-10-13: the agent walked 12.7 m round a
            room because a 0.5 m pinch on the 7.5 m way read as 0.25 m
            short of the preferred standoff (Corozal/000076, actions
            394-486). 0 restores "the safest feasible route first".
        graph_period_steps: How often the scene graph's GEOMETRY is refreshed
            -- rooms, object->room association, doors, per-room search time
            and frontier counts. Every action by default: the room-search
            loop reads the room masks and frontier counts on every action,
            and a ten-action-old mask confines routes to a room that has
            since moved. This is not the LLM cadence; the room LLM runs at
            the loop's own points (see ``room_search_loop.LoopSettings``).
        transit_graph_period_steps: The cadence instead while the loop walks
            a COMMITTED transit route under its commitment rule
            (``LoopSettings.transit_commitment``): the follower needs the map
            and its route, not a fresh partition, and the full rebuild --
            watershed, registry, frontier inventory, per-room facts -- was
            1.4 s of a 1.7 s median decision on Corozal/000003
            (``c1af8954a378``, 2026-10-22). 5 by default; the loop forces a
            rebuild on the action a transit ends, and a loop point always
            reasons over geometry refreshed on its own action. A value at or
            under ``graph_period_steps`` disables the throttle.
        warmup_steps: Upper bound on the warm-up rotation taken in place
            where the agent stands before any room is chosen (and again on
            every storey first entered). The rotation itself is as long as
            the camera's field of view needs to have seen every bearing
            once -- ten turns at the benchmark's 30-degree turn with its
            79-degree camera (``RPTSearchPolicy.rotation_turns``) -- never
            more than this; the position is then recorded as a completed
            scan, so the spawn room is finished before the search begins.
        doorway_peek: The one-shot doorway inspections. OFF by default since
            2026-10-04: under the scan visit a peek -- approach, a half
            turn from the threshold, the walk back -- costs as much as a
            room's own scan and finishes nothing, and the four peeks of the
            last Ranchester recording were all cancelled. ``{"enabled":
            True}`` restores them; the floor-departure gate they fed is a
            further option of their own (``gate_floor_departure``).
        allow_stair_traversal: Whether the search may change storeys. OFF
            by default: the standard ZSON benchmarks (Gibson SemExp v1.1,
            HM3D) score an episode on the spawn storey, so a staircase is
            never a node of the room-search loop, never the exploration
            fallback's last resort, never committed or climbed, and an
            unplanned height departure starts no traversal; every seen
            staircase and every observed drop below the storey plane is
            impassable for frontiers, openings, peeks and routes, and no goal
            off the spawn plane is accepted (``core/planning/object_search/multifloor/spawn_floor_guard.py``).
            ``True`` restores the multi-storey search exactly as before.
        floor_plane_bound_m: With stair traversal forbidden, how far the
            storey the agent stands on may lie from the spawn height before
            every goal on it is refused -- the ``Z_spawn +/- 0.5 m`` plane.
            Larger than the atlas's ``departure_m`` (0.45): a plateau the
            atlas still calls this floor is this floor.
        step_down_m: With stair traversal forbidden, an observed surface
            more than this far BELOW the storey plane is a drop and is
            written occupied for every plan (``spawn_floor_guard``). 0.30 m,
            the atlas's floor-match tolerance, as before -- a knob since
            2026-10-14, and NOT lower: at 0.15 m Corozal/000142 painted
            3,316 cells of its own storey as drops (the plane is the pose
            height, and the floor the camera sees lies 0.15-0.30 m under it
            there; the toilet stands across a one-step sunken area the
            earlier runs walked into) and the episode was lost from the
            spawn point. The three-tread flight Darden/000034 walked down
            (-0.11, -0.18, -0.21 m, the floor -0.45 m seen by five samples)
            is inside this tolerance and remains the open item.
        stair_margin_m: The preferred half-width of the capsule written
            impassable around a SEEN staircase's centreline
            (``world_model/peek_stairs.stair_peek_mask``): a flight's
            half-width plus room for the body, which the planner's own
            body-radius inflation then surrounds like any wall. Stairs
            only -- walls, furniture and observed drops are inflated as
            before. 0.45 m (five cells drawn) since 2026-10-21, down from
            the implicit ``body_radius + goal_tolerance + cell`` (0.53 m,
            six cells drawn): with the planner's 0.18 m on top the old
            capsule kept the body 0.78 m from a flight's centreline and,
            in Markleeville, narrowed the passages beside its stairwell on
            the 2D grid; the adaptive capsule still narrows it further,
            down to the body radius, wherever a corridor would close.
            Never below ``body_radius_m``.
        segmentation: Whether the TARGET's boxes are segmented (box-prompted
            Segment Anything, ``tasks/planning/objnav_benchmark_runtime/services/http_segmenter.HttpSegmenter``) and
            their depth read on the object's own pixels rather than on the
            box (``perception.observed_objects``). OFF by default -- the
            experiment of 2026-10-12: a box admits whatever stands in front
            of the object, so the terminal range to a sofa with a chair
            before it was the chair's, and the agent STOPped a metre short.
            With it on, a target box the service could not mask is NOT read
            the old way (``segmentation_unavailable``), so the two pipelines
            differ in exactly one thing: which pixels count as the object.
            The policy needs a segmenter when this is on and refuses to
            start without one.
        centreline_weight: The planner's medial-axis layer
            (``WeightedAStarParams.centreline_weight``, since 2026-10-28):
            inside a passage narrower than ``2 * channel_half_width_m`` the
            cost rises linearly from the ridge of the distance transform to
            the walls, so a route threads a doorway through its middle and
            a corridor down its centreline instead of cutting the jamb. 4.0:
            at a 0.8 m doorway's centre a cell costs 2.3 times open floor,
            at the jamb-side cell 3.0; the soft band alone told them apart by
            a tenth. 0 restores the soft band alone.
        channel_half_width_m: Half the passage width below which the layer
            applies and a route is in a "channel" for the scrape guard and
            the look-down. 0.6 m: every doorway, no room.
        kinematic: The discrete-action lattice that turns every committed
            route into the exact actions that walk it
            (``navigation/kinematic_follower.py``).
        doorway_scrape: When a refused step in a doorway is a scrape, not a
            wall, and stays off the map (``navigation/scrape_guard.py``).
        doorway_look_down: The camera pitched on the approach to a
            doorway and the re-plan on what it saw
            (``navigation/doorway_look_down.py``).
    """

    map_size_m: float = 80.0
    map_resolution_m: float = 0.1
    depth_stride: int = 8
    graph_period_steps: int = 1
    transit_graph_period_steps: int = 5
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
    # Retraction (since 2026-10-21, ``landmarks.py`` ``max_misses``): a confirmed landmark that
    # stood in ``landmark_max_misses`` CLEAR views -- inside the frame by ``landmark_miss_margin_px``,
    # between ``landmark_miss_min_range_m`` and ``landmark_miss_max_range_m`` away, the depth at its
    # pixel showing nothing in front of it -- without a detection of its class leaves the confirmed
    # set, and the room it stood in is re-classified without it; the next detection reinstates it.
    # Four, or as many as the landmark's observations if more (``landmarks.py``: the negative evidence
    # must outweigh the positive), each from a DISTINCT viewpoint -- the camera moved
    # ``landmark_miss_viewpoint_m`` or turned 30 degrees since the last counted miss -- so four frames
    # from one bad angle are one miss. 0 disables (the one-way confirmation of before).
    landmark_max_misses: int = 4
    landmark_miss_margin_px: int = 40
    landmark_miss_min_range_m: float = 0.8
    landmark_miss_max_range_m: float = 3.5
    landmark_miss_viewpoint_m: float = 0.5
    stop_distance_m: float = 0.75
    target_memory_steps: int = 30
    seed: int = 0
    max_rooms: int = 12
    body_height_m: float = 0.88
    body_radius_m: float = 0.18
    preferred_clearance_m: float = 0.30
    # The tightest standoff from a confirmed obstacle a route may relax to (``WeightedAStarParams.inflate_floor_m``,
    # and the "passable" map every count, entry and frontier is measured on). Until 2026-10-25 this was the body
    # radius itself, which on the 0.1 m grid admits a 0.4 m gap -- and the agent is a 0.36 m body that HM3D moves
    # WITHOUT sliding (``allow_sliding`` false, unlike Gibson): any contact refuses the whole step, and a heading
    # quantised to 30 degrees cannot thread a 0.4 m gap without one. 00824/000066 walked a route between a bed and
    # a dresser 4-5 cells apart and spent 460 actions wedged there; 00835/000006 began its episode boxed in a
    # hallway the same way. 0.25 m: a 0.5 m gap (five free cells, 0.3 m of clearance at its centre) is still a
    # passage, a 0.4 m one is a wall. Doorways (0.7 m and up) are untouched.
    min_clearance_m: float = 0.25
    detour_factor: float = 1.5
    # What a refused MOVE_FORWARD costs the goal it was taken toward (``RPTSearchPolicy._bumped_toward``, since
    # 2026-10-28). The route is re-planned round the bump at once; the GOAL is retired -- taken out of the frontier
    # memory with every frontier within the sweep's ``visited_radius_m`` of it -- only when ``goal_bump_retire``
    # refused steps toward the same place (``goal_bump_match_m`` cells) fall within ``goal_bump_window`` actions.
    # 1 restores the one-bump rule of before, which the GT-oracle sweep of 2026-10-26 showed sealing whole wings:
    # same-storey success 97 % with no refused step, 51 % with six to ten.
    goal_bump_retire: int = 3
    goal_bump_window: int = 40
    goal_bump_match_m: float = 0.5
    # How long the STICKY bar a second bump writes across the heading stays forced occupied
    # (``ObservedMap.notify_blocked``), in actions; a further bump at the place renews it. 0 keeps it for the rest
    # of the episode, as before 2026-10-28. A bar at a doorway the body did pass -- the jamb clipped twice by a
    # quantised heading -- closed the doorway for good; a ledge the camera reads as floor earns its bar again.
    sticky_bump_actions: int = 120
    local_exploration: str = "frontier"
    warmup_steps: int = 12
    allow_stair_traversal: bool = False
    floor_plane_bound_m: float = 0.5
    step_down_m: float = 0.30
    stair_margin_m: float = 0.45
    segmentation: bool = False
    centreline_weight: float = 4.0
    channel_half_width_m: float = 0.6
    kinematic: KinematicSettings = field(default_factory=KinematicSettings)
    doorway_scrape: ScrapeSettings = field(default_factory=ScrapeSettings)
    doorway_look_down: LookDownSettings = field(default_factory=LookDownSettings)
    target_closing: TargetClosingSettings = field(default_factory=TargetClosingSettings)
    doorway_peek: PeekSettings = field(default_factory=lambda: PeekSettings(enabled=False))
    falcon: FalconParams = field(default_factory=lambda: FalconParams(burst_actions=10))
    multifloor: MultiFloorParams = field(default_factory=MultiFloorParams)

    def __post_init__(self):
        for key in ("map_size_m", "map_resolution_m", "action_time_s", "stop_distance_m", "body_height_m", "body_radius_m",
                    "preferred_clearance_m", "min_clearance_m", "landmark_dedupe_radius_m", "floor_plane_bound_m",
                    "step_down_m", "stair_margin_m"):
            value = getattr(self, key)
            if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value) or value <= 0:
                raise ValueError("%s must be positive and finite" % key)
        if self.stair_margin_m < self.body_radius_m:
            raise ValueError("stair_margin_m cannot be smaller than the robot radius")
        if type(self.allow_stair_traversal) is not bool:
            raise ValueError("allow_stair_traversal must be boolean; no silent fallback")
        if type(self.segmentation) is not bool:
            raise ValueError("segmentation must be boolean; no silent fallback")
        if (isinstance(self.landmark_footprint_iou, bool) or not math.isfinite(self.landmark_footprint_iou)
                or not 0 < self.landmark_footprint_iou <= 1):
            raise ValueError("landmark_footprint_iou must lie in (0, 1]")
        for name in ("landmark_max_misses", "landmark_miss_margin_px"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 0:
                raise ValueError("%s must be a non-negative integer" % name)
        for name in ("landmark_miss_min_range_m", "landmark_miss_max_range_m", "landmark_miss_viewpoint_m"):
            value = getattr(self, name)
            if isinstance(value, bool) or not math.isfinite(value) or value <= 0:
                raise ValueError("%s must be positive and finite" % name)
        if self.landmark_miss_max_range_m <= self.landmark_miss_min_range_m:
            raise ValueError("landmark_miss_max_range_m must exceed landmark_miss_min_range_m")
        for key in ("depth_stride", "graph_period_steps", "transit_graph_period_steps", "replan_steps",
                    "target_memory_steps", "max_rooms", "warmup_steps", "goal_bump_retire", "goal_bump_window"):
            if type(getattr(self, key)) is not int or getattr(self, key) <= 0:
                raise ValueError("%s must be a positive integer" % key)
        if type(self.sticky_bump_actions) is not int or self.sticky_bump_actions < 0:
            raise ValueError("sticky_bump_actions must be a non-negative integer (0 keeps a bar for the episode)")
        if (isinstance(self.goal_bump_match_m, bool) or not math.isfinite(self.goal_bump_match_m)
                or self.goal_bump_match_m <= 0):
            raise ValueError("goal_bump_match_m must be positive and finite")
        if not 0 <= self.detection_confidence <= 1 or type(self.seed) is not int:
            raise ValueError("Invalid confidence or seed")
        if self.preferred_clearance_m < self.body_radius_m:
            raise ValueError("Preferred clearance cannot be smaller than the robot radius")
        if not self.body_radius_m <= self.min_clearance_m <= self.preferred_clearance_m:
            raise ValueError("min_clearance_m must lie between body_radius_m and preferred_clearance_m")
        if isinstance(self.detour_factor, bool) or not math.isfinite(self.detour_factor) or self.detour_factor < 0:
            raise ValueError("detour_factor must be finite and non-negative (0 disables the squeeze rule)")
        if self.local_exploration not in ("frontier", "falcon"):
            raise ValueError("local_exploration must be frontier or falcon; no silent fallback")
        if (isinstance(self.centreline_weight, bool) or not math.isfinite(self.centreline_weight)
                or self.centreline_weight < 0):
            raise ValueError("centreline_weight must be finite and non-negative (0 disables the medial-axis layer)")
        if (isinstance(self.channel_half_width_m, bool) or not math.isfinite(self.channel_half_width_m)
                or self.channel_half_width_m <= self.body_radius_m):
            raise ValueError("channel_half_width_m must be finite and wider than the body radius")
        for name, cls in (("kinematic", KinematicSettings), ("doorway_scrape", ScrapeSettings),
                          ("doorway_look_down", LookDownSettings)):
            value = getattr(self, name)
            if isinstance(value, dict):
                value = cls(**value)
                object.__setattr__(self, name, value)
            if not isinstance(value, cls):
                raise ValueError("%s must be %s or parameter overrides" % (name, cls.__name__))
        if not self.body_radius_m <= self.kinematic.lattice_floor_m <= self.min_clearance_m:
            raise ValueError("kinematic.lattice_floor_m must lie between body_radius_m and min_clearance_m")
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
