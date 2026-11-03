"""Spawn-floor confinement when stair traversal is forbidden (``RPTSettings.allow_stair_traversal``).

Standard ZSON benchmarks (Gibson, HM3D) score an episode on the storey the
agent spawns on, so by default the search must never leave it. The guard
does this at the one place every consumer shares -- the world the policy
plans on -- rather than in each frontier, opening, peek and route module:

* the footprint of every SEEN staircase near the storey's level
  (:func:`peek_stairs.stair_peek_mask`, the same cells the room partition
  already excludes) is written occupied, so no frontier survives on a
  landing and no A* route crosses a flight;
* every cell whose observed surface lies more than ``drop_m`` BELOW the
  storey plane -- a descending flight, a stairwell, a ledge -- is written
  occupied too. The observed map's free band is the plane +/- 0.10 m and its
  obstacle band starts 0.15 m above it, so a drop is otherwise *unknown*,
  and unknown next to free is a frontier: that is how a floor-wide frontier
  walked the Pomaria agent down a flight it had never chosen. Rises beyond
  the first tread are obstacles already. ``drop_m`` is
  ``RPTSettings.step_down_m`` (0.30 m): the plane is the pose height and
  the floor the camera sees lies up to 0.3 m under it in some scenes, so a
  tighter bar masks the storey itself (Corozal/000142, 2026-10-14, 3,316
  cells); a shallow flight inside the tolerance (Darden/000034's three
  treads at -0.11, -0.18, -0.21 m) is the open item;
* **a verdict is fixed when the cell is measured; the plane moving changes
  nothing the camera has not seen again** (since 2026-11-03). Per floor and
  cell the guard keeps the LOWEST height a depth point ever landed at
  (:attr:`landed_z`), the lowest height a depth ray ever ran through the
  cell's column without meeting a surface (:attr:`clear_z`), and the three
  thresholds in force the last time the cell was measured -- the drop line
  (:attr:`line_z`), the crossing line (:attr:`deep_z`) and the top of the
  surface band (:attr:`band_z`). :meth:`_cut` compares every cell with ITS
  OWN thresholds: a drop where the landing is below the cell's drop line, a
  crossing where the ray ran below the cell's step reach, a surface where
  the landing lies between the drop line and the obstacle band. A cell
  measured again takes the thresholds of the plane in force then, and its
  verdict may change -- that is an observation. A re-measure of the storey
  plane (below), a rebase from a provisional spawn storey, the storey
  confirmed: none of them touches a cell the camera does not see. Before:
  the guard re-thresholded EVERY measured cell of the grid against the
  plane in force on every action (and until the same day emptied the
  layers at a re-measure), and the plane follows the agent's own level
  (``_follow_step_plane``). 00878-XB4GS9ShBRE/000000 is split-level --
  floors at 0.00, +0.20 and +0.30 m and a sunken room at -0.20 m -- and
  re-measured 14 times in 500 actions; each time 270-730 cells changed, none
  of them in the camera's view: at action 84 (plane +0.20 -> 0.00) the
  sunken room, 0.40 m under the hall and a drop from there, was released;
  at 150 (plane -> +0.30) it was 0.50 m under and the whole room was painted
  occupied, its three rooms gone from the search; at 159 (plane -> 0.00)
  released again -- and the agent went back to places it had searched. The
  ledger (``world_model/map_ledger.py``) counts changes out of view per
  action; the regression is ``test_spawn_floor_guard.py``'s out-of-view
  stability test;
* **a drop along the ray** (since 2026-10-31, :mod:`drop_rays`): a depth ray
  that lands deeper than the storey's step reach (:attr:`step_reach_m`, and
  never less than ``drop_m``) ran that deep, without meeting a surface, from
  where it sank under that line -- the cells it crossed there are drops too,
  their surface deeper than any step the storey owns. Nothing else measures
  them: the observed map's free-space rays toward the walls beyond a
  stairwell carve its treads free, and a level camera never sees treads that
  close (00835-q3zU7Yy5E5s/000006: a frontier route down a flight whose
  every cell was free and none a drop, then 18 actions of ``RETREAT``). A
  sunken floor inside the reach (00878/000035, 0.40 m down) is crossed only
  at its own height, never below the line. Two more limits: a cell where a
  depth point ever landed between the drop line and the obstacle band
  (``surfaced``) is not a crossing drop -- the ray may have run on under the
  slab beyond the stairwell, and that floor is the storey -- and a camera
  under the storey plane (down a flight by accident) draws no crossing;
* a goal is refused when its cell is masked, or when the storey the agent
  stands on is further than ``floor_plane_bound_m`` from the spawn plane
  (``Z_spawn +/- 0.5 m``) -- then nothing on it is a destination;
* **a spawn on the stairs** (since 2026-10-26): the spawn plane is the first
  pose's height, which on a tread is the middle of a flight -- nothing is
  level with it and every flight off it is masked, so 00873/000002 stood
  boxed in for 447 actions. While the coordinator still calls the spawn
  storey provisional (``MultiFloorSearch.storey_confirmed`` False: never
  walked ``stable_distance_m`` level), the first storey the agent settles on
  becomes the spawn plane (:meth:`SpawnFloorGuard.rebase`): a settled
  plateau of floor size is adopted at once rather than after
  ``HALF_LEVEL_ACTIONS``, and a storey the atlas creates on its own
  (``min_floor_separation_m`` away) is not "off the spawn plane" but the
  plane itself;
* a **half-level** (since 2026-10-07): a split-level house has floor
  0.5-1.5 m off the spawn plane that is neither another storey nor a
  staircase -- the agent walked up a step or a ramp onto it. The atlas
  reads it as a landing and stays in transition, which froze the map and
  left every route unplannable (Klickitat 2026-10-05: 361 idle turns).
  After ``HALF_LEVEL_ACTIONS`` actions on such a settled plateau the guard
  has the atlas adopt it as a level of its own (:meth:`FloorAtlas.adopt_level`)
  and treats it as part of the spawn storey: goals on it are allowed, and
  drops are measured from the lower of the two planes so the storey below
  stays walkable. A plateau ``min_floor_separation_m`` or more from the
  spawn plane is another storey and is never adopted.

* **steps** (since 2026-10-31, ``RPTSettings.step_surface_band_m``): the
  map's bands are fixed offsets from the plane -- free within 0.10 m,
  obstacle from 0.15 m above, drop beyond ``drop_m`` below -- so a single
  step up, a raised dais or a sunken living room was a wall or a drop, and a
  spawn on a tread a box. Support the stair terrain can WALK to from the
  agent (``StairTerrain.step_surfaces``: risers of at most ``max_step_m``,
  clear to body height) within the band of the plane is kept per floor, and
  the part of it within :attr:`SpawnFloorGuard.step_reach_m` of the plane --
  the whole band from a provisional spawn storey, the atlas's
  ``departure_m`` from a storey walked level -- is free on the planning
  grid, never a drop, never a stair mask, and where the closing's standoffs
  may lie. One bound for all four, and the atlas's own: from a storey walked
  level, support further off than ``departure_m`` is where the atlas starts
  a transition and the trail retreat walks the agent back, so it is not
  offered (00835/000006 on the first cut, which freed the whole band: a
  frontier route down a flight's treads toward a frontier 1.7 m below, 0.59 m
  off the plane in one stride, then 18 actions of ``RETREAT``). Nothing
  reachable is withheld by it: on habitat-sim's recomputed navmesh no
  single-storey HM3D v1 val episode that starts on a level of floor size
  (1,495 of 1,586) has a shortest path more than 0.45 m off the start
  height, and every one whose goal lies 0.45-0.8 m off (20) starts on a
  tread, a landing or a ramp -- the provisional spawn storey, whose
  departure is walked as a traversal and rebased, not retreated from. The
  GT-oracle sweep of 2026-10-28 solved 14 % of the single-storey episodes
  whose shortest path changes height by 0.25-1.0 m. The step's verdict --
  within the reach of the plane or not -- is taken when the terrain reports
  the cell (:attr:`step_ok`, since 2026-11-03) and kept until it reports it
  again: the terrain reports what it can reach from where the agent stands,
  within its own radius and memory, so a plane re-measured on the far side
  of the house re-judges no step here.
  When the agent has stood level on such a plane of floor size, off the
  storey plane by more than a step but inside ``departure_m`` (where the
  atlas starts no transition), the atlas re-measures the storey there
  (``FloorAtlas.remeasure``): on a sunken floor measured from the step above
  it, a toilet's lid lay in the free band. The re-measure sets the plane for
  what is measured from then on; nothing measured before is re-judged by it
  (above);

Two rules keep the mask from trapping the agent itself (since 2026-10-08;
Collierville/000000 walked past the foot of a flight 0.6 m from its
centreline, the capsule covered its cell as a GENUINE obstacle -- which the
planner's start relaxation never opens -- and the fallback held for 309
actions): the floor run of a flight within ``peek_stairs.FLOOR_RUN_M`` of
the storey plane is not drawn, and the cells the agent stands on or has
stood on this storey (``peek_stairs.walked_exemption``: the sight ledger's
trail plus the current pose, body radius and one cell around each) are
carved out of the combined mask -- the agent was physically there, and
nothing written for planning may outvote that (``walked_exempt_cells``).

All of this is a constraint on what the agent may choose, not knowledge
about the target. Under the default ``stair_source == "observed"`` (since
2026-10-21) nothing here is privileged at all: the flights masked are the
ones the robot's own depth measured rising (``stair_terrain`` candidates,
placed as portals by the coordinator), the drops are its own depth reads.
Under the ``ground_truth`` ablation the one privileged input is the stair
geometry the perfect detector hands the coordinator once a staircase is in
frame. With ``allow_stair_traversal`` true the guard is inert and returns
the world untouched.
"""
from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np

from sparx_agency.core.planning.environment import OccupancyGrid2D
from sparx_agency.core.planning.objnav.camera_geometry import backproject_depth, camera_position_world
from sparx_agency.core.planning.object_search.multifloor.drop_rays import cells_crossed_below
from sparx_agency.core.planning.object_search.world_model.peek_stairs import stair_peek_mask, walked_exemption

#: Refusal events kept for the recording; the counters are complete.
MAX_EVENTS = 50
#: Actions a settled plateau inside the separation rule may stay "in transition" before it is adopted.
HALF_LEVEL_ACTIONS = 30
#: Consecutive actions within this height of each other, off the plane by more than the atlas's ``stable_height_m``
#: (the height that is still "level" with it), that make the agent "standing level" on another plane for a re-measure.
#: The floor was a fixed 0.15 m until 2026-11-02: 00843/000083 spawned 0.149 m above its storey, stepped down onto it
#: and stood there unconfirmed -- not level with the plane (more than 0.12 m off), not off it enough to re-measure.
REMEASURE_SAMPLES = 3
REMEASURE_LEVEL_M = 0.05
#: Consecutive actions at one height (within the atlas's ``stable_height_m``) that make a plateau of a spot the
#: agent cannot translate on -- the atlas's own test wants ``stable_distance_m`` of walking, and the map is frozen
#: while the transition stands (since 2026-10-28).
STALLED_SAMPLES = 8
#: Height above the storey plane where the observed map's obstacle band starts (``ObservedMap.update``'s ``z_min``):
#: a depth point between the drop line and it is a surface the storey may stand on.
OBSTACLE_BAND_M = 0.15


class SpawnFloorGuard:
    """The spawn plane, the impassable cells of the floor in force, and the goal gate."""

    def __init__(self, policy):
        self.policy = policy
        settings = policy.settings
        self.enabled: bool = not settings.allow_stair_traversal
        self.bound_m: float = settings.floor_plane_bound_m
        # Below the plane by more than ``RPTSettings.step_down_m`` (0.30 m, the atlas's same-floor tolerance) is a
        # drop. Not lower: the plane is the pose height, and the floor the camera sees lies up to 0.3 m under it
        # in some scenes (Corozal, 2026-10-14: 0.15 m masked the agent's own storey).
        self.drop_m: float = settings.step_down_m
        self.spawn_z: Optional[float] = None
        self.spawn_floor: Optional[int] = None  # the atlas floor the spawn plane belongs to (rebased with it)
        # The measurements, heights per cell that no plane enters (NaN: never measured) -- see the module docstring:
        self.landed_z: Dict[int, np.ndarray] = {}      # floor id -> the lowest height a depth point landed at
        self.clear_z: Dict[int, np.ndarray] = {}       # floor id -> the lowest height a ray ran through the column, unmet
        # The thresholds in force when the cell was LAST measured (NaN: never): its drop line, its crossing line
        # (the plane less the step reach) and the top of its surface band. A cell is judged against these alone.
        self.line_z: Dict[int, np.ndarray] = {}
        self.deep_z: Dict[int, np.ndarray] = {}
        self.band_z: Dict[int, np.ndarray] = {}
        # Their cut, refreshed for the floor in force on every ``observe`` (``_cut``), kept for the record and the tests:
        self.drops: Dict[int, np.ndarray] = {}         # floor id -> landed below the cell's drop line
        self.crossed: Dict[int, np.ndarray] = {}       # floor id -> a ray ran below the cell's crossing line
        self.surfaced: Dict[int, np.ndarray] = {}      # floor id -> landed above the drop line, under the obstacle band
        self.step_band_m: float = float(settings.step_surface_band_m)
        self.steps: Dict[int, np.ndarray] = {}     # floor id -> walkable step support heights (NaN: none)
        # The plane the camera last measured the cell under (NaN: never) -- the step's reach is judged against it,
        # as the drops are judged against their own thresholds; a cell the camera never measured is anchored to the
        # plane in force when the terrain first reported it.
        self.step_ref_z: Dict[int, np.ndarray] = {}
        self._level_heights: List[float] = []
        self.mask: Optional[np.ndarray] = None
        self.freed: Optional[np.ndarray] = None    # the step cells written free on the last action (the ledger reads it)
        self.off_plane: bool = False
        self.half_levels: set = set()          # atlas floor ids adopted as half-levels of the spawn storey
        self.events: List[dict] = []
        self.stats = {"goals_refused": 0, "drop_cells": 0, "drop_cells_crossed": 0, "stair_cells": 0, "off_plane_actions": 0,
                      "half_levels_adopted": 0, "walked_exempt_cells": 0, "rebased": 0, "stalled_adoptions": 0,
                      "stalled_returns": 0, "step_cells": 0, "step_cells_freed": 0, "step_cells_withheld": 0,
                      "step_drop_exempt": 0, "step_stair_exempt": 0, "plane_remeasured": 0}
        self._transition_heights: List[float] = []   # the pose height on every action of the transition in force
        self.stair_margins_m = {}               # portal id -> capsule margin drawn, metres (``adaptive_capsule``)

    # -- per action -----------------------------------------------------------
    def observe(self, obs, world):
        """Record the spawn height, accumulate observed drops, and return the world the policy may plan on."""
        if self.spawn_z is None:
            self.spawn_z = float(obs.pose.z)
            self.spawn_floor = int(self.policy.mapping.floor_id)
        if not self.enabled:
            self.mask = None
            return world
        if self.provisional and int(self.policy.mapping.floor_id) != self.spawn_floor:
            # The atlas arrived on a storey of its own from a spawn storey never walked level: THIS is the spawn storey.
            self.rebase(obs, int(self.policy.mapping.floor_id), float(self.policy.mapping.atlas.elevation_m),
                        "the atlas settled on a storey from the provisional spawn storey")
        self._adopt_half_level(obs)
        self._follow_step_plane(obs)
        self._track_plane(obs)
        floor_id = self.policy.mapping.floor_id
        landed_z = self._heights(self.landed_z, floor_id, world)
        clear_z = self._heights(self.clear_z, floor_id, world)
        line_z = self._heights(self.line_z, floor_id, world)
        deep_z = self._heights(self.deep_z, floor_id, world)
        band_z = self._heights(self.band_z, floor_id, world)
        step_ref = self._heights(self.step_ref_z, floor_id, world)
        self._mark_drops(obs, world, landed_z, clear_z, line_z, deep_z, band_z, step_ref)
        landed, crossed, surfaced = self._cut(floor_id, landed_z, clear_z, line_z, deep_z, band_z)
        # A crossing is a drop only where no surface above the drop line was ever measured: a ray that sank into a
        # stairwell can run on under the floor slab beyond it, and that floor is the storey.
        crossed_only = crossed & ~surfaced & ~landed
        drops = landed | crossed_only
        steps = self._step_surfaces(world)
        walkable = self._walkable(steps)
        self.stats["step_cells_withheld"] = int(np.isfinite(steps).sum()) - int(walkable.sum())
        self.stats["step_drop_exempt"] = int((drops & walkable).sum())
        mask = drops & ~walkable
        margins = {}
        stairs = (stair_peek_mask(self.policy, world, obs.pose, carve=False, report=margins)
                  if getattr(self.policy, "building", None) is not None else None)
        self.stair_margins_m = margins
        if stairs is not None:
            self.stats["step_stair_exempt"] = int((stairs & walkable).sum())
            mask |= stairs & ~walkable
        exempt = 0
        if mask.any():
            # Where the agent stands or has stood is passable, whatever the depth said about it
            # (a drop painted under a step the agent then took): never a cell A* cannot start from.
            walked = walked_exemption(self.policy, world, obs.pose)
            exempt = int((mask & walked).sum())
            mask &= ~walked
        self.stats["drop_cells"] = int(landed.sum())
        self.stats["drop_cells_crossed"] = int(crossed_only.sum())
        self.stats["stair_cells"] = 0 if stairs is None else int(stairs.sum())
        self.stats["walked_exempt_cells"] = exempt
        self.mask = mask if mask.any() else None
        free = walkable & ~mask & (world.grid != world.values.free)
        touched = getattr(self.policy.mapping, "_touched", {}).get(self.policy.mapping.floor_id) or {}
        for tx, ty in touched:
            if 0 <= ty < free.shape[0] and 0 <= tx < free.shape[1]:
                free[ty, tx] = False                # what the body touched twice outranks the terrain too
        self.stats["step_cells_freed"] = int(free.sum())
        self.freed = free if free.any() else None
        if self.mask is None and not free.any():
            return world
        grid = world.grid.copy()
        grid[free] = world.values.free
        if self.mask is not None:
            grid[self.mask] = world.values.occupied
        return OccupancyGrid2D(grid, world.params, values=world.values)

    def goal_allowed(self, obs, world, goal, kind) -> bool:
        """False for a goal off the spawn plane or on masked cells; every refusal is counted."""
        if not self.enabled:
            return True
        if self.off_plane:
            return self._refuse(obs, goal, kind, "storey %.2f m from the spawn plane" % self._plane_offset())
        if self.mask is not None:
            gx, gy = world.world_to_grid(goal[0], goal[1])
            if world.in_bounds(gx, gy) and self.mask[gy, gx]:
                return self._refuse(obs, goal, kind, "goal on a staircase or an observed drop")
        return True

    def diagnostics(self) -> dict:
        return {"enabled": self.enabled, "spawn_z_m": self.spawn_z, "spawn_floor": self.spawn_floor,
                "floor_plane_bound_m": self.bound_m,
                "drop_m": self.drop_m, "off_plane": self.off_plane, "stats": dict(self.stats),
                "stair_margins_m": dict(self.stair_margins_m),
                "half_levels": sorted(self.half_levels), "events": list(self.events)}

    # -- the provisional spawn storey -----------------------------------------
    @property
    def provisional(self) -> bool:
        """Whether the spawn storey may be a tread: observed stairs, and the agent never walked it level (see the module docstring)."""
        building = getattr(self.policy, "building", None)
        return building is not None and building.ground_truth is None and not building.storey_confirmed

    @property
    def step_reach_m(self) -> float:
        """How far off the storey plane walkable step support is still the storey (see the module docstring).

        The whole ``step_surface_band_m`` from a provisional spawn storey: a
        departure from it is the agent leaving the flight it spawned on,
        walked as a traversal to the storey it settles on. The atlas's
        ``departure_m`` from a storey walked level: past it the atlas starts a
        transition and the trail retreat walks the agent back, so the guard
        never offers what the retreat would undo.
        """
        if self.provisional:
            return self.step_band_m
        return min(self.step_band_m, float(self.policy.mapping.atlas.params.departure_m))

    def rebase(self, obs, floor, height, reason, after_actions=None):
        """Make ``height`` -- the storey the agent settled on from a provisional spawn storey -- the spawn plane."""
        offset = float(height) - float(self.spawn_z) if self.spawn_z is not None else 0.0
        self.spawn_z, self.spawn_floor, self.off_plane = float(height), int(floor), False
        self.half_levels.discard(int(floor))           # the plane itself, not a half-level of a storey never stood on
        # The mapping takes the atlas's elevation on its next frame; the drop plane and the offset need it now.
        self.policy.mapping._anchor = float(height)
        self.stats["rebased"] += 1
        self._record({"action": obs.step, "event": "spawn_plane_rebased", "floor_id": int(floor),
                      "height_m": round(float(height), 3), "offset_m": round(offset, 3),
                      "after_actions": after_actions, "reason": reason})

    # -- internals ------------------------------------------------------------
    def _plane_offset(self) -> float:
        plane = getattr(self.policy.mapping, "_anchor", None)
        if plane is None or self.spawn_z is None:
            return 0.0
        if self.policy.mapping.floor_id in self.half_levels:
            return 0.0                              # an adopted half-level is the spawn storey
        return float(plane) - self.spawn_z

    def _adopt_half_level(self, obs):
        """Adopt a settled plateau the atlas cannot resolve as a level of the spawn storey (see the module docstring).

        From a PROVISIONAL spawn storey the wait is skipped once the plateau
        is of floor size (``min_floor_area_m2`` of connected level support
        under the agent, the atlas's own storey criterion): a tread is no
        storey to wait thirty actions on, and the plateau becomes the spawn
        plane itself rather than a half-level of it.
        """
        atlas = getattr(self.policy.mapping, "atlas", None)
        if atlas is None or not atlas.params.enabled or not atlas.in_transition or self.spawn_z is None:
            self._transition_heights = []
            return
        self._transition_heights.append(float(obs.pose.z))
        height = atlas.destination_height_m
        stalled = height is None
        if stalled:
            height = self._stalled_plateau(obs, atlas)
            if height is None:
                return
        match_m = float(self.policy.settings.multifloor.floor_match_m)
        known = [f for f in atlas.floors.values() if abs(f.elevation_m - float(height)) <= match_m]
        building = getattr(self.policy, "building", None)
        if building is not None and getattr(building.switch, "state", None) == "traverse":
            # The spawn switch arrives by its own rule (``spawn_switch.accept_arrival``) or retreats: every floor
            # known when it set off is one it left, and settling on the spawn's own tread on the way past took
            # 00843/000022 back to the height it started at (v3, 2026-11-02).
            return
        if known:
            # The plateau IS a floor the atlas knows -- the source of a traversal that never took a tread. Adopting
            # it made a second "floor" 0.01-0.12 m from the first (00802/000056, 00814/000004, 00800/000086,
            # 2026-11-02); after the half-level wait the transition settles back on that floor instead.
            if atlas.transition_ticks >= HALF_LEVEL_ACTIONS:
                floor = min(known, key=lambda f: abs(f.elevation_m - float(height)))
                atlas.settle(obs.pose, float(floor.elevation_m))
                self._transition_heights = []
                self._record({"action": obs.step, "event": "transition_settled_on_known_floor", "floor_id": int(floor.id),
                              "height_m": round(float(height), 3), "after_actions": int(HALF_LEVEL_ACTIONS)})
            return
        provisional = self.provisional
        floor_size = (atlas.plateau_observer is not None
                      and atlas.plateau_observer(obs.pose) >= float(self.policy.settings.multifloor.min_floor_area_m2))
        if atlas.transition_ticks < HALF_LEVEL_ACTIONS and not (provisional and floor_size):
            return
        if provisional and atlas.plateau_observer is not None and not floor_size:
            # From a stair spawn a plateau smaller than a floor is a landing, however long the agent stands on it:
            # adopted as the spawn storey it armed the switch, which walked 00843/000022 back down to its spawn
            # tread (v3, 2026-11-02). The traversal finishes, or retreats, on its own terms.
            return
        separation = float(self.policy.settings.multifloor.min_floor_separation_m)
        if abs(float(height) - self.spawn_z) >= separation:
            return                                  # another storey: confined, as the protocol says
        waited = int(atlas.transition_ticks)              # adopt_level resets the atlas's count
        floor = atlas.adopt_level(obs.pose, float(height))
        self._transition_heights = []
        if stalled:
            self.stats["stalled_adoptions"] += 1
        if provisional:
            self.rebase(obs, floor, float(height), "settled on a storey from the provisional spawn storey", waited)
            return
        self.half_levels.add(int(floor))
        self.stats["half_levels_adopted"] += 1
        self._record({"action": obs.step, "event": "half_level_adopted", "floor_id": int(floor),
                      "height_m": round(float(height), 3), "offset_m": round(float(height) - self.spawn_z, 3),
                      "after_actions": waited, "stalled": stalled})

    def _stalled_plateau(self, obs, atlas):
        """The height of a transition that cannot settle because the agent cannot move, once it has stood long enough.

        The atlas confirms a plateau on ``stable_samples`` TRANSLATED
        samples; the map is not integrated while a transition stands, so
        an agent whose planning world froze on a step or a ledge takes no
        step that would settle it, and until 2026-10-28 the transition
        stood for the rest of the episode (the ``RETREAT`` episodes of the
        GT-oracle sweep: SR 6.6 %). After ``HALF_LEVEL_ACTIONS`` actions in
        transition, with the retreat not in force and the last
        ``STALLED_SAMPLES`` pose heights within ``stable_height_m`` of each
        other, that height is the plateau. Within ``floor_match_m`` of the
        storey it is the storey, confirmed in place (``stalled_returns``);
        otherwise the caller adopts it as any settled plateau
        (``stalled_adoptions``). None while the agent still moves in height,
        the retreat still walks, or the wait is not over.
        """
        building = getattr(self.policy, "building", None)
        if building is not None and getattr(building, "retreating", False):
            return None
        if atlas.transition_ticks < HALF_LEVEL_ACTIONS or len(self._transition_heights) < STALLED_SAMPLES:
            return None
        heights = self._transition_heights[-STALLED_SAMPLES:]
        if max(heights) - min(heights) > float(atlas.params.stable_height_m):
            return None
        height = sum(heights) / len(heights)
        if abs(height - float(atlas.elevation_m)) <= float(atlas.params.floor_match_m):
            # The storey itself, a step's worth off its plane: the return the plateau test never confirmed.
            if atlas.settle(obs.pose, float(atlas.elevation_m)):
                self.stats["stalled_returns"] += 1
                self._record({"action": obs.step, "event": "transition_settled_in_place", "height_m": round(height, 3),
                              "offset_m": round(height - float(atlas.elevation_m), 3),
                              "after_actions": int(self.policy.mapping.atlas.transition_ticks)})
                self._transition_heights = []
            return None
        return height

    def _track_plane(self, obs):
        offset = self._plane_offset()
        off_plane = abs(offset) > self.bound_m
        if off_plane:
            self.stats["off_plane_actions"] += 1
        if off_plane != self.off_plane:
            self._record({"action": obs.step, "event": "spawn_plane_left" if off_plane else "spawn_plane_regained",
                          "plane_offset_m": round(offset, 3), "height_m": round(float(obs.pose.z), 3)})
        self.off_plane = off_plane

    def _step_surfaces(self, world) -> np.ndarray:
        """The floor's walkable step support heights on the planning grid, NaN elsewhere; this action's terrain added.

        The terrain reports what it can reach from where the agent stands,
        out of its own memory -- not what the camera sees this action -- so
        a report changes a cell's HEIGHT only. The reference the height is
        judged against (:attr:`step_ref_z`) is the plane the camera last
        measured the cell under (``_mark_drops``); a cell the camera never
        measured is anchored to the plane in force at its first report.
        """
        floor_id = self.policy.mapping.floor_id
        heights = self.steps.get(floor_id)
        if heights is None or heights.shape != world.grid.shape:
            heights = np.full(world.grid.shape, np.nan, dtype=np.float32)
            self.steps[floor_id] = heights
        reference = self._heights(self.step_ref_z, floor_id, world)
        terrain = getattr(getattr(self.policy, "building", None), "terrain", None)
        plane = getattr(self.policy.mapping, "_anchor", None)
        atlas = getattr(self.policy.mapping, "atlas", None)
        if terrain is not None and plane is not None and not (atlas is not None and atlas.in_transition):
            xy, h = terrain.step_surfaces(float(plane), self.step_band_m)
            if len(h):
                xs = np.floor((xy[:, 0] - world.origin_x) / world.resolution).astype(int)
                ys = np.floor((xy[:, 1] - world.origin_y) / world.resolution).astype(int)
                inside = (xs >= 0) & (xs < world.width) & (ys >= 0) & (ys < world.height)
                xs, ys = xs[inside], ys[inside]
                heights[ys, xs] = h[inside]
                unanchored = ~np.isfinite(reference[ys, xs])
                reference[ys[unanchored], xs[unanchored]] = float(plane)
        self.stats["step_cells"] = int(np.isfinite(heights).sum())
        return heights

    def _walkable(self, steps) -> np.ndarray:
        """The step cells that are this storey: support within :attr:`step_reach_m` of the plane the cell was measured under.

        The reference is the cell's own (:attr:`step_ref_z`), never the plane
        in force: a storey re-measured on another level re-judges no step the
        camera is not looking at. The reach is the storey's -- the whole band
        while the spawn storey is provisional, ``departure_m`` once it is
        confirmed by walking it.
        """
        reference = self.step_ref_z.get(self.policy.mapping.floor_id)
        if reference is None or reference.shape != steps.shape:
            return np.zeros(steps.shape, dtype=bool)
        with np.errstate(invalid="ignore"):
            return np.isfinite(steps) & np.isfinite(reference) & (np.abs(steps - reference) <= self.step_reach_m)

    def _follow_step_plane(self, obs):
        """Re-measure the storey on the plane the agent stands level on, a step or more off it but inside ``departure_m``.

        The plane must be of floor size (the terrain's connected flat support
        under the agent, ``min_floor_area_m2``). From a provisional spawn
        storey -- a spawn on a tread -- the spawn plane moves with it; from a
        storey walked level, the spawn plane stays and the bound measures
        the step against it. Nothing already measured is re-judged here: the
        new plane sets the thresholds of what the camera measures from now
        on (:meth:`_mark_drops`), and a cell keeps the verdict of its last
        measurement until it is seen again (the module docstring). Until
        2026-11-03 the layers were emptied here, then for a few hours re-cut
        whole against the new plane -- either way 00878-XB4GS9ShBRE/000000
        rewrote hundreds of cells out of view on each of its re-measures.
        """
        mapping = self.policy.mapping
        atlas = getattr(mapping, "atlas", None)
        terrain = getattr(getattr(self.policy, "building", None), "terrain", None)
        if atlas is None or not atlas.params.enabled or atlas.in_transition or terrain is None or not atlas.floors:
            self._level_heights = []
            return
        plane = float(atlas.elevation_m)
        offset = float(obs.pose.z) - plane
        if not float(atlas.params.stable_height_m) < abs(offset) <= float(atlas.params.departure_m):
            self._level_heights = []
            return
        self._level_heights = (self._level_heights + [float(obs.pose.z)])[-REMEASURE_SAMPLES:]
        if (len(self._level_heights) < REMEASURE_SAMPLES
                or max(self._level_heights) - min(self._level_heights) > REMEASURE_LEVEL_M):
            return
        area = float(terrain.plateau_area(obs.pose))
        if area < float(atlas.params.min_floor_area_m2):
            return
        height = sum(self._level_heights) / len(self._level_heights)
        provisional = self.provisional
        if not atlas.remeasure(height):
            return
        mapping._anchor = height
        if provisional and self.spawn_floor == int(mapping.floor_id):
            self.spawn_z = height
            building = getattr(self.policy, "building", None)
            if building is not None and building.flights_seen_from_spawn:
                # A step off a flight onto the floor: the flight's other end is the switch's storey (spawn_switch.py).
                building.switch.settled(obs, int(mapping.floor_id), plane, height, "remeasure")
        self._level_heights = []
        self.stats["plane_remeasured"] += 1
        self._record({"action": obs.step, "event": "storey_plane_remeasured", "floor_id": int(mapping.floor_id),
                      "height_m": round(height, 3), "offset_m": round(height - plane, 3),
                      "plateau_m2": round(area, 2), "spawn_plane_moved": bool(provisional)})

    @staticmethod
    def _heights(layers, floor_id, world) -> np.ndarray:
        """The floor's height layer on the planning grid, NaN where nothing was measured; created on first use."""
        layer = layers.get(floor_id)
        if layer is None or layer.shape != world.grid.shape:
            layer = np.full(world.grid.shape, np.nan, dtype=np.float32)
            layers[floor_id] = layer
        return layer

    def _drop_plane(self):
        """The plane the drop line hangs from, and the plane the obstacle band stands on (the storey's own).

        On an adopted half-level the drop line hangs from the lower of the two
        planes -- the storey below is not a drop -- while the obstacle band
        stays the half-level's.
        """
        plane = getattr(self.policy.mapping, "_anchor", None)
        if plane is None:
            return None, None
        band_plane = float(plane)
        if self.policy.mapping.floor_id in self.half_levels and self.spawn_z is not None:
            plane = min(float(plane), self.spawn_z)
        return float(plane), band_plane

    def _mark_drops(self, obs, world, landed_z, clear_z, line_z, deep_z, band_z, step_ref):
        """This frame's depth into the height layers, and this frame's thresholds onto the cells it measured.

        Where every point landed (the lowest kept), and how deep the rays of
        those landing below the drop line ran through the cells on their way
        without meeting a surface. Every cell a point landed in or a ray
        crossed takes the drop line, the crossing line and the surface band
        of the plane in force NOW, and the plane itself as the reference its
        step support is judged against (``step_ref``): it has been measured,
        and is judged by this measurement until the next.
        """
        plane, band_plane = self._drop_plane()
        if plane is None:
            return
        points = backproject_depth(obs.depth_m, obs.camera, obs.pose, self.policy.settings.depth_stride)
        if not len(points):
            return
        line = float(plane) - self.drop_m
        deep = float(plane) - max(self.drop_m, self.step_reach_m)
        band = float(band_plane) + OBSTACLE_BAND_M
        xs, ys, zs = self._cells(points, world)
        self._lower(landed_z, xs, ys, zs)
        line_z[ys, xs], deep_z[ys, xs], band_z[ys, xs], step_ref[ys, xs] = line, deep, band, float(band_plane)
        below = points[points[:, 2] < line]
        if not len(below):
            return
        camera = camera_position_world(obs.pose, obs.camera)
        if camera[2] <= float(plane):
            return                                  # a ray from under the storey's floor may run on under it
        # From where each ray sinks under the drop line to where it lands: the column above the ray is empty there,
        # so the surface under every cell crossed is lower than the ray's height in it. The crossing line cuts it
        # (``_cut``): crossed below the step reach, the surface is deeper than any step the storey owns (a sunken
        # floor 0.4 m down, 00878/000035, is crossed only at its height, never below it).
        xs, ys, zs = cells_crossed_below(camera, below, line, (world.origin_x, world.origin_y), world.resolution,
                                         world.grid.shape)
        self._lower(clear_z, xs, ys, zs)
        line_z[ys, xs], deep_z[ys, xs], band_z[ys, xs], step_ref[ys, xs] = line, deep, band, float(band_plane)

    def _cut(self, floor_id, landed_z, clear_z, line_z, deep_z, band_z):
        """Every cell against its own thresholds: ``(landed, crossed, surfaced)``, kept per floor for the record.

        Landed: the lowest point under the cell is below the cell's drop
        line. Crossed: a ray ran through the cell's column below the cell's
        crossing line (the plane less the step reach, never above the drop
        line). Surfaced: the lowest point lies between the drop line and the
        obstacle band -- a surface the storey may stand on, which outvotes a
        crossing (``observe``). The thresholds are the ones in force when
        the cell was last measured (``_mark_drops``), so the plane moving
        since changes no verdict here; a cell never measured has none.
        """
        with np.errstate(invalid="ignore"):
            landed = landed_z < line_z
            surfaced = (landed_z >= line_z) & (landed_z < band_z)
            crossed = clear_z < deep_z
        self.drops[floor_id], self.crossed[floor_id], self.surfaced[floor_id] = landed, crossed, surfaced
        return landed, crossed, surfaced

    @staticmethod
    def _lower(layer, xs, ys, zs):
        """Keep in ``layer`` the lowest of ``zs`` per cell against what it holds (NaN: nothing yet); repeats welcome."""
        if not len(xs):
            return
        flat = ys.astype(np.int64) * layer.shape[1] + xs.astype(np.int64)
        order = np.lexsort((zs, flat))                # by cell, the lowest height of each first
        flat, zs = flat[order], zs[order]
        first = np.ones(len(flat), dtype=bool)
        first[1:] = flat[1:] != flat[:-1]
        cells, lowest = flat[first], zs[first]
        rows, cols = np.divmod(cells, layer.shape[1])
        layer[rows, cols] = np.fmin(layer[rows, cols], lowest.astype(layer.dtype))

    @staticmethod
    def _cells(points, world):
        xs = np.floor((points[:, 0] - world.origin_x) / world.resolution).astype(int)
        ys = np.floor((points[:, 1] - world.origin_y) / world.resolution).astype(int)
        inside = (xs >= 0) & (xs < world.width) & (ys >= 0) & (ys < world.height)
        return xs[inside], ys[inside], points[inside, 2]

    def _refuse(self, obs, goal, kind, reason) -> bool:
        self.stats["goals_refused"] += 1
        self._record({"action": obs.step, "event": "goal_refused", "kind": kind, "reason": reason,
                      "goal": [round(float(goal[0]), 2), round(float(goal[1]), 2)]})
        return False

    def _record(self, event):
        if len(self.events) < MAX_EVENTS:
            self.events.append(event)
        elif len(self.events) == MAX_EVENTS:
            self.events.append({"event": "events_truncated", "limit": MAX_EVENTS})
