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
from sparx_agency.core.planning.objnav.camera_geometry import backproject_depth
from sparx_agency.core.planning.object_search.world_model.peek_stairs import stair_peek_mask, walked_exemption

#: Refusal events kept for the recording; the counters are complete.
MAX_EVENTS = 50
#: Actions a settled plateau inside the separation rule may stay "in transition" before it is adopted.
HALF_LEVEL_ACTIONS = 30
#: Consecutive actions at one height (within the atlas's ``stable_height_m``) that make a plateau of a spot the
#: agent cannot translate on -- the atlas's own test wants ``stable_distance_m`` of walking, and the map is frozen
#: while the transition stands (since 2026-10-28).
STALLED_SAMPLES = 8


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
        self.drops: Dict[int, np.ndarray] = {}
        self.mask: Optional[np.ndarray] = None
        self.off_plane: bool = False
        self.half_levels: set = set()          # atlas floor ids adopted as half-levels of the spawn storey
        self.events: List[dict] = []
        self.stats = {"goals_refused": 0, "drop_cells": 0, "stair_cells": 0, "off_plane_actions": 0,
                      "half_levels_adopted": 0, "walked_exempt_cells": 0, "rebased": 0, "stalled_adoptions": 0,
                      "stalled_returns": 0}
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
        self._track_plane(obs)
        drops = self._drops_for(self.policy.mapping.floor_id, world)
        self._mark_drops(obs, world, drops)
        mask = drops.copy()
        margins = {}
        stairs = (stair_peek_mask(self.policy, world, obs.pose, carve=False, report=margins)
                  if getattr(self.policy, "building", None) is not None else None)
        self.stair_margins_m = margins
        if stairs is not None:
            mask |= stairs
        exempt = 0
        if mask.any():
            # Where the agent stands or has stood is passable, whatever the depth said about it
            # (a drop painted under a step the agent then took): never a cell A* cannot start from.
            walked = walked_exemption(self.policy, world, obs.pose)
            exempt = int((mask & walked).sum())
            mask &= ~walked
        self.stats["drop_cells"] = int(drops.sum())
        self.stats["stair_cells"] = 0 if stairs is None else int(stairs.sum())
        self.stats["walked_exempt_cells"] = exempt
        self.mask = mask if mask.any() else None
        if self.mask is None:
            return world
        grid = world.grid.copy()
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
        provisional = self.provisional
        if atlas.transition_ticks < HALF_LEVEL_ACTIONS:
            if (not provisional or atlas.plateau_observer is None
                    or atlas.plateau_observer(obs.pose) < float(self.policy.settings.multifloor.min_floor_area_m2)):
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

    def _drops_for(self, floor_id, world) -> np.ndarray:
        drops = self.drops.get(floor_id)
        if drops is None or drops.shape != world.grid.shape:
            drops = np.zeros(world.grid.shape, dtype=bool)
            self.drops[floor_id] = drops
        return drops

    def _mark_drops(self, obs, world, drops):
        plane = getattr(self.policy.mapping, "_anchor", None)
        if plane is None:
            return
        if self.policy.mapping.floor_id in self.half_levels and self.spawn_z is not None:
            plane = min(float(plane), self.spawn_z)   # the storey below an adopted half-level is not a drop
        points = backproject_depth(obs.depth_m, obs.camera, obs.pose, self.policy.settings.depth_stride)
        if not len(points):
            return
        below = points[points[:, 2] < float(plane) - self.drop_m]
        if not len(below):
            return
        xs = np.floor((below[:, 0] - world.origin_x) / world.resolution).astype(int)
        ys = np.floor((below[:, 1] - world.origin_y) / world.resolution).astype(int)
        inside = (xs >= 0) & (xs < world.width) & (ys >= 0) & (ys < world.height)
        drops[ys[inside], xs[inside]] = True

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
