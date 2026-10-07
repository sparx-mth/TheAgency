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
  the first tread are obstacles already;
* a goal is refused when its cell is masked, or when the storey the agent
  stands on is further than ``floor_plane_bound_m`` from the spawn plane
  (``Z_spawn +/- 0.5 m``) -- then nothing on it is a destination.

All of this is a constraint on what the agent may choose, not knowledge
about the target: the only privileged input is the stair geometry the
perfect detector already hands the coordinator once a staircase is in frame.
With ``allow_stair_traversal`` true the guard is inert and returns the world
untouched.
"""
from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np

from sparx_agency.core.planning.environment import OccupancyGrid2D
from sparx_agency.core.planning.objnav.camera_geometry import backproject_depth
from sparx_agency.tasks.planning.objnav_benchmark_runtime.methods.peek_stairs import stair_peek_mask

#: Refusal events kept for the recording; the counters are complete.
MAX_EVENTS = 50


class SpawnFloorGuard:
    """The spawn plane, the impassable cells of the floor in force, and the goal gate."""

    def __init__(self, policy):
        self.policy = policy
        settings = policy.settings
        self.enabled: bool = not settings.allow_stair_traversal
        self.bound_m: float = settings.floor_plane_bound_m
        # Below the plane by more than the atlas's same-floor tolerance is another level.
        self.drop_m: float = settings.multifloor.floor_match_m
        self.spawn_z: Optional[float] = None
        self.drops: Dict[int, np.ndarray] = {}
        self.mask: Optional[np.ndarray] = None
        self.off_plane: bool = False
        self.events: List[dict] = []
        self.stats = {"goals_refused": 0, "drop_cells": 0, "stair_cells": 0, "off_plane_actions": 0}

    # -- per action -----------------------------------------------------------
    def observe(self, obs, world):
        """Record the spawn height, accumulate observed drops, and return the world the policy may plan on."""
        if self.spawn_z is None:
            self.spawn_z = float(obs.pose.z)
        if not self.enabled:
            self.mask = None
            return world
        self._track_plane(obs)
        drops = self._drops_for(self.policy.mapping.floor_id, world)
        self._mark_drops(obs, world, drops)
        mask = drops.copy()
        stairs = stair_peek_mask(self.policy, world) if getattr(self.policy, "building", None) is not None else None
        if stairs is not None:
            mask |= stairs
        self.stats["drop_cells"] = int(drops.sum())
        self.stats["stair_cells"] = 0 if stairs is None else int(stairs.sum())
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
        return {"enabled": self.enabled, "spawn_z_m": self.spawn_z, "floor_plane_bound_m": self.bound_m,
                "drop_m": self.drop_m, "off_plane": self.off_plane, "stats": dict(self.stats),
                "events": list(self.events)}

    # -- internals ------------------------------------------------------------
    def _plane_offset(self) -> float:
        plane = getattr(self.policy.mapping, "_anchor", None)
        if plane is None or self.spawn_z is None:
            return 0.0
        return float(plane) - self.spawn_z

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
