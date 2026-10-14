"""Observed-only RGB-D mapping using the existing core log-odds integrator."""
from __future__ import annotations

import hashlib
import math
import numpy as np

from sparx_agency.core.mapping.costmap.depth_to_grid import update_grid_from_depth
from sparx_agency.core.mapping.costmap.log_odds_grid import LogOddsGridConfig, LogOddsGridCostmap
from sparx_agency.core.planning.environment import OccupancyGrid2D, OccupancyGrid2DParams, OccupancyValues
from sparx_agency.core.planning.exploration.floor_atlas import FloorAtlas, MultiFloorParams
from sparx_agency.core.planning.objnav.camera_geometry import world_T_camera_optical, backproject_depth


class ObservedMap:
    """Independent persistent grids at a shared observed XY origin.

    Intermediate stair observations never enter semantic floor maps. No scene
    bounds, surveyed heights or configured display-slot count are policy inputs.
    """

    def __init__(self, size_m=80.0, resolution_m=0.1, stride=8, body_height_m=0.88, body_radius_m=0.18, multifloor=None,
                 repeat_bump_m=0.5):
        self.grid = LogOddsGridCostmap(LogOddsGridConfig(size_m=size_m, resolution_m=resolution_m))
        self.stride = stride
        self.body_height_m = body_height_m
        self.body_radius_m = body_radius_m
        self.repeat_bump_m = float(repeat_bump_m)
        self._anchor = None
        self.floor_revision = 0
        self.atlas = FloorAtlas(multifloor or MultiFloorParams())
        self.floor_id = 0
        self.maps, self.worlds = {0: self.grid}, {}
        self._bumps = {}          # floor id -> [(x, y) the body touched]
        self._touched = {}        # floor id -> {(gx, gy)} forced occupied in every export (see notify_blocked)
        self.sticky_bumps = 0

    def update(self, observation, integrate=True, *, arrival_allowed=True):
        pose, camera = observation.pose, observation.camera
        if self._anchor is None:
            self._anchor = pose.z
            self.grid.origin_x = pose.x - self.grid.cfg.size_m / 2
            self.grid.origin_y = pose.y - self.grid.cfg.size_m / 2
        if self.atlas.params.enabled:
            floor_id = self.atlas.update(pose, arrival_allowed=arrival_allowed)
            if floor_id not in self.maps:
                grid = LogOddsGridCostmap(self.grid.cfg)
                grid.origin_x, grid.origin_y = self.grid.origin_x, self.grid.origin_y
                self.maps[floor_id] = grid
            self.floor_id = floor_id
            self.grid = self.maps[floor_id]
            self.floor_revision = self.atlas.revision
            self._anchor = self.atlas.elevation_m
            if self.atlas.in_transition or not integrate:
                if floor_id not in self.worlds:
                    spec, probabilities = self.grid.get_grid()
                    self.worlds[floor_id] = OccupancyGrid2D(
                        np.full(probabilities.shape, -1, np.int8),
                        OccupancyGrid2DParams(spec.resolution_m, spec.origin_x, spec.origin_y, "world"),
                        values=OccupancyValues(free=0, occupied=100, unknown=-1))
                return self.worlds[floor_id]
        elif abs(pose.z - self._anchor) > 0.6:
            # Explicit legacy ablation only, recorded in method configuration.
            self.grid.reset()
            self._anchor = pose.z
            self.floor_revision += 1
            self.floor_id = self.floor_revision
        margin = camera.max_depth_m
        if (abs(pose.x - self.grid.origin_x - self.grid.cfg.size_m / 2) > self.grid.cfg.size_m / 2 - margin
                or abs(pose.y - self.grid.origin_y - self.grid.cfg.size_m / 2) > self.grid.cfg.size_m / 2 - margin):
            raise ValueError("Observed map exhausted; increase method map_size_m")
        k = camera.intrinsics
        matrix = np.array([[k.fx, 0, k.cx], [0, k.fy, k.cy], [0, 0, 1]])
        update_grid_from_depth(
            self.grid, observation.depth_m, matrix, world_T_camera_optical(pose, camera),
            z_min_world=self._anchor + 0.15, z_max_world=self._anchor + self.body_height_m + 0.05,
            depth_min_m=camera.min_depth_m, depth_max_m=camera.max_depth_m,
            downsample=self.stride, stamp_sec=float(observation.step), raytrace=True, raytrace_stride=1)
        spec, probabilities = self.grid.get_grid()
        # Visible floor samples clear only their cells, never unseen rays under
        # furniture or previously occupied columns.
        points = backproject_depth(observation.depth_m, camera, pose, self.stride)
        floor = points[np.abs(points[:, 2] - self._anchor) <= 0.10]
        free_mask = np.zeros(probabilities.shape, dtype=bool)
        if len(floor):
            xs = np.floor((floor[:, 0] - spec.origin_x) / spec.resolution_m).astype(int)
            ys = np.floor((floor[:, 1] - spec.origin_y) / spec.resolution_m).astype(int)
            inside = (xs >= 0) & (xs < spec.width) & (ys >= 0) & (ys < spec.height)
            free_mask[ys[inside], xs[inside]] = True
            free_mask &= probabilities < 65
            self.grid.apply_free_mask(free_mask)
            spec, probabilities = self.grid.get_grid()
        data = np.full(probabilities.shape, -1, dtype=np.int8)
        data[(probabilities >= 0) & (probabilities <= 45)] = 0
        data[probabilities >= 65] = 100
        gx = int((pose.x - spec.origin_x) / spec.resolution_m)
        gy = int((pose.y - spec.origin_y) / spec.resolution_m)
        # Only the actual measured footprint supplies embodiment evidence.
        radius = int(math.ceil(self.body_radius_m / spec.resolution_m))
        yy, xx = np.ogrid[max(0, gy - radius):min(data.shape[0], gy + radius + 1),
                          max(0, gx - radius):min(data.shape[1], gx + radius + 1)]
        footprint = (xx - gx) ** 2 + (yy - gy) ** 2 <= (self.body_radius_m / spec.resolution_m) ** 2
        block = data[max(0, gy - radius):min(data.shape[0], gy + radius + 1),
                     max(0, gx - radius):min(data.shape[1], gx + radius + 1)]
        block[footprint & (block != 100)] = 0
        touched = self._touched.get(self.floor_id)
        if touched:
            # What the body learned by touching it twice outranks every ray: see notify_blocked.
            for tx, ty in touched:
                if 0 <= ty < data.shape[0] and 0 <= tx < data.shape[1]:
                    data[ty, tx] = 100
        data[gy, gx] = 0
        world = OccupancyGrid2D(data, OccupancyGrid2DParams(spec.resolution_m, spec.origin_x, spec.origin_y, "world"),
                                values=OccupancyValues(free=0, occupied=100, unknown=-1))
        self.worlds[self.floor_id] = world
        return world

    def notify_blocked(self, observation, forward_step_m):
        """Record a failed forward step as local evidence, not a GT wall lookup.

        The cell a forward step would have put the agent's centre on is set
        occupied outright (:meth:`LogOddsGridCostmap.mark_occupied`): the body
        touched what is there. A single hit's increment could not outvote the
        free-space rays that had cleared the cell, and the planner kept routing
        through it -- Corozal/000142 (``51eff3fa1637``, 2026-10-13) pushed into
        the same wall ten actions running before the cell flipped. A cell at
        ``l_max`` also survives the body-footprint clearing below, which keeps
        occupied cells.

        **A second bump within ``repeat_bump_m`` of an earlier one leaves a
        STICKY mark** (since 2026-10-14 evening): the cells of a bar one body
        wide across the heading at the step's end are forced occupied in every
        export of this storey's world for the rest of the episode
        (:attr:`_touched`), whatever the rays say afterwards. The ``l_max``
        cell still decays under the free-space rays of the next frames -- a
        ledge the camera looks over, a one-step drop the navmesh refuses, is
        free floor to every ray -- and the planner was back through the same
        0.1 m cell, or the one beside it, within a dozen actions: Corozal/
        000019 (``79d7126045c3``, 2026-10-14) bumped three times at one spot
        in 25 actions and lost the three nodes it was walking to, one by one,
        to the releases. The first bump stays a single decaying cell, so one
        spurious refusal walls nothing off.
        """
        p = observation.pose
        ahead = (p.x + forward_step_m * math.cos(p.yaw), p.y + forward_step_m * math.sin(p.yaw))
        self.grid.mark_occupied(np.asarray([list(ahead)]))
        bumps = self._bumps.setdefault(self.floor_id, [])
        repeat = any(math.dist(ahead, earlier) <= self.repeat_bump_m for earlier in bumps)
        bumps.append(ahead)
        if not repeat:
            return
        grid = self.grid
        resolution = float(grid.cfg.resolution_m)
        normal = (-math.sin(p.yaw), math.cos(p.yaw))
        count = int(math.ceil(2.0 * self.body_radius_m / resolution)) + 1
        bar = [(ahead[0] + t * normal[0], ahead[1] + t * normal[1])
               for t in np.linspace(-self.body_radius_m, self.body_radius_m, count)]
        touched = self._touched.setdefault(self.floor_id, set())
        for x, y in bar:
            gx = int(math.floor((x - grid.origin_x) / resolution))
            gy = int(math.floor((y - grid.origin_y) / resolution))
            if 0 <= gx < grid.width and 0 <= gy < grid.height:
                touched.add((gx, gy))
        self.grid.mark_occupied(np.asarray(bar))
        self.sticky_bumps += 1

    def integrity(self):
        """Observed map identities for recording; no surveyed geometry or levels."""
        rows = []
        for floor_id, grid in sorted(self.maps.items()):
            spec, probabilities = grid.get_grid()
            world = self.worlds.get(floor_id)
            rows.append({"floor_id": floor_id, "shape": list(probabilities.shape),
                         "origin": [spec.origin_x, spec.origin_y], "resolution_m": spec.resolution_m,
                         "occupancy_sha256": hashlib.sha256(probabilities.tobytes()).hexdigest(),
                         "known_cells": int(np.count_nonzero(world.grid >= 0)) if world is not None else 0,
                         "occupied_cells": int(np.count_nonzero(world.grid == 100)) if world is not None else 0,
                         "touched_cells": len(self._touched.get(floor_id, ())), "bumps": len(self._bumps.get(floor_id, ()))})
        return rows
