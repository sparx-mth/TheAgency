"""Observed-only RGB-D mapping using the existing core log-odds integrator."""
from __future__ import annotations

import math

import numpy as np

from sparx_agency.core.mapping.costmap.depth_to_grid import update_grid_from_depth
from sparx_agency.core.mapping.costmap.log_odds_grid import LogOddsGridConfig, LogOddsGridCostmap
from sparx_agency.core.planning.environment import OccupancyGrid2D, OccupancyGrid2DParams, OccupancyValues
from sparx_agency.core.planning.exploration.floor_atlas import FloorAtlas, MultiFloorParams
from sparx_agency.core.planning.objnav.camera_geometry import world_T_camera_optical, backproject_depth


class ObservedMap:
    """A fresh map per episode, centred on its first observed position.

    Each settled floor owns an independent grid at a shared observed XY origin.
    Intermediate stair observations are not fused into either floor. This is
    NOT the scoring map; no scene bounds or floor heights are supplied by GT.
    """

    def __init__(self, size_m=80.0, resolution_m=0.1, stride=8, body_height_m=0.88, body_radius_m=0.18, multifloor=None):
        self.grid = LogOddsGridCostmap(LogOddsGridConfig(
            size_m=size_m, resolution_m=resolution_m))
        self.stride = stride
        self.body_height_m = body_height_m
        self.body_radius_m = body_radius_m
        self._anchor = None
        self.floor_revision = 0
        self.atlas = FloorAtlas(multifloor or MultiFloorParams())
        self.floor_id = 0
        self.maps, self.worlds = {0: self.grid}, {}

    def update(self, observation, integrate=True):
        pose, camera = observation.pose, observation.camera
        if self._anchor is None:
            self._anchor = pose.z
            self.grid.origin_x = pose.x - self.grid.cfg.size_m / 2
            self.grid.origin_y = pose.y - self.grid.cfg.size_m / 2
        if self.atlas.params.enabled:
            floor_id = self.atlas.update(pose)
            if floor_id not in self.maps:
                grid = LogOddsGridCostmap(self.grid.cfg)
                grid.origin_x, grid.origin_y = self.grid.origin_x, self.grid.origin_y
                self.maps[floor_id] = grid
            self.floor_id = floor_id
            self.grid = self.maps[floor_id]
            self.floor_revision = self.atlas.revision
            self._anchor = self.atlas.elevation_m
            if (self.atlas.in_transition or not integrate) and floor_id in self.worlds:
                return self.worlds[floor_id]
        elif abs(pose.z - self._anchor) > 0.6:
            # Explicit legacy ablation only, recorded in method configuration.
            self.grid.reset()
            self._anchor = pose.z
            self.floor_revision += 1
            self.floor_id = self.floor_revision
        margin = camera.max_depth_m
        if (abs(pose.x - self.grid.origin_x - self.grid.cfg.size_m / 2)
                > self.grid.cfg.size_m / 2 - margin
                or abs(pose.y - self.grid.origin_y - self.grid.cfg.size_m / 2)
                > self.grid.cfg.size_m / 2 - margin):
            raise ValueError("Observed map exhausted; increase method map_size_m")
        k = camera.intrinsics
        matrix = np.array([[k.fx, 0, k.cx], [0, k.fy, k.cy], [0, 0, 1]])
        update_grid_from_depth(
            self.grid, observation.depth_m, matrix,
            world_T_camera_optical(pose, camera),
            z_min_world=self._anchor + 0.15,
            z_max_world=self._anchor + self.body_height_m + 0.05,
            depth_min_m=camera.min_depth_m, depth_max_m=camera.max_depth_m,
            downsample=self.stride, stamp_sec=float(observation.step),
            raytrace=True, raytrace_stride=1)
        spec, probabilities = self.grid.get_grid()
        # A visible floor sample is free evidence at its OWN cell, not an
        # unobserved free ray under/through furniture. Never erase an occupied
        # column: robot-height obstacle evidence takes precedence.
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
        # Explicit ternary encoding; raw probabilities are not occupied=100.
        data = np.full(probabilities.shape, -1, dtype=np.int8)
        data[(probabilities >= 0) & (probabilities <= 45)] = 0
        data[probabilities >= 65] = 100
        gx = int((pose.x - spec.origin_x) / spec.resolution_m)
        gy = int((pose.y - spec.origin_y) / spec.resolution_m)
        # The occupied footprint at the measured pose is direct embodiment
        # evidence, not an opening panorama or a free-space oracle. Never clear
        # observed obstacles or carve a disk around a future destination.
        radius = int(math.ceil(self.body_radius_m / spec.resolution_m))
        yy, xx = np.ogrid[max(0, gy - radius):min(data.shape[0], gy + radius + 1),
                          max(0, gx - radius):min(data.shape[1], gx + radius + 1)]
        footprint = (xx - gx) ** 2 + (yy - gy) ** 2 <= (self.body_radius_m / spec.resolution_m) ** 2
        block = data[max(0, gy - radius):min(data.shape[0], gy + radius + 1),
                     max(0, gx - radius):min(data.shape[1], gx + radius + 1)]
        block[footprint & (block != 100)] = 0
        data[gy, gx] = 0  # the observed base itself is reachable
        world = OccupancyGrid2D(
            data, OccupancyGrid2DParams(spec.resolution_m, spec.origin_x,
                                        spec.origin_y, "world"),
            values=OccupancyValues(free=0, occupied=100, unknown=-1))
        self.worlds[self.floor_id] = world
        return world

    def notify_blocked(self, observation, forward_step_m):
        """Record a failed forward step as local evidence, not a GT wall lookup."""
        p = observation.pose
        ahead = [[p.x + forward_step_m * math.cos(p.yaw),
                  p.y + forward_step_m * math.sin(p.yaw)]]
        self.grid.update_from_points_xy(np.repeat(ahead, 20, axis=0))


