#!/usr/bin/env python3
"""
Pygame-based Drone Tracking Simulation.

Uses ALL algorithms from sparx_agency.core:
- RRTStarOmplPlanner (path planning)
- HermiteSmoother / MinSnapSmoother (trajectory smoothing)
- PurePursuitTracker (trajectory tracking)

Only the drone physics simulator comes from tasks (it's not an algorithm).

Usage:
    python run_pygame_sim.py                    # Run scenario 1
    python run_pygame_sim.py --scenario 2       # Run scenario 2
    python run_pygame_sim.py --smoother minsnap # Use MinSnap instead of Hermite
    python run_pygame_sim.py --no-wind          # Disable wind/gusts

Controls:
    SPACE   - Pause/Resume
    Q/ESC   - Quit
"""
from __future__ import annotations

import argparse
import math
from dataclasses import dataclass, field
from typing import List, Tuple, Optional
from pathlib import Path

import yaml
import numpy as np
import matplotlib.pyplot as plt
import pygame
import colorsys

# ============================================================================
# CORE IMPORTS - All algorithms from sparx_agency.core
# ============================================================================

# Types
from sparx_agency.core.common.types import (
    Pose2D, Path2D, Pose3D, Twist3D, State3D, TrajectoryPoint
)

# Planning interfaces
from sparx_agency.core.planning.interfaces.planner import PlanRequest
from sparx_agency.core.planning.interfaces.smoother import SmootherRequest
from sparx_agency.core.planning.interfaces.tracker import TrackerRequest

# Planner (from core)
from sparx_agency.core.planning.planners.rrtstar import RRTStarOmplPlanner, RRTStarOmplParams

# Smoothers (from core)
from sparx_agency.core.planning.smoothers.hermite import HermiteSmoother, HermiteParams
from sparx_agency.core.planning.smoothers.minsnap import MinSnapSmoother, MinSnapParams

# Tracker (from core)
from sparx_agency.core.planning.trackers.pure_pursuit import PurePursuitTracker, PurePursuitParams

# Environment / Costmap (from core)
from sparx_agency.core.planning.environment import Costmap2D, CostmapParams
from sparx_agency.core.mapping.costmap.occupancy import occupancy_from_grayscale, OccupancyThresholds
from sparx_agency.core.mapping.costmap.inflation import inflate_occupancy, InflationParams
from sparx_agency.core.mapping.costmap.distance_field import compute_clearance_field, DistanceFieldParams

# ============================================================================
# SIMULATION (physics only - not an algorithm)
# ============================================================================
from sparx_agency.tasks.planning.simulation.drone_sim import DroneSimulator, DroneSimParams


# ============================================================================
# PGM MAP LOADING (for real maps like hospital_map)
# ============================================================================

def load_pgm_map(pgm_path: str, yaml_path: str, inflate_radius: float = 0.1) -> Costmap2D:
    """
    Load PGM map with YAML metadata (same as run_pipeline.py).

    Args:
        pgm_path: Path to .pgm file
        yaml_path: Path to .yaml metadata file
        inflate_radius: Obstacle inflation radius (meters)

    Returns:
        Costmap2D ready for planning
    """
    with open(yaml_path, 'r') as f:
        config = yaml.safe_load(f)

    resolution = config['resolution']
    origin = config['origin']

    img = plt.imread(pgm_path)
    if img.max() <= 1.0:
        img = (img * 255).astype(np.uint8)

    thresholds = OccupancyThresholds(
        occupied_if_below=249,
        free_if_above=250,
        unknown_as_occupied=True
    )
    occupancy = occupancy_from_grayscale(img, thresholds)

    if inflate_radius > 0:
        occupancy = inflate_occupancy(
            occupancy,
            resolution=resolution,
            params=InflationParams(radius_m=inflate_radius)
        )

    clearance = compute_clearance_field(
        occupancy,
        resolution=resolution,
        params=DistanceFieldParams()
    )

    params = CostmapParams(
        resolution=resolution,
        origin_x=origin[0],
        origin_y=origin[1],
        frame_id="map"
    )
    return Costmap2D(occupancy, params, clearance=clearance)


# ============================================================================
# OBSTACLE MAP -> COSTMAP CONVERSION
# ============================================================================

@dataclass
class ObstacleMap:
    """Simple obstacle map for defining the environment."""
    width: float = 10.0
    height: float = 10.0
    origin_x: float = -5.0
    origin_y: float = -5.0
    resolution: float = 0.05

    rectangles: List[Tuple[float, float, float, float]] = field(default_factory=list)
    circles: List[Tuple[float, float, float]] = field(default_factory=list)

    def add_rectangle(self, x: float, y: float, w: float, h: float):
        self.rectangles.append((x, y, w, h))

    def add_circle(self, cx: float, cy: float, r: float):
        self.circles.append((cx, cy, r))

    def is_occupied(self, x: float, y: float, margin: float = 0.0) -> bool:
        for rx, ry, rw, rh in self.rectangles:
            if (rx - margin <= x <= rx + rw + margin and
                    ry - margin <= y <= ry + rh + margin):
                return True
        for cx, cy, r in self.circles:
            if math.sqrt((x - cx) ** 2 + (y - cy) ** 2) <= r + margin:
                return True
        return False

    def to_costmap(self, inflate_radius: float = 0.1) -> Costmap2D:
        """Convert obstacle map to Costmap2D for core planners."""
        # Create occupancy grid
        nx = int(self.width / self.resolution)
        ny = int(self.height / self.resolution)
        occupancy = np.zeros((ny, nx), dtype=np.uint8)

        # Fill occupancy grid
        for iy in range(ny):
            for ix in range(nx):
                x = self.origin_x + ix * self.resolution
                y = self.origin_y + iy * self.resolution
                if self.is_occupied(x, y):
                    occupancy[iy, ix] = 255  # Occupied

        # Inflate obstacles
        if inflate_radius > 0:
            occupancy = inflate_occupancy(
                occupancy,
                resolution=self.resolution,
                params=InflationParams(radius_m=inflate_radius)
            )

        # Compute clearance field
        clearance = compute_clearance_field(
            occupancy,
            resolution=self.resolution,
            params=DistanceFieldParams()
        )

        # Create costmap
        params = CostmapParams(
            resolution=self.resolution,
            origin_x=self.origin_x,
            origin_y=self.origin_y,
            frame_id="map"
        )

        return Costmap2D(occupancy, params, clearance=clearance)


# ============================================================================
# PYGAME VISUALIZATION (display only)
# ============================================================================

@dataclass
class ViewSettings:
    width: int = 1200
    height: int = 800
    hud_width: int = 250
    margin: float = 0.5

    bg_color: Tuple[int, int, int] = (240, 240, 245)
    grid_color: Tuple[int, int, int] = (200, 200, 200)
    obstacle_color: Tuple[int, int, int] = (100, 100, 100)
    obstacle_border: Tuple[int, int, int] = (50, 50, 50)
    trajectory_color: Tuple[int, int, int] = (100, 100, 255)
    path_color: Tuple[int, int, int] = (0, 200, 200)
    start_color: Tuple[int, int, int] = (0, 200, 0)
    goal_color: Tuple[int, int, int] = (255, 50, 50)
    drone_color: Tuple[int, int, int] = (255, 80, 80)
    drone_border: Tuple[int, int, int] = (150, 0, 0)
    lookahead_color: Tuple[int, int, int] = (0, 200, 100)
    hud_bg: Tuple[int, int, int] = (50, 50, 60)
    hud_text: Tuple[int, int, int] = (220, 220, 220)
    warning_color: Tuple[int, int, int] = (255, 200, 0)
    error_color: Tuple[int, int, int] = (255, 50, 50)
    success_color: Tuple[int, int, int] = (50, 255, 100)


class WorldToScreen:
    def __init__(self, world_bounds: Tuple[float, float, float, float], screen_rect: pygame.Rect):
        self.world_bounds = world_bounds
        world_width = world_bounds[2] - world_bounds[0]
        world_height = world_bounds[3] - world_bounds[1]
        scale_x = screen_rect.width / world_width
        scale_y = screen_rect.height / world_height
        self.scale = min(scale_x, scale_y)
        scaled_width = world_width * self.scale
        scaled_height = world_height * self.scale
        self.offset_x = screen_rect.x + (screen_rect.width - scaled_width) / 2
        self.offset_y = screen_rect.y + (screen_rect.height - scaled_height) / 2

    def to_screen(self, world_x: float, world_y: float) -> Tuple[int, int]:
        screen_x = self.offset_x + (world_x - self.world_bounds[0]) * self.scale
        screen_y = self.offset_y + (self.world_bounds[3] - world_y) * self.scale
        return int(screen_x), int(screen_y)

    def scale_distance(self, world_dist: float) -> int:
        return max(1, int(world_dist * self.scale))


def speed_to_color(speed: float, max_speed: float = 0.6) -> Tuple[int, int, int]:
    t = min(1.0, speed / max_speed)
    hue = 0.6 * (1 - t)
    r, g, b = colorsys.hsv_to_rgb(hue, 0.8, 0.9)
    return int(r * 255), int(g * 255), int(b * 255)


class DroneVisualizer:
    """Pygame-based visualization with 60 FPS rendering."""

    def __init__(self, obstacle_map: Optional[ObstacleMap] = None,
                 trajectory=None, raw_path: Optional[Path2D] = None,
                 settings: Optional[ViewSettings] = None,
                 costmap: Optional[Costmap2D] = None):
        self.settings = settings or ViewSettings()
        self.obstacle_map = obstacle_map
        self.trajectory = trajectory
        self.raw_path = raw_path
        self.costmap = costmap  # For scenario 4 (PGM maps)

        pygame.init()
        pygame.font.init()

        self.screen = pygame.display.set_mode(
            (self.settings.width, self.settings.height), pygame.RESIZABLE
        )
        pygame.display.set_caption("Drone Simulation (sparx_agency.core) - SPACE=pause, Q=quit")

        self.font_large = pygame.font.SysFont('monospace', 24, bold=True)
        self.font_medium = pygame.font.SysFont('monospace', 18)
        self.font_small = pygame.font.SysFont('monospace', 14)

        self._calculate_world_bounds()
        map_rect = pygame.Rect(10, 10, self.settings.width - self.settings.hud_width - 20,
                               self.settings.height - 20)
        self.transform = WorldToScreen(self.world_bounds, map_rect)

        self.drone_pos = (0.0, 0.0, 0.0)
        self.drone_yaw = 0.0
        self.drone_speed = 0.0
        self.lookahead_point = None
        self.trail: List[Tuple[float, float, float]] = []
        self.max_trail_points = 5000

        self.time = 0.0
        self.cross_track_error = 0.0
        self.progress_idx = 0
        self.gust_active = False
        self.collision = False
        self.done = False
        self.failed = False
        self.paused = False

        self.collisions_count = 0
        self.gusts_count = 0
        self.max_cte = 0.0
        self._was_gust_active = False

        self._render_static_elements()
        self.clock = pygame.time.Clock()
        self.running = True

    def _calculate_world_bounds(self):
        s = self.settings
        if self.obstacle_map:
            x_min = self.obstacle_map.origin_x - s.margin
            x_max = self.obstacle_map.origin_x + self.obstacle_map.width + s.margin
            y_min = self.obstacle_map.origin_y - s.margin
            y_max = self.obstacle_map.origin_y + self.obstacle_map.height + s.margin
        elif self.costmap:
            # For scenario 4 with PGM costmap
            x_min = self.costmap.origin_x - s.margin
            x_max = self.costmap.origin_x + self.costmap.width * self.costmap.resolution + s.margin
            y_min = self.costmap.origin_y - s.margin
            y_max = self.costmap.origin_y + self.costmap.height * self.costmap.resolution + s.margin
        else:
            x_min, x_max, y_min, y_max = -5, 15, -5, 15
        self.world_bounds = (x_min, y_min, x_max, y_max)

    def _render_static_elements(self):
        map_width = self.settings.width - self.settings.hud_width
        self.static_surface = pygame.Surface((map_width, self.settings.height))
        self.static_surface.fill(self.settings.bg_color)
        self._draw_grid(self.static_surface)
        if self.obstacle_map:
            self._draw_obstacles(self.static_surface)
        elif self.costmap:
            self._draw_costmap(self.static_surface)
        if self.raw_path:
            self._draw_raw_path(self.static_surface)
        if self.trajectory:
            self._draw_trajectory(self.static_surface)

    def _draw_grid(self, surface):
        s = self.settings
        world_w = self.world_bounds[2] - self.world_bounds[0]
        world_h = self.world_bounds[3] - self.world_bounds[1]
        spacing = max(0.5, round(max(world_w, world_h) / 20 * 2) / 2)

        x = math.ceil(self.world_bounds[0] / spacing) * spacing
        while x <= self.world_bounds[2]:
            p1 = self.transform.to_screen(x, self.world_bounds[1])
            p2 = self.transform.to_screen(x, self.world_bounds[3])
            pygame.draw.line(surface, s.grid_color, p1, p2, 1)
            x += spacing
        y = math.ceil(self.world_bounds[1] / spacing) * spacing
        while y <= self.world_bounds[3]:
            p1 = self.transform.to_screen(self.world_bounds[0], y)
            p2 = self.transform.to_screen(self.world_bounds[2], y)
            pygame.draw.line(surface, s.grid_color, p1, p2, 1)
            y += spacing

    def _draw_obstacles(self, surface):
        s = self.settings
        for rx, ry, rw, rh in self.obstacle_map.rectangles:
            tl = self.transform.to_screen(rx, ry + rh)
            br = self.transform.to_screen(rx + rw, ry)
            rect = pygame.Rect(tl[0], tl[1], br[0] - tl[0], br[1] - tl[1])
            pygame.draw.rect(surface, s.obstacle_color, rect)
            pygame.draw.rect(surface, s.obstacle_border, rect, 2)
        for cx, cy, r in self.obstacle_map.circles:
            center = self.transform.to_screen(cx, cy)
            radius = self.transform.scale_distance(r)
            pygame.draw.circle(surface, s.obstacle_color, center, radius)
            pygame.draw.circle(surface, s.obstacle_border, center, radius, 2)

    def _draw_costmap(self, surface):
        """Draw costmap occupancy grid (for scenario 4 with PGM maps)."""
        if not self.costmap:
            return

        s = self.settings
        occupancy = self.costmap.occupancy

        # Get screen coordinates for the costmap corners
        x0 = self.costmap.origin_x
        y0 = self.costmap.origin_y
        res = self.costmap.resolution

        # Draw each occupied cell
        for iy in range(occupancy.shape[0]):
            for ix in range(occupancy.shape[1]):
                if occupancy[iy, ix] > 200:  # Occupied
                    world_x = x0 + ix * res
                    world_y = y0 + iy * res

                    tl = self.transform.to_screen(world_x, world_y + res)
                    br = self.transform.to_screen(world_x + res, world_y)

                    w = max(1, br[0] - tl[0])
                    h = max(1, br[1] - tl[1])

                    rect = pygame.Rect(tl[0], tl[1], w, h)
                    pygame.draw.rect(surface, s.obstacle_color, rect)

    def _draw_raw_path(self, surface):
        """Draw the raw RRT* path (before smoothing)."""
        s = self.settings
        if not self.raw_path or len(self.raw_path.points) < 2:
            return
        points = [self.transform.to_screen(p.x, p.y) for p in self.raw_path.points]
        pygame.draw.lines(surface, s.path_color, False, points, 2)
        for p in self.raw_path.points:
            pygame.draw.circle(surface, s.path_color, self.transform.to_screen(p.x, p.y), 4)

    def _draw_trajectory(self, surface):
        """Draw the smoothed trajectory."""
        s = self.settings
        samples = self.trajectory.sample_by_time(0.05)
        if len(samples) < 2:
            return
        points = [self.transform.to_screen(p.x, p.y) for p in samples]
        pygame.draw.lines(surface, s.trajectory_color, False, points, 3)

        # Start and goal
        start = (samples[0].x, samples[0].y)
        goal = (samples[-1].x, samples[-1].y)
        pygame.draw.circle(surface, s.start_color, self.transform.to_screen(*start), 15)
        pygame.draw.circle(surface, (0, 100, 0), self.transform.to_screen(*start), 15, 3)
        pygame.draw.circle(surface, s.goal_color, self.transform.to_screen(*goal), 18)
        pygame.draw.circle(surface, (150, 0, 0), self.transform.to_screen(*goal), 18, 3)

    def _draw_trail(self, surface):
        if len(self.trail) < 2:
            return
        for i in range(1, len(self.trail)):
            x1, y1, s1 = self.trail[i - 1]
            x2, y2, s2 = self.trail[i]
            color = speed_to_color((s1 + s2) / 2)
            pygame.draw.line(surface, color, self.transform.to_screen(x1, y1),
                             self.transform.to_screen(x2, y2), 3)

    def _draw_drone(self, surface):
        s = self.settings
        pos = self.transform.to_screen(self.drone_pos[0], self.drone_pos[1])
        pygame.draw.circle(surface, s.drone_color, pos, 12)
        pygame.draw.circle(surface, s.drone_border, pos, 12, 3)
        arrow_len = 25
        tip = (pos[0] + arrow_len * math.cos(-self.drone_yaw),
               pos[1] + arrow_len * math.sin(-self.drone_yaw))
        angle = math.atan2(tip[1] - pos[1], tip[0] - pos[0])
        head_len, head_angle = 8, math.pi / 6
        left = (tip[0] - head_len * math.cos(angle - head_angle),
                tip[1] - head_len * math.sin(angle - head_angle))
        right = (tip[0] - head_len * math.cos(angle + head_angle),
                 tip[1] - head_len * math.sin(angle + head_angle))
        pygame.draw.line(surface, s.drone_border, pos, tip, 3)
        pygame.draw.polygon(surface, s.drone_border, [tip, left, right])

    def _draw_lookahead(self, surface):
        if not self.lookahead_point:
            return
        s = self.settings
        d_scr = self.transform.to_screen(self.drone_pos[0], self.drone_pos[1])
        l_scr = self.transform.to_screen(self.lookahead_point[0], self.lookahead_point[1])
        pygame.draw.line(surface, s.lookahead_color, d_scr, l_scr, 2)
        pygame.draw.circle(surface, s.lookahead_color, l_scr, 8)

    def _draw_hud(self, surface):
        s = self.settings
        hud_rect = pygame.Rect(self.settings.width - self.settings.hud_width, 0,
                               self.settings.hud_width, self.settings.height)
        pygame.draw.rect(surface, s.hud_bg, hud_rect)

        x, y = hud_rect.x + 15, 20
        surface.blit(self.font_large.render("FLIGHT STATUS", True, s.hud_text), (x, y))
        y += 40
        pygame.draw.line(surface, s.hud_text, (x, y), (hud_rect.right - 15, y), 1)
        y += 20

        surface.blit(self.font_medium.render(f"Time: {self.time:.1f}s", True, s.hud_text), (x, y))
        y += 30
        surface.blit(self.font_medium.render(f"Speed: {self.drone_speed:.2f} m/s", True,
                                             speed_to_color(self.drone_speed)), (x, y))
        y += 30

        cte_color = s.error_color if self.cross_track_error > 0.5 else (
            s.warning_color if self.cross_track_error > 0.3 else s.hud_text)
        surface.blit(self.font_medium.render(f"CTE: {self.cross_track_error:.3f} m", True, cte_color), (x, y))
        y += 30
        surface.blit(self.font_small.render(f"Pos: ({self.drone_pos[0]:.1f}, {self.drone_pos[1]:.1f})",
                                            True, s.hud_text), (x, y))
        y += 45

        pygame.draw.line(surface, s.hud_text, (x, y), (hud_rect.right - 15, y), 1)
        y += 15
        surface.blit(self.font_medium.render("STATISTICS", True, s.hud_text), (x, y))
        y += 25
        surface.blit(self.font_small.render(f"Collisions: {self.collisions_count}", True,
                                            s.error_color if self.collisions_count else s.hud_text), (x, y))
        y += 22
        surface.blit(self.font_small.render(f"Gusts: {self.gusts_count}", True, s.hud_text), (x, y))
        y += 22
        surface.blit(self.font_small.render(f"Max CTE: {self.max_cte:.3f} m", True, s.hud_text), (x, y))
        y += 35

        pygame.draw.line(surface, s.hud_text, (x, y), (hud_rect.right - 15, y), 1)
        y += 20

        if self.gust_active:
            surface.blit(self.font_medium.render("⚠ GUST!", True, s.warning_color), (x, y))
            y += 30
        if self.collision:
            surface.blit(self.font_medium.render("⛔ COLLISION!", True, s.error_color), (x, y))
            y += 30
        if self.paused:
            surface.blit(self.font_large.render("⏸ PAUSED", True, s.warning_color), (x, y))
            y += 35
        if self.done:
            surface.blit(self.font_large.render("✓ SUCCESS!", True, s.success_color), (x, y))
        elif self.failed:
            surface.blit(self.font_large.render("✗ FAILED", True, s.error_color), (x, y))

        y = self.settings.height - 60
        pygame.draw.line(surface, s.hud_text, (x, y), (hud_rect.right - 15, y), 1)
        y += 12
        for ctrl in ["SPACE - Pause", "Q - Quit"]:
            surface.blit(self.font_small.render(ctrl, True, (180, 180, 180)), (x, y))
            y += 18

    def update(self, drone_position, drone_velocity, drone_yaw, time, cross_track_error,
               progress_idx, lookahead_point=None, gust_active=False, collision=False,
               done=False, failed=False) -> bool:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
                return False
            elif event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_q, pygame.K_ESCAPE):
                    self.running = False
                    return False
                elif event.key == pygame.K_SPACE:
                    self.paused = not self.paused
            elif event.type == pygame.VIDEORESIZE:
                self.settings.width, self.settings.height = event.w, event.h
                self.screen = pygame.display.set_mode((event.w, event.h), pygame.RESIZABLE)
                map_rect = pygame.Rect(10, 10, self.settings.width - self.settings.hud_width - 20,
                                       self.settings.height - 20)
                self.transform = WorldToScreen(self.world_bounds, map_rect)
                self._render_static_elements()

        if self.paused:
            self._render_frame()
            self.clock.tick(30)
            return True

        self.drone_pos = drone_position
        self.drone_yaw = drone_yaw
        self.drone_speed = math.sqrt(drone_velocity[0] ** 2 + drone_velocity[1] ** 2)
        self.time = time
        self.cross_track_error = cross_track_error
        self.progress_idx = progress_idx
        self.lookahead_point = lookahead_point
        self.gust_active = gust_active
        self.collision = collision
        self.done = done
        self.failed = failed

        if collision:
            self.collisions_count += 1
        if gust_active and not self._was_gust_active:
            self.gusts_count += 1
        self._was_gust_active = gust_active
        self.max_cte = max(self.max_cte, cross_track_error)

        self.trail.append((drone_position[0], drone_position[1], self.drone_speed))
        if len(self.trail) > self.max_trail_points:
            self.trail = self.trail[-self.max_trail_points:]

        self._render_frame()
        self.clock.tick(60)
        return self.running

    def _render_frame(self):
        self.screen.fill(self.settings.bg_color)
        self.screen.blit(self.static_surface, (0, 0))
        map_surface = self.screen.subsurface(pygame.Rect(
            0, 0, self.settings.width - self.settings.hud_width, self.settings.height))
        self._draw_trail(map_surface)
        self._draw_lookahead(map_surface)
        self._draw_drone(map_surface)
        self._draw_hud(self.screen)
        pygame.display.flip()

    def close(self):
        pygame.quit()

    def wait_for_close(self):
        while self.running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and
                                                 event.key in (pygame.K_q, pygame.K_ESCAPE, pygame.K_RETURN)):
                    self.running = False
            self._render_frame()
            self.clock.tick(30)
        self.close()


# ============================================================================
# MAIN SIMULATION - Uses all core algorithms
# ============================================================================

def run_simulation(
        obstacle_map: Optional[ObstacleMap],
        start: Pose2D,
        goal: Pose2D,
        planner_params: RRTStarOmplParams,
        smoother_type: str,  # "hermite" or "minsnap"
        smoother_params,
        tracker_params: PurePursuitParams,
        sim_params: DroneSimParams,
        max_time: float = 100.0,
        collision_margin: float = 0.3,
        seed: Optional[int] = None,
        costmap: Optional[Costmap2D] = None,
) -> bool:
    """
    Run full planning + tracking simulation.

    All algorithms from sparx_agency.core:
    1. RRTStarOmplPlanner - path planning
    2. HermiteSmoother/MinSnapSmoother - trajectory smoothing
    3. PurePursuitTracker - trajectory tracking
    """

    # ========== STEP 1: Create/Use Costmap ==========
    if costmap is not None:
        # Scenario 4: costmap provided directly (from PGM file)
        print("Using provided costmap...")
    elif obstacle_map is not None:
        print("Creating costmap from obstacles...")
        costmap = obstacle_map.to_costmap(inflate_radius=0.1)
    else:
        print("ERROR: No obstacle_map or costmap provided")
        return False

    print(f"  Costmap: {costmap.width}x{costmap.height}, res={costmap.resolution}m")

    # ========== STEP 2: Plan path with RRT* (from CORE) ==========
    print(f"Planning path: ({start.x}, {start.y}) → ({goal.x}, {goal.y})...")
    planner = RRTStarOmplPlanner(params=planner_params)
    plan_request = PlanRequest(start=start, goal=goal, frame_id="map")
    plan_result = planner.plan(plan_request, costmap)

    print(f"  Status: {plan_result.status}")
    if not plan_result.ok:
        print(f"  Planning failed: {plan_result.message}")
        return False

    raw_path = plan_result.path
    print(f"  Path: {len(raw_path.points)} waypoints, length={raw_path.length():.2f}m")

    # ========== STEP 3: Smooth trajectory (from CORE) ==========
    print(f"Smoothing with {smoother_type}...")
    smooth_request = SmootherRequest(path=raw_path)

    if smoother_type == "hermite":
        smoother = HermiteSmoother(params=smoother_params)
    else:
        smoother = MinSnapSmoother(params=smoother_params)

    trajectory = smoother.smooth(smooth_request)
    print(f"  Trajectory duration: {trajectory.total_time:.2f}s")

    # ========== STEP 4: Create tracker (from CORE) ==========
    tracker = PurePursuitTracker(params=tracker_params)
    tracker.reset()

    # ========== STEP 5: Create drone simulator (physics only) ==========
    # Collision function - use costmap for scenario 4, obstacle_map otherwise
    if obstacle_map is not None:
        collision_fn = lambda x, y: obstacle_map.is_occupied(x, y, collision_margin)
    else:
        # Use costmap occupancy check for scenario 4
        def collision_fn(x: float, y: float) -> bool:
            ix = int((x - costmap.origin_x) / costmap.resolution)
            iy = int((y - costmap.origin_y) / costmap.resolution)
            if 0 <= ix < costmap.width and 0 <= iy < costmap.height:
                return costmap.occupancy[iy, ix] > 200  # Occupied threshold
            return True  # Out of bounds = collision

    sim = DroneSimulator(params=sim_params, obstacle_fn=collision_fn, seed=seed)
    sim.reset(x=start.x, y=start.y, z=0.0)

    # ========== STEP 6: Create visualizer ==========
    vis = DroneVisualizer(
        obstacle_map=obstacle_map,
        trajectory=trajectory,
        raw_path=raw_path,
        costmap=costmap if obstacle_map is None else None,  # Show costmap for scenario 4
    )

    # ========== STEP 7: Run simulation loop ==========
    dt = sim.params.dt
    t = 0.0

    print("\n" + "=" * 50)
    print("SIMULATION RUNNING")
    print("  Planner: RRTStarOmplPlanner (core)")
    print(f"  Smoother: {smoother_type.capitalize()}Smoother (core)")
    print("  Tracker: PurePursuitTracker (core)")
    print("Controls: SPACE=pause, Q=quit")
    print("=" * 50 + "\n")

    running = True
    result_success = False

    while running and t < max_time:
        # Get measured state
        x, y, z, vx, vy, vz, yaw = sim.get_measured_state()

        # Build state (core types)
        state = State3D(
            pose=Pose3D(x=x, y=y, z=z, yaw=yaw),
            twist=Twist3D(vx=vx, vy=vy, vz=vz, yaw_rate=0.0),
        )

        # Run tracker (core TrackerRequest)
        request = TrackerRequest(state=state, trajectory=trajectory, t=t)
        tracker_result = tracker.step(request)

        # Extract command and step simulator
        cmd = tracker_result.command
        sim_state, info = sim.step(cmd.x, cmd.y, cmd.z, cmd.yaw_rate)

        # Get visualization data
        lookahead_pt = None
        if tracker_result.reference:
            ref = tracker_result.reference
            lookahead_pt = (ref.x, ref.y, ref.z)

        cte = tracker_result.metadata.get("cross_track_error", 0.0)
        done = tracker_result.metadata.get("done", False)
        failed = tracker_result.metadata.get("failed", False)

        # Update visualization
        running = vis.update(
            drone_position=(sim_state.x, sim_state.y, sim_state.z),
            drone_velocity=(sim_state.vx, sim_state.vy, sim_state.vz),
            drone_yaw=sim_state.yaw,
            time=t,
            cross_track_error=cte,
            progress_idx=tracker_result.metadata.get("progress_idx", 0),
            lookahead_point=lookahead_pt,
            gust_active=info["gust_active"],
            collision=info["collision"],
            done=done,
            failed=failed,
        )

        if done:
            result_success = True
            print("\n✓ GOAL REACHED!")
            break

        if failed:
            print(f"\n✗ TRACKING FAILED: {tracker_result.metadata.get('reason', 'unknown')}")
            break

        t += dt

    # Wait for user to close
    if running:
        print("\nSimulation complete. Press Q or close window to exit.")
        vis.wait_for_close()
    else:
        vis.close()

    return result_success


# ============================================================================
# SCENARIOS - Parameter configuration only
# ============================================================================

def create_scenario_1(smoother_type: str = "hermite"):
    """Scenario 1: Basic Navigation with Wind."""
    print("\n" + "=" * 60)
    print("SCENARIO 1: Basic Navigation with Wind")
    print("=" * 60)

    # Environment
    obs_map = ObstacleMap(width=12.0, height=12.0, origin_x=-2.0, origin_y=-2.0)
    obs_map.add_rectangle(2.0, 1.0, 1.5, 0.5)
    obs_map.add_rectangle(4.0, 3.0, 0.5, 2.5)
    obs_map.add_rectangle(1.0, 4.0, 2.0, 0.5)
    obs_map.add_circle(6.0, 5.0, 0.8)
    obs_map.add_circle(3.0, 6.5, 0.5)

    start = Pose2D(x=-1.0, y=-1.0)
    goal = Pose2D(x=8.0, y=7.0)

    # RRT* parameters (core)
    planner_params = RRTStarOmplParams(
        timeout=3.0,
        use_clearance_objective=True,
        clearance_weight=10.0,
        interpolation_spacing=3.0,
    )

    # Smoother parameters (core)
    if smoother_type == "hermite":
        smoother_params = HermiteParams(dt=0.02, nominal_speed_xy=0.4, tangent_scale=0.5)
    else:
        smoother_params = MinSnapParams(dt=0.02, nominal_speed_xy=0.4)

    # Pure Pursuit parameters (core)
    tracker_params = PurePursuitParams(
        holonomic=True,
        base_lookahead=0.5,
        min_lookahead=0.3,
        max_lookahead=1.2,
        cruise_speed=0.4,
        goal_tolerance=0.2,
        path_tolerance=1.0,
    )

    # Simulator parameters (physics)
    sim_params = DroneSimParams(
        dt=0.02,
        wind_enabled=True,
        wind_mean=(0.03, 0.01, 0.0),
        wind_std=0.05,
        gust_enabled=True,
        gust_probability=0.003,
        gust_magnitude=0.15,
        process_noise_std=0.01,
        position_noise_std=0.005,
    )

    return obs_map, start, goal, planner_params, smoother_params, tracker_params, sim_params


def create_scenario_2(smoother_type: str = "hermite"):
    """Scenario 2: Tight Corridors + Strong Wind."""
    print("\n" + "=" * 60)
    print("SCENARIO 2: Tight Corridors + Strong Wind")
    print("=" * 60)

    obs_map = ObstacleMap(width=14.0, height=10.0, origin_x=-2.0, origin_y=-2.0)
    obs_map.add_rectangle(1.0, -1.0, 0.3, 3.5)
    obs_map.add_rectangle(1.0, 4.5, 0.3, 3.5)
    obs_map.add_rectangle(4.0, 0.0, 0.3, 3.0)
    obs_map.add_rectangle(4.0, 5.5, 0.3, 2.5)
    obs_map.add_rectangle(7.0, -1.0, 0.3, 4.5)
    obs_map.add_rectangle(7.0, 6.0, 0.3, 2.0)
    obs_map.add_circle(2.5, 2.0, 0.35)
    obs_map.add_circle(5.5, 6.5, 0.4)

    start = Pose2D(x=-1.0, y=0.0)
    goal = Pose2D(x=10.0, y=6.0)

    planner_params = RRTStarOmplParams(
        timeout=5.0,
        use_clearance_objective=True,
        clearance_weight=15.0,
        interpolation_spacing=2.0,
    )

    if smoother_type == "hermite":
        smoother_params = HermiteParams(dt=0.02, nominal_speed_xy=0.35, tangent_scale=0.4)
    else:
        smoother_params = MinSnapParams(dt=0.02, nominal_speed_xy=0.35)

    tracker_params = PurePursuitParams(
        holonomic=True,
        base_lookahead=0.4,
        min_lookahead=0.25,
        max_lookahead=0.8,
        cruise_speed=0.3,
        max_speed=0.4,
        goal_tolerance=0.2,
        path_tolerance=0.6,
    )

    sim_params = DroneSimParams(
        dt=0.02,
        wind_enabled=True,
        wind_mean=(0.06, 0.03, 0.0),
        wind_std=0.08,
        wind_tau=1.5,
        gust_enabled=True,
        gust_probability=0.005,
        gust_magnitude=0.2,
        gust_duration=0.5,
        process_noise_std=0.015,
    )

    return obs_map, start, goal, planner_params, smoother_params, tracker_params, sim_params


def create_scenario_3(smoother_type: str = "hermite"):
    """Scenario 3: Open Area with Frequent Gusts."""
    print("\n" + "=" * 60)
    print("SCENARIO 3: Open Area with Frequent Gusts")
    print("=" * 60)

    obs_map = ObstacleMap(width=15.0, height=12.0, origin_x=-2.0, origin_y=-2.0)
    obs_map.add_circle(3.0, 3.0, 0.7)
    obs_map.add_circle(7.0, 2.0, 0.5)
    obs_map.add_circle(5.0, 6.0, 0.8)
    obs_map.add_circle(9.0, 5.0, 0.6)
    obs_map.add_rectangle(1.0, 7.0, 2.0, 0.4)

    start = Pose2D(x=-1.0, y=-1.0)
    goal = Pose2D(x=11.0, y=8.0)

    planner_params = RRTStarOmplParams(
        timeout=2.0,
        use_clearance_objective=True,
        clearance_weight=5.0,
        interpolation_spacing=4.0,
    )

    if smoother_type == "hermite":
        smoother_params = HermiteParams(dt=0.02, nominal_speed_xy=0.5, tangent_scale=0.5)
    else:
        smoother_params = MinSnapParams(dt=0.02, nominal_speed_xy=0.5)

    tracker_params = PurePursuitParams(
        holonomic=True,
        base_lookahead=0.7,
        min_lookahead=0.4,
        max_lookahead=1.5,
        cruise_speed=0.5,
        max_speed=0.6,
        goal_tolerance=0.2,
        path_tolerance=1.2,
    )

    sim_params = DroneSimParams(
        dt=0.02,
        wind_enabled=True,
        wind_mean=(0.0, 0.0, 0.0),
        wind_std=0.03,
        gust_enabled=True,
        gust_probability=0.01,
        gust_magnitude=0.25,
        gust_duration=0.6,
        process_noise_std=0.01,
    )

    return obs_map, start, goal, planner_params, smoother_params, tracker_params, sim_params


def create_scenario_4(smoother_type: str = "hermite", map_dir: Optional[str] = None):
    """
    Scenario 4: Hospital Map (real-world map from run_pipeline.py).

    Uses PGM map file instead of synthetic obstacles.
    """
    print("\n" + "=" * 60)
    print("SCENARIO 4: Hospital Map (Real-World)")
    print("=" * 60)

    # Find map files
    if map_dir is None:
        # Try common locations
        possible_dirs = [
            Path(__file__).parent / "maps",
            Path(__file__).parent.parent / "maps",
            Path.cwd() / "maps",
            Path.cwd(),
        ]
        for d in possible_dirs:
            if (d / "hospital_map_cropped.pgm").exists():
                map_dir = d
                break

    if map_dir is None:
        print("ERROR: Could not find hospital_map_cropped.pgm")
        print("Please provide --map-dir argument or place maps in ./maps/")
        return None

    map_dir = Path(map_dir)
    pgm_file = map_dir / "hospital_map_cropped.pgm"
    yaml_file = map_dir / "hospital_map_cropped.yaml"

    if not pgm_file.exists() or not yaml_file.exists():
        print(f"ERROR: Map files not found in {map_dir}")
        return None

    print(f"Loading map: {pgm_file}")

    # Load the costmap using core functions
    costmap = load_pgm_map(str(pgm_file), str(yaml_file), inflate_radius=0.1)
    print(f"  Costmap: {costmap.width}x{costmap.height}, res={costmap.resolution}m")

    # Start and goal (same as run_pipeline.py)
    start = Pose2D(x=-2.0, y=-2.5)
    goal = Pose2D(x=5.0, y=5.0)

    # RRT* parameters (same as run_pipeline.py)
    planner_params = RRTStarOmplParams(
        timeout=3.0,
        use_clearance_objective=True,
        clearance_weight=10.0,
        interpolation_spacing=3.0,
    )

    # Smoother parameters
    if smoother_type == "hermite":
        smoother_params = HermiteParams(dt=0.02, nominal_speed_xy=0.5, tangent_scale=0.5)
    else:
        smoother_params = MinSnapParams(dt=0.02, nominal_speed_xy=0.5)

    # Pure Pursuit parameters
    tracker_params = PurePursuitParams(
        holonomic=True,
        base_lookahead=0.6,
        min_lookahead=0.3,
        max_lookahead=1.5,
        cruise_speed=0.4,
        goal_tolerance=0.2,
        path_tolerance=1.0,
    )

    # Simulator parameters
    sim_params = DroneSimParams(
        dt=0.02,
        wind_enabled=True,
        wind_mean=(0.02, 0.01, 0.0),
        wind_std=0.04,
        gust_enabled=True,
        gust_probability=0.002,
        gust_magnitude=0.12,
        process_noise_std=0.01,
        position_noise_std=0.005,
    )

    # Return costmap directly (scenario 4 uses costmap, not ObstacleMap)
    return costmap, start, goal, planner_params, smoother_params, tracker_params, sim_params


def main():
    parser = argparse.ArgumentParser(
        description="Drone Simulation using sparx_agency.core algorithms"
    )
    parser.add_argument('--scenario', '-s', type=int, choices=[1, 2, 3, 4], default=1)
    parser.add_argument('--smoother', type=str, choices=['hermite', 'minsnap'], default='hermite',
                        help='Trajectory smoother: hermite or minsnap')
    parser.add_argument('--no-wind', action='store_true', help='Disable wind and gusts')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--max-time', type=float, default=100.0, help='Max simulation time')
    parser.add_argument('--map-dir', type=str, default="/home/nadavc/PycharmProjects/TheAgency_workspace/sparx_agency/tasks/planning/rrt_smoothing_check/maps/",
                        help='Directory containing map files (for scenario 4)')
    args = parser.parse_args()

    print("\n" + "#" * 60)
    print("# DRONE TRACKING SIMULATION")
    print("# All algorithms from sparx_agency.core:")
    print("#   - RRTStarOmplPlanner")
    print(f"#   - {args.smoother.capitalize()}Smoother")
    print("#   - PurePursuitTracker")
    print("#" * 60)

    # Create scenario
    if args.scenario == 4:
        result = create_scenario_4(args.smoother, args.map_dir)
        if result is None:
            return
        costmap, start, goal, planner_params, smoother_params, tracker_params, sim_params = result
        obs_map = None  # Scenario 4 uses costmap directly
    else:
        scenario_fns = {1: create_scenario_1, 2: create_scenario_2, 3: create_scenario_3}
        obs_map, start, goal, planner_params, smoother_params, tracker_params, sim_params = \
            scenario_fns[args.scenario](args.smoother)
        costmap = None

    # Apply options
    if args.no_wind:
        sim_params = DroneSimParams(
            **{k: v for k, v in sim_params.__dict__.items()
               if k not in ('wind_enabled', 'gust_enabled')},
            wind_enabled=False, gust_enabled=False
        )
        print("\nWind and gusts DISABLED")

    collision_margin = 0.35 if args.scenario == 2 else 0.3

    # Run simulation
    success = run_simulation(
        obstacle_map=obs_map,
        costmap=costmap,
        start=start,
        goal=goal,
        planner_params=planner_params,
        smoother_type=args.smoother,
        smoother_params=smoother_params,
        tracker_params=tracker_params,
        sim_params=sim_params,
        max_time=args.max_time,
        collision_margin=collision_margin,
        seed=args.seed,
    )

    print("\n" + "=" * 60)
    print("SIMULATION COMPLETED!" if success else "SIMULATION ENDED")
    print("=" * 60)


if __name__ == "__main__":
    main()