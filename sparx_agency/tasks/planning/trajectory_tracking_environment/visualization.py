"""
Pygame-based visualization for drone simulation.

Provides real-time rendering of drone state, trajectory, and obstacles.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Tuple, Optional

import colorsys
import pygame

from sparx_agency.core.common.types import Path2D

from map_loading import ObstacleMap


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
                 drone_radius: float = 0.15):
        self.settings = settings or ViewSettings()
        self.obstacle_map = obstacle_map
        self.trajectory = trajectory
        self.raw_path = raw_path
        self.drone_radius = drone_radius  # Actual collision radius for accurate visualization

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
        self.wind = (0.0, 0.0, 0.0)  # Current wind velocity (vx, vy, vz)

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

    def _draw_raw_path(self, surface):
        """Draw the raw RRT* waypoints with thin connecting lines."""
        s = self.settings
        if not self.raw_path or len(self.raw_path.points) < 2:
            return
        # Draw thin straight lines connecting waypoints
        points = [self.transform.to_screen(p.x, p.y) for p in self.raw_path.points]
        pygame.draw.lines(surface, s.path_color, False, points, 1)  # 1 = thin line

    def _draw_trajectory(self, surface):
        """Draw the smoothed trajectory."""
        s = self.settings
        samples = self.trajectory.sample_by_time(0.05)
        if len(samples) < 2:
            return
        points = [self.transform.to_screen(p.x, p.y) for p in samples]
        pygame.draw.lines(surface, s.trajectory_color, False, points, 4)

        # Start and goal
        start = (samples[0].x, samples[0].y)
        goal = (samples[-1].x, samples[-1].y)
        pygame.draw.circle(surface, s.start_color, self.transform.to_screen(*start), 12)
        pygame.draw.circle(surface, (0, 100, 0), self.transform.to_screen(*start), 12, 3)
        pygame.draw.circle(surface, s.goal_color, self.transform.to_screen(*goal), 14)
        pygame.draw.circle(surface, (150, 0, 0), self.transform.to_screen(*goal), 14, 3)

    def _draw_trail(self, surface):
        if len(self.trail) < 2:
            return
        for i in range(1, len(self.trail)):
            x1, y1, s1 = self.trail[i - 1]
            x2, y2, s2 = self.trail[i]
            color = speed_to_color((s1 + s2) / 2)
            pygame.draw.line(surface, color, self.transform.to_screen(x1, y1),
                             self.transform.to_screen(x2, y2), 4)

    def _draw_drone(self, surface):
        s = self.settings
        pos = self.transform.to_screen(self.drone_pos[0], self.drone_pos[1])
        # Draw drone with its actual collision radius
        drone_screen_radius = self.transform.scale_distance(self.drone_radius) if hasattr(self, 'drone_radius') else 10
        drone_screen_radius = max(10, drone_screen_radius)  # Minimum 10px for visibility
        pygame.draw.circle(surface, s.drone_color, pos, drone_screen_radius)
        pygame.draw.circle(surface, s.drone_border, pos, drone_screen_radius, 2)
        arrow_len = drone_screen_radius + 10
        tip = (pos[0] + arrow_len * math.cos(-self.drone_yaw),
               pos[1] + arrow_len * math.sin(-self.drone_yaw))
        angle = math.atan2(tip[1] - pos[1], tip[0] - pos[0])
        head_len, head_angle = max(6, drone_screen_radius // 2), math.pi / 6
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
        pygame.draw.line(surface, s.lookahead_color, d_scr, l_scr, 3)
        pygame.draw.circle(surface, s.lookahead_color, l_scr, 10)

    def _draw_wind_indicator(self, surface):
        """Draw wind direction indicator in top-left corner."""
        s = self.settings
        wind_x, wind_y = self.wind[0], self.wind[1]
        wind_speed = math.sqrt(wind_x**2 + wind_y**2)

        # Indicator position and size
        center_x, center_y = 60, 60
        radius = 40

        # Background circle
        pygame.draw.circle(surface, (60, 60, 70), (center_x, center_y), radius + 5)
        pygame.draw.circle(surface, (40, 40, 50), (center_x, center_y), radius)
        pygame.draw.circle(surface, (100, 100, 110), (center_x, center_y), radius, 2)

        # Draw compass directions
        for i, label in enumerate(['N', 'E', 'S', 'W']):
            angle = -math.pi/2 + i * math.pi/2
            lx = center_x + (radius - 12) * math.cos(angle)
            ly = center_y + (radius - 12) * math.sin(angle)
            text = self.font_small.render(label, True, (150, 150, 160))
            text_rect = text.get_rect(center=(lx, ly))
            surface.blit(text, text_rect)

        # Draw wind arrow if there's wind
        if wind_speed > 0.001:
            # Wind direction (angle from +X axis, but Y is inverted in screen coords)
            wind_angle = math.atan2(-wind_y, wind_x)  # Negative Y because screen Y is inverted

            # Arrow length proportional to wind speed (max at ~0.3 m/s)
            arrow_len = min(radius - 5, (wind_speed / 0.3) * (radius - 5))
            arrow_len = max(10, arrow_len)  # Minimum arrow length

            # Arrow tip
            tip_x = center_x + arrow_len * math.cos(wind_angle)
            tip_y = center_y + arrow_len * math.sin(wind_angle)

            # Arrow color based on strength (blue to red)
            intensity = min(1.0, wind_speed / 0.3)
            if self.gust_active:
                arrow_color = (255, 100, 50)  # Orange for gust
            else:
                r = int(100 + 155 * intensity)
                g = int(200 - 100 * intensity)
                b = int(255 - 155 * intensity)
                arrow_color = (r, g, b)

            # Draw arrow line
            pygame.draw.line(surface, arrow_color, (center_x, center_y), (tip_x, tip_y), 3)

            # Arrow head
            head_len = 10
            head_angle = math.pi / 6
            left = (tip_x - head_len * math.cos(wind_angle - head_angle),
                    tip_y - head_len * math.sin(wind_angle - head_angle))
            right = (tip_x - head_len * math.cos(wind_angle + head_angle),
                     tip_y - head_len * math.sin(wind_angle + head_angle))
            pygame.draw.polygon(surface, arrow_color, [(tip_x, tip_y), left, right])

        # Wind speed text
        speed_text = f"{wind_speed:.2f} m/s"
        text_surface = self.font_small.render(speed_text, True, (200, 200, 210))
        text_rect = text_surface.get_rect(center=(center_x, center_y + radius + 15))
        surface.blit(text_surface, text_rect)

        # "WIND" label
        label_surface = self.font_small.render("WIND", True, (180, 180, 190))
        label_rect = label_surface.get_rect(center=(center_x, center_y - radius - 12))
        surface.blit(label_surface, label_rect)

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
               done=False, failed=False, wind=None) -> bool:
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
        self.wind = wind if wind else (0.0, 0.0, 0.0)

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

        # Calculate map surface rect safely
        map_width = self.settings.width - self.settings.hud_width
        map_height = self.settings.height

        # Ensure dimensions are valid
        if map_width > 0 and map_height > 0:
            map_rect = pygame.Rect(0, 0, min(map_width, self.screen.get_width()),
                                   min(map_height, self.screen.get_height()))
            try:
                map_surface = self.screen.subsurface(map_rect)
                self._draw_trail(map_surface)
                self._draw_lookahead(map_surface)
                self._draw_drone(map_surface)
                self._draw_wind_indicator(map_surface)
            except ValueError:
                pass  # Skip drawing if subsurface is invalid

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
                elif event.type == pygame.VIDEORESIZE:
                    self.settings.width, self.settings.height = event.w, event.h
                    self.screen = pygame.display.set_mode((event.w, event.h), pygame.RESIZABLE)
                    map_rect = pygame.Rect(10, 10, self.settings.width - self.settings.hud_width - 20,
                                           self.settings.height - 20)
                    self.transform = WorldToScreen(self.world_bounds, map_rect)
                    self._render_static_elements()
            if self.running:
                try:
                    self._render_frame()
                except ValueError:
                    # Handle subsurface error on resize
                    pass
            self.clock.tick(30)
        self.close()