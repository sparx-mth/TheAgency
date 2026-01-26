"""
Pygame visualization for the dynamic tracking environment.

Enhancement:
- Left click (no drag) => spawn dynamic circle with "random-ish" velocity.
- Left click + drag A->B => spawn dynamic circle that patrols along segment A<->B.
  When reaching B, it reverses toward A, and vice versa.
- If drag distance is tiny (or A==B) => treated like simple click => random velocity.

Controls:
- Left Click: spawn random-moving dynamic circle
- Left Drag: spawn segment-patrol dynamic circle
- C: clear dynamic obstacles
- R: toggle local radius overlay
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict, Any

import colorsys
import pygame

from sparx_agency.core.common.types import Path2D

from .map_dynamic import ObstacleMapDynamic


@dataclass
class ViewSettings:
    width: int = 1200
    height: int = 800
    hud_width: int = 290
    margin: float = 0.5

    bg_color: Tuple[int, int, int] = (240, 240, 245)
    grid_color: Tuple[int, int, int] = (200, 200, 200)

    obstacle_color: Tuple[int, int, int] = (100, 100, 100)
    obstacle_border: Tuple[int, int, int] = (50, 50, 50)

    dynamic_obstacle_color: Tuple[int, int, int] = (120, 80, 200)
    dynamic_obstacle_border: Tuple[int, int, int] = (70, 40, 140)

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

    local_radius_color: Tuple[int, int, int] = (80, 180, 255)  # base RGB

    # Spawn tuning
    dyn_default_radius_m: float = 0.25
    dyn_default_speed_mps: float = 0.6
    drag_threshold_px: int = 6  # below this => treat as click (random)


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

    def to_world(self, screen_x: int, screen_y: int) -> Tuple[float, float]:
        wx = (screen_x - self.offset_x) / self.scale + self.world_bounds[0]
        wy = self.world_bounds[3] - (screen_y - self.offset_y) / self.scale
        return float(wx), float(wy)

    def scale_distance(self, world_dist: float) -> int:
        return max(1, int(world_dist * self.scale))


def speed_to_color(speed: float, max_speed: float = 0.6) -> Tuple[int, int, int]:
    t = min(1.0, speed / max_speed) if max_speed > 1e-9 else 0.0
    hue = 0.6 * (1 - t)
    r, g, b = colorsys.hsv_to_rgb(hue, 0.8, 0.9)
    return int(r * 255), int(g * 255), int(b * 255)


class DroneVisualizer:
    """
    Pygame visualizer with:
    - static obstacles (from map)
    - dynamic obstacles (from map)
    - drone local interaction radius overlay
    - hazard flags in HUD (environment-only)
    - drag-to-define dynamic obstacle patrol path
    """

    def __init__(
        self,
        obstacle_map: Optional[ObstacleMapDynamic] = None,
        trajectory=None,
        raw_path: Optional[Path2D] = None,
        settings: Optional[ViewSettings] = None,
        drone_radius: float = 0.15,
        local_radius_m: float = 1.5,
        show_local_radius: bool = True,
    ):
        self.settings = settings or ViewSettings()
        self.obstacle_map = obstacle_map
        self.trajectory = trajectory
        self.raw_path = raw_path

        self.drone_radius = float(drone_radius)
        self.local_radius_m = float(local_radius_m)
        self.show_local_radius = bool(show_local_radius)

        pygame.init()
        pygame.font.init()

        self.screen = pygame.display.set_mode((self.settings.width, self.settings.height), pygame.RESIZABLE)
        pygame.display.set_caption("Trajectory Tracking Dynamic Environment - SPACE=pause, Q=quit")

        self.font_large = pygame.font.SysFont("monospace", 22, bold=True)
        self.font_medium = pygame.font.SysFont("monospace", 18)
        self.font_small = pygame.font.SysFont("monospace", 14)

        self._calculate_world_bounds()
        map_rect = pygame.Rect(10, 10, self.settings.width - self.settings.hud_width - 20, self.settings.height - 20)
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
        self.wind = (0.0, 0.0, 0.0)

        # Environment-only hazard flags (set by simulation loop)
        self.hazards: Dict[str, Any] = {
            "in_local_radius": False,
            "near_path_ahead": False,
            "dynamic_count": 0,
        }

        # Drag state for spawning
        self._drag_active = False
        self._drag_start_px: Optional[Tuple[int, int]] = None
        self._drag_current_px: Optional[Tuple[int, int]] = None

        self._render_static_elements()

        self.clock = pygame.time.Clock()
        self.running = True

    def _calculate_world_bounds(self) -> None:
        s = self.settings
        if self.obstacle_map:
            x_min = self.obstacle_map.origin_x - s.margin
            x_max = self.obstacle_map.origin_x + self.obstacle_map.width + s.margin
            y_min = self.obstacle_map.origin_y - s.margin
            y_max = self.obstacle_map.origin_y + self.obstacle_map.height + s.margin
        else:
            x_min, x_max, y_min, y_max = -5, 15, -5, 15
        self.world_bounds = (x_min, y_min, x_max, y_max)

    def _render_static_elements(self) -> None:
        map_width = self.settings.width - self.settings.hud_width
        self.static_surface = pygame.Surface((map_width, self.settings.height))
        self.static_surface.fill(self.settings.bg_color)
        self._draw_grid(self.static_surface)
        if self.obstacle_map:
            self._draw_static_obstacles(self.static_surface)
        if self.raw_path:
            self._draw_raw_path(self.static_surface)
        if self.trajectory:
            self._draw_trajectory(self.static_surface)

    def _draw_grid(self, surface: pygame.Surface) -> None:
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

    def _draw_static_obstacles(self, surface: pygame.Surface) -> None:
        s = self.settings
        if not self.obstacle_map:
            return

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

    def _draw_dynamic_obstacles(self, surface: pygame.Surface) -> None:
        s = self.settings
        if not self.obstacle_map:
            return
        for o in self.obstacle_map.dynamic_circles:
            center = self.transform.to_screen(o.cx, o.cy)
            radius = self.transform.scale_distance(o.r)
            pygame.draw.circle(surface, s.dynamic_obstacle_color, center, radius)
            pygame.draw.circle(surface, s.dynamic_obstacle_border, center, radius, 2)

    def _draw_raw_path(self, surface: pygame.Surface) -> None:
        s = self.settings
        if not self.raw_path or len(self.raw_path.points) < 2:
            return
        points = [self.transform.to_screen(p.x, p.y) for p in self.raw_path.points]
        pygame.draw.lines(surface, s.path_color, False, points, 1)

    def _draw_trajectory(self, surface: pygame.Surface) -> None:
        s = self.settings
        samples = self.trajectory.sample_by_time(0.05)
        if len(samples) < 2:
            return
        points = [self.transform.to_screen(p.x, p.y) for p in samples]
        pygame.draw.lines(surface, s.trajectory_color, False, points, 4)

        start = (samples[0].x, samples[0].y)
        goal = (samples[-1].x, samples[-1].y)
        pygame.draw.circle(surface, s.start_color, self.transform.to_screen(*start), 12)
        pygame.draw.circle(surface, (0, 100, 0), self.transform.to_screen(*start), 12, 3)
        pygame.draw.circle(surface, s.goal_color, self.transform.to_screen(*goal), 14)
        pygame.draw.circle(surface, (150, 0, 0), self.transform.to_screen(*goal), 14, 3)

    def _draw_trail(self, surface: pygame.Surface) -> None:
        if len(self.trail) < 2:
            return
        for i in range(1, len(self.trail)):
            x1, y1, s1 = self.trail[i - 1]
            x2, y2, s2 = self.trail[i]
            color = speed_to_color((s1 + s2) / 2)
            pygame.draw.line(surface, color, self.transform.to_screen(x1, y1), self.transform.to_screen(x2, y2), 4)

    def _draw_drone(self, surface: pygame.Surface) -> None:
        s = self.settings
        pos = self.transform.to_screen(self.drone_pos[0], self.drone_pos[1])
        drone_screen_radius = max(10, self.transform.scale_distance(self.drone_radius))
        pygame.draw.circle(surface, s.drone_color, pos, drone_screen_radius)
        pygame.draw.circle(surface, s.drone_border, pos, drone_screen_radius, 2)

        arrow_len = drone_screen_radius + 10
        tip = (pos[0] + arrow_len * math.cos(-self.drone_yaw), pos[1] + arrow_len * math.sin(-self.drone_yaw))
        angle = math.atan2(tip[1] - pos[1], tip[0] - pos[0])
        head_len, head_angle = max(6, drone_screen_radius // 2), math.pi / 6
        left = (tip[0] - head_len * math.cos(angle - head_angle), tip[1] - head_len * math.sin(angle - head_angle))
        right = (tip[0] - head_len * math.cos(angle + head_angle), tip[1] - head_len * math.sin(angle + head_angle))
        pygame.draw.line(surface, s.drone_border, pos, tip, 3)
        pygame.draw.polygon(surface, s.drone_border, [tip, left, right])

    def _draw_local_radius(self, surface: pygame.Surface) -> None:
        if not self.show_local_radius:
            return
        s = self.settings
        center = self.transform.to_screen(self.drone_pos[0], self.drone_pos[1])
        radius_px = max(1, self.transform.scale_distance(self.local_radius_m))

        overlay = pygame.Surface((radius_px * 2 + 2, radius_px * 2 + 2), pygame.SRCALPHA)
        fill_rgba = (*s.local_radius_color, 35)
        border_rgba = (*s.local_radius_color, 120)
        pygame.draw.circle(overlay, fill_rgba, (radius_px + 1, radius_px + 1), radius_px)
        pygame.draw.circle(overlay, border_rgba, (radius_px + 1, radius_px + 1), radius_px, 2)
        surface.blit(overlay, (center[0] - radius_px - 1, center[1] - radius_px - 1))

    def _draw_lookahead(self, surface: pygame.Surface) -> None:
        if not self.lookahead_point:
            return
        s = self.settings
        d_scr = self.transform.to_screen(self.drone_pos[0], self.drone_pos[1])
        l_scr = self.transform.to_screen(self.lookahead_point[0], self.lookahead_point[1])
        pygame.draw.line(surface, s.lookahead_color, d_scr, l_scr, 3)
        pygame.draw.circle(surface, s.lookahead_color, l_scr, 10)

    def _draw_drag_preview(self, surface: pygame.Surface) -> None:
        if not self._drag_active or self._drag_start_px is None or self._drag_current_px is None:
            return
        # Only draw inside the map surface, so convert to its local coordinates:
        # Note: we call this on the map subsurface, so event coords are already in screen coords.
        a = self._drag_start_px
        b = self._drag_current_px
        pygame.draw.line(surface, (0, 0, 0), a, b, 2)
        pygame.draw.circle(surface, (0, 0, 0), a, 5)
        pygame.draw.circle(surface, (0, 0, 0), b, 5)

    def _draw_hud(self, surface: pygame.Surface) -> None:
        s = self.settings
        hud_rect = pygame.Rect(self.settings.width - self.settings.hud_width, 0, self.settings.hud_width, self.settings.height)
        pygame.draw.rect(surface, s.hud_bg, hud_rect)

        x, y = hud_rect.x + 15, 18
        surface.blit(self.font_large.render("ENV STATUS", True, s.hud_text), (x, y))
        y += 35
        pygame.draw.line(surface, s.hud_text, (x, y), (hud_rect.right - 15, y), 1)
        y += 16

        surface.blit(self.font_medium.render(f"Time: {self.time:.1f}s", True, s.hud_text), (x, y))
        y += 26
        surface.blit(self.font_medium.render(f"Speed: {self.drone_speed:.2f} m/s", True, speed_to_color(self.drone_speed)), (x, y))
        y += 26

        cte_color = s.error_color if self.cross_track_error > 0.5 else (s.warning_color if self.cross_track_error > 0.3 else s.hud_text)
        surface.blit(self.font_medium.render(f"CTE: {self.cross_track_error:.3f} m", True, cte_color), (x, y))
        y += 26
        surface.blit(self.font_small.render(f"Pos: ({self.drone_pos[0]:.2f}, {self.drone_pos[1]:.2f})", True, s.hud_text), (x, y))
        y += 34

        pygame.draw.line(surface, s.hud_text, (x, y), (hud_rect.right - 15, y), 1)
        y += 16

        in_rad = bool(self.hazards.get("in_local_radius", False))
        near_path = bool(self.hazards.get("near_path_ahead", False))
        dyn_count = int(self.hazards.get("dynamic_count", 0))

        surface.blit(self.font_medium.render("LOCAL INTERACTION", True, s.hud_text), (x, y))
        y += 24

        color_rad = s.warning_color if in_rad else s.hud_text
        surface.blit(self.font_small.render(f"In radius: {in_rad}", True, color_rad), (x, y))
        y += 20

        color_path = s.warning_color if near_path else s.hud_text
        surface.blit(self.font_small.render(f"Near path-ahead: {near_path}", True, color_path), (x, y))
        y += 20

        surface.blit(self.font_small.render(f"Dynamic obstacles: {dyn_count}", True, s.hud_text), (x, y))
        y += 28

        pygame.draw.line(surface, s.hud_text, (x, y), (hud_rect.right - 15, y), 1)
        y += 16

        if self.gust_active:
            surface.blit(self.font_medium.render("⚠ GUST", True, s.warning_color), (x, y))
            y += 26
        if self.collision:
            surface.blit(self.font_medium.render("⛔ COLLISION", True, s.error_color), (x, y))
            y += 26
        if self.paused:
            surface.blit(self.font_medium.render("⏸ PAUSED", True, s.warning_color), (x, y))
            y += 26

        y = self.settings.height - 155
        pygame.draw.line(surface, s.hud_text, (x, y), (hud_rect.right - 15, y), 1)
        y += 14

        controls = [
            "SPACE - Pause",
            "Q/ESC - Quit",
            "L-Click - Spawn random",
            "L-Drag - Spawn A<->B patrol",
            "C - Clear dynamic",
            "R - Toggle radius",
        ]
        for c in controls:
            surface.blit(self.font_small.render(c, True, (180, 180, 180)), (x, y))
            y += 18

    def _clamp_to_map(self, wx: float, wy: float) -> Tuple[float, float]:
        if not self.obstacle_map:
            return float(wx), float(wy)
        wx = float(max(self.obstacle_map.origin_x, min(wx, self.obstacle_map.origin_x + self.obstacle_map.width)))
        wy = float(max(self.obstacle_map.origin_y, min(wy, self.obstacle_map.origin_y + self.obstacle_map.height)))
        return wx, wy

    def _spawn_random_circle_at(self, wx: float, wy: float) -> None:
        if not self.obstacle_map:
            return
        wx, wy = self._clamp_to_map(wx, wy)

        # Deterministic "random-ish" direction from position (no RNG needed)
        angle = (wx * 12.345 + wy * 7.89) % (2.0 * math.pi)
        speed = float(self.settings.dyn_default_speed_mps)
        vx = speed * math.cos(angle)
        vy = speed * math.sin(angle)

        self.obstacle_map.add_dynamic_circle(
            cx=wx,
            cy=wy,
            r=float(self.settings.dyn_default_radius_m),
            vx=vx,
            vy=vy,
        )

    def _spawn_patrol_circle(self, a_screen: Tuple[int, int], b_screen: Tuple[int, int]) -> None:
        if not self.obstacle_map:
            return

        ax, ay = self.transform.to_world(a_screen[0], a_screen[1])
        bx, by = self.transform.to_world(b_screen[0], b_screen[1])
        ax, ay = self._clamp_to_map(ax, ay)
        bx, by = self._clamp_to_map(bx, by)

        # If A and B are basically the same -> treat like click => random
        if (ax - bx) ** 2 + (ay - by) ** 2 < 1e-6:
            self._spawn_random_circle_at(ax, ay)
            return

        # Spawn at A, patrol A<->B
        speed = float(self.settings.dyn_default_speed_mps)
        self.obstacle_map.add_dynamic_circle(
            cx=ax,
            cy=ay,
            r=float(self.settings.dyn_default_radius_m),
            vx=0.0,
            vy=0.0,
            patrol_a=(ax, ay),
            patrol_b=(bx, by),
            patrol_speed=speed,
        )

    def update(
        self,
        drone_position,
        drone_velocity,
        drone_yaw,
        time,
        cross_track_error,
        progress_idx,
        lookahead_point=None,
        gust_active=False,
        collision=False,
        done=False,
        failed=False,
        wind=None,
        hazards: Optional[Dict[str, Any]] = None,
    ) -> bool:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
                return False

            if event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_q, pygame.K_ESCAPE):
                    self.running = False
                    return False
                if event.key == pygame.K_SPACE:
                    self.paused = not self.paused
                if event.key == pygame.K_c and self.obstacle_map:
                    self.obstacle_map.clear_dynamic()
                if event.key == pygame.K_r:
                    self.show_local_radius = not self.show_local_radius

            # Drag-to-spawn handling
            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                self._drag_active = True
                self._drag_start_px = event.pos
                self._drag_current_px = event.pos

            if event.type == pygame.MOUSEMOTION and self._drag_active:
                self._drag_current_px = event.pos

            if event.type == pygame.MOUSEBUTTONUP and event.button == 1 and self._drag_active:
                start_px = self._drag_start_px
                end_px = event.pos
                self._drag_active = False
                self._drag_current_px = None
                self._drag_start_px = None

                if start_px is not None:
                    dx = end_px[0] - start_px[0]
                    dy = end_px[1] - start_px[1]
                    dist2 = dx * dx + dy * dy

                    if dist2 <= self.settings.drag_threshold_px * self.settings.drag_threshold_px:
                        # Treat as click -> random velocity
                        wx, wy = self.transform.to_world(start_px[0], start_px[1])
                        self._spawn_random_circle_at(wx, wy)
                    else:
                        # Treat as drag -> A<->B patrol
                        self._spawn_patrol_circle(start_px, end_px)

            if event.type == pygame.VIDEORESIZE:
                self.settings.width, self.settings.height = event.w, event.h
                self.screen = pygame.display.set_mode((event.w, event.h), pygame.RESIZABLE)
                map_rect = pygame.Rect(10, 10, self.settings.width - self.settings.hud_width - 20, self.settings.height - 20)
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

        if hazards is not None:
            self.hazards = hazards

        self.trail.append((drone_position[0], drone_position[1], self.drone_speed))
        if len(self.trail) > self.max_trail_points:
            self.trail = self.trail[-self.max_trail_points:]

        self._render_frame()
        self.clock.tick(60)
        return self.running

    def _render_frame(self) -> None:
        self.screen.fill(self.settings.bg_color)
        self.screen.blit(self.static_surface, (0, 0))

        map_width = self.settings.width - self.settings.hud_width
        map_height = self.settings.height

        if map_width > 0 and map_height > 0:
            map_rect = pygame.Rect(0, 0, min(map_width, self.screen.get_width()), min(map_height, self.screen.get_height()))
            try:
                map_surface = self.screen.subsurface(map_rect)
                self._draw_trail(map_surface)
                self._draw_dynamic_obstacles(map_surface)
                self._draw_local_radius(map_surface)
                self._draw_lookahead(map_surface)
                self._draw_drone(map_surface)

                # Drag preview (screen coords are fine because map_surface is a subsurface at (0,0))
                if self._drag_active and self._drag_start_px and self._drag_current_px:
                    self._draw_drag_preview(map_surface)
            except ValueError:
                pass

        self._draw_hud(self.screen)
        pygame.display.flip()

    def close(self) -> None:
        pygame.quit()

    def wait_for_close(self) -> None:
        while self.running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False
                elif event.type == pygame.KEYDOWN and event.key in (pygame.K_q, pygame.K_ESCAPE, pygame.K_RETURN):
                    self.running = False
                elif event.type == pygame.VIDEORESIZE:
                    self.settings.width, self.settings.height = event.w, event.h
                    self.screen = pygame.display.set_mode((event.w, event.h), pygame.RESIZABLE)
                    map_rect = pygame.Rect(10, 10, self.settings.width - self.settings.hud_width - 20, self.settings.height - 20)
                    self.transform = WorldToScreen(self.world_bounds, map_rect)
                    self._render_static_elements()

            if self.running:
                try:
                    self._render_frame()
                except ValueError:
                    pass
            self.clock.tick(30)
        self.close()
