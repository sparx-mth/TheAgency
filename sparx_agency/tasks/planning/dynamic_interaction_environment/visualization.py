"""
Pygame visualization for trajectory tracking - CLEAN VERSION.

SIMPLIFIED:
- Click to place static obstacles
- No dynamic/moving obstacles
- No local planner visualization
- Clean HUD with essential info

Controls:
- Left Click: Place obstacle at cursor position
- Right Click: Remove obstacle at cursor position (if any)
- C: Clear all placed obstacles
- SPACE: Pause/resume
- Q/ESC: Quit
"""
from __future__ import annotations

import math
import colorsys
from dataclasses import dataclass
from typing import List, Tuple, Optional, Any

import pygame

from sparx_agency.core.common.types import Path2D

from .obstacle_map import ObstacleMap


@dataclass
class ViewSettings:
    """Visualization settings."""
    width: int = 1200
    height: int = 800
    hud_width: int = 280
    margin: float = 0.5

    # Colors
    bg_color: Tuple[int, int, int] = (240, 240, 245)
    grid_color: Tuple[int, int, int] = (200, 200, 200)

    obstacle_color: Tuple[int, int, int] = (100, 100, 100)
    obstacle_border: Tuple[int, int, int] = (50, 50, 50)

    placed_obstacle_color: Tuple[int, int, int] = (180, 100, 180)
    placed_obstacle_border: Tuple[int, int, int] = (120, 60, 120)

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
    success_color: Tuple[int, int, int] = (50, 255, 100)


class WorldToScreen:
    """Transform between world coordinates and screen pixels."""

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
    """Convert speed to a color (blue=slow, red=fast)."""
    t = min(1.0, speed / max_speed) if max_speed > 1e-9 else 0.0
    hue = 0.6 * (1 - t)
    r, g, b = colorsys.hsv_to_rgb(hue, 0.8, 0.9)
    return int(r * 255), int(g * 255), int(b * 255)


class Visualizer:
    """
    Pygame visualizer with:
    - Static obstacles
    - Click-to-place obstacles (static)
    - Trajectory and path
    - Drone with trail
    - Simple HUD
    """

    def __init__(
        self,
        obstacle_map: Optional[ObstacleMap] = None,
        trajectory: Any = None,
        raw_path: Optional[Path2D] = None,
        settings: Optional[ViewSettings] = None,
        drone_radius: float = 0.15,
        click_obstacle_radius: float = 0.25,
        click_obstacles_enabled: bool = True,
    ):
        self.settings = settings or ViewSettings()
        self.obstacle_map = obstacle_map
        self.trajectory = trajectory
        self.raw_path = raw_path

        self.drone_radius = float(drone_radius)
        self.click_obstacle_radius = float(click_obstacle_radius)
        self.click_obstacles_enabled = click_obstacles_enabled

        pygame.init()
        pygame.font.init()

        self.screen = pygame.display.set_mode(
            (self.settings.width, self.settings.height),
            pygame.RESIZABLE
        )
        pygame.display.set_caption("Trajectory Tracking - Click to place obstacles")

        self.font_large = pygame.font.SysFont("monospace", 22, bold=True)
        self.font_medium = pygame.font.SysFont("monospace", 18)
        self.font_small = pygame.font.SysFont("monospace", 14)

        self._calculate_world_bounds()
        map_rect = pygame.Rect(
            10, 10,
            self.settings.width - self.settings.hud_width - 20,
            self.settings.height - 20
        )
        self.transform = WorldToScreen(self.world_bounds, map_rect)

        # Drone state
        self.drone_pos = (0.0, 0.0, 0.0)
        self.drone_yaw = 0.0
        self.drone_speed = 0.0
        self.lookahead_point = None
        self.trail: List[Tuple[float, float, float]] = []
        self.max_trail_points = 5000

        # Simulation state
        self.time = 0.0
        self.cross_track_error = 0.0
        self.progress_idx = 0
        self.gust_active = False
        self.collision = False
        self.done = False
        self.failed = False
        self.paused = False
        self.wind = (0.0, 0.0, 0.0)

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
        """Render static elements to a surface (obstacles, trajectory)."""
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

    def _draw_placed_obstacles(self, surface: pygame.Surface) -> None:
        """Draw click-placed obstacles."""
        s = self.settings
        if not self.obstacle_map:
            return
        for o in self.obstacle_map.placed_obstacles:
            center = self.transform.to_screen(o.cx, o.cy)
            radius = self.transform.scale_distance(o.r)
            pygame.draw.circle(surface, s.placed_obstacle_color, center, radius)
            pygame.draw.circle(surface, s.placed_obstacle_border, center, radius, 2)

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
            pygame.draw.line(
                surface,
                color,
                self.transform.to_screen(x1, y1),
                self.transform.to_screen(x2, y2),
                4
            )

    def _draw_drone(self, surface: pygame.Surface) -> None:
        s = self.settings
        pos = self.transform.to_screen(self.drone_pos[0], self.drone_pos[1])
        drone_screen_radius = max(10, self.transform.scale_distance(self.drone_radius))
        pygame.draw.circle(surface, s.drone_color, pos, drone_screen_radius)
        pygame.draw.circle(surface, s.drone_border, pos, drone_screen_radius, 2)

        # Direction arrow
        arrow_len = drone_screen_radius + 10
        tip = (
            pos[0] + arrow_len * math.cos(-self.drone_yaw),
            pos[1] + arrow_len * math.sin(-self.drone_yaw)
        )
        angle = math.atan2(tip[1] - pos[1], tip[0] - pos[0])
        head_len, head_angle = max(6, drone_screen_radius // 2), math.pi / 6
        left = (
            tip[0] - head_len * math.cos(angle - head_angle),
            tip[1] - head_len * math.sin(angle - head_angle)
        )
        right = (
            tip[0] - head_len * math.cos(angle + head_angle),
            tip[1] - head_len * math.sin(angle + head_angle)
        )
        pygame.draw.line(surface, s.drone_border, pos, tip, 3)
        pygame.draw.polygon(surface, s.drone_border, [tip, left, right])

    def _draw_lookahead(self, surface: pygame.Surface) -> None:
        if not self.lookahead_point:
            return
        s = self.settings
        d_scr = self.transform.to_screen(self.drone_pos[0], self.drone_pos[1])
        l_scr = self.transform.to_screen(self.lookahead_point[0], self.lookahead_point[1])
        pygame.draw.line(surface, s.lookahead_color, d_scr, l_scr, 3)
        pygame.draw.circle(surface, s.lookahead_color, l_scr, 10)

    def _draw_hud(self, surface: pygame.Surface) -> None:
        s = self.settings
        hud_x = self.settings.width - self.settings.hud_width
        hud_rect = pygame.Rect(hud_x, 0, self.settings.hud_width, self.settings.height)
        pygame.draw.rect(surface, s.hud_bg, hud_rect)

        y = 15
        line_height = 24

        # Title
        title = self.font_large.render("TRAJECTORY TRACKING", True, s.hud_text)
        surface.blit(title, (hud_x + 10, y))
        y += line_height + 10

        # Time
        time_txt = self.font_medium.render(f"Time: {self.time:.2f}s", True, s.hud_text)
        surface.blit(time_txt, (hud_x + 10, y))
        y += line_height

        # Position
        pos_txt = self.font_small.render(
            f"Pos: ({self.drone_pos[0]:.2f}, {self.drone_pos[1]:.2f})",
            True, s.hud_text
        )
        surface.blit(pos_txt, (hud_x + 10, y))
        y += line_height

        # Speed
        speed_txt = self.font_small.render(f"Speed: {self.drone_speed:.2f} m/s", True, s.hud_text)
        surface.blit(speed_txt, (hud_x + 10, y))
        y += line_height

        # Cross-track error
        cte_txt = self.font_small.render(f"CTE: {self.cross_track_error:.3f}m", True, s.hud_text)
        surface.blit(cte_txt, (hud_x + 10, y))
        y += line_height + 10

        # Wind
        wind_txt = self.font_small.render(
            f"Wind: ({self.wind[0]:.2f}, {self.wind[1]:.2f})",
            True, s.hud_text
        )
        surface.blit(wind_txt, (hud_x + 10, y))
        y += line_height

        # Gust indicator
        if self.gust_active:
            gust_txt = self.font_medium.render("GUST!", True, s.warning_color)
            surface.blit(gust_txt, (hud_x + 10, y))
        y += line_height + 10

        # Placed obstacles count
        placed_count = len(self.obstacle_map.placed_obstacles) if self.obstacle_map else 0
        placed_txt = self.font_small.render(f"Placed obstacles: {placed_count}", True, s.hud_text)
        surface.blit(placed_txt, (hud_x + 10, y))
        y += line_height + 20

        # Status
        if self.done:
            status_txt = self.font_large.render("GOAL REACHED!", True, s.success_color)
            surface.blit(status_txt, (hud_x + 10, y))
        elif self.collision:
            status_txt = self.font_medium.render("COLLISION!", True, s.warning_color)
            surface.blit(status_txt, (hud_x + 10, y))
        elif self.paused:
            status_txt = self.font_medium.render("PAUSED", True, s.warning_color)
            surface.blit(status_txt, (hud_x + 10, y))

        # Controls help at bottom
        y = self.settings.height - 120
        pygame.draw.line(surface, s.hud_text, (hud_x + 10, y), (hud_x + self.settings.hud_width - 10, y), 1)
        y += 10

        controls = [
            "Controls:",
            "  Click: Place obstacle",
            "  Right-click: Remove",
            "  C: Clear all",
            "  SPACE: Pause",
            "  Q: Quit",
        ]
        for line in controls:
            txt = self.font_small.render(line, True, s.hud_text)
            surface.blit(txt, (hud_x + 10, y))
            y += 18

    def _handle_click(self, pos: Tuple[int, int], button: int) -> None:
        """Handle mouse click - place or remove obstacle."""
        if not self.obstacle_map or not self.click_obstacles_enabled:
            return

        wx, wy = self.transform.to_world(pos[0], pos[1])

        # Check if click is within map bounds
        if not (self.obstacle_map.origin_x <= wx <= self.obstacle_map.origin_x + self.obstacle_map.width and
                self.obstacle_map.origin_y <= wy <= self.obstacle_map.origin_y + self.obstacle_map.height):
            return

        if button == 1:  # Left click - place
            self.obstacle_map.place_obstacle(wx, wy, self.click_obstacle_radius)
        elif button == 3:  # Right click - remove
            obs_id = self.obstacle_map.get_obstacle_at(wx, wy)
            if obs_id is not None:
                self.obstacle_map.remove_obstacle_by_id(obs_id)

    def update(
        self,
        drone_position: Tuple[float, float, float],
        drone_velocity: Tuple[float, float, float],
        drone_yaw: float,
        time: float,
        cross_track_error: float,
        progress_idx: int,
        lookahead_point: Optional[Tuple[float, float, float]] = None,
        gust_active: bool = False,
        collision: bool = False,
        done: bool = False,
        failed: bool = False,
        wind: Optional[Tuple[float, float, float]] = None,
    ) -> bool:
        """Update visualization. Returns False if window was closed."""

        # Handle events
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
                    self.obstacle_map.clear_placed_obstacles()

            if event.type == pygame.MOUSEBUTTONDOWN:
                if event.button in (1, 3):  # Left or right click
                    self._handle_click(event.pos, event.button)

            if event.type == pygame.VIDEORESIZE:
                self.settings.width, self.settings.height = event.w, event.h
                self.screen = pygame.display.set_mode((event.w, event.h), pygame.RESIZABLE)
                map_rect = pygame.Rect(
                    10, 10,
                    self.settings.width - self.settings.hud_width - 20,
                    self.settings.height - 20
                )
                self.transform = WorldToScreen(self.world_bounds, map_rect)
                self._render_static_elements()

        if self.paused:
            self._render_frame()
            self.clock.tick(30)
            return True

        # Update state
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

        # Update trail
        self.trail.append((drone_position[0], drone_position[1], self.drone_speed))
        if len(self.trail) > self.max_trail_points:
            self.trail = self.trail[-self.max_trail_points:]

        self._render_frame()
        self.clock.tick(60)
        return self.running

    def _render_frame(self) -> None:
        """Render a single frame."""
        self.screen.fill(self.settings.bg_color)
        self.screen.blit(self.static_surface, (0, 0))

        map_width = self.settings.width - self.settings.hud_width
        map_height = self.settings.height

        if map_width > 0 and map_height > 0:
            map_rect = pygame.Rect(
                0, 0,
                min(map_width, self.screen.get_width()),
                min(map_height, self.screen.get_height())
            )
            try:
                map_surface = self.screen.subsurface(map_rect)
                self._draw_trail(map_surface)
                self._draw_placed_obstacles(map_surface)
                self._draw_lookahead(map_surface)
                self._draw_drone(map_surface)
            except ValueError:
                pass

        self._draw_hud(self.screen)
        pygame.display.flip()

    def close(self) -> None:
        """Close the visualizer."""
        pygame.quit()

    def wait_for_close(self) -> None:
        """Wait for user to close the window."""
        while self.running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False
                elif event.type == pygame.KEYDOWN and event.key in (pygame.K_q, pygame.K_ESCAPE, pygame.K_RETURN):
                    self.running = False
                elif event.type == pygame.VIDEORESIZE:
                    self.settings.width, self.settings.height = event.w, event.h
                    self.screen = pygame.display.set_mode((event.w, event.h), pygame.RESIZABLE)
                    map_rect = pygame.Rect(
                        10, 10,
                        self.settings.width - self.settings.hud_width - 20,
                        self.settings.height - 20
                    )
                    self.transform = WorldToScreen(self.world_bounds, map_rect)
                    self._render_static_elements()

            if self.running:
                try:
                    self._render_frame()
                except ValueError:
                    pass
            self.clock.tick(30)
        self.close()