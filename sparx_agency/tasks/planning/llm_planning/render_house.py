#!/usr/bin/env python3
"""
render_house_dynamic.py - Headless House Renderer

Renders house map to an image file (no Pygame window).
Auto-reloads to show real-time updates via saved images.
HEADLESS: Uses off-screen surface only — no display window.
"""

import os
# Force SDL to use a dummy video driver (no window)
os.environ["SDL_VIDEODRIVER"] = "dummy"

import pygame
import numpy as np
import json
import sys
import re
from house_config import get_config

cfg = get_config()


class DynamicHouseRenderer:
    """Headless renderer — draws to an off-screen surface and saves to PNG."""

    def __init__(self, unified_json=None, map_txt=None, cell_size=None):
        """Initialize the renderer."""
        unified_json = unified_json or cfg.unified_rooms_json
        map_txt = map_txt or cfg.house_map_txt
        cell_size = cell_size or cfg.cell_size

        # Load structure
        with open(unified_json, 'r') as f:
            self.structure = json.load(f)

        # Get dimensions
        self.house_width_m = self.structure["house_dimensions_m"]["width"]
        self.house_height_m = self.structure["house_dimensions_m"]["height"]
        self.grid_resolution = self.structure["grid_resolution"]

        # Load dynamic tile registry
        self.tile_registry = {}
        self.tile_colors = {}
        self.load_tile_registry()

        # Load grid and derive dimensions from actual array shape
        try:
            self.grid = np.loadtxt(map_txt, dtype=np.int8)
        except:
            fallback_w = int(round(self.house_width_m / self.grid_resolution))
            fallback_h = int(round(self.house_height_m / self.grid_resolution))
            self.grid = np.full((fallback_h, fallback_w), 0, dtype=np.int8)

        # Use actual grid shape — never recompute from floats
        self.grid_height, self.grid_width = self.grid.shape

        # Display parameters
        self.cell_size = max(5, min(50, cell_size))
        self.legend_width = cfg.legend_width
        self.window_width = self.grid_width * self.cell_size + self.legend_width
        self.window_height = self.grid_height * self.cell_size

        # Initialize pygame in headless mode
        pygame.init()
        # Create an off-screen surface instead of a display window
        self.screen = pygame.Surface((self.window_width, self.window_height))

        # Multiple fonts for different purposes
        self.font_title = pygame.font.Font(None, 26)
        self.font_stats = pygame.font.Font(None, 22)
        self.font_objects = pygame.font.Font(None, 40)

        # CHANGE DETECTION - Track grid state
        self.last_grid_hash = None
        self.last_structure_hash = None
        self.last_save_time = 0
        self.FORCE_SAVE_INTERVAL = 20  # seconds — keep web fresh

    def load_tile_registry(self):
        """Load tile types and colors from JSON (single source of truth)."""
        self.tile_registry = self.structure.get("tile_registry", {})
        hex_colors = self.structure.get("tile_colors", {})
        for name, hex_c in hex_colors.items():
            tid = self.tile_registry.get(name)
            if tid is not None:
                h = hex_c.lstrip('#')
                self.tile_colors[tid] = (int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16))
        self.tile_colors.setdefault(-1, (20, 20, 20))

    def save_map_image(self, filename=None):
        """Save current surface to file for web display."""
        filename = filename or cfg.current_map_png
        os.makedirs(os.path.dirname(filename) or '.', exist_ok=True)
        pygame.image.save(self.screen, filename)
        print(f"[Map saved: {filename}]")

    def format_object_name(self, name):
        """Format object names with proper spacing."""
        display_name = name.replace('_', ' ')
        display_name = re.sub(r'([a-z])([A-Z])', r'\1 \2', display_name)
        display_name = re.sub(r'([A-Z]+)([A-Z][a-z])', r'\1 \2', display_name)
        display_name = re.sub(r'([a-zA-Z])(And)([A-Z])', r'\1 \2 \3', display_name)
        display_name = ' '.join(display_name.split())

        words = display_name.split()
        formatted_words = []
        for word in words:
            if word.lower() in ['and', 'or', 'the', 'of', 'in', 'on', 'at']:
                formatted_words.append(word.lower())
            else:
                formatted_words.append(word.capitalize())
        if formatted_words:
            formatted_words[0] = formatted_words[0].capitalize()
        return ' '.join(formatted_words)

    def render(self):
        """Render the grid to the off-screen surface."""
        self.screen.fill((30, 30, 30))

        # Draw grid
        for y in range(self.grid_height):
            for x in range(self.grid_width):
                tile_type = self.grid[y, x]
                color = self.tile_colors.get(tile_type, (50, 50, 50))

                rect = pygame.Rect(x * self.cell_size, y * self.cell_size,
                                   self.cell_size, self.cell_size)
                pygame.draw.rect(self.screen, color, rect)
                pygame.draw.rect(self.screen, (60, 60, 60), rect, 1)

        # Draw legend
        self.draw_legend()

        # CHANGE DETECTION - Only save when grid or structure changes, or every 20s
        import time as _time
        current_grid_hash = hash(self.grid.tobytes())
        current_structure_hash = hash(json.dumps(self.structure, sort_keys=True))
        now = _time.time()

        changed = (current_grid_hash != self.last_grid_hash or
                   current_structure_hash != self.last_structure_hash)
        force = (now - self.last_save_time) >= self.FORCE_SAVE_INTERVAL

        if changed or force:
            self.save_map_image()
            self.last_grid_hash = current_grid_hash
            self.last_structure_hash = current_structure_hash
            self.last_save_time = now

    def wrap_text(self, text, max_width, font=None):
        """Wrap text to fit within max_width pixels."""
        if font is None:
            font = self.font_objects

        words = text.split(' ')
        lines = []
        current_line = []

        for word in words:
            test_line = ' '.join(current_line + [word])
            if font.size(test_line)[0] <= max_width:
                current_line.append(word)
            else:
                if current_line:
                    lines.append(' '.join(current_line))
                    current_line = [word]
                else:
                    lines.append(word)

        if current_line:
            lines.append(' '.join(current_line))

        return lines if lines else [text]

    def draw_legend(self):
        """Draw legend with dynamic tiles."""
        legend_rect = pygame.Rect(self.grid_width * self.cell_size, 0,
                                  self.legend_width, self.window_height)
        pygame.draw.rect(self.screen, (40, 40, 40), legend_rect)

        present_types = set(self.grid.flatten())

        y_offset = 10
        x_base = self.grid_width * self.cell_size + 10

        title = self.font_title.render("DETECTED OBJECTS", True, (255, 255, 255))
        self.screen.blit(title, (x_base, y_offset))
        y_offset += 35

        stats_text = f"Total: {len(self.structure.get('rooms', {}).get('main_room', {}).get('objects', []))} objects"
        stats = self.font_stats.render(stats_text, True, (180, 180, 180))
        self.screen.blit(stats, (x_base, y_offset))
        y_offset += 25

        pygame.draw.line(self.screen, (80, 80, 80),
                         (x_base, y_offset), (x_base + self.legend_width - 20, y_offset))
        y_offset += 15

        sorted_tiles = sorted([(name, tid) for name, tid in self.tile_registry.items()
                               if tid in present_types], key=lambda x: x[0])

        for name, tile_id in sorted_tiles:
            if y_offset > self.window_height - 40:
                break

            color = self.tile_colors[tile_id]
            box_rect = pygame.Rect(x_base, y_offset + 2, 22, 22)
            pygame.draw.rect(self.screen, color, box_rect)
            pygame.draw.rect(self.screen, (200, 200, 200), box_rect, 1)

            display_name = self.format_object_name(name)
            if display_name.lower() == "free space":
                display_name = "Empty"
            elif display_name.lower() == "entry point":
                display_name = "Entry Point"

            max_text_width = self.legend_width - 60
            wrapped_lines = self.wrap_text(display_name, max_text_width, self.font_objects)

            line_height = 26
            for i, line in enumerate(wrapped_lines):
                label = self.font_objects.render(line, True, (220, 220, 220))
                self.screen.blit(label, (x_base + 35, y_offset + (i * line_height)))

            y_offset += max(32, len(wrapped_lines) * line_height + 10)

    def reload(self):
        """Reload map and structure."""
        try:
            new_grid = np.loadtxt(cfg.house_map_txt, dtype=np.int8)
            self.grid = new_grid
            # Update dimensions from actual grid shape
            new_h, new_w = new_grid.shape
            if new_w != self.grid_width or new_h != self.grid_height:
                self.grid_width, self.grid_height = new_w, new_h
                self.window_width = self.grid_width * self.cell_size + self.legend_width
                self.window_height = self.grid_height * self.cell_size
                self.screen = pygame.Surface((self.window_width, self.window_height))

            with open(cfg.unified_rooms_json, 'r') as f:
                self.structure = json.load(f)
                self.load_tile_registry()
        except:
            pass

    def render_once(self):
        """Render a single frame and save. Use this for one-shot image generation."""
        self.render()
        print(f"Headless render complete -> {cfg.current_map_png}")

    def run(self):
        """Main loop — headless polling mode (no window)."""
        import time

        print("Headless House Renderer - No Display Window")
        print("-" * 60)
        print(f"Grid: {self.grid_width}x{self.grid_height}")
        print(f"Cell size: {self.cell_size}px")
        print(f"Legend width: {self.legend_width}px")
        print(f"Output: {cfg.current_map_png}")
        print(f"Reload interval: {cfg.reload_interval}ms")
        print("\nRunning headless (Ctrl+C to stop)...")
        print("-" * 60)

        try:
            while True:
                self.reload()
                self.render()
                time.sleep(cfg.reload_interval / 1000.0)
        except KeyboardInterrupt:
            print("\nStopped.")
        finally:
            pygame.quit()


if __name__ == "__main__":
    try:
        renderer = DynamicHouseRenderer()
        renderer.run()
    except FileNotFoundError:
        print(f"Error: {cfg.unified_rooms_json} or {cfg.house_map_txt} not found")
        print("Run the pixel_room_mapper.py first!")
    except Exception as e:
        print(f"Error: {e}")