#!/usr/bin/env python3
"""
render_house_dynamic.py - Dynamic Pygame House Renderer

Renders house with dynamic tiles loaded from the JSON file.
Auto-reloads to show real-time updates.
NOW SAVES IMAGES FOR WEB DISPLAY ONLY WHEN MAP CHANGES
"""

import pygame
import numpy as np
import json
import sys
import os  # ADDED FOR WEB INTEGRATION


class DynamicHouseRenderer:
    """Pygame renderer with dynamic tile support."""

    def __init__(self, unified_json="unified_rooms.json", map_txt="house_map.txt", cell_size=20):
        """Initialize the renderer."""
        # Load structure
        with open(unified_json, 'r') as f:
            self.structure = json.load(f)

        # Get dimensions
        self.house_width_m = self.structure["house_dimensions_m"]["width"]
        self.house_height_m = self.structure["house_dimensions_m"]["height"]
        self.grid_resolution = self.structure["grid_resolution"]

        # Calculate grid size
        self.grid_width = int(self.house_width_m / self.grid_resolution)
        self.grid_height = int(self.house_height_m / self.grid_resolution)

        # Load dynamic tile registry
        self.tile_registry = {}
        self.tile_colors = {}
        self.load_tile_registry()

        # Display parameters
        self.cell_size = max(5, min(50, cell_size))
        self.legend_width = 200
        self.window_width = self.grid_width * self.cell_size + self.legend_width
        self.window_height = self.grid_height * self.cell_size

        # Initialize pygame
        pygame.init()
        self.screen = pygame.display.set_mode((self.window_width, self.window_height))
        pygame.display.set_caption("Dynamic House Map")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(None, 14)

        # Load grid
        try:
            self.grid = np.loadtxt(map_txt, dtype=np.int8)
        except:
            self.grid = np.full((self.grid_height, self.grid_width), 0, dtype=np.int8)

        # Auto-reload
        self.last_reload = pygame.time.get_ticks()
        self.reload_interval = 500  # Reload every 500ms

        # CHANGE DETECTION - Track grid state
        self.last_grid_hash = None
        self.last_structure_hash = None

    def load_tile_registry(self):
        """Load dynamic tile types and generate colors."""
        if "tile_registry" in self.structure:
            # Load from JSON
            registry = self.structure["tile_registry"]

            # Generate colors for each tile
            for name, tile_id in registry.items():
                # Special colors for reserved types
                if name == "free_space":
                    self.tile_colors[tile_id] = (200, 200, 200)
                elif name == "wall":
                    self.tile_colors[tile_id] = (100, 100, 100)
                elif name == "camera":
                    self.tile_colors[tile_id] = (0, 255, 255)
                elif name == "door":
                    self.tile_colors[tile_id] = (139, 69, 19)  # Brown
                else:
                    # Generate color from hash
                    hash_val = hash(name)
                    r = max(50, (hash_val & 0xFF0000) >> 16)
                    g = max(50, (hash_val & 0x00FF00) >> 8)
                    b = max(50, hash_val & 0x0000FF)
                    self.tile_colors[tile_id] = (r, g, b)

                self.tile_registry[name] = tile_id

    def save_map_image(self, filename="static/current_map.png"):
        """Save current pygame screen to file for web display"""
        os.makedirs('static', exist_ok=True)
        pygame.image.save(self.screen, filename)
        print(f"[Map saved: {filename}]")  # Debug message

    def render(self):
        """Render the grid."""
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
        pygame.display.flip()

        # CHANGE DETECTION - Only save when grid or structure changes
        current_grid_hash = hash(self.grid.tobytes())
        current_structure_hash = hash(json.dumps(self.structure, sort_keys=True))

        if (current_grid_hash != self.last_grid_hash or
                current_structure_hash != self.last_structure_hash):
            # Something changed, save the image
            self.save_map_image()
            self.last_grid_hash = current_grid_hash
            self.last_structure_hash = current_structure_hash

    def wrap_text(self, text, max_width):
        """Wrap text to fit within max_width pixels."""
        words = text.split(' ')
        lines = []
        current_line = []

        for word in words:
            test_line = ' '.join(current_line + [word])
            if self.font.size(test_line)[0] <= max_width:
                current_line.append(word)
            else:
                if current_line:
                    lines.append(' '.join(current_line))
                    current_line = [word]
                else:
                    # Single word too long, add it as is
                    lines.append(word)

        if current_line:
            lines.append(' '.join(current_line))

        return lines if lines else [text]

    def draw_legend(self):
        """Draw legend with dynamic tiles."""
        # Background
        legend_rect = pygame.Rect(self.grid_width * self.cell_size, 0,
                                  self.legend_width, self.window_height)
        pygame.draw.rect(self.screen, (40, 40, 40), legend_rect)

        # Find present tile types
        present_types = set(self.grid.flatten())

        # Draw items
        y_offset = 10
        x_base = self.grid_width * self.cell_size + 10

        # Title
        title = self.font.render("DETECTED OBJECTS", True, (255, 255, 255))
        self.screen.blit(title, (x_base, y_offset))
        y_offset += 25

        # Stats
        stats = self.font.render(
            f"Total: {len(self.structure.get('rooms', {}).get('main_room', {}).get('objects', []))} objects",
            True, (180, 180, 180))
        self.screen.blit(stats, (x_base, y_offset))
        y_offset += 20

        # Separator
        pygame.draw.line(self.screen, (80, 80, 80),
                         (x_base, y_offset), (x_base + 170, y_offset))
        y_offset += 10

        # Sort by name for consistent display
        sorted_tiles = sorted([(name, tid) for name, tid in self.tile_registry.items()
                               if tid in present_types], key=lambda x: x[0])

        for name, tile_id in sorted_tiles:
            if y_offset > self.window_height - 25:
                break

            # Color box
            color = self.tile_colors[tile_id]
            box_rect = pygame.Rect(x_base, y_offset, 16, 16)
            pygame.draw.rect(self.screen, color, box_rect)
            pygame.draw.rect(self.screen, (200, 200, 200), box_rect, 1)

            # Label with text wrapping
            # Capitalize and clean up name
            display_name = name.replace('_', ' ').title()
            if display_name == "Free Space":
                display_name = "Empty"

            # Wrap text to fit in available width (legend_width - margins - color box)
            max_text_width = self.legend_width - 40  # Leave space for margins and color box
            wrapped_lines = self.wrap_text(display_name, max_text_width)

            # Render each line
            for i, line in enumerate(wrapped_lines):
                label = self.font.render(line, True, (220, 220, 220))
                self.screen.blit(label, (x_base + 25, y_offset + 1 + (i * 14)))

            # Adjust y_offset based on number of lines
            y_offset += max(20, len(wrapped_lines) * 14 + 6)

    def reload(self):
        """Reload map and structure."""
        try:
            # Reload grid
            new_grid = np.loadtxt("house_map.txt", dtype=np.int8)
            if new_grid.shape == self.grid.shape:
                self.grid = new_grid

            # Reload structure and tiles
            with open("unified_rooms.json", 'r') as f:
                self.structure = json.load(f)
                self.load_tile_registry()
        except:
            pass  # Silent fail during file writes

    def check_auto_reload(self):
        """Auto-reload check."""
        current = pygame.time.get_ticks()
        if current - self.last_reload > self.reload_interval:
            self.reload()
            self.last_reload = current

    def run(self):
        """Main loop."""
        print("Dynamic House Renderer")
        print("-" * 30)
        print(f"Grid: {self.grid_width}x{self.grid_height}")
        print(f"Cell size: {self.cell_size}px")
        print(f"Auto-reload: {self.reload_interval}ms")
        print(f"Web save: Only when map changes (to static/current_map.png)")  # UPDATED
        print("\nControls:")
        print("  ESC - Exit")
        print("  R   - Manual reload")
        print("  +/- - Zoom")
        print("-" * 30)

        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
                    elif event.key == pygame.K_r:
                        self.reload()
                        print(f"Reloaded - {len(self.tile_registry)} tile types")
                    elif event.key in [pygame.K_PLUS, pygame.K_EQUALS]:
                        self.cell_size = min(50, self.cell_size + 2)
                        self.window_width = self.grid_width * self.cell_size + self.legend_width
                        self.window_height = self.grid_height * self.cell_size
                        self.screen = pygame.display.set_mode((self.window_width, self.window_height))
                    elif event.key == pygame.K_MINUS:
                        self.cell_size = max(5, self.cell_size - 2)
                        self.window_width = self.grid_width * self.cell_size + self.legend_width
                        self.window_height = self.grid_height * self.cell_size
                        self.screen = pygame.display.set_mode((self.window_width, self.window_height))

            self.check_auto_reload()
            self.render()
            self.clock.tick(30)

        pygame.quit()


if __name__ == "__main__":
    try:
        renderer = DynamicHouseRenderer()
        renderer.run()
    except FileNotFoundError:
        print("Error: unified_rooms.json or house_map.txt not found")
        print("Run the pixel_room_mapper.py first!")
    except Exception as e:
        print(f"Error: {e}")