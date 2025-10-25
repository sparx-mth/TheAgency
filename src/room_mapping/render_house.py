#!/usr/bin/env python3
"""
render_house_dynamic.py - Dynamic Pygame House Renderer

Renders house with dynamic tiles loaded from the JSON file.
Auto-reloads to show real-time updates.
NOW SAVES IMAGES FOR WEB DISPLAY ONLY WHEN MAP CHANGES
IMPROVED: Better text spacing for detected objects
FIXED: Using predefined color scheme for consistency
"""

import pygame
import numpy as np
import json
import sys
import os  # ADDED FOR WEB INTEGRATION
import re  # ADDED FOR TEXT FORMATTING


class DynamicHouseRenderer:
    """Pygame renderer with dynamic tile support."""

    def __init__(self, unified_json="data/unified_rooms.json", map_txt="data/house_map.txt", cell_size=25):
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
        self.legend_width = 400  # Increased width for bigger text display
        self.window_width = self.grid_width * self.cell_size + self.legend_width
        self.window_height = self.grid_height * self.cell_size

        # Initialize pygame
        pygame.init()
        self.screen = pygame.display.set_mode((self.window_width, self.window_height))
        pygame.display.set_caption("Dynamic House Map")
        self.clock = pygame.time.Clock()

        # Multiple fonts for different purposes
        self.font_title = pygame.font.Font(None, 26)  # Title font
        self.font_stats = pygame.font.Font(None, 22)  # Stats font
        self.font_objects = pygame.font.Font(None, 40)  # BIGGER font for object list

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
        """Load dynamic tile types and use fixed colors."""
        # Define fixed colors for tile IDs (matching DynamicTileManager)
        fixed_tile_colors = {
            0: (240, 240, 240),  # free_space - light gray
            1: (50, 50, 50),  # wall - dark gray
            2: (0, 255, 0),  # camera - green
            3: (139, 69, 19),  # door - brown
            4: (139, 90, 43),  # table - tan
            5: (165, 42, 42),  # chair - brown-red
            6: (100, 100, 200),  # table and monitor - blue-gray
            7: (75, 75, 150),  # tv - dark blue
            8: (255, 255, 0),  # entry point - yellow
            9: (128, 0, 128),  # bicycle - purple
            10: (80, 80, 80),  # black suitcase - dark gray
            11: (0, 128, 255),  # bottle - light blue
            12: (255, 128, 0),  # mug - orange
            13: (100, 100, 200),  # monitor - blue
            14: (64, 64, 64),  # keyboard - gray
            15: (150, 50, 150),  # bicycle and chair - purple-red
            16: (80, 80, 160),  # keyboard and monitor - gray-blue
            -1: (20, 20, 20),  # unknown - very dark gray
        }

        if "tile_registry" in self.structure:
            # Load from JSON
            registry = self.structure["tile_registry"]

            # Apply colors from fixed scheme
            for name, tile_id in registry.items():
                # Use fixed color if available, otherwise use a default
                if tile_id in fixed_tile_colors:
                    self.tile_colors[tile_id] = fixed_tile_colors[tile_id]
                else:
                    # Fallback for any undefined tile IDs
                    # Generate a distinguishable color based on the ID
                    base_color = 100 + (tile_id * 10) % 155
                    self.tile_colors[tile_id] = (base_color, base_color, base_color)

                self.tile_registry[name] = tile_id

        # Also add the fixed colors that might not be in the registry
        for tile_id, color in fixed_tile_colors.items():
            if tile_id not in self.tile_colors:
                self.tile_colors[tile_id] = color

    def save_map_image(self, filename="data/current_map.png"):
        """Save current pygame screen to file for web display"""
        os.makedirs('data', exist_ok=True)
        pygame.image.save(self.screen, filename)
        print(f"[Map saved: {filename}]")  # Debug message

    def format_object_name(self, name):
        """Format object names with proper spacing."""
        # First, replace underscores with spaces
        display_name = name.replace('_', ' ')

        # Handle CamelCase and concatenated words
        # Add space before uppercase letters that follow lowercase letters
        display_name = re.sub(r'([a-z])([A-Z])', r'\1 \2', display_name)

        # Add space between consecutive uppercase letters followed by lowercase
        display_name = re.sub(r'([A-Z]+)([A-Z][a-z])', r'\1 \2', display_name)

        # Handle "And" specifically - ensure it has spaces around it
        display_name = re.sub(r'([a-zA-Z])(And)([A-Z])', r'\1 \2 \3', display_name)

        # Clean up any multiple spaces
        display_name = ' '.join(display_name.split())

        # Title case for better readability
        words = display_name.split()
        formatted_words = []

        for word in words:
            # Keep "And", "Or", "The" etc. in proper case
            if word.lower() in ['and', 'or', 'the', 'of', 'in', 'on', 'at']:
                formatted_words.append(word.lower())
            else:
                formatted_words.append(word.capitalize())

        # Capitalize first word regardless
        if formatted_words:
            formatted_words[0] = formatted_words[0].capitalize()

        return ' '.join(formatted_words)

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

    def wrap_text(self, text, max_width, font=None):
        """Wrap text to fit within max_width pixels."""
        if font is None:
            font = self.font_objects  # Default to objects font

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

        # Title (using title font)
        title = self.font_title.render("DETECTED OBJECTS", True, (255, 255, 255))
        self.screen.blit(title, (x_base, y_offset))
        y_offset += 35

        # Stats (using stats font)
        stats_text = f"Total: {len(self.structure.get('rooms', {}).get('main_room', {}).get('objects', []))} objects"
        stats = self.font_stats.render(stats_text, True, (180, 180, 180))
        self.screen.blit(stats, (x_base, y_offset))
        y_offset += 25

        # Separator
        pygame.draw.line(self.screen, (80, 80, 80),
                         (x_base, y_offset), (x_base + self.legend_width - 20, y_offset))
        y_offset += 15

        # Sort by name for consistent display
        sorted_tiles = sorted([(name, tid) for name, tid in self.tile_registry.items()
                               if tid in present_types], key=lambda x: x[0])

        for name, tile_id in sorted_tiles:
            if y_offset > self.window_height - 40:
                break

            # Color box (slightly larger to match bigger text)
            color = self.tile_colors[tile_id]
            box_rect = pygame.Rect(x_base, y_offset + 2, 22, 22)
            pygame.draw.rect(self.screen, color, box_rect)
            pygame.draw.rect(self.screen, (200, 200, 200), box_rect, 1)

            # Format the display name with proper spacing
            display_name = self.format_object_name(name)

            # Special cases
            if display_name.lower() == "free space":
                display_name = "Empty"
            elif display_name.lower() == "entry point":
                display_name = "Entry Point"

            # Wrap text to fit in available width (using bigger font)
            max_text_width = self.legend_width - 60  # Leave space for margins and color box
            wrapped_lines = self.wrap_text(display_name, max_text_width, self.font_objects)

            # Render each line with BIGGER FONT
            line_height = 26  # Increased line height for bigger font
            for i, line in enumerate(wrapped_lines):
                label = self.font_objects.render(line, True, (220, 220, 220))
                self.screen.blit(label, (x_base + 35, y_offset + (i * line_height)))

            # Adjust y_offset based on number of lines
            y_offset += max(32, len(wrapped_lines) * line_height + 10)

    def reload(self):
        """Reload map and structure."""
        try:
            # Reload grid
            new_grid = np.loadtxt("data/house_map.txt", dtype=np.int8)
            if new_grid.shape == self.grid.shape:
                self.grid = new_grid

            # Reload structure and tiles
            with open("data/unified_rooms.json", 'r') as f:
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
        print("Dynamic House Renderer - Fixed Color Scheme")
        print("-" * 60)
        print(f"Grid: {self.grid_width}x{self.grid_height}")
        print(f"Cell size: {self.cell_size}px")
        print(f"Legend width: {self.legend_width}px")
        print(f"Object list font size: 32pt (bigger)")
        print(f"Auto-reload: {self.reload_interval}ms")
        print(f"Web save: Only when map changes (to data/current_map.png)")
        print("\nColors: Using predefined color scheme for consistency")
        print("\nControls:")
        print("  ESC - Exit")
        print("  R   - Manual reload")
        print("  +/- - Zoom")
        print("-" * 60)

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