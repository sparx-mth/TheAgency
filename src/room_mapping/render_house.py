#!/usr/bin/env python3
"""
render_house.py - Pygame House Renderer

Renders the unified house structure using pygame,
showing all rooms, objects, doors, and camera positions.
"""

import pygame
import numpy as np
import json
import sys
from tile_definitions import TileType, TILE_COLORS, TILE_NAMES, OBJECT_SIZES


class HouseRenderer:
    """Simple pygame renderer for unified house structure."""

    def __init__(self, unified_json="unified_rooms.json", map_txt="house_map.txt", cell_size=12):
        """
        Initialize the house renderer.

        Args:
            unified_json: Path to unified rooms JSON file
            map_txt: Path to house map text file
            cell_size: Pixels per grid cell (default 12, range 5-50 recommended)
        """
        # Load structure to get dimensions
        with open(unified_json, 'r') as f:
            self.structure = json.load(f)

        # Get house and grid parameters
        self.house_width_m = self.structure["house_dimensions_m"]["width"]
        self.house_height_m = self.structure["house_dimensions_m"]["height"]
        self.grid_resolution = self.structure["grid_resolution"]

        # Calculate grid size
        self.grid_width = int(self.house_width_m / self.grid_resolution)
        self.grid_height = int(self.house_height_m / self.grid_resolution)

        # Set display parameters
        self.cell_size = max(5, min(50, cell_size))  # Clamp between 5 and 50
        self.legend_width = 200  # Width of legend panel
        self.window_width = self.grid_width * self.cell_size + self.legend_width
        self.window_height = self.grid_height * self.cell_size

        # Initialize pygame
        pygame.init()
        self.screen = pygame.display.set_mode((self.window_width, self.window_height))
        pygame.display.set_caption("House Map Viewer")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(None, 16)

        # Load grid map
        try:
            self.grid = np.loadtxt(map_txt, dtype=np.int8)
            print(f"Loaded map from {map_txt}")
        except:
            print(f"Could not load {map_txt}, creating empty grid")
            self.grid = np.full((self.grid_height, self.grid_width),
                                TileType.FREE_SPACE, dtype=np.int8)

    def render(self):
        """Render the house grid."""
        # Clear screen
        self.screen.fill((30, 30, 30))

        # Draw grid cells
        for y in range(self.grid_height):
            for x in range(self.grid_width):
                tile_type = self.grid[y, x]
                color = TILE_COLORS.get(tile_type, (50, 50, 50))

                rect = pygame.Rect(
                    x * self.cell_size,
                    y * self.cell_size,
                    self.cell_size,
                    self.cell_size
                )
                pygame.draw.rect(self.screen, color, rect)

                # Draw grid lines
                pygame.draw.rect(self.screen, (60, 60, 60), rect, 1)

        # Draw legend on the right
        self.draw_legend()

        pygame.display.flip()

    def draw_legend(self):
        """Draw color legend on the right side."""
        # Background for legend
        legend_rect = pygame.Rect(self.grid_width * self.cell_size, 0,
                                  self.legend_width, self.window_height)
        pygame.draw.rect(self.screen, (40, 40, 40), legend_rect)

        # Find all tile types present in the grid
        present_types = set()
        for y in range(self.grid_height):
            for x in range(self.grid_width):
                present_types.add(self.grid[y, x])

        # Sort tile types for consistent display
        sorted_types = sorted(present_types)

        # Draw legend items
        y_offset = 20
        x_base = self.grid_width * self.cell_size + 15

        # Title
        title_text = self.font.render("LEGEND", True, (255, 255, 255))
        self.screen.blit(title_text, (x_base, y_offset))
        y_offset += 30

        for tile_type in sorted_types:
            if tile_type in TILE_COLORS and tile_type in TILE_NAMES:
                # Draw color box
                color = TILE_COLORS[tile_type]
                box_rect = pygame.Rect(x_base, y_offset, 20, 20)
                pygame.draw.rect(self.screen, color, box_rect)
                pygame.draw.rect(self.screen, (200, 200, 200), box_rect, 1)

                # Draw label
                name = TILE_NAMES[tile_type]
                label_text = self.font.render(name, True, (220, 220, 220))
                self.screen.blit(label_text, (x_base + 30, y_offset + 2))

                y_offset += 25

                # Stop if we run out of space
                if y_offset > self.window_height - 30:
                    break

    def reload(self):
        """Reload the map from file."""
        try:
            self.grid = np.loadtxt("house_map.txt", dtype=np.int8)
            print("Reloaded house_map.txt")

            # Also reload structure
            with open("unified_rooms.json", 'r') as f:
                self.structure = json.load(f)
            print("Reloaded unified_rooms.json")
        except Exception as e:
            print(f"Error reloading: {e}")

    def resize_window(self, new_cell_size):
        """Resize the window with new cell size."""
        self.cell_size = max(5, min(50, new_cell_size))
        self.window_width = self.grid_width * self.cell_size + self.legend_width
        self.window_height = self.grid_height * self.cell_size
        self.screen = pygame.display.set_mode((self.window_width, self.window_height))
        print(f"Cell size: {self.cell_size}px, Window: {self.window_width}x{self.window_height}")

    def run(self):
        """Main rendering loop."""
        print(f"Rendering {self.grid_width}x{self.grid_height} grid")
        print(f"Cell size: {self.cell_size}px")
        print(f"Window size: {self.window_width}x{self.window_height} pixels")

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
                    elif event.key == pygame.K_PLUS or event.key == pygame.K_EQUALS:
                        self.resize_window(self.cell_size + 2)
                    elif event.key == pygame.K_MINUS:
                        self.resize_window(self.cell_size - 2)

            self.render()
            self.clock.tick(30)

        pygame.quit()
        sys.exit()


def main():
    """Run the house renderer."""
    import argparse

    parser = argparse.ArgumentParser(description="House Map Renderer")
    parser.add_argument("--cell-size", type=int, default=20,
                        help="Size of each cell in pixels (5-50, default: 12)")
    parser.add_argument("--json", default="unified_rooms.json",
                        help="Path to unified rooms JSON file")
    parser.add_argument("--map", default="house_map.txt",
                        help="Path to house map text file")
    args = parser.parse_args()

    print("House Map Renderer")
    print("-" * 40)
    print("Controls:")
    print("  ESC     - Exit")
    print("  R       - Reload files")
    print("  +/-     - Adjust cell size")
    print("-" * 40)

    try:
        renderer = HouseRenderer(args.json, args.map, args.cell_size)
        renderer.run()
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please run room_unifier.py first to generate the required files.")
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()