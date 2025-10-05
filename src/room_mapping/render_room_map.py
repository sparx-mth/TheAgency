"""
render_room_map.py

Simple script to render a room map text file using pygame.
"""

import numpy as np
import pygame
import sys
from tile_definitions import TileType, TILE_COLORS, TILE_NAMES


def render_map(map_file: str, tile_size: int = 20):
    """
    Render a room map from a text file.

    Args:
        map_file: Path to the map text file
        tile_size: Size of each tile in pixels
    """
    # Load map
    room_map = np.loadtxt(map_file, dtype=np.int8)
    height, width = room_map.shape

    # Initialize pygame
    pygame.init()
    screen_width = width * tile_size
    screen_height = height * tile_size
    screen = pygame.display.set_mode((screen_width, screen_height))
    pygame.display.set_caption(f"Room Map - {map_file}")
    clock = pygame.time.Clock()

    print(f"Rendering {map_file} ({width}x{height})")
    print("Press ESC or close window to exit")

    # Main loop
    running = True
    while running:
        # Handle events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False

        # Clear screen
        screen.fill((50, 50, 50))

        # Draw tiles
        for y in range(height):
            for x in range(width):
                tile = int(room_map[y, x])
                color = TILE_COLORS.get(tile, (150, 150, 150))
                rect = pygame.Rect(x * tile_size, y * tile_size,
                                   tile_size - 1, tile_size - 1)
                pygame.draw.rect(screen, color, rect)

        # Highlight entry point with a circle
        for y in range(height):
            for x in range(width):
                if room_map[y, x] == TileType.ENTRY_POINT:
                    center = (x * tile_size + tile_size // 2,
                              y * tile_size + tile_size // 2)
                    pygame.draw.circle(screen, (0, 255, 0), center, tile_size // 3)

        # Update display
        pygame.display.flip()
        clock.tick(30)

    pygame.quit()


def render_with_legend(map_file: str, tile_size: int = 20):
    """
    Render a room map with a legend showing tile types.

    Args:
        map_file: Path to the map text file
        tile_size: Size of each tile in pixels
    """
    # Load map
    room_map = np.loadtxt(map_file, dtype=np.int8)
    height, width = room_map.shape

    # Initialize pygame
    pygame.init()
    screen_width = width * tile_size + 200  # Extra space for legend
    screen_height = height * tile_size
    screen = pygame.display.set_mode((screen_width, screen_height))
    pygame.display.set_caption(f"Room Map - {map_file}")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("Arial", 14)

    # Build legend from tiles actually present in the map
    unique_tiles = np.unique(room_map)
    legend_items = []

    # Priority order for legend display
    priority_order = [
        TileType.WALL,
        TileType.FREE_SPACE,
        TileType.ENTRY_POINT,
        TileType.DOOR_OPEN,
        TileType.DOOR_CLOSED,
        TileType.WINDOW,
        TileType.CHAIR,
        TileType.PLASTIC_CHAIR,
        TileType.TABLE,
        TileType.COUCH,
        TileType.TV,
        TileType.BED,
        TileType.DESK,
        TileType.PLANT,
        TileType.CABINET,
        TileType.APPLIANCE,
        TileType.REFRIGERATOR,
        TileType.SUITCASE,
        TileType.BOX,
        TileType.CARDBOARD_BOX,
        TileType.COMPUTER,
        TileType.KEYBOARD,
        TileType.MOUSE,
        TileType.SOCKET,
        TileType.BOTTLE,
        TileType.WEAPON,
    ]

    # Add tiles in priority order if they exist in the map
    for tile_type in priority_order:
        if tile_type in unique_tiles:
            name = TILE_NAMES.get(tile_type, f"Type {tile_type}")
            legend_items.append((name, tile_type))

    print(f"Rendering {map_file} ({width}x{height})")
    print("Press ESC or close window to exit")

    # Main loop
    running = True
    while running:
        # Handle events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False

        # Clear screen
        screen.fill((30, 30, 30))

        # Draw tiles
        for y in range(height):
            for x in range(width):
                tile = int(room_map[y, x])
                color = TILE_COLORS.get(tile, (150, 150, 150))
                rect = pygame.Rect(x * tile_size, y * tile_size,
                                   tile_size - 1, tile_size - 1)
                pygame.draw.rect(screen, color, rect)

        # Draw grid lines
        for x in range(width + 1):
            pygame.draw.line(screen, (60, 60, 60),
                             (x * tile_size, 0),
                             (x * tile_size, height * tile_size))
        for y in range(height + 1):
            pygame.draw.line(screen, (60, 60, 60),
                             (0, y * tile_size),
                             (width * tile_size, y * tile_size))

        # Highlight entry point
        for y in range(height):
            for x in range(width):
                if room_map[y, x] == TileType.ENTRY_POINT:
                    center = (x * tile_size + tile_size // 2,
                              y * tile_size + tile_size // 2)
                    pygame.draw.circle(screen, (0, 255, 0), center, tile_size // 3, 2)

        # Draw legend
        legend_x = width * tile_size + 20
        for i, (name, tile_type) in enumerate(legend_items):
            y_pos = 20 + i * 30

            # Draw color box
            color = TILE_COLORS.get(tile_type, (150, 150, 150))
            pygame.draw.rect(screen, color, (legend_x, y_pos, 20, 20))
            pygame.draw.rect(screen, (255, 255, 255), (legend_x, y_pos, 20, 20), 1)

            # Draw text
            text = font.render(name, True, (255, 255, 255))
            screen.blit(text, (legend_x + 25, y_pos + 2))

        # Update display
        pygame.display.flip()
        clock.tick(30)

    pygame.quit()


if __name__ == "__main__":
    # Get map file from command line or use default
    map_file = sys.argv[1] if len(sys.argv) > 1 else "room_map.txt"

    # Render with legend by default
    if "--simple" in sys.argv:
        render_map(map_file)
    else:
        render_with_legend(map_file)