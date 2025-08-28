"""
Utility functions to pre-compute room boundaries and doorways.
"""

import numpy as np
from typing import Dict, Set, Tuple, List
from environments.base.constants import TileType


def precompute_room_data(map_path: str) -> Dict:
    """
    Pre-compute all rooms and doorways from a map file.

    Args:
        map_path: Path to the map file

    Returns:
        Dictionary containing:
        - 'doorways': Set of all doorway positions
        - 'rooms': List of rooms (each room is a set of positions)
        - 'room_boundaries': Dict mapping room_id to its boundary walls
        - 'map_shape': (height, width) of the map
    """
    # Load map
    true_map = np.loadtxt(map_path, dtype=np.int8)
    height, width = true_map.shape

    # Find all doorways efficiently
    doorways = set()
    for y in range(1, height - 1):
        for x in range(1, width - 1):
            center = true_map[y, x]
            if center not in [TileType.FREE_SPACE, TileType.DOOR_OPEN]:
                continue

            # Check doorway patterns
            left = true_map[y, x - 1]
            right = true_map[y, x + 1]
            top = true_map[y - 1, x]
            bottom = true_map[y + 1, x]

            # Horizontal doorway
            if left == TileType.WALL and right == TileType.WALL:
                doorways.add((x, y))
            # Vertical doorway
            elif top == TileType.WALL and bottom == TileType.WALL:
                doorways.add((x, y))

    # Find rooms using flood fill (rooms are separated by doorways)
    visited = np.zeros((height, width), dtype=bool)
    rooms = []
    room_boundaries = {}

    for y in range(height):
        for x in range(width):
            # Skip if already visited or not free space
            if visited[y, x] or true_map[y, x] not in [TileType.FREE_SPACE, TileType.DOOR_OPEN, TileType.ENTRY_POINT]:
                continue

            # Skip doorways as room starting points
            if (x, y) in doorways:
                continue

            # Flood fill to find room
            room = set()
            boundaries = set()
            queue = [(x, y)]

            while queue:
                cx, cy = queue.pop(0)
                if visited[cy, cx]:
                    continue

                # Don't cross doorways
                if (cx, cy) in doorways:
                    continue

                visited[cy, cx] = True

                # Check if traversable
                if true_map[cy, cx] in [TileType.FREE_SPACE, TileType.DOOR_OPEN, TileType.ENTRY_POINT]:
                    room.add((cx, cy))

                    # Check neighbors
                    for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                        nx, ny = cx + dx, cy + dy
                        if 0 <= nx < width and 0 <= ny < height:
                            if not visited[ny, nx]:
                                if true_map[ny, nx] == TileType.WALL:
                                    boundaries.add((nx, ny))
                                else:
                                    queue.append((nx, ny))

            if room:
                room_id = len(rooms)
                rooms.append(room)
                room_boundaries[room_id] = boundaries

    return {
        'doorways': doorways,
        'rooms': rooms,
        'room_boundaries': room_boundaries,
        'map_shape': (height, width)
    }