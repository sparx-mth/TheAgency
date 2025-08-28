"""
Utility function to pre-compute doorways from a map.
This should be called once in the training script.
"""

import numpy as np
from typing import Dict, Tuple
from environments.base.constants import TileType


def precompute_doorways(map_path: str) -> Dict[Tuple[int, int], str]:
    """
    Pre-compute all doorways from a map file.

    Args:
        map_path: Path to the map file

    Returns:
        Dictionary mapping (x, y) positions to doorway orientations ('horizontal' or 'vertical')
    """
    # Load the map
    true_map = np.loadtxt(map_path, dtype=np.int8)
    height, width = true_map.shape

    doorways = {}

    # Check all positions for doorway patterns
    for y in range(height):
        for x in range(width):
            # Check horizontal doorway pattern: wall-passage-wall
            if 1 <= x < width - 1:
                center = true_map[y, x]
                left = true_map[y, x - 1]
                right = true_map[y, x + 1]

                if (left == TileType.WALL and
                        center in [TileType.FREE_SPACE, TileType.DOOR_OPEN] and
                        right == TileType.WALL):
                    doorways[(x, y)] = 'horizontal'

            # Check vertical doorway pattern: wall-passage-wall
            # Only add if not already marked as horizontal
            if (x, y) not in doorways and 1 <= y < height - 1:
                center = true_map[y, x]
                top = true_map[y - 1, x]
                bottom = true_map[y + 1, x]

                if (top == TileType.WALL and
                        center in [TileType.FREE_SPACE, TileType.DOOR_OPEN] and
                        bottom == TileType.WALL):
                    doorways[(x, y)] = 'vertical'

    return doorways


def precompute_doorway_visibility_masks(doorways: Dict[Tuple[int, int], str],
                                        width: int,
                                        height: int) -> Dict[Tuple[int, int], np.ndarray]:
    """
    Pre-compute visibility masks for each doorway.
    A doorway is considered "visible" when the doorway cell and its adjacent walls are discovered.

    Args:
        doorways: Dictionary of doorway positions and orientations
        width: Map width
        height: Map height

    Returns:
        Dictionary mapping each doorway position to a boolean mask of cells that must be visible
    """
    visibility_masks = {}

    for (x, y), orientation in doorways.items():
        # Create a mask for this doorway
        mask = np.zeros((height, width), dtype=bool)

        # The doorway cell itself must be visible
        mask[y, x] = True

        if orientation == 'horizontal':
            # Left and right cells must be visible
            if x > 0:
                mask[y, x - 1] = True
            if x < width - 1:
                mask[y, x + 1] = True
        else:  # vertical
            # Top and bottom cells must be visible
            if y > 0:
                mask[y - 1, x] = True
            if y < height - 1:
                mask[y + 1, x] = True

        visibility_masks[(x, y)] = mask

    return visibility_masks