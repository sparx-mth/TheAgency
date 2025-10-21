"""
environments/constants.py

This file contains all shared constants, enums, and configuration values used throughout
the SLAM simulation. It centralizes all magic numbers and provides type-safe enums for
tile types, actions, and directions.

Key Components:
- TileType: Enum for different map tile types
- Action: Enum for drone actions
- Direction constants and mappings
- Rendering parameters
- Default sensor parameters
"""

from enum import IntEnum
from typing import Dict, Tuple

# ============= Tile Types =============
class TileType(IntEnum):
    """
    Enumeration of all possible tile types in the environment.
    These values are used in the grid map representation.
    """
    FREE_SPACE = 0      # Walkable empty space
    WALL = 1            # Solid wall (blocks movement and vision)
    ENTRY_POINT = 2     # Drone spawn location
    DOOR_CLOSED = 3     # Closed door (blocks movement and vision)
    DOOR_OPEN = 4       # Open door (allows movement)
    WINDOW = 5          # Window (blocks movement, allows vision)
    OUT_OF_BOUNDS = 6   # Out of bounds area
    UNKNOWN = -1        # Unexplored area


# ============= Action Types =============
class Action(IntEnum):
    """
    Enumeration of possible drone actions.
    These map directly to the Gymnasium action space.
    """
    FORWARD = 0     # Move one step forward
    TURN_LEFT = 1   # Rotate 90 degrees counter-clockwise
    TURN_RIGHT = 2  # Rotate 90 degrees clockwise
    STAY = 3        # Remain in place


# ============= Direction Constants =============
# Cardinal directions
DIRECTIONS = ['NORTH', 'EAST', 'SOUTH', 'WEST']

# Direction to movement delta mapping
DIRECTION_DELTAS: Dict[str, Tuple[int, int]] = {
    'NORTH': (0, -1),   # Up
    'EAST': (1, 0),     # Right
    'SOUTH': (0, 1),    # Down
    'WEST': (-1, 0),    # Left
}


# ============= Rendering Constants =============
# Display parameters
TILE_SIZE = 20  # Size of each tile in pixels
FPS = 1        # Frames per second for rendering

# Color mapping for tiles (RGB values)
TILE_COLORS = {
    TileType.UNKNOWN: (50, 50, 50),        # Dark gray
    TileType.FREE_SPACE: (200, 200, 200),  # Light gray
    TileType.WALL: (100, 100, 100),        # Medium gray
    TileType.ENTRY_POINT: (0, 255, 255),   # Cyan
    TileType.DOOR_CLOSED: (255, 0, 0),     # Red
    TileType.DOOR_OPEN: (0, 200, 0),       # Green
    TileType.WINDOW: (0, 0, 255),          # Blue
    TileType.OUT_OF_BOUNDS: (0, 0, 0),     # Black
}

# Drone colors for multi-agent visualization
DRONE_COLORS = [
    (255, 255, 0),   # Yellow
    (0, 255, 255),   # Cyan
    (255, 0, 255),   # Magenta
    (0, 255, 0),     # Green
    (255, 128, 0),   # Orange
]


# ============= Default Parameters =============
# Environment defaults
DEFAULT_ENV_PARAMS = {
    'width': 32,
    'height': 32,
    'max_steps': 1000,
    'num_agents': 3,
}

# Sensor defaults
DEFAULT_CAMERA_PARAMS = {
    'max_range': 10,
    'fov_deg': 45,
    'num_rays': 30,
}

DEFAULT_LIDAR_PARAMS = {
    'max_range': 15,
    'num_rays': 360,
}

# Reward defaults
DEFAULT_REWARD_PARAMS = {
    'discovery_reward': 0.1,
    'collision_penalty': -1.0,
    'step_penalty': -0.001,
    'completion_bonus': 10.0,
}