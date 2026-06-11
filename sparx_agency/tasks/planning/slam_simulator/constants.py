from enum import IntEnum
from typing import Dict, Tuple

class TileType(IntEnum):
    UNKNOWN = -1
    FREE_SPACE = 0
    WALL = 1
    ENTRY_POINT = 2
    DOOR_CLOSED = 3
    DOOR_OPEN = 4
    WINDOW = 5
    OUT_OF_BOUNDS = 6

class Action(IntEnum):
    FORWARD = 0
    TURN_LEFT = 1
    TURN_RIGHT = 2
    STAY = 3

DIRECTIONS = ['NORTH', 'EAST', 'SOUTH', 'WEST']

DIRECTION_DELTAS: Dict[str, Tuple[int, int]] = {
    'NORTH': (0, -1),
    'EAST': (1, 0),
    'SOUTH': (0, 1),
    'WEST': (-1, 0),
}

# Rendering
TILE_SIZE = 20
FPS = 10

TILE_COLORS = {
    TileType.UNKNOWN: (50, 50, 50),
    TileType.FREE_SPACE: (200, 200, 200),
    TileType.WALL: (100, 100, 100),
    TileType.ENTRY_POINT: (0, 255, 255),
    TileType.DOOR_CLOSED: (255, 0, 0),
    TileType.DOOR_OPEN: (0, 200, 0),
    TileType.WINDOW: (0, 0, 255),
    TileType.OUT_OF_BOUNDS: (0, 0, 0),
}

DRONE_COLORS = [
    (255, 255, 0),
    (0, 255, 255),
    (255, 0, 255),
    (0, 255, 0),
    (255, 128, 0),
]

# Default configs
DEFAULT_SENSOR_CONFIG = {
    'max_range': 10,
    'fov_deg': 90,
    'num_rays': 30,
}

DEFAULT_REWARDS = {
    'discovery': 0.1,
    'collision': -1.0,
    'step': -0.001,
    'completion': 10.0,
}