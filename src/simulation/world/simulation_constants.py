"""
This module defines constants for use in the simulation environment of a multi-agent SLAM system.

Tile Types:
-----------
Integer codes are assigned to different types of tiles on the map grid.
These constants are used to describe the environment's layout, obstacles, and special zones.

- FREE_SPACE:      Walkable space (value = 0)
- WALL:            Solid, non-traversable wall (value = 1)
- ENTRY_POINT:     Starting location where drones can enter (value = 2)
- DOOR_CLOSED:     Impassable door (value = 3)
- DOOR_OPEN:       Passable door (value = 4)
- WINDOW:          Transparent and traversable tile (value = 5)
- OUT_OF_BOUNDS:   Region beyond the defined environment (value = 6)

The dictionary `TILE_NAME` provides human-readable names for UI and debugging.

Movement Directions:
--------------------
Defines **drone facing directions** and their corresponding (dx, dy) offsets:

- 'NORTH': (0, -1)
- 'EAST':  (1, 0)
- 'SOUTH': (0, 1)
- 'WEST':  (-1, 0)

These are used to determine which way a drone is facing and how it moves forward.

Control Commands:
-----------------
Drones are controlled using one of the following actions:

- 'FORWARD': Move one step forward in the current facing direction
- 'TURN_LEFT': Rotate 90° left (counter-clockwise)
- 'TURN_RIGHT': Rotate 90° right (clockwise)
- 'STAY': Remain in place and sense surroundings

The list `FACING_DIRECTIONS` is used in orientation logic and turning behavior.
The list `DIRECTION_COMMANDS` is used by the planner and controller to determine drone actions.

Usage:
------
These constants are imported across modules (e.g., drone, environment, planner)
to ensure consistent handling of tile values and movements.
"""

from typing import Literal, Tuple, Dict

# === Tile Types ===
FREE_SPACE: int = 0
WALL: int = 1
ENTRY_POINT: int = 2
DOOR_CLOSED: int = 3
DOOR_OPEN: int = 4
WINDOW: int = 5
OUT_OF_BOUNDS: int = 6

TILE_NAME: Dict[int, str] = {
    FREE_SPACE: "Free",
    WALL: "Wall",
    ENTRY_POINT: "Entry Point",
    DOOR_CLOSED: "Door (Closed)",
    DOOR_OPEN: "Door (Open)",
    WINDOW: "Window",
    OUT_OF_BOUNDS: "Out of Bounds"
}

# === Movement Directions ===
FACING_DIRECTION = Literal['NORTH', 'EAST', 'SOUTH', 'WEST']
FACING_DIRECTIONS = ['NORTH', 'EAST', 'SOUTH', 'WEST']

FACING_TO_DELTA: Dict[FACING_DIRECTION, Tuple[int, int]] = {
    'NORTH': (0, -1),
    'EAST': (1, 0),
    'SOUTH': (0, 1),
    'WEST': (-1, 0),
}

DIRECTIONS = Literal['FORWARD', 'TURN_LEFT', 'TURN_RIGHT', 'STAY']
DIRECTION_COMMANDS = ['FORWARD', 'TURN_LEFT', 'TURN_RIGHT', 'STAY']

# === Simulation Parameters ===
TILE_SIZE: int = 20
FPS: int = 90
MAX_TIME: int = 90
CAMERA_RANGE: int = 10
FOV: int = 45