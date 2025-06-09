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
Defines drone movement commands and their corresponding (dx, dy) offsets:

- 'UP':    (0, -1)
- 'DOWN':  (0, 1)
- 'LEFT':  (-1, 0)
- 'RIGHT': (1, 0)
- 'STAY':  (0, 0)

The `DIRECTIONS` dictionary is used by the planner and controller to determine how drones move.
`DIRECTION_LIST` provides a list of all movement keys for random choice or iteration.

Usage:
------
These constants are imported across modules (e.g., drone, environment, planner)
to ensure consistent handling of tile values and movements.
"""

# === Tile Types ===
FREE_SPACE = 0
WALL = 1
ENTRY_POINT = 2
DOOR_CLOSED = 3
DOOR_OPEN = 4
WINDOW = 5
OUT_OF_BOUNDS = 6

TILE_NAME = {
    FREE_SPACE: "Free",
    WALL: "Wall",
    ENTRY_POINT: "Entry Point",
    DOOR_CLOSED: "Door (Closed)",
    DOOR_OPEN: "Door (Open)",
    WINDOW: "Window",
    OUT_OF_BOUNDS: "Out of Bounds"
}

# === Movement Directions ===
DIRECTIONS = {
    'UP': (0, -1),
    'DOWN': (0, 1),
    'LEFT': (-1, 0),
    'RIGHT': (1, 0),
    'STAY': (0, 0)
}

DIRECTION_LIST = list(DIRECTIONS.keys())

# === Simulation Parameters ===
TILE_SIZE = 20
FPS = 180
MAX_TIME = 50
