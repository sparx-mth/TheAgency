"""
This module defines the `GridMapEnv` class, which simulates a 2D environment for multi-agent SLAM.
It supports loading predefined maps or generating randomized grid maps with obstacles and entry points.
The environment manages drone placement, collision detection, and tile querying.

Key Responsibilities:
---------------------
- Load or generate a grid-based map containing:
    * FREE_SPACE, WALL, DOOR_CLOSED, DOOR_OPEN, WINDOW, OUT_OF_BOUNDS, ENTRY_POINT tiles
- Place drones at entry points with optional entry time delay
- Detect collisions with walls, closed doors, other drones, or out-of-bounds areas
- Provide sensing support to drones via `get_tile`
- Maintain a list of active drones initialized with communication support

Main Methods:
-------------
- `__init__(...)`: Initializes the environment and drones
- `load_map(path)`: Loads map from a file
- `generate_random_map(...)`: Creates a randomized grid environment
- `is_collision(x, y)`: Checks for collisions at a given position
- `get_tile(x, y)`: Returns the tile type at (x, y)
- `find_entry_points()`: Identifies or forces entry points on the map
- `print_legend()`: Utility to print tile names and their values

Usage:
------
This environment serves as the base simulation world for drones performing SLAM.
Each drone queries the environment for tile types and attempts to explore without collisions.
The map can either be loaded from a `.txt` file or generated on the fly for experimentation.
"""

from typing import List, Tuple, Optional, TYPE_CHECKING
import numpy as np
import random

from planner.simulation.simulation_constants import (
    FREE_SPACE, WALL, DOOR_CLOSED, DOOR_OPEN, WINDOW, OUT_OF_BOUNDS, ENTRY_POINT, TILE_NAME
)
from planner.simulation.sensors.camera_sensor import CameraSensor
from planner.simulation.sensors.bresenham_fov import *
from planner.simulation.drone import Drone

if TYPE_CHECKING:
    from planner.communication.comm_interface import CommunicationInterface


class GridMapEnv:
    """
    Simulates a 2D grid-based environment for multi-agent SLAM experiments.

    The environment supports static or randomly generated maps and manages drone
    placement, movement validation, and tile querying for sensing. It tracks entry
    points and instantiates drones with preconfigured sensors and communication interfaces.

    Attributes:
    -----------
    grid (np.ndarray): 2D grid map of the environment with tile types.
    width (int): Width of the map.
    height (int): Height of the map.
    entry_points (List[Tuple[int, int]]): Coordinates where drones enter.
    comm (CommunicationInterface): Interface for sending/receiving drone data.
    drones (List[Drone]): List of active drones within the environment.

    Key Tile Types:
    ---------------
    - FREE_SPACE: Traversable
    - WALL / DOOR_CLOSED / OUT_OF_BOUNDS: Non-traversable
    - ENTRY_POINT: Spawn location
    - DOOR_OPEN / WINDOW: Traversable special tiles

    Typical Usage:
    --------------
    The environment is initialized once per simulation and provides utility
    methods for sensing (`get_tile`) and collision checking (`is_collision`).
    """
    def __init__(
        self,
        comm_interface: "CommunicationInterface",
        width: int = 32,
        height: int = 32,
        randomize: bool = False,
        map_path: Optional[str] = None,
        num_entry_points: int = 2,
        num_drones: int = 3,
        camera_range: int = 10,
        fov: int = 45
    ):
        """
        Initialize the environment by loading or generating a map and placing drones.

        Args:
            comm_interface (CommunicationInterface): Communication interface for drones.
            width (int): Width of the grid if generating a map.
            height (int): Height of the grid if generating a map.
            randomize (bool): Whether to generate a random map.
            map_path (Optional[str]): Path to map file if loading a static map.
            num_entry_points (int): Number of entry points to place.
            num_drones (int): Number of drones to initialize.
            camera_range (int): Maximum sensing range of the drone's camera.
            fov (float): Field of view for the drone camera sensor (in degrees).
        """
        if map_path:
            self.grid = self.load_map(map_path)
        elif randomize:
            self.grid = self.generate_random_map(width, height, num_entry_points)
        else:
            self.grid = np.zeros((height, width), dtype=np.int8)

        self.height, self.width = self.grid.shape
        self.entry_points: List[Tuple[int, int]] = self.find_entry_points()
        self.comm: "CommunicationInterface" = comm_interface
        self.drones: List["Drone"] = []

        for i in range(num_drones):
            y, x = self.entry_points[i % len(self.entry_points)]
            entry_time = i * 2
            drone = Drone(
                drone_id=i,
                start_pos=(x, y),
                entry_time=entry_time,
                comm_interface=self.comm,
                sensors=[CameraSensor(camera_range, fov)]
            )
            drone.initialize_map(self.grid.shape)
            self.drones.append(drone)

    @staticmethod
    def load_map(path: str) -> np.ndarray:
        """
        Load a map from a file into a numpy array.

        Args:
            path (str): Path to the map file.

        Returns:
            np.ndarray: Loaded map as a 2D integer array.
        """
        grid = np.loadtxt(path, dtype=np.int8)
        return grid

    @staticmethod
    def generate_random_map(width: int, height: int, num_entry_points: int = 2) -> np.ndarray:
        """
        Generates a randomized map containing walls, doors, windows, and entry points.

        Args:
            width (int): Width of the map.
            height (int): Height of the map.
            num_entry_points (int): Number of entry points to place.

        Returns:
            np.ndarray: The generated map as a 2D grid.
        """
        grid = np.zeros((height, width), dtype=np.int8)

        # Border walls
        grid[0, :] = WALL
        grid[-1, :] = WALL
        grid[:, 0] = WALL
        grid[:, -1] = WALL

        # Random internal walls
        for _ in range(int(width * height * 0.1)):
            x = random.randint(1, width - 2)
            y = random.randint(1, height - 2)
            grid[y, x] = WALL

        # Random closed doors
        for _ in range(int(width * height * 0.01)):
            x = random.randint(1, width - 2)
            y = random.randint(1, height - 2)
            if grid[y, x] == FREE_SPACE:
                grid[y, x] = DOOR_CLOSED

        # Random open doors
        for _ in range(int(width * height * 0.01)):
            x = random.randint(1, width - 2)
            y = random.randint(1, height - 2)
            if grid[y, x] == FREE_SPACE:
                grid[y, x] = DOOR_OPEN

        # Random windows
        for _ in range(int(width * height * 0.01)):
            x = random.randint(1, width - 2)
            y = random.randint(1, height - 2)
            if grid[y, x] == FREE_SPACE:
                grid[y, x] = WINDOW

        # Random out-of-bounds areas (blackout zones)
        for _ in range(int(width * height * 0.005)):
            x = random.randint(1, width - 2)
            y = random.randint(1, height - 2)
            if grid[y, x] == FREE_SPACE:
                grid[y, x] = OUT_OF_BOUNDS

        # Force entry points on borders
        entries = set()
        attempts = 0
        max_attempts = 100

        while len(entries) < num_entry_points and attempts < max_attempts:
            attempts += 1
            side = random.choice(['top', 'bottom', 'left', 'right'])

            if side == 'top':
                x, y = random.randint(1, width - 2), 0
            elif side == 'bottom':
                x, y = random.randint(1, width - 2), height - 1
            elif side == 'left':
                x, y = 0, random.randint(1, height - 2)
            else:
                x, y = width - 1, random.randint(1, height - 2)

            grid[y, x] = ENTRY_POINT
            entries.add((y, x))
        return grid

    def is_collision(self, x: int, y: int) -> bool:
        """
        Check if the given coordinate causes a collision (wall, door, drone, etc.).

        Args:
            x (int): X coordinate.
            y (int): Y coordinate.

        Returns:
            bool: True if collision detected, else False.
        """
        if not (0 <= x < self.width and 0 <= y < self.height):
            return True
        if self.grid[y, x] in {WALL, DOOR_CLOSED, OUT_OF_BOUNDS}:
            return True
        for drone in self.drones:
            if drone.get_position() == (x, y) and drone.active:
                return True
        return False

    def get_tile(self, x: int, y: int) -> int:
        """
        Retrieve the tile value at the given position, or OUT_OF_BOUNDS if invalid.

        Args:
            x (int): X coordinate.
            y (int): Y coordinate.

        Returns:
            int: Tile value at the location.
        """
        if 0 <= x < self.width and 0 <= y < self.height:
            return int(self.grid[y, x])
        return OUT_OF_BOUNDS

    def find_entry_points(self) -> List[Tuple[int, int]]:
        """
        Locate all entry points in the map. If none exist, create one at a valid cell.

        Returns:
            List[Tuple[int, int]]: List of (y, x) entry point coordinates.
        """
        entry_points = [(y, x) for y in range(self.height)
                        for x in range(self.width)
                        if self.grid[y, x] == ENTRY_POINT]

        if not entry_points:
            # Find all cells with values 0, 1, or 2
            candidates = [(y, x) for y in range(self.height)
                          for x in range(self.width)
                          if self.grid[y, x] in [FREE_SPACE, DOOR_OPEN, WINDOW]]

            if candidates:
                y, x = random.choice(candidates)
                self.grid[y, x] = ENTRY_POINT
                entry_points = [(y, x)]

        return entry_points

    @staticmethod
    def print_legend() -> None:
        """
        Print the legend showing tile values and their meanings.
        """
        for k, v in TILE_NAME.items():
            print(f"{k}: {v}")
