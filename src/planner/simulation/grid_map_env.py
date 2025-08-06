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

        # Ensure we have enough entry points for the requested number
        self.entry_points: List[Tuple[int, int]] = self.find_or_create_entry_points(num_entry_points)

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

    def find_or_create_entry_points(self, num_requested: int) -> List[Tuple[int, int]]:
        """
        Find existing entry points and create additional ones if needed.

        Args:
            num_requested (int): Minimum number of entry points needed.

        Returns:
            List[Tuple[int, int]]: List of (y, x) entry point coordinates.
        """
        # First, find all existing entry points
        entry_points = [(y, x) for y in range(self.height)
                        for x in range(self.width)
                        if self.grid[y, x] == ENTRY_POINT]

        # If we have enough, return them
        if len(entry_points) >= num_requested:
            return entry_points[:num_requested]

        # Otherwise, we need to create more entry points
        existing_count = len(entry_points)
        needed = num_requested - existing_count

        # print(f"Found {existing_count} entry points, creating {needed} more...")

        # Strategy 1: Try to add entry points on borders (only on free spaces)
        border_candidates = []

        # Top and bottom borders
        for x in range(1, self.width - 1):
            if self.grid[0, x] == FREE_SPACE:
                border_candidates.append((0, x, 'top'))
            if self.grid[self.height - 1, x] == FREE_SPACE:
                border_candidates.append((self.height - 1, x, 'bottom'))

        # Left and right borders
        for y in range(1, self.height - 1):
            if self.grid[y, 0] == FREE_SPACE:
                border_candidates.append((y, 0, 'left'))
            if self.grid[y, self.width - 1] == FREE_SPACE:
                border_candidates.append((y, self.width - 1, 'right'))

        # Shuffle to get random distribution
        random.shuffle(border_candidates)

        # Add new entry points from border candidates
        for y, x, side in border_candidates:
            if needed <= 0:
                break
            if (y, x) not in entry_points:
                self.grid[y, x] = ENTRY_POINT
                entry_points.append((y, x))
                needed -= 1
                # print(f"  Added entry point at ({x}, {y}) on {side} border")

        # Strategy 2: If still need more, use any free space
        if needed > 0:
            free_spaces = [(y, x) for y in range(self.height)
                          for x in range(self.width)
                          if self.grid[y, x] in [FREE_SPACE, DOOR_OPEN, WINDOW]
                          and (y, x) not in entry_points]

            random.shuffle(free_spaces)

            for y, x in free_spaces:
                if needed <= 0:
                    break
                self.grid[y, x] = ENTRY_POINT
                entry_points.append((y, x))
                needed -= 1
                # print(f"  Added entry point at ({x}, {y}) in free space")

        # Final check
        # if len(entry_points) < num_requested:
        #     print(f"Warning: Could only create {len(entry_points)} entry points out of {num_requested} requested")

        return entry_points

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