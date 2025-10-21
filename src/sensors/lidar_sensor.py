"""
sensors/lidar_sensor.py

This file implements a 360-degree LIDAR sensor for SLAM drones. Unlike the camera
sensor which has a limited field of view, the LIDAR sensor can observe in all
directions simultaneously, making it ideal for rapid environment mapping.

The LIDAR sensor simulates laser range finding by casting rays in all directions
and detecting obstacles. It provides more comprehensive coverage than a camera
but may be more computationally expensive.
"""

import math
from typing import List, Tuple
import numpy as np

from .base_sensor import BaseSensor
from src.environments.base.constants import TileType


class LidarSensor(BaseSensor):
    """
    360-degree LIDAR sensor for omnidirectional sensing.

    This sensor simulates a LIDAR scanner that rotates and captures
    distance measurements in all directions. It ignores the drone's
    facing direction and provides complete surrounding awareness.

    Attributes:
        max_range: Maximum sensing distance in grid cells
        num_rays: Number of rays to cast (e.g., 360 for 1-degree resolution)
        ray_step: Step size for ray marching
    """

    def __init__(
            self,
            max_range: int = 15,
            num_rays: int = 360
    ):
        """
        Initialize the LIDAR sensor.

        Args:
            max_range: Maximum sensing distance in grid cells
            num_rays: Number of rays to cast in 360 degrees
                     (e.g., 360 = 1 ray per degree)
        """
        self.max_range = max_range
        self.num_rays = num_rays
        self.ray_step = 0.1

    def sense(
            self,
            pos: Tuple[int, int],
            facing: str,
            grid: np.ndarray
    ) -> List[Tuple[int, int, int]]:
        """
        Perform 360-degree sensing.

        The facing parameter is ignored as LIDAR senses in all directions.
        Rays are cast uniformly around the full circle.

        Args:
            pos: Current (x, y) position of the drone
            facing: Current facing direction (ignored for LIDAR)
            grid: The environment grid

        Returns:
            List of (x, y, tile_value) tuples for all visible cells
        """
        observations = set()

        # Cast rays in all directions
        for i in range(self.num_rays):
            # Calculate angle for this ray (uniform distribution around circle)
            angle = 2 * math.pi * i / self.num_rays

            # Cast ray and collect observations
            ray_obs = self._cast_ray(pos[0], pos[1], angle, grid)
            observations.update(ray_obs)

        return list(observations)

    def _cast_ray(
            self,
            x0: int,
            y0: int,
            angle: float,
            grid: np.ndarray
    ) -> List[Tuple[int, int, int]]:
        """
        Cast a single ray and return all cells it observes.

        Similar to camera ray casting but typically with longer range.

        Args:
            x0: Starting x coordinate
            y0: Starting y coordinate
            angle: Ray direction in radians
            grid: The environment grid

        Returns:
            List of (x, y, tile_value) tuples along the ray
        """
        dx = math.cos(angle)
        dy = math.sin(angle)

        # Start from center of drone's tile
        x, y = float(x0) + 0.5, float(y0) + 0.5
        observations = []
        visited = set()

        # Maximum number of steps
        max_steps = int(self.max_range / self.ray_step) + 50

        for _ in range(max_steps):
            # Get current tile coordinates
            tile_x, tile_y = int(x), int(y)

            # Check bounds
            if not (0 <= tile_x < grid.shape[1] and 0 <= tile_y < grid.shape[0]):
                break

            # Process each cell once
            if (tile_x, tile_y) not in visited:
                visited.add((tile_x, tile_y))

                # Skip drone's position
                if tile_x != x0 or tile_y != y0:
                    cell_value = grid[tile_y, tile_x]
                    observations.append((tile_x, tile_y, int(cell_value)))

                    # Check if this cell blocks the laser
                    if self._blocks_laser(cell_value):
                        break

            # Step forward
            x += dx * self.ray_step
            y += dy * self.ray_step

            # Check range
            dist = math.sqrt((x - (x0 + 0.5)) ** 2 + (y - (y0 + 0.5)) ** 2)
            if dist >= self.max_range:
                break

        return observations

    def _blocks_laser(self, tile_value: int) -> bool:
        """
        Determine if a tile type blocks the LIDAR laser.

        Args:
            tile_value: The tile type value

        Returns:
            True if the tile blocks the laser
        """
        # LIDAR is blocked by solid obstacles
        # Windows may or may not block LIDAR depending on the type
        return tile_value in {TileType.WALL, TileType.DOOR_CLOSED, TileType.WINDOW}

    def get_max_range(self) -> int:
        """Return the maximum sensing range."""
        return self.max_range

    def get_sensor_type(self) -> str:
        """Return sensor type identifier."""
        return "lidar"

    def get_sensor_params(self) -> dict:
        """Return sensor configuration parameters."""
        return {
            'max_range': self.max_range,
            'num_rays': self.num_rays,
            'ray_step': self.ray_step
        }