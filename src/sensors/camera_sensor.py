"""
sensors/camera_sensor.py

This file implements a directional camera sensor for SLAM drones. The camera sensor
simulates realistic vision with a limited field of view (FOV) and range, using
ray-casting to detect visible cells within its cone of vision.

The sensor respects line-of-sight rules, where walls and closed doors block vision,
while windows allow seeing through them. This creates realistic sensing behavior
for vision-based SLAM algorithms.
"""

import math
from typing import List, Tuple
import numpy as np

from .base_sensor import BaseSensor
from environments.constants import TileType, DIRECTION_DELTAS


class CameraSensor(BaseSensor):
    """
    Directional camera sensor with configurable FOV and range.

    This sensor simulates a forward-facing camera that can only see within
    a cone defined by its field of view angle. It uses ray-casting to
    determine which cells are visible, respecting line-of-sight blocking.

    Attributes:
        max_range: Maximum sensing distance in grid cells
        fov_deg: Field of view angle in degrees
        fov_rad: Field of view angle in radians
        num_rays: Number of rays to cast across the FOV
        ray_step: Step size for ray marching (smaller = more accurate)
    """

    def __init__(
        self,
        max_range: int = 10,
        fov_deg: int = 45,
        num_rays: int = 30
    ):
        """
        Initialize the camera sensor.

        Args:
            max_range: Maximum sensing distance in grid cells
            fov_deg: Field of view angle in degrees (e.g., 45 for narrow, 90 for wide)
            num_rays: Number of rays to cast across the FOV (more = better coverage)
        """
        self.max_range = max_range
        self.fov_deg = fov_deg
        self.fov_rad = math.radians(fov_deg)
        self.num_rays = num_rays
        self.ray_step = 0.1  # Step size for ray marching

    def sense(
        self,
        pos: Tuple[int, int],
        facing: str,
        grid: np.ndarray
    ) -> List[Tuple[int, int, int]]:
        """
        Perform directional sensing using ray-casting.

        Casts multiple rays distributed across the field of view to detect
        all visible cells. Each ray stops when it hits an obstacle that
        blocks vision.

        Args:
            pos: Current (x, y) position of the drone
            facing: Current facing direction
            grid: The environment grid

        Returns:
            List of (x, y, tile_value) tuples for all visible cells
        """
        # Get the center direction angle
        dx, dy = DIRECTION_DELTAS[facing]
        center_angle = math.atan2(dy, dx)

        # Collect observations from all rays
        observations = set()

        # IMPORTANT: Always include the drone's current position
        # The drone can always see where it currently is!
        x0, y0 = pos
        if 0 <= x0 < grid.shape[1] and 0 <= y0 < grid.shape[0]:
            observations.add((x0, y0, int(grid[y0, x0])))

        for i in range(self.num_rays):
            # Calculate angle for this ray
            if self.num_rays == 1:
                angle = center_angle
            else:
                # Distribute rays evenly across FOV
                angle_offset = (i / (self.num_rays - 1) - 0.5) * self.fov_rad
                angle = center_angle + angle_offset

            # Cast the ray and collect observations
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

        Uses ray marching with small steps to ensure no cells are missed.
        The ray continues until it hits a vision-blocking obstacle or
        reaches maximum range.

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

        # Start from center of the drone's tile
        x, y = float(x0) + 0.5, float(y0) + 0.5
        observations = []
        visited = set()

        # Maximum number of steps to prevent infinite loops
        max_steps = int(self.max_range / self.ray_step) + 50

        for _ in range(max_steps):
            # Get current tile coordinates
            tile_x, tile_y = int(x), int(y)

            # Check bounds
            if not (0 <= tile_x < grid.shape[1] and 0 <= tile_y < grid.shape[0]):
                break

            # Only process each cell once per ray
            if (tile_x, tile_y) not in visited:
                visited.add((tile_x, tile_y))

                # Don't add the drone's position here (already added in sense())
                # This avoids duplicates since we always add it in the main sense() method
                if tile_x != x0 or tile_y != y0:
                    cell_value = grid[tile_y, tile_x]
                    observations.append((tile_x, tile_y, int(cell_value)))

                    # Check if this cell blocks vision
                    if self._blocks_vision(cell_value):
                        break

            # Step forward along the ray
            x += dx * self.ray_step
            y += dy * self.ray_step

            # Check if we've exceeded maximum range
            dist = math.sqrt((x - (x0 + 0.5))**2 + (y - (y0 + 0.5))**2)
            if dist >= self.max_range:
                break

        return observations

    def _blocks_vision(self, tile_value: int) -> bool:
        """
        Determine if a tile type blocks vision.

        Args:
            tile_value: The tile type value

        Returns:
            True if the tile blocks vision to cells behind it
        """
        # Walls and closed doors block vision
        # Windows and open doors allow seeing through
        return tile_value in {TileType.WALL, TileType.DOOR_CLOSED}

    def get_max_range(self) -> int:
        """Return the maximum sensing range."""
        return self.max_range

    def get_sensor_type(self) -> str:
        """Return sensor type identifier."""
        return "camera"

    def get_sensor_params(self) -> dict:
        """Return sensor configuration parameters."""
        return {
            'max_range': self.max_range,
            'fov_deg': self.fov_deg,
            'num_rays': self.num_rays,
            'ray_step': self.ray_step
        }