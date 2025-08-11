"""
sensors/base_sensor.py

This file defines the abstract base class for all sensor types in the SLAM simulation.
The BaseSensor interface ensures that all sensor implementations provide consistent
methods for sensing the environment, regardless of their specific implementation details.

This abstraction allows different drones to be equipped with different sensor types
(camera, LIDAR, ultrasonic, etc.) while maintaining a uniform interface for the
environment to interact with them.
"""

from abc import ABC, abstractmethod
from typing import List, Tuple
import numpy as np


class BaseSensor(ABC):
    """
    Abstract base class for drone sensors.

    This interface defines the contract that all sensor implementations must follow.
    It enables the simulation to support heterogeneous sensor configurations where
    different drones can have different types of sensors with varying capabilities.

    Implementing classes must provide:
    - sense(): Perform environmental sensing
    - get_max_range(): Return maximum sensing distance
    - get_sensor_type(): Return sensor type identifier
    - get_sensor_params(): Return sensor configuration parameters
    """

    @abstractmethod
    def sense(
        self,
        pos: Tuple[int, int],
        facing: str,
        grid: np.ndarray
    ) -> List[Tuple[int, int, int]]:
        """
        Perform sensing from the given position and orientation.

        This method simulates the sensor's perception of the environment,
        returning all cells that the sensor can observe from its current state.

        Args:
            pos: Current (x, y) position of the drone
            facing: Current facing direction ('NORTH', 'EAST', 'SOUTH', 'WEST')
            grid: The environment grid (2D numpy array of tile values)

        Returns:
            List of (x, y, tile_value) tuples representing observed cells.
            Each tuple contains:
            - x: X coordinate of the observed cell
            - y: Y coordinate of the observed cell
            - tile_value: The type of tile at that position
        """
        pass

    @abstractmethod
    def get_max_range(self) -> int:
        """
        Return the maximum sensing range of the sensor.

        Returns:
            Maximum distance (in grid cells) that the sensor can observe
        """
        pass

    @abstractmethod
    def get_sensor_type(self) -> str:
        """
        Return a string identifier for the sensor type.

        This is used for logging, debugging, and sensor-specific logic.

        Returns:
            String identifier (e.g., 'camera', 'lidar', 'ultrasonic')
        """
        pass

    @abstractmethod
    def get_sensor_params(self) -> dict:
        """
        Return the sensor's configuration parameters.

        This method provides access to sensor-specific parameters for
        debugging, logging, or dynamic reconfiguration.

        Returns:
            Dictionary of sensor parameters (e.g., {'fov': 45, 'range': 10})
        """
        pass

    def is_in_range(self, drone_pos: Tuple[int, int], target_pos: Tuple[int, int]) -> bool:
        """
        Check if a target position is within sensor range.

        Helper method to determine if a given position could potentially
        be sensed from the drone's current position.

        Args:
            drone_pos: Current position of the drone
            target_pos: Position to check

        Returns:
            True if the target is within maximum sensor range
        """
        distance = abs(drone_pos[0] - target_pos[0]) + abs(drone_pos[1] - target_pos[1])
        return distance <= self.get_max_range()