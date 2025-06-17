"""
BaseSensor Interface for SLAM Simulation

This module defines the abstract base class `BaseSensor`, which represents
a generic sensor mounted on a drone within the SLAM simulation framework.

Key Responsibilities:
- Enforces implementation of the `sense` method for environmental perception.
- Encapsulates directional sensing behavior depending on the drone's orientation and position.

Concrete implementations of this class must define how sensing is performed
(e.g., forward-facing cameras, 360° LiDAR, etc.).
"""

from abc import ABC, abstractmethod
from typing import Tuple, List, TYPE_CHECKING
from planner.simulation.simulation_constants import FACING_DIRECTION

if TYPE_CHECKING:
    from planner.simulation.grid_map_env import GridMapEnv


class BaseSensor(ABC):
    """
    Abstract base class for drone sensors in the SLAM simulation.

    This interface specifies the required `sense` method that all concrete
    sensor types must implement. Sensors simulate environmental perception
    based on drone location and orientation.

    Implementing subclasses may define different sensor shapes and behaviors,
    such as directional FOV or omnidirectional sensing.
    """
    @abstractmethod
    def sense(self, pos: Tuple[int, int], facing: FACING_DIRECTION, env: "GridMapEnv") -> List[Tuple[int, int, int]]:
        """
        Perform sensing from the current position and facing direction.

        Args:
            pos (Tuple[int, int]): Current (x, y) position of the drone.
            facing (FACING_DIRECTION): Current facing direction (e.g., 'NORTH').
            env (GridMapEnv): The simulation environment (used to query observed tiles).

        Returns:
            List[Tuple[int, int, int]]: List of (x, y, value) tuples representing
            the coordinates and values of observed cells.
        """
        pass
