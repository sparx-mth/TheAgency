"""
Implements a directional CameraSensor for SLAM drones.

This sensor casts a ray in the drone's current facing direction and returns all
cells in a straight line—up to a maximum distance—stopping at obstacles like walls or doors.

Used in directional vision-based SLAM simulations to mimic a camera's forward view.
"""

from typing import Tuple, List, TYPE_CHECKING
from planner.simulation.sensors.base_sensor import BaseSensor
from planner.simulation.simulation_constants import FACING_TO_DELTA, WALL, DOOR_CLOSED, FACING_DIRECTION
if TYPE_CHECKING:
    from planner.simulation.grid_map_env import GridMapEnv


class CameraSensor(BaseSensor):
    """
    A sensor that simulates a directional camera mounted on a SLAM drone.

    The camera casts a ray from the drone’s position in its current facing direction
    and returns all visible cells in a straight line up to a specified distance.
    Visibility is blocked by obstacles such as walls or closed doors.
    """
    def __init__(self, max_distance: int):
        """
        Initializes the sensor.

        Args:
            max_distance (int): Maximum range of the camera in tiles.
        """
        self.max_distance: int = max_distance

    def sense(self, pos: Tuple[int, int], facing: FACING_DIRECTION, env: "GridMapEnv") -> List[Tuple[int, int, int]]:
        """
        Simulates a forward-looking camera sensor that returns all tiles
        in a straight line in the current facing direction up to max_distance.

        Args:
            pos (Tuple[int, int]): Current (x, y) position of the drone.
            facing (str): Facing direction ('NORTH', 'EAST', etc.).
            env (GridMapEnv): The environment object with map access.

        Returns:
            List[Tuple[int, int, int]]: List of (x, y, value) tuples for each
            visible tile ahead, stopping at obstacles or max range.
        """
        dx, dy = FACING_TO_DELTA[facing]
        x, y = pos
        observations = []

        for i in range(1, self.max_distance + 1):
            nx, ny = x + i * dx, y + i * dy
            if not (0 <= nx < env.width and 0 <= ny < env.height):
                break
            val = env.get_tile(nx, ny)
            observations.append((nx, ny, val))

            # Stop if vision is blocked
            if val in {WALL, DOOR_CLOSED}:
                break

        return observations
