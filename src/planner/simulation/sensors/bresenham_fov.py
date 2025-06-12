"""
Implements `BresenhamFOVSensor`, which simulates a circular field-of-view using
Bresenham's line algorithm for ray-casting from the drone's position.

Each ray is cast outward from the drone's location within a specified radius, stopping
when it hits an obstacle such as a wall or closed door. This simulates how a drone might
sense its surroundings using a camera or other directional sensor.

This sensor is intended for use in multi-agent SLAM simulations.
"""

import math
from typing import Tuple, List, TYPE_CHECKING
from src.planner.simulation.sensors.base_sensor import BaseSensor
from src.planner.simulation.simulation_constants import WALL, DOOR_CLOSED, FACING_DIRECTION
if TYPE_CHECKING:
    from src.planner.simulation.grid_map_env import GridMapEnv


class BresenhamFOVSensor(BaseSensor):
    """
    A concrete sensor that performs circular field-of-view sensing using Bresenham's algorithm.

    Rays are cast in all directions from the drone's current position, up to a given radius.
    The sensor stops sensing along each ray once it encounters a blocking tile (wall or closed door).

    Attributes:
        radius (int): The sensing radius in tiles.
    """
    def __init__(self, radius: int):
        """
        Initializes the sensor with a circular field-of-view.

        Args:
            radius (int): The maximum sensing radius (in tiles).
        """
        self.radius: int = radius

    def sense(self, pos: Tuple[int, int], facing: FACING_DIRECTION, env: "GridMapEnv") -> List[Tuple[int, int, int]]:
        """
        Perform sensing using Bresenham ray-casting from the current position.

        Args:
            pos (Tuple[int, int]): Drone's current (x, y) position.
            facing (str): Current facing direction (unused in this implementation).
            env (GridMapEnv): The simulation environment for map access.

        Returns:
            List[Tuple[int, int, int]]: List of sensed (x, y, value) cells.
        """
        def bresenham(x0, y0, x1, y1):
            dx = abs(x1 - x0)
            dy = -abs(y1 - y0)
            sx = 1 if x0 < x1 else -1
            sy = 1 if y0 < y1 else -1
            err = dx + dy
            while True:
                yield x0, y0
                if x0 == x1 and y0 == y1:
                    break
                e2 = 2 * err
                if e2 >= dy:
                    err += dy
                    x0 += sx
                if e2 <= dx:
                    err += dx
                    y0 += sy

        cx, cy = pos
        observations = []

        for offset_y in range(-self.radius, self.radius + 1):
            for offset_x in range(-self.radius, self.radius + 1):
                x, y = cx + offset_x, cy + offset_y
                if not (0 <= x < env.width and 0 <= y < env.height):
                    continue
                if offset_x ** 2 + offset_y ** 2 > self.radius ** 2:
                    continue

                for lx, ly in bresenham(cx, cy, x, y):
                    if not (0 <= lx < env.width and 0 <= ly < env.height):
                        break
                    val = env.get_tile(lx, ly)
                    observations.append((lx, ly, val))
                    if val in {WALL, DOOR_CLOSED}:
                        break

        return observations
