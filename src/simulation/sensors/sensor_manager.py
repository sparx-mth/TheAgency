"""
Manages a collection of sensors for a drone in the SLAM simulation.

Allows adding multiple sensors (e.g., camera, FOV) and invoking them all
to collect combined observations at a given position and facing direction.
"""

from typing import List, Tuple, TYPE_CHECKING
from simulation.sensors.base_sensor import BaseSensor
from simulation.world.simulation_constants import FACING_DIRECTION
if TYPE_CHECKING:
    from envs.grid_map_env import GridMapEnv


class SensorManager:
    """
    SensorManager is responsible for managing and coordinating multiple sensors
    attached to a drone in the SLAM simulation.

    It supports adding arbitrary sensors and aggregating their outputs into a unified
    observation set when the drone performs sensing.
    """
    def __init__(self):
        """
        Initializes the sensor manager with an empty sensor list.
        """
        self.sensors: List[BaseSensor] = []

    def add_sensor(self, sensor: BaseSensor) -> None:
        """
        Adds a sensor to the manager.

        Args:
            sensor (BaseSensor): A sensor implementing the sense() interface.
        """
        self.sensors.append(sensor)

    def sense_all(self, pos: Tuple[int, int], facing: FACING_DIRECTION, env: "GridMapEnv") -> List[Tuple[int, int, int]]:
        """
        Calls all sensors and aggregates their observations.

        Args:
            pos (Tuple[int, int]): The drone's current (x, y) position.
            facing (str): The current facing direction.
            env (GridMapEnv): The simulation environment.

        Returns:
            List[Tuple[int, int, int]]: All observed cells from all sensors.
        """
        all_observations = []
        for sensor in self.sensors:
            observations = sensor.sense(pos, facing, env)
            all_observations.extend(observations)
        return all_observations
