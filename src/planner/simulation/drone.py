"""
This module defines the `Drone` class, which models an autonomous agent in a
2D grid environment performing SLAM (Simultaneous Localization and Mapping).

Each drone maintains:
- A position and movement history
- A local map of the environment (initialized as unknown)
- A field-of-view (FOV) radius for sensing
- A communication interface to interact with a central controller (e.g., master)

Core Methods:
-------------
- `activate(current_time)`: Activates the drone if its entry time has arrived.
- `move(direction, env)`: Moves the drone in a specified direction and updates its state.
- `sense(env)`: Uses Bresenham's algorithm to perform FOV-limited sensing and updates the local map.
- `initialize_map(map_shape)`: Initializes the local map with all unknown tiles.
- `get_observed_map()`: Returns the drone's local map.
- `get_position()`: Returns the current position.
- `get_history()`: Returns the movement path history.

Usage:
------
This class is used as part of a multi-agent SLAM simulation. The drone interacts
with the environment and reports its findings to a communication bus, which is used
by a centralized planner or controller.
"""

import numpy as np
import random
from typing import List, Tuple, Optional, Dict, Literal, TYPE_CHECKING
from src.planner.simulation.simulation_constants import DIRECTIONS, FACING_DIRECTION, FACING_TO_DELTA, FACING_DIRECTIONS, DIRECTION_COMMANDS
from src.planner.simulation.sensors.sensor_manager import SensorManager
from src.planner.communication.comm_interface import CommunicationInterface
if TYPE_CHECKING:
    from src.planner.simulation.grid_map_env import GridMapEnv


def turn(
    facing: FACING_DIRECTION,
    action: Literal['TURN_LEFT', 'TURN_RIGHT']
) -> FACING_DIRECTION:
    """
    Computes the new facing direction after applying a turn action.

    Args:
        facing (FACING_DIRECTION): Current direction.
        action (Literal): 'TURN_LEFT' or 'TURN_RIGHT'.

    Returns:
        FACING_DIRECTION: Updated direction.
    """
    idx = FACING_DIRECTIONS.index(facing)
    if action == 'TURN_LEFT':
        return FACING_DIRECTIONS[(idx - 1) % 4]
    elif action == 'TURN_RIGHT':
        return FACING_DIRECTIONS[(idx + 1) % 4]
    return facing


class Drone:
    """
    Represents a mobile SLAM drone operating within a 2D grid environment.

    Each drone is responsible for navigating the environment, maintaining a local
    map based on its sensors, and reporting its state and discoveries to a central
    controller via a communication interface.

    Key Attributes:
    - id: Unique identifier.
    - pos: Current position on the grid.
    - facing_direction: Direction the drone is currently facing.
    - entry_time: Simulation tick when the drone becomes active.
    - active: Whether the drone is currently active.
    - path_history: List of visited positions.
    - local_map: The internal map built from the drone’s observations.
    - sensor_manager: Manages one or more attached sensors.
    - comm: Interface to send/receive state and instructions.

    This class is used in multi-agent SLAM simulations to explore and map unknown environments.
    """
    def __init__(
        self,
        drone_id: int,
        start_pos: Tuple[int, int],
        comm_interface: CommunicationInterface,
        entry_time: int = 0,
        facing_direction: FACING_DIRECTION = 'NORTH',
        sensors: Optional[List] = None
    ):
        """
        Initialize a new drone.

        Args:
            drone_id (int): Unique identifier.
            start_pos (Tuple[int, int]): Starting position.
            comm_interface (CommunicationInterface): Messaging interface.
            entry_time (int): Tick when the drone becomes active.
            facing_direction (FACING_DIRECTION): Initial direction.
            sensors (Optional[List]): Optional list of sensors.
        """
        self.id: int = drone_id
        self.pos: Tuple[int, int] = start_pos
        self.entry_time: int = entry_time
        self.active: bool = False
        self.local_map: Optional[np.ndarray] = None
        self.path_history: List[Tuple[int, int]] = [start_pos]
        self.collided: bool = False
        self.comm: CommunicationInterface = comm_interface
        self.facing_direction: FACING_DIRECTION = facing_direction
        self.sensor_manager: SensorManager = SensorManager()

        if sensors:
            for sensor in sensors:
                self.sensor_manager.add_sensor(sensor)

    def initialize_map(self, map_shape: Tuple[int, int]) -> None:
        """Initialize the local map with unknown (-1) values."""
        self.local_map = np.full(map_shape, -1, dtype=np.int8)  # -1 = unknown

    def activate(self, current_time: int) -> None:
        """Activate the drone if current_time ≥ entry_time."""
        if not self.active and current_time >= self.entry_time:
            self.active = True
            self.comm.broadcast_state(self.id, self._make_state([]))

    def move(self, action: DIRECTIONS, env: "GridMapEnv") -> List[Tuple[int, int, int]]:
        """
        Move the drone based on an action and update discoveries.

        Args:
            action (DIRECTIONS): Movement or rotation command.
            env (GridMapEnv): The environment for movement and sensing.

        Returns:
            List[Tuple[int, int, int]]: List of newly discovered (x, y, val) tiles.
        """
        if not self.active:
            return []

        if action in ['TURN_LEFT', 'TURN_RIGHT']:
            self.facing_direction = turn(self.facing_direction, action)  # type: ignore
            new_discoveries = self.sense(env)
            self.comm.broadcast_state(self.id, self._make_state(new_discoveries))
            return new_discoveries

        elif action == 'FORWARD':
            dx, dy = FACING_TO_DELTA[self.facing_direction]
            new_x, new_y = self.pos[0] + dx, self.pos[1] + dy

            if env.is_collision(new_x, new_y):
                self.collided = True
                return []

            self.pos = (new_x, new_y)
            self.path_history.append(self.pos)
            self.collided = False

            new_discoveries = self.sense(env)
            self.comm.broadcast_state(self.id, self._make_state(new_discoveries))
            return new_discoveries

        elif action == 'STAY':
            new_discoveries = self.sense(env)
            self.comm.broadcast_state(self.id, self._make_state(new_discoveries))
            return new_discoveries

        else:
            raise ValueError(f"Invalid action for constrained drone: {action}")

    def sense(self, env: "GridMapEnv") -> List[Tuple[int, int, int]]:
        """
        Use the drone's sensors to scan and update the local map.

        Returns:
            List of newly discovered (x, y, val) tiles.
        """
        if not self.active:
            return []

        observations = self.sensor_manager.sense_all(self.pos, self.facing_direction, env)
        new_discoveries = []

        for x, y, val in observations:
            if self.local_map[y, x] != val:
                self.local_map[y, x] = val
                new_discoveries.append((x, y, val))

        return new_discoveries

    def _make_state(self, new_discoveries: List[Tuple[int, int, int]]) -> Dict:
        """
        Package the drone’s current state for broadcasting to the controller.

        Args:
            new_discoveries (List[Tuple[int, int, int]]):
                List of newly observed tiles as (x, y, val) tuples.

        Returns:
            Dict[str, object]: A dictionary containing the drone's current position,
                facing direction, entry time, activation status, and discoveries.
        """
        return {
            'pos': self.pos,
            'facing_direction': self.facing_direction,
            'entry_time': self.entry_time,
            'active': self.active,
            'new_discoveries': new_discoveries
        }

    def get_observed_map(self) -> Optional[np.ndarray]:
        """Returns the current local map observed by the drone."""
        return self.local_map

    def get_position(self) -> Tuple[int, int]:
        """Returns the drone's current position on the map."""
        return self.pos

    def get_history(self) -> List[Tuple[int, int]]:
        """Returns the history of all positions the drone has visited."""
        return self.path_history

    def get_facing_arrow_vector(self):
        """Returns the direction vector the drone is facing, for rendering."""
        dx, dy = FACING_TO_DELTA[self.facing_direction]
        return dx, dy
