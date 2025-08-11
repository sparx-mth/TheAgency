"""
core/drone_state.py

This file defines the DroneState dataclass which represents the complete state
of a single drone in the simulation. It encapsulates position, orientation,
sensor configuration, and tracking information.

The DroneState is used by the environment to manage drone states without
the complexity of a full drone class, maintaining clean separation between
state and behavior.
"""

from dataclasses import dataclass, field
from typing import Tuple, List, TYPE_CHECKING

from .constants import DIRECTIONS

if TYPE_CHECKING:
    from sensors.base_sensor import BaseSensor


@dataclass
class DroneState:
    """
    Complete state representation of a single drone.

    This dataclass holds all the information needed to track a drone's
    state in the simulation, including its physical properties, sensor
    configuration, and performance metrics.

    Attributes:
        drone_id: Unique identifier for the drone
        pos: Current (x, y) position on the grid
        facing: Current facing direction (NORTH, EAST, SOUTH, WEST)
        active: Whether the drone is currently active in the simulation
        entry_time: Simulation step when the drone becomes active
        sensor: The sensor attached to this drone
        collision_count: Number of collisions experienced
        discoveries: List of newly discovered tiles in the last step
        total_discoveries: Total number of tiles discovered by this drone
        path_history: History of positions visited
    """

    # Required fields
    drone_id: int
    pos: Tuple[int, int]
    facing: str
    active: bool
    entry_time: int
    sensor: 'BaseSensor'

    # Optional fields with defaults
    collision_count: int = 0
    discoveries: List[Tuple[int, int, int]] = field(default_factory=list)
    total_discoveries: int = 0
    path_history: List[Tuple[int, int]] = field(default_factory=list)

    def get_facing_idx(self) -> int:
        """
        Get the index of the current facing direction.

        Returns:
            Index in DIRECTIONS list (0=NORTH, 1=EAST, 2=SOUTH, 3=WEST)
        """
        return DIRECTIONS.index(self.facing)

    def turn(self, action: str) -> None:
        """
        Update facing direction based on turn action.

        Args:
            action: Either 'TURN_LEFT' or 'TURN_RIGHT'
        """
        idx = self.get_facing_idx()
        if action == 'TURN_LEFT':
            self.facing = DIRECTIONS[(idx - 1) % 4]
        elif action == 'TURN_RIGHT':
            self.facing = DIRECTIONS[(idx + 1) % 4]

    def update_position(self, new_pos: Tuple[int, int]) -> None:
        """
        Update the drone's position and add to path history.

        Args:
            new_pos: New (x, y) position
        """
        self.pos = new_pos
        self.path_history.append(new_pos)

    def add_collision(self) -> None:
        """Increment the collision counter."""
        self.collision_count += 1

    def add_discoveries(self, new_discoveries: List[Tuple[int, int, int]]) -> None:
        """
        Update discoveries for this step.

        Args:
            new_discoveries: List of (x, y, tile_value) tuples discovered this step
        """
        self.discoveries = new_discoveries
        self.total_discoveries += len(new_discoveries)

    def reset_step_discoveries(self) -> None:
        """Clear the discoveries list for the new step."""
        self.discoveries = []

    def to_dict(self) -> dict:
        """
        Convert drone state to dictionary for communication.

        Returns:
            Dictionary representation of the drone state
        """
        return {
            'drone_id': self.drone_id,
            'position': self.pos,
            'facing': self.facing,
            'facing_idx': self.get_facing_idx(),
            'active': self.active,
            'collision_count': self.collision_count,
            'total_discoveries': self.total_discoveries,
            'sensor_type': type(self.sensor).__name__,
        }