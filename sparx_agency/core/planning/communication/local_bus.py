"""
Implements `LocalCommBus`, a concrete class for local (in-memory) communication between
drones and the master controller in the SLAM simulation.

This class adheres to the `CommunicationInterface` and provides:
- Storage and retrieval of drone states and instructions
- Simple command broadcasting without networking
- Global map synchronization across all agents

This implementation is suitable for simulation and testing environments where
communication can be simulated via shared memory.
"""
from planner.communication.comm_interface import CommunicationInterface
from typing import Dict, Any, Optional, Literal
import numpy as np
from planner.simulation.simulation_constants import DIRECTIONS


class LocalCommBus(CommunicationInterface):
    """
    Concrete implementation of the `CommunicationInterface` for in-memory communication.

    `LocalCommBus` enables message passing between drones and the master controller
    without any real network layer. It supports:
    - State broadcasting: Drones update their current state.
    - Instruction delivery: The master controller assigns and retrieves movement commands.
    - Global map synchronization: Drones share the updated map globally.

    This implementation is primarily used for local simulations and testing.
    """
    def __init__(self):
        self.states: Dict[int, Dict[str, Any]] = {}
        self.instructions: Dict[int, DIRECTIONS] = {}
        self.global_map: Optional[np.ndarray] = None

    def broadcast_state(self, drone_id: int, state: Dict[str, Any]) -> None:
        """
        Store the current state of a drone in shared memory.

        Args:
            drone_id (int): The drone's unique identifier.
            state (Dict[str, Any]): The drone's state dictionary.
        """
        self.states[drone_id] = state

    def receive_instructions(self, drone_id: int) -> DIRECTIONS | None:
        """
        Get the latest instruction assigned to the specified drone.

        Args:
            drone_id (int): The drone's unique identifier.

        Returns:
            DIRECTIONS | None: The movement instruction assigned to the drone, or None if unavailable.
        """
        return self.instructions.get(drone_id, None)

    def send_instruction(self, drone_id: int, instruction: DIRECTIONS) -> None:
        """
        Store the instruction to be retrieved by the specified drone.

        Args:
            drone_id (int): The drone's unique identifier.
            instruction (Literal): One of the valid movement instructions.
        """
        self.instructions[drone_id] = instruction

    def send_global_map(self, global_map: np.ndarray) -> None:
        """
        Set the global map state shared across all drones.

        Args:
            global_map (np.ndarray): The updated global map.
        """
        self.global_map = global_map

    def get_all_drones_state(self) -> Dict[int, Dict[str, Any]]:
        """
        Get a copy of the current state of all drones.

        Returns:
            Dict[int, Dict[str, Any]]: A snapshot of all drone states.
        """
        return self.states.copy()
