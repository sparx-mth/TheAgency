"""
This module defines an abstract communication interface for a multi-agent SLAM simulation.

It provides a contract for bidirectional communication between autonomous drones and
a central controller, including state reporting, command dispatching, and map sharing.

Concrete implementations (e.g., `LocalCommBus`) must implement all abstract methods.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Literal
import numpy as np
from src.planner.simulation.simulation_constants import DIRECTIONS


class CommunicationInterface(ABC):
    """
    Abstract base class for drone-controller communication in the SLAM simulation.

    This interface defines how drones share their states with the central controller
    and how the controller issues commands and synchronizes the global map.

    All subclasses must implement:
    - Broadcasting state information from drones
    - Receiving movement instructions for each drone
    - Synchronizing a global map across the system
    - Providing access to the states of all drones
    """
    @abstractmethod
    def broadcast_state(self, drone_id: int, state: Dict[str, Any]) -> None:
        """
        Broadcast the current state of a drone to the central controller.

        Args:
            drone_id (int): Unique identifier for the drone.
            state (dict): Dictionary containing position, discoveries, facing, etc.
        """
        pass

    @abstractmethod
    def receive_instructions(self, drone_id: int) -> DIRECTIONS:
        """
        Retrieve the next instruction intended for a specific drone.

        Args:
            drone_id (int): Unique identifier for the drone.

        Returns:
            Optional[str]: One of 'FORWARD', 'TURN_LEFT', 'TURN_RIGHT', 'STAY', or None if not found.
        """
        pass

    @abstractmethod
    def send_instruction(self, drone_id: int, instruction: DIRECTIONS) -> None:
        """
        Send a specific instruction to a drone.

        Args:
            drone_id (int or str): Unique identifier for the drone.
            instruction (Literal): One of 'FORWARD', 'TURN_LEFT', 'TURN_RIGHT', or 'STAY'.
        """
        pass

    @abstractmethod
    def send_global_map(self, global_map: np.ndarray) -> None:
        """
        Send the updated global map to the communication system.

        Args:
            global_map (np.ndarray): The environment map updated by the controller.
        """
        pass

    @abstractmethod
    def get_all_drones_state(self) -> Dict[int, Dict[str, Any]]:
        """
        Retrieve the current state of all drones for centralized planning.

        Returns:
            dict: A dictionary mapping drone IDs to their current states.
        """
        pass
