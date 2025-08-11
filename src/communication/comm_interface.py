"""
communication/comm_interface.py

This file defines the abstract communication interface for the multi-agent SLAM system.
The interface provides a contract for different communication implementations, enabling
the system to switch between local memory communication (for simulation) and networked
communication (e.g., ROS2) without changing the core logic.

This abstraction is crucial for:
- Testing and simulation with local communication
- Deployment with ROS2 or other middleware
- Supporting different communication patterns (centralized, decentralized, etc.)
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List, Tuple
import numpy as np


class CommunicationInterface(ABC):
    """
    Abstract communication interface for drone coordination.

    This interface defines how drones and the environment share information,
    including state broadcasting, command distribution, and map synchronization.

    Implementations of this interface can provide:
    - Local memory communication (for single-machine simulation)
    - ROS2 topics and services (for distributed systems)
    - TCP/UDP networking (for custom protocols)
    - Shared memory IPC (for high-performance local communication)

    All methods should be thread-safe if used in a multi-threaded environment.
    """

    @abstractmethod
    def broadcast_state(self, drone_id: int, state: Dict[str, Any]) -> None:
        """
        Broadcast a drone's state to the communication system.

        This method is called by drones to share their current state with
        other agents or a central controller.

        Args:
            drone_id: Unique identifier of the broadcasting drone
            state: Dictionary containing the drone's state information
                  Expected keys may include:
                  - 'position': (x, y) tuple
                  - 'facing': direction string
                  - 'active': boolean
                  - 'discoveries': list of discovered tiles
                  - 'sensor_data': sensor readings
        """
        pass

    @abstractmethod
    def receive_command(self, drone_id: int) -> Optional[int]:
        """
        Receive a command for a specific drone.

        This method is called by drones to check if there are any
        pending commands for them.

        Args:
            drone_id: Unique identifier of the drone requesting commands

        Returns:
            Command as an integer (Action enum value) if available,
            None if no command is pending
        """
        pass

    @abstractmethod
    def send_command(self, drone_id: int, command: int) -> None:
        """
        Send a command to a specific drone.

        This method is called by agents or controllers to send
        action commands to drones.

        Args:
            drone_id: Unique identifier of the target drone
            command: Action command as integer (Action enum value)
        """
        pass

    @abstractmethod
    def get_all_states(self) -> Dict[int, Dict[str, Any]]:
        """
        Get the current states of all drones.

        This method provides a snapshot of all drone states for
        centralized planning or monitoring.

        Returns:
            Dictionary mapping drone_id to state dictionary
        """
        pass

    @abstractmethod
    def broadcast_map_update(self, discoveries: List[Tuple[int, int, int]]) -> None:
        """
        Broadcast newly discovered map cells.

        This method shares map discoveries with all agents in the system,
        enabling collaborative mapping.

        Args:
            discoveries: List of (x, y, tile_value) tuples representing
                        newly discovered cells
        """
        pass

    @abstractmethod
    def get_global_map(self) -> Optional[np.ndarray]:
        """
        Get the current global map.

        Returns the shared global map that represents the collective
        knowledge of all drones.

        Returns:
            2D numpy array representing the global map,
            None if no map is available
        """
        pass

    @abstractmethod
    def set_global_map(self, global_map: np.ndarray) -> None:
        """
        Set or update the global map.

        This method updates the shared global map, typically called
        by the environment or a map fusion component.

        Args:
            global_map: 2D numpy array representing the updated global map
        """
        pass

    @abstractmethod
    def reset(self) -> None:
        """
        Reset all communication state.

        This method clears all stored states, commands, and map data.
        Typically called at the start of a new episode.
        """
        pass

    @abstractmethod
    def is_connected(self) -> bool:
        """
        Check if the communication system is connected and operational.

        Returns:
            True if communication is working, False otherwise
        """
        pass