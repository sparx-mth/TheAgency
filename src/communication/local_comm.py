"""
communication/local_comm.py

This file implements local in-memory communication for the SLAM simulation.
It provides a simple, efficient communication mechanism for single-machine
simulations where all components run in the same process.

This implementation is ideal for:
- Development and testing
- Single-machine simulations
- Benchmarking and evaluation
- Situations where network communication overhead should be avoided
"""

from typing import Dict, Any, Optional, List, Tuple
import numpy as np
from threading import Lock

from .comm_interface import CommunicationInterface


class LocalCommunication(CommunicationInterface):
    """
    Local in-memory communication implementation.

    This class provides communication through shared memory structures,
    making it efficient for single-process simulations. It maintains
    thread safety through locks to support potential multi-threaded usage.

    Attributes:
        states: Dictionary storing current state of each drone
        commands: Dictionary storing pending commands for each drone
        global_map: Shared global map array
        map_updates: List of recent map discoveries
        lock: Threading lock for thread-safe operations
    """

    def __init__(self):
        """Initialize the local communication system."""
        self.states: Dict[int, Dict[str, Any]] = {}
        self.commands: Dict[int, int] = {}
        self.global_map: Optional[np.ndarray] = None
        self.map_updates: List[Tuple[int, int, int]] = []
        self.lock = Lock()  # For thread safety
        self._connected = True

    def broadcast_state(self, drone_id: int, state: Dict[str, Any]) -> None:
        """
        Store a drone's state in shared memory.

        Args:
            drone_id: Unique identifier of the drone
            state: State dictionary to store
        """
        with self.lock:
            self.states[drone_id] = state.copy()

    def receive_command(self, drone_id: int) -> Optional[int]:
        """
        Retrieve a pending command for a drone.

        Args:
            drone_id: Unique identifier of the drone

        Returns:
            Command integer if available, None otherwise
        """
        with self.lock:
            return self.commands.get(drone_id)

    def send_command(self, drone_id: int, command: int) -> None:
        """
        Store a command for a drone.

        Args:
            drone_id: Target drone identifier
            command: Command to send
        """
        with self.lock:
            self.commands[drone_id] = command

    def get_all_states(self) -> Dict[int, Dict[str, Any]]:
        """
        Get a snapshot of all drone states.

        Returns:
            Copy of all drone states
        """
        with self.lock:
            return self.states.copy()

    def broadcast_map_update(self, discoveries: List[Tuple[int, int, int]]) -> None:
        """
        Add new discoveries to the update list.

        Args:
            discoveries: List of newly discovered cells
        """
        with self.lock:
            self.map_updates.extend(discoveries)

    def get_global_map(self) -> Optional[np.ndarray]:
        """
        Get the current global map.

        Returns:
            Copy of the global map if available
        """
        with self.lock:
            if self.global_map is not None:
                return self.global_map.copy()
            return None

    def set_global_map(self, global_map: np.ndarray) -> None:
        """
        Update the global map.

        Args:
            global_map: New global map array
        """
        with self.lock:
            self.global_map = global_map.copy()

    def get_map_updates(self) -> List[Tuple[int, int, int]]:
        """
        Get and clear pending map updates.

        Returns:
            List of map updates since last call
        """
        with self.lock:
            updates = self.map_updates.copy()
            self.map_updates.clear()
            return updates

    def clear_commands(self) -> None:
        """Clear all pending commands."""
        with self.lock:
            self.commands.clear()

    def reset(self) -> None:
        """Reset all communication state."""
        with self.lock:
            self.states.clear()
            self.commands.clear()
            self.global_map = None
            self.map_updates.clear()

    def is_connected(self) -> bool:
        """
        Check connection status.

        Always returns True for local communication.

        Returns:
            True (local communication is always connected)
        """
        return self._connected

    def disconnect(self) -> None:
        """Simulate disconnection (for testing)."""
        self._connected = False

    def reconnect(self) -> None:
        """Simulate reconnection (for testing)."""
        self._connected = True