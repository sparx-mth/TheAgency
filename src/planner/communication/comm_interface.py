"""
Defines the abstract base class `CommunicationInterface` used to standardize communication
between drones and the central controller in the SLAM simulation.

This interface ensures consistent implementation of:
- State broadcasting from drones to the controller
- Instruction reception from the controller to drones
- Global map synchronization
- Access to all drone states for centralized planning

Concrete implementations of this interface (e.g., `LocalCommBus`) must implement all methods.
"""
from abc import ABC, abstractmethod


class CommunicationInterface(ABC):
    @abstractmethod
    def broadcast_state(self, drone_id, state):
        """
        Broadcast the current state of a drone to the central controller.

        Args:
            drone_id (int or str): Unique identifier for the drone.
            state (dict): Dictionary containing position, discoveries, facing, etc.
        """
        pass

    @abstractmethod
    def receive_instructions(self, drone_id):
        """
        Retrieve the next instruction intended for a specific drone.

        Args:
            drone_id (int or str): Unique identifier for the drone.

        Returns:
            dict or any instruction format: The instruction to execute.
        """
        pass

    @abstractmethod
    def send_instruction(self, drone_id, instruction):
        """
        Send a specific instruction to a drone.

        Args:
            drone_id (int or str): Unique identifier for the drone.
            instruction (any): The movement or action command for the drone.
        """
        pass

    @abstractmethod
    def send_global_map(self, global_map):
        """
        Send the updated global map to the communication system.

        Args:
            global_map (np.ndarray): The environment map updated by the controller.
        """
        pass

    @abstractmethod
    def get_all_drones_state(self):
        """
        Retrieve the current state of all drones for centralized planning.

        Returns:
            dict: A dictionary mapping drone IDs to their current states.
        """
        pass
