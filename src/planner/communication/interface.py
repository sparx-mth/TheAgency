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


class CommunicationInterface:
    def broadcast_state(self, drone_id, state):  # position, map update, etc.
        raise NotImplementedError

    def receive_instructions(self, drone_id):
        raise NotImplementedError

    def send_global_map(self, global_map):
        raise NotImplementedError

    def get_all_drones_state(self):
        raise NotImplementedError
