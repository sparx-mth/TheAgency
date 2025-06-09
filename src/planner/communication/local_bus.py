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
from src.planner.communication.interface import CommunicationInterface


class LocalCommBus(CommunicationInterface):
    def __init__(self):
        self.states = {}
        self.instructions = {}
        self.global_map = None

    def broadcast_state(self, drone_id, state):
        self.states[drone_id] = state

    def receive_instructions(self, drone_id):
        return self.instructions.get(drone_id, {})

    def send_instruction(self, drone_id, instruction):
        # print(f"[DEBUG] Drone {drone_id} gets instruction: {instruction}")

        self.instructions[drone_id] = instruction

    def send_global_map(self, global_map):
        self.global_map = global_map

    def get_all_drones_state(self):
        return self.states.copy()
