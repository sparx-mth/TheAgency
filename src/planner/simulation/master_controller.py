"""
This module defines the `MasterController` class, which coordinates the behavior of multiple drones
in a SLAM environment. It acts as the central planner and communicator, assigning exploration tasks to drones and
aggregating their local discoveries into a global map.

Key Responsibilities:
---------------------
- Maintain a global map updated from drones' discoveries
- Detect frontiers (exploration boundaries) in the map
- Assign directions to drones based on the selected planning strategy ("random" or "frontier")
- Track each drone’s current goal, planned path, and wait time if blocked
- Communicate motion instructions to drones via the communication interface

Constructor Parameters:
------------------------
- env: GridMapEnv object representing the simulation environment
- discoverable_mask: A 2D boolean mask indicating which cells are discoverable
- comm_interface: Object implementing `CommunicationInterface`, used to send/receive data
- mode: Planning mode ("random" for random walk or "frontier" for frontier-based planning)

Main Methods:
-------------
- step(current_time): Main update loop called each simulation tick. Processes drone states, updates the map,
  assigns new directions, and communicates with drones.
- _update_frontiers(): Detects all valid frontier cells that border unexplored areas. Used in frontier-based planning.

Notes:
------
- The controller supports both naive random-walk and frontier-based planning.
- The class supports scalable multi-agent coordination and is designed for centralized control of SLAM drones.

Usage:
------
This module should be instantiated once per simulation and called regularly via `step()` to control drone behavior.
"""

import numpy as np
from src.planner.algorithm.naive_planner import plan_random_walk, plan_frontier
from src.planner.simulation.simulation_constants import *



class MasterController:
    def __init__(self, env, discoverable_mask, comm_interface, mode="frontier"):
        self.env = env
        self.comm = comm_interface
        self.global_map = np.full(self.env.grid.shape, -1, dtype=np.int8)
        self.frontiers = set()
        self.discoverable_mask = discoverable_mask
        self.mode = mode  # "random" or "frontier"
        self.goals = {}
        self.paths = {}
        self.wait_counters = {}
        self.max_wait = 3

    def step(self, current_time):
        all_states = self.comm.get_all_drones_state()

        # Update global map with new discoveries
        for state in all_states.values():
            for x, y, val in state.get("new_discoveries", []):
                if 0 <= y < self.global_map.shape[0] and 0 <= x < self.global_map.shape[1]:
                    if self.global_map[y, x] == -1:
                        self.global_map[y, x] = val

        self._update_frontiers()
        assigned_goals = set()

        for drone_id, state in all_states.items():
            if not state.get("active", False):
                continue

            current_pos = state["pos"]
            if drone_id not in self.goals:
                self.goals[drone_id] = None
            if drone_id not in self.paths:
                self.paths[drone_id] = []
            if drone_id not in self.wait_counters:
                self.wait_counters[drone_id] = 0

            if self.mode == "random":
                direction = plan_random_walk(current_pos, self.env)

            elif self.mode == "frontier":
                direction, new_goal, new_path, new_wait_counter = plan_frontier(
                    drone_id, current_pos, self.goals[drone_id], self.paths[drone_id],
                    self.frontiers, assigned_goals, all_states,
                    self.global_map, self.wait_counters[drone_id], self.max_wait, self.env
                )
                self.goals[drone_id] = new_goal
                self.paths[drone_id] = new_path
                self.wait_counters[drone_id] = new_wait_counter

            else:
                raise ValueError("Unknown mode")

            if direction not in DIRECTIONS:
                print(f"[ERROR] Invalid direction sent: {direction} for Drone {drone_id}")
                direction = 'STAY'

            self.comm.send_instruction(drone_id, direction)

    def _update_frontiers(self):
        self.frontiers = set()
        for y in range(self.env.grid.shape[0]):
            for x in range(self.env.grid.shape[1]):
                if self.global_map[y, x] == -1:
                    continue
                if self.global_map[y, x] in {1, 3, 6}:  # WALL, DOOR_CLOSED, OUT_OF_BOUNDS
                    continue
                for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < self.env.width and 0 <= ny < self.env.height:
                        if self.global_map[ny, nx] == -1 and self.discoverable_mask[ny, nx]:
                            self.frontiers.add((x, y))
                            break
