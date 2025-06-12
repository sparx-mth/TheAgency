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
from src.planner.simulation.simulation_constants import WALL, DOOR_CLOSED, DIRECTIONS, OUT_OF_BOUNDS, FACING_DIRECTION
from typing import Dict, Tuple, List, Set, Optional, TYPE_CHECKING, Literal
if TYPE_CHECKING:
    from src.planner.simulation.grid_map_env import GridMapEnv
    from src.planner.communication.comm_interface import CommunicationInterface


class MasterController:
    """
    Centralized SLAM controller that coordinates multiple drones in a grid environment.

    This class manages drone coordination, global map aggregation, and exploration planning.
    It supports both random-walk and frontier-based planning strategies and uses a shared
    communication interface to interact with drones.

    Attributes:
        env (GridMapEnv): The simulation environment.
        comm (CommunicationInterface): Interface for drone communication.
        discoverable_mask (np.ndarray): Mask indicating discoverable tiles.
        mode (str): Planning strategy ("random" or "frontier").
        global_map (np.ndarray): Shared map updated from drone observations.
        frontiers (Set[Tuple[int, int]]): Current frontier tiles to explore.
        goals (Dict[int, Optional[Tuple[int, int]]]): Current goals per drone.
        paths (Dict[int, List[Tuple[int, int]]]): Paths to goals per drone.
        wait_counters (Dict[int, int]): Consecutive wait times per drone.
        max_wait (int): Threshold for reassigning goals when drones are blocked.

    Use `step(current_time)` to update the system state at each simulation tick.
    """
    def __init__(
        self,
        env: "GridMapEnv",
        discoverable_mask: np.ndarray,
        comm_interface: "CommunicationInterface",
        mode: Literal["random", "frontier"] = "frontier"
    ):
        """
        Initialize the MasterController for coordinating multiple SLAM drones.

        Args:
            env (GridMapEnv): The simulation environment containing map data and drones.
            comm_interface (CommunicationInterface): Communication interface for sending/receiving data.
            discoverable_mask (np.ndarray): A boolean mask indicating which tiles can be discovered.
            mode (Literal['random', 'frontier'], optional):
                Planning strategy for drones.
                'random' uses random walk, 'frontier' uses a goal-based frontier strategy.
                Defaults to 'frontier'.
        """
        self.env: GridMapEnv = env
        self.comm: CommunicationInterface = comm_interface
        self.discoverable_mask: np.ndarray = discoverable_mask
        self.mode: Literal["random", "frontier"] = mode  # Planning strategy
        self.global_map: np.ndarray = np.full(self.env.grid.shape, -1, dtype=np.int8)
        self.frontiers: Set[Tuple[int, int]] = set()
        self.goals: Dict[int, Optional[Tuple[int, int]]] = {}
        self.paths: Dict[int, List[Tuple[int, int]]] = {}
        self.wait_counters: Dict[int, int] = {}
        self.max_wait: int = 3

    def step(self, current_time: int) -> None:
        """
        Perform one control step: collect drone discoveries, update global map, assign new actions.

        Args:
            current_time (int): The current simulation tick.
        """
        all_states = self.comm.get_all_drones_state()

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
            facing = state["facing_direction"]
            if drone_id not in self.goals:
                self.goals[drone_id] = None
            if drone_id not in self.paths:
                self.paths[drone_id] = []
            if drone_id not in self.wait_counters:
                self.wait_counters[drone_id] = 0

            if self.mode == "random":
                direction = plan_random_walk(
                    pos=current_pos,
                    facing=facing,
                    env=self.env
                )

            elif self.mode == "frontier":
                direction, new_goal, new_path, new_wait_counter = plan_frontier(
                    drone_id=drone_id,
                    current_pos=current_pos,
                    facing=facing,
                    goal=self.goals[drone_id],
                    path=self.paths[drone_id],
                    frontiers=self.frontiers,
                    assigned_goals=assigned_goals,
                    all_states=all_states,
                    global_map=self.global_map,
                    wait_counter=self.wait_counters[drone_id],
                    max_wait=self.max_wait,
                    env=self.env
                )
                self.goals[drone_id] = new_goal
                self.paths[drone_id] = new_path
                self.wait_counters[drone_id] = new_wait_counter

            else:
                raise ValueError("Unknown mode")

            self.comm.send_instruction(drone_id, direction)

    def _update_frontiers(self) -> None:
        """Scan the map for frontiers — known free cells that border unexplored, discoverable areas."""
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
