"""
Frontier SLAM Agent - Intelligent frontier-based exploration

This agent implements a frontier-based exploration strategy where drones
are assigned to explore the boundaries between known and unknown areas.
It maintains a global map and coordinates multiple drones efficiently.
"""

import numpy as np
import random
from typing import Dict, List, Tuple, Optional, Set, Any
from .base_slam_agent import BaseSLAMAgent

# Import constants and utilities
try:
    from planner.simulation.simulation_constants import (
        WALL, DOOR_CLOSED, OUT_OF_BOUNDS, FREE_SPACE, ENTRY_POINT,
        FACING_DIRECTIONS, FACING_TO_DELTA, DIRECTION_COMMANDS
    )
    from planner.algorithm.naive_planner import a_star
except ImportError:
    # Define minimal constants if imports fail
    WALL = 1
    DOOR_CLOSED = 3
    OUT_OF_BOUNDS = 6
    FREE_SPACE = 0
    ENTRY_POINT = 2
    FACING_DIRECTIONS = ['NORTH', 'EAST', 'SOUTH', 'WEST']
    FACING_TO_DELTA = {
        'NORTH': (0, -1),
        'EAST': (1, 0),
        'SOUTH': (0, 1),
        'WEST': (-1, 0),
    }
    DIRECTION_COMMANDS = ['FORWARD', 'TURN_LEFT', 'TURN_RIGHT', 'STAY']


class FrontierAgent(BaseSLAMAgent):
    """
    Frontier-based exploration agent.

    This agent maintains a global map and assigns drones to explore frontiers
    (boundaries between known and unknown areas). It uses A* pathfinding to
    navigate efficiently and coordinates multiple drones to avoid redundant
    exploration.
    """

    def __init__(self, num_agents: int, camera_range: int = 10):
        """
        Initialize the frontier agent.

        Args:
            num_agents: Number of agents in the environment
            camera_range: Maximum sensing range of the drones
        """
        super().__init__(num_agents)
        self.camera_range = camera_range
        self.global_map = None
        self.frontiers = set()
        self.goals = {}
        self.paths = {}
        self.wait_counters = {}
        self.max_wait = 3
        self.assigned_goals = set()

    def reset(self):
        """Reset agent state for new episode."""
        self.global_map = None
        self.frontiers = set()
        self.goals = {}
        self.paths = {}
        self.wait_counters = {}
        self.assigned_goals = set()

    def get_actions(self, observations: Dict[int, Any], info: Dict[str, Any]) -> Dict[int, int]:
        """Get actions using frontier-based exploration strategy."""
        # Initialize global map if needed
        if self.global_map is None:
            first_obs = next(iter(observations.values()))
            map_shape = first_obs['local_map'].shape
            self.global_map = np.full(map_shape, -1, dtype=np.int8)

        # Update global map from all drone observations
        all_drone_states = info.get('all_drone_states', {})
        for drone_id, state in all_drone_states.items():
            for x, y, val in state.get("new_discoveries", []):
                if 0 <= y < self.global_map.shape[0] and 0 <= x < self.global_map.shape[1]:
                    if self.global_map[y, x] == -1:
                        self.global_map[y, x] = val

        # Update frontiers
        self._update_frontiers(info.get('reachable_mask', np.ones_like(self.global_map, dtype=bool)))

        actions = {}

        for agent_id, obs in observations.items():
            if not obs['active']:
                actions[agent_id] = 3  # STAY
                continue

            # Get drone state
            current_pos = tuple(obs['position'])
            facing_idx = obs['facing_direction']
            facing = FACING_DIRECTIONS[facing_idx]

            # Initialize agent tracking if needed
            if agent_id not in self.goals:
                self.goals[agent_id] = None
            if agent_id not in self.paths:
                self.paths[agent_id] = []
            if agent_id not in self.wait_counters:
                self.wait_counters[agent_id] = 0

            # Plan action
            action = self._plan_frontier_action(
                agent_id, current_pos, facing, obs,
                all_drone_states, info
            )

            # Convert action string to index
            action_idx = DIRECTION_COMMANDS.index(action)
            actions[agent_id] = action_idx

        return actions

    def _update_frontiers(self, reachable_mask: np.ndarray):
        """Update the set of frontier cells."""
        self.frontiers = set()
        height, width = self.global_map.shape

        for y in range(height):
            for x in range(width):
                if self.global_map[y, x] == -1:
                    continue
                if self.global_map[y, x] in {WALL, DOOR_CLOSED, OUT_OF_BOUNDS}:
                    continue

                # Check if this cell borders unknown reachable area
                for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < width and 0 <= ny < height:
                        if self.global_map[ny, nx] == -1 and reachable_mask[ny, nx]:
                            self.frontiers.add((x, y))
                            break

    def _plan_frontier_action(
            self, agent_id: int, current_pos: Tuple[int, int],
            facing: str, obs: Dict[str, Any],
            all_drone_states: Dict[int, Any], info: Dict[str, Any]
    ) -> str:
        """Plan action for a single agent using frontier strategy."""
        # First check if we can discover new cells by rotating
        for direction in FACING_DIRECTIONS:
            ddx, ddy = FACING_TO_DELTA[direction]
            for step in range(1, self.camera_range + 1):
                x = current_pos[0] + ddx * step
                y = current_pos[1] + ddy * step

                if not (0 <= x < self.global_map.shape[1] and 0 <= y < self.global_map.shape[0]):
                    break

                val = self.global_map[y, x]
                if val in {WALL, DOOR_CLOSED, OUT_OF_BOUNDS}:
                    break

                if val == -1:  # Unexplored
                    if direction != facing:
                        # Turn toward that direction
                        if self._turn_direction(facing, 'TURN_LEFT') == direction:
                            return 'TURN_LEFT'
                        elif self._turn_direction(facing, 'TURN_RIGHT') == direction:
                            return 'TURN_RIGHT'
                        else:
                            return 'TURN_RIGHT'
                    else:
                        return 'STAY'

        # Check if current goal is still valid
        goal = self.goals[agent_id]
        path = self.paths[agent_id]

        goal_still_valid = (goal and path and
                            (goal in self.frontiers or
                             abs(goal[0] - current_pos[0]) + abs(goal[1] - current_pos[1]) <= 2))

        # Assign new goal if needed
        if not goal_still_valid:
            new_goal, new_path = self._assign_new_goal(agent_id, current_pos)
            self.goals[agent_id] = new_goal
            self.paths[agent_id] = new_path

            if new_goal is None:
                # No frontiers available, random walk
                return self._random_walk_action(current_pos, facing)

        # Follow path to goal
        return self._follow_path(agent_id, current_pos, facing, all_drone_states)

    def _assign_new_goal(
            self, agent_id: int, current_pos: Tuple[int, int]
    ) -> Tuple[Optional[Tuple[int, int]], List[Tuple[int, int]]]:
        """Assign a new frontier goal to the agent."""
        # Remove old goal from assigned set
        old_goal = self.goals.get(agent_id)
        if old_goal and old_goal in self.assigned_goals:
            self.assigned_goals.remove(old_goal)

        # Find available frontiers
        available_frontiers = [f for f in self.frontiers if f not in self.assigned_goals]

        if not available_frontiers:
            return None, []

        # Find closest frontier with valid path
        best_goal = None
        best_path = []
        min_dist = float('inf')

        for frontier in available_frontiers:
            dist = abs(frontier[0] - current_pos[0]) + abs(frontier[1] - current_pos[1])
            if dist < min_dist:
                path = a_star(current_pos, frontier, self.global_map)
                if path:
                    best_goal = frontier
                    best_path = path
                    min_dist = dist

        if best_goal:
            self.assigned_goals.add(best_goal)

        return best_goal, best_path

    def _follow_path(
            self, agent_id: int, current_pos: Tuple[int, int],
            facing: str, all_drone_states: Dict[int, Any]
    ) -> str:
        """Follow the planned path."""
        path = self.paths[agent_id]

        if not path:
            return self._random_walk_action(current_pos, facing)

        next_pos = path[0]

        # Check if next position is blocked by another drone
        blocked = False
        for other_id, state in all_drone_states.items():
            if other_id != agent_id and state.get('pos') == next_pos:
                blocked = True
                break

        if blocked:
            self.wait_counters[agent_id] += 1
            if self.wait_counters[agent_id] >= self.max_wait:
                # Waited too long, try random action
                self.wait_counters[agent_id] = 0
                return self._random_walk_action(current_pos, facing)
            else:
                return 'STAY'

        # Reset wait counter
        self.wait_counters[agent_id] = 0

        # Determine direction to next position
        dx = next_pos[0] - current_pos[0]
        dy = next_pos[1] - current_pos[1]

        # Get current facing delta
        fdx, fdy = FACING_TO_DELTA[facing]

        if (dx, dy) == (fdx, fdy):
            # Already facing the right direction
            self.paths[agent_id].pop(0)
            return 'FORWARD'

        # Need to turn
        for action in ['TURN_LEFT', 'TURN_RIGHT']:
            new_facing = self._turn_direction(facing, action)
            new_fdx, new_fdy = FACING_TO_DELTA[new_facing]
            if (new_fdx, new_fdy) == (dx, dy):
                return action

        # If we can't turn to face the target in one move, just turn right
        return 'TURN_RIGHT'

    def _random_walk_action(self, current_pos: Tuple[int, int], facing: str) -> str:
        """Generate a random walk action."""
        if random.random() < 0.7:
            return 'FORWARD'
        else:
            return random.choice(['TURN_LEFT', 'TURN_RIGHT', 'STAY'])

    def _turn_direction(self, facing: str, action: str) -> str:
        """Calculate new facing after a turn action."""
        idx = FACING_DIRECTIONS.index(facing)
        if action == 'TURN_LEFT':
            return FACING_DIRECTIONS[(idx - 1) % 4]
        elif action == 'TURN_RIGHT':
            return FACING_DIRECTIONS[(idx + 1) % 4]
        return facing