"""
agents/frontier_agent.py

This file implements a frontier-based exploration agent for SLAM. The frontier
agent uses the concept of frontiers (boundaries between known and unknown areas)
to systematically explore the environment.

This is a classic approach in robotic exploration that balances between:
- Exploring new areas (frontiers)
- Efficient path planning to reach frontiers
- Coordination between multiple agents to avoid redundant exploration
"""

import random
from typing import Any, Dict, Union, List, Tuple, Set, Optional
from heapq import heappush, heappop
import numpy as np

from .base_agent import BaseSLAMAgent
from core.constants import Action, TileType, DIRECTIONS, DIRECTION_DELTAS


class FrontierAgent(BaseSLAMAgent):
    """
    Frontier-based exploration agent with sensor-aware planning.

    This agent first checks if unexplored cells can be discovered by rotating
    in place using the drone's directional field-of-view, and only then assigns
    the drone to explore the closest unexplored frontier using A* pathfinding.

    For multi-agent scenarios, it coordinates frontier assignments to minimize
    overlap and redundant exploration.

    Attributes:
        goals: Current goal positions for each agent
        paths: Planned paths to goals for each agent
        assigned_frontiers: Set of frontiers currently assigned to agents
        stuck_counters: Counters to detect when agents are stuck
        last_positions: Previous positions to detect lack of progress
        wait_counters: Counters for waiting when path is blocked
        max_wait: Maximum wait time before replanning
        camera_range: Range of the camera sensor (default 10)
    """

    def __init__(self, num_agents: int = 1, camera_range: int = 10):
        """
        Initialize the frontier agent.

        Args:
            num_agents: Number of agents to control
            camera_range: Range of camera sensor for exploration
        """
        super().__init__(num_agents)
        self.goals: Dict[int, Optional[Tuple[int, int]]] = {}
        self.paths: Dict[int, List[Tuple[int, int]]] = {}
        self.assigned_frontiers: Set[Tuple[int, int]] = set()
        self.stuck_counters: Dict[int, int] = {}
        self.last_positions: Dict[int, Tuple[int, int]] = {}
        self.wait_counters: Dict[int, int] = {}
        self.max_wait: int = 3
        self.camera_range: int = camera_range
        self.random_walk_counter: Dict[int, int] = {}

    def get_actions(
        self,
        observations: Union[Dict, Any],
        info: Dict[str, Any]
    ) -> Union[int, Dict[int, int]]:
        """
        Get actions using frontier-based strategy with sensor-aware exploration.

        Args:
            observations: Current observations from environment
            info: Additional environment information

        Returns:
            Actions targeting frontier exploration
        """
        if self.is_single_agent:
            return self._get_single_action(observations, info)
        else:
            return self._get_multi_actions(observations, info)

    def _get_single_action(self, obs: Dict, info: Dict) -> int:
        """Get action for single agent."""
        if not obs.get('active', 1):  # Default to active if not present
            return Action.STAY

        return self._compute_action(0, obs, self.assigned_frontiers, {})

    def _get_multi_actions(self, observations: Dict, info: Dict) -> Dict[int, int]:
        """Get actions for multiple agents."""
        actions = {}

        # Build all_states dict for coordination
        all_states = {}
        for agent_id in range(self.num_agents):
            obs = observations[agent_id]
            all_states[agent_id] = {
                'pos': tuple(obs['position']),
                'facing': obs['facing'],
                'active': obs.get('active', True)
            }

        # Don't clear assigned frontiers here - manage them individually per agent

        for agent_id in range(self.num_agents):
            obs = observations[agent_id]

            if not obs.get('active', True):
                actions[agent_id] = Action.STAY
            else:
                actions[agent_id] = self._compute_action(
                    agent_id, obs, self.assigned_frontiers, all_states
                )

        return actions

    def _compute_action(
        self,
        agent_id: int,
        obs: Dict,
        assigned: Set[Tuple[int, int]],
        all_states: Dict[int, Dict[str, Any]]
    ) -> int:
        """
        Compute action for a single agent using sensor-aware frontier exploration.

        Args:
            agent_id: ID of the agent
            obs: Observation dictionary
            assigned: Set of already assigned frontiers
            all_states: States of all agents for coordination

        Returns:
            Action integer
        """
        global_map = obs['global_map']
        pos = tuple(obs['position'])
        facing_idx = obs['facing']

        # Initialize counters if needed
        if agent_id not in self.wait_counters:
            self.wait_counters[agent_id] = 0
        if agent_id not in self.random_walk_counter:
            self.random_walk_counter[agent_id] = 0

        # Check if stuck
        if agent_id in self.last_positions:
            if self.last_positions[agent_id] == pos:
                self.stuck_counters[agent_id] = self.stuck_counters.get(agent_id, 0) + 1
            else:
                self.stuck_counters[agent_id] = 0
        else:
            self.stuck_counters[agent_id] = 0

        self.last_positions[agent_id] = pos

        # If stuck for too long, reset everything and do random walk
        if self.stuck_counters.get(agent_id, 0) > 5:
            # Remove old goal from assigned set
            old_goal = self.goals.get(agent_id)
            if old_goal and old_goal in assigned:
                assigned.discard(old_goal)

            self.goals[agent_id] = None
            self.paths[agent_id] = []
            self.stuck_counters[agent_id] = 0
            self.wait_counters[agent_id] = 0
            self.random_walk_counter[agent_id] = 10  # Do random walk for next 10 steps
            return random.choice([Action.FORWARD, Action.TURN_LEFT, Action.TURN_RIGHT])

        # If in random walk mode, continue random walking
        if self.random_walk_counter.get(agent_id, 0) > 0:
            self.random_walk_counter[agent_id] -= 1
            return self._random_walk_action(pos, facing_idx, global_map)

        # SENSOR-AWARE EXPLORATION: Check if we can discover unexplored cells by rotating
        action = self._check_sensor_exploration(pos, facing_idx, global_map)
        if action is not None:
            return action

        # Find ALL frontiers (not just unassigned ones initially)
        all_frontiers = self._find_frontiers(global_map)

        # Check if current goal is still valid (exists and is reachable)
        current_goal = self.goals.get(agent_id)
        goal_still_valid = False

        if current_goal:
            # Check if goal is still a frontier or we're very close to it
            if current_goal in all_frontiers or self._manhattan_distance(current_goal, pos) <= 2:
                # Check if we still have a valid path
                if self.paths.get(agent_id):
                    goal_still_valid = True
                else:
                    # Try to replan path to same goal
                    new_path = self._plan_path(pos, current_goal, global_map)
                    if new_path:
                        self.paths[agent_id] = new_path
                        goal_still_valid = True

        # If goal is no longer valid, find a new one
        if not goal_still_valid:
            # Remove old goal from assigned set
            if current_goal and current_goal in assigned:
                assigned.discard(current_goal)

            # Filter out assigned frontiers for selection
            available_frontiers = [f for f in all_frontiers if f not in assigned or f == current_goal]

            if available_frontiers:
                best_goal, best_path = self._select_best_frontier(
                    pos, available_frontiers, global_map, agent_id, all_states
                )

                if best_goal:
                    self.goals[agent_id] = best_goal
                    self.paths[agent_id] = best_path
                    assigned.add(best_goal)
                    self.wait_counters[agent_id] = 0
                else:
                    # No reachable frontier found
                    self.goals[agent_id] = None
                    self.paths[agent_id] = []
                    return self._random_walk_action(pos, facing_idx, global_map)
            else:
                # No frontiers available at all
                self.goals[agent_id] = None
                self.paths[agent_id] = []
                return self._random_walk_action(pos, facing_idx, global_map)

        # Follow path if we have one
        if self.paths.get(agent_id):
            return self._follow_path_with_collision_avoidance(
                agent_id, pos, facing_idx, all_states, global_map
            )

        # No path available, explore randomly
        return self._random_walk_action(pos, facing_idx, global_map)

    def _random_walk_action(
        self,
        pos: Tuple[int, int],
        facing_idx: int,
        global_map: np.ndarray
    ) -> int:
        """
        Perform random walk exploration.

        Args:
            pos: Current position
            facing_idx: Current facing direction
            global_map: The current map

        Returns:
            Random exploration action
        """
        # 25% chance to turn
        if random.random() < 0.25:
            return random.choice([Action.TURN_LEFT, Action.TURN_RIGHT])

        # Try to move forward
        dx, dy = [(0, -1), (1, 0), (0, 1), (-1, 0)][facing_idx]
        new_x, new_y = pos[0] + dx, pos[1] + dy

        height, width = global_map.shape
        if 0 <= new_x < width and 0 <= new_y < height:
            tile = global_map[new_y, new_x]
            if tile not in {TileType.WALL, TileType.DOOR_CLOSED, TileType.OUT_OF_BOUNDS}:
                return Action.FORWARD

        # Can't move forward, turn instead
        return random.choice([Action.TURN_LEFT, Action.TURN_RIGHT])

    def _check_sensor_exploration(
        self,
        pos: Tuple[int, int],
        facing_idx: int,
        global_map: np.ndarray
    ) -> Optional[int]:
        """
        Check if unexplored cells can be discovered by rotating in place.

        This implements the sensor-aware exploration from the old algorithm:
        First checks all four directions to see if any unexplored cells (-1/UNKNOWN)
        are within camera range. If found, either turns toward them or stays to sense.

        Args:
            pos: Current position
            facing_idx: Current facing direction index
            global_map: The current map

        Returns:
            Action to take for sensor exploration, or None if no unexplored cells nearby
        """
        height, width = global_map.shape

        # Check all four directions for unexplored cells
        for dir_idx, (dx, dy) in enumerate([(0, -1), (1, 0), (0, 1), (-1, 0)]):
            # Scan in this direction up to camera range
            for step in range(1, self.camera_range + 1):
                x = pos[0] + dx * step
                y = pos[1] + dy * step

                # Check bounds
                if not (0 <= x < width and 0 <= y < height):
                    break

                # Stop scanning if we hit an obstacle
                val = global_map[y, x]
                if val in {TileType.WALL, TileType.DOOR_CLOSED, TileType.OUT_OF_BOUNDS}:
                    break

                # Found unexplored cell
                if val == TileType.UNKNOWN:
                    # If not facing this direction, turn toward it
                    if dir_idx != facing_idx:
                        # Calculate turn direction
                        diff = (dir_idx - facing_idx) % 4
                        if diff == 1:
                            return Action.TURN_RIGHT
                        elif diff == 3:
                            return Action.TURN_LEFT
                        else:
                            # Need 180 turn, turn right
                            return Action.TURN_RIGHT
                    else:
                        # Already facing unexplored area, stay to sense
                        return Action.STAY

        return None

    def _select_best_frontier(
        self,
        pos: Tuple[int, int],
        available_frontiers: List[Tuple[int, int]],
        global_map: np.ndarray,
        agent_id: int,
        all_states: Dict[int, Dict[str, Any]]
    ) -> Tuple[Optional[Tuple[int, int]], List[Tuple[int, int]]]:
        """
        Select the best frontier considering distance and spacing from other agents.

        Args:
            pos: Current position
            available_frontiers: List of available frontier positions
            global_map: The current map
            agent_id: ID of current agent
            all_states: States of all agents

        Returns:
            Tuple of (best frontier position, path to it) or (None, [])
        """
        # Try to find paths to all frontiers and sort by distance
        reachable_frontiers = []

        for frontier in available_frontiers:
            path = self._plan_path(pos, frontier, global_map)
            if path:
                # Use path length as distance (more accurate than Manhattan)
                reachable_frontiers.append((len(path), frontier, path))

        if not reachable_frontiers:
            return None, []

        # Sort by distance
        reachable_frontiers.sort(key=lambda x: x[0])

        # If single agent or no other agents, just pick closest
        if not all_states or len(all_states) <= 1:
            _, best_frontier, best_path = reachable_frontiers[0]
            return best_frontier, best_path

        # For multi-agent, consider spacing for the closest few frontiers
        candidates = reachable_frontiers[:min(5, len(reachable_frontiers))]

        best_score = float('-inf')
        best_frontier = None
        best_path = []

        for dist, frontier, path in candidates:
            # Calculate spacing from other agents
            min_spacing = float('inf')
            for other_id, other in all_states.items():
                if other_id != agent_id and other.get('active', False):
                    other_pos = other.get('pos', (0, 0))
                    spacing = self._manhattan_distance(frontier, other_pos)
                    min_spacing = min(min_spacing, spacing)

            # Score combines distance (lower is better) and spacing (higher is better)
            # Normalize distance to [0, 1] range
            max_dist = max(d for d, _, _ in candidates)
            normalized_dist = dist / max_dist if max_dist > 0 else 0

            # Score: prefer closer frontiers but with good spacing
            score = -normalized_dist + 0.5 * (min_spacing / 10.0)  # Adjust weight as needed

            if score > best_score:
                best_score = score
                best_frontier = frontier
                best_path = path

        return best_frontier, best_path

    def _follow_path_with_collision_avoidance(
        self,
        agent_id: int,
        pos: Tuple[int, int],
        facing_idx: int,
        all_states: Dict[int, Dict[str, Any]],
        global_map: np.ndarray
    ) -> int:
        """
        Follow the planned path with collision avoidance for other agents.

        Args:
            agent_id: ID of the agent
            pos: Current position
            facing_idx: Current facing direction index
            all_states: States of all agents
            global_map: The current map

        Returns:
            Action to take
        """
        path = self.paths.get(agent_id, [])
        if not path:
            return Action.STAY

        # Clean up path - remove current position if we're on it
        while path and pos == path[0]:
            path.pop(0)

        if not path:
            # Reached goal
            self.goals[agent_id] = None
            return Action.STAY

        next_pos = path[0]

        # Check if another agent is blocking the next position
        blocked = False
        if all_states:
            for other_id, other in all_states.items():
                if other_id != agent_id and other.get('active', False):
                    if other.get('pos') == next_pos:
                        blocked = True
                        break

        if blocked:
            self.wait_counters[agent_id] += 1
            if self.wait_counters[agent_id] >= self.max_wait:
                # Waited too long, try to replan
                goal = self.goals.get(agent_id)
                if goal:
                    new_path = self._plan_path(pos, goal, global_map)
                    if new_path:
                        self.paths[agent_id] = new_path
                        self.wait_counters[agent_id] = 0
                        return self._follow_path_with_collision_avoidance(
                            agent_id, pos, facing_idx, all_states, global_map
                        )

                # Can't replan, reset goal
                self.goals[agent_id] = None
                self.paths[agent_id] = []
                self.wait_counters[agent_id] = 0
                return self._random_walk_action(pos, facing_idx, global_map)
            else:
                return Action.STAY

        # Reset wait counter if not blocked
        self.wait_counters[agent_id] = 0

        # Determine direction to next position
        dx = next_pos[0] - pos[0]
        dy = next_pos[1] - pos[1]

        # Determine target direction
        target_dir = None
        if dy < 0:
            target_dir = 0  # NORTH
        elif dx > 0:
            target_dir = 1  # EAST
        elif dy > 0:
            target_dir = 2  # SOUTH
        elif dx < 0:
            target_dir = 3  # WEST
        else:
            return Action.STAY

        # If already facing target, move forward
        if facing_idx == target_dir:
            return Action.FORWARD

        # Calculate turn direction
        diff = (target_dir - facing_idx) % 4
        if diff == 1:
            return Action.TURN_RIGHT
        elif diff == 3:
            return Action.TURN_LEFT
        else:
            # Need 180 turn, choose randomly
            return random.choice([Action.TURN_LEFT, Action.TURN_RIGHT])

    def _find_frontiers(self, global_map: np.ndarray) -> List[Tuple[int, int]]:
        """
        Find frontier cells (known free cells adjacent to unknown cells).

        Args:
            global_map: The current map

        Returns:
            List of frontier positions
        """
        frontiers = []
        height, width = global_map.shape

        for y in range(height):
            for x in range(width):
                # Must be a known free space
                if global_map[y, x] == TileType.UNKNOWN:
                    continue
                if global_map[y, x] in {TileType.WALL, TileType.DOOR_CLOSED, TileType.OUT_OF_BOUNDS}:
                    continue

                # Check if adjacent to unknown
                is_frontier = False
                for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < width and 0 <= ny < height:
                        if global_map[ny, nx] == TileType.UNKNOWN:
                            is_frontier = True
                            break

                if is_frontier:
                    frontiers.append((x, y))

        return frontiers

    def _plan_path(
        self,
        start: Tuple[int, int],
        goal: Tuple[int, int],
        global_map: np.ndarray
    ) -> List[Tuple[int, int]]:
        """
        Plan a path from start to goal using A* search.

        Args:
            start: Starting position
            goal: Goal position
            global_map: The current map

        Returns:
            List of positions from start to goal (excluding start)
        """
        if start == goal:
            return []

        height, width = global_map.shape

        # A* search
        open_set = [(0, start)]
        came_from = {}
        g_score = {start: 0}
        f_score = {start: self._manhattan_distance(start, goal)}
        closed_set = set()

        while open_set:
            current = heappop(open_set)[1]

            if current in closed_set:
                continue
            closed_set.add(current)

            if current == goal:
                # Reconstruct path
                path = []
                while current in came_from:
                    path.append(current)
                    current = came_from[current]
                path.reverse()
                return path

            for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                neighbor = (current[0] + dx, current[1] + dy)

                # Check bounds
                if not (0 <= neighbor[0] < width and 0 <= neighbor[1] < height):
                    continue

                # Check if traversable (unknown cells are not traversable for planning)
                tile = global_map[neighbor[1], neighbor[0]]
                if tile == TileType.UNKNOWN:
                    continue
                if tile in {TileType.WALL, TileType.DOOR_CLOSED, TileType.OUT_OF_BOUNDS}:
                    continue

                if neighbor in closed_set:
                    continue

                tentative_g = g_score[current] + 1

                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f_score[neighbor] = tentative_g + self._manhattan_distance(neighbor, goal)
                    heappush(open_set, (f_score[neighbor], neighbor))

        return []  # No path found

    def _manhattan_distance(self, p1: Tuple[int, int], p2: Tuple[int, int]) -> int:
        """Calculate Manhattan distance between two points."""
        return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])

    def reset(self) -> None:
        """Reset the agent's internal state."""
        self.goals.clear()
        self.paths.clear()
        self.assigned_frontiers.clear()
        self.stuck_counters.clear()
        self.last_positions.clear()
        self.wait_counters.clear()
        self.random_walk_counter.clear()

    def get_metrics(self) -> Dict[str, Any]:
        """Get agent metrics."""
        return {
            'num_assigned_goals': len([g for g in self.goals.values() if g is not None]),
            'num_stuck_agents': sum(1 for c in self.stuck_counters.values() if c > 0),
            'total_assigned_frontiers': len(self.assigned_frontiers),
            'num_waiting_agents': sum(1 for c in self.wait_counters.values() if c > 0),
            'num_random_walking': sum(1 for c in self.random_walk_counter.values() if c > 0),
        }