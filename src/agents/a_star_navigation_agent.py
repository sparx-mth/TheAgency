"""
A* Navigation Agent for SLAM environments

This agent uses A* pathfinding to navigate to goal positions, treating unknown
cells as passable until proven otherwise.

ADDED: State tracking for agent execution status
"""

import numpy as np
from typing import Dict, Any, Tuple, List, Optional
from heapq import heappush, heappop

from src.agents.base_agent import BaseSLAMAgent, AgentState
from src.environments.base.constants import Action, TileType


class AStarNavigationAgent(BaseSLAMAgent):
    """
    Simplified A* pathfinding agent that navigates to goals efficiently.

    Key principles:
    - Minimalist design with clear logic flow
    - Treats unknown cells as passable (optimistic planning)
    - Replans only when necessary (path blocked or no path)
    - Tracks execution state for coordination with other agents
    """

    def __init__(self, num_agents: int = 1):
        super().__init__(num_agents)
        self.path = []
        self.goal = None
        self.steps_taken = 0
        self.max_steps = 1000  # Maximum steps before considering task complete

    def get_actions(self, observations: Dict[str, Any], info: Dict[str, Any]) -> np.ndarray:
        """Get navigation actions using A* pathfinding."""
        try:
            # Update state to in progress when first called
            if self.execution_state == AgentState.NOT_YET_STARTED:
                self.execution_state = AgentState.IN_PROGRESS

            # Single agent only for simplicity
            if self.num_agents != 1:
                self.set_error("This agent only supports single agent navigation")
                raise ValueError("This agent only supports single agent navigation")

            # Extract current state
            map_grid = observations['global_map']
            pos = tuple(observations['positions'][0])
            facing = observations['facings'][0]
            goal = observations.get('goal_position', None)

            # Update step counter
            self.steps_taken += 1

            # Check for completion conditions
            if self.steps_taken >= self.max_steps:
                self.execution_state = AgentState.COMPLETED
                return np.array([Action.STAY])

            # Check for valid goal
            if goal is None or np.array_equal(goal, [-1, -1]):
                # No goal means exploration mode
                action = self._explore(pos, facing, map_grid)
                return np.array([action])

            self.goal = tuple(goal)

            # Check if at goal
            if pos == self.goal:
                self.execution_state = AgentState.COMPLETED
                return np.array([Action.STAY])

            # Check if we need a new path
            if not self._is_path_valid(pos, map_grid):
                self.path = self._find_path(pos, self.goal, map_grid)

            # Follow path or explore if no path exists
            if self.path:
                action = self._follow_path(pos, facing)
            else:
                action = self._explore(pos, facing, map_grid)

            return np.array([action])

        except Exception as e:
            self.set_error(str(e))
            return np.array([Action.STAY])

    def _find_path(self, start: Tuple[int, int], goal: Tuple[int, int],
                   map_grid: np.ndarray) -> List[Tuple[int, int]]:
        """
        Find path using A* algorithm.
        Returns list of positions from start to goal (exclusive of start).
        """
        if start == goal:
            return []

        height, width = map_grid.shape

        # Priority queue: (f_score, counter, position)
        counter = 0
        open_set = [(0, counter, start)]
        came_from = {}
        g_score = {start: 0}
        closed = set()

        while open_set:
            _, _, current = heappop(open_set)

            if current in closed:
                continue
            closed.add(current)

            if current == goal:
                # Reconstruct path
                path = []
                while current in came_from:
                    path.append(current)
                    current = came_from[current]
                path.reverse()
                return path

            # Check neighbors
            for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                nx, ny = current[0] + dx, current[1] + dy

                # Check bounds
                if not (0 <= nx < width and 0 <= ny < height):
                    continue

                # Skip if already visited
                if (nx, ny) in closed:
                    continue

                # Check if passable (unknown cells are considered passable)
                if not self._is_passable(map_grid[ny, nx]):
                    continue

                # Calculate cost
                new_g = g_score[current] + 1

                if (nx, ny) not in g_score or new_g < g_score[(nx, ny)]:
                    came_from[(nx, ny)] = current
                    g_score[(nx, ny)] = new_g
                    f_score = new_g + abs(nx - goal[0]) + abs(ny - goal[1])
                    counter += 1
                    heappush(open_set, (f_score, counter, (nx, ny)))

        return []  # No path found

    def _is_passable(self, tile: int) -> bool:
        """Check if a tile can be traversed."""
        return tile in {TileType.UNKNOWN, TileType.FREE_SPACE,
                       TileType.ENTRY_POINT, TileType.DOOR_OPEN}

    def _is_path_valid(self, current_pos: Tuple[int, int],
                      map_grid: np.ndarray) -> bool:
        """Check if current path is still valid."""
        if not self.path:
            return False

        # Check if we're on the path
        if current_pos not in self.path:
            return False

        # Check if any future waypoint is blocked
        idx = self.path.index(current_pos) if current_pos in self.path else 0
        for pos in self.path[idx:]:
            x, y = pos
            if not self._is_passable(map_grid[y, x]):
                return False

        return True

    def _follow_path(self, current_pos: Tuple[int, int], facing: int) -> int:
        """Get action to follow the planned path."""
        # Find next waypoint
        if current_pos in self.path:
            idx = self.path.index(current_pos)
            if idx + 1 < len(self.path):
                next_pos = self.path[idx + 1]
            else:
                # Reached end of path but not at goal
                self.path = []
                return Action.FORWARD
        else:
            # Not on path, target first waypoint
            next_pos = self.path[0] if self.path else current_pos

        # Calculate direction to next position
        dx = next_pos[0] - current_pos[0]
        dy = next_pos[1] - current_pos[1]

        # Determine required facing (0=N, 1=E, 2=S, 3=W)
        target_facing = None
        if dy < 0:
            target_facing = 0  # North
        elif dx > 0:
            target_facing = 1  # East
        elif dy > 0:
            target_facing = 2  # South
        elif dx < 0:
            target_facing = 3  # West
        else:
            return Action.STAY

        # Turn or move
        if facing == target_facing:
            return Action.FORWARD

        # Calculate shortest turn
        turn_diff = (target_facing - facing) % 4
        if turn_diff == 1:
            return Action.TURN_RIGHT
        elif turn_diff == 3:
            return Action.TURN_LEFT
        else:
            # 180 degree turn needed
            return Action.TURN_RIGHT

    def _explore(self, pos: Tuple[int, int], facing: int,
                map_grid: np.ndarray) -> int:
        """Simple exploration when no goal or path available."""
        height, width = map_grid.shape
        x, y = pos

        # Check what's ahead
        facing_deltas = [(0, -1), (1, 0), (0, 1), (-1, 0)]  # N, E, S, W
        dx, dy = facing_deltas[facing]
        ahead_x, ahead_y = x + dx, y + dy

        # Go forward if possible
        if 0 <= ahead_x < width and 0 <= ahead_y < height:
            if self._is_passable(map_grid[ahead_y, ahead_x]):
                return Action.FORWARD

        # Otherwise turn randomly
        return np.random.choice([Action.TURN_LEFT, Action.TURN_RIGHT])

    def reset(self) -> None:
        """Reset agent state."""
        super().reset()  # Reset execution state
        self.path = []
        self.goal = None
        self.steps_taken = 0

    def get_metrics(self) -> Dict[str, Any]:
        """Get agent metrics."""
        metrics = super().get_metrics()  # Get base metrics including execution state
        metrics.update({
            'has_path': len(self.path) > 0,
            'path_length': len(self.path),
            'goal': self.goal,
            'steps_taken': self.steps_taken,
            'max_steps': self.max_steps
        })
        return metrics