"""
Doorway Traversal Agent that extends the navigation agent.

This agent reuses the A* navigation logic and adds doorway-specific behavior.

State tracking for agent execution status
"""

import numpy as np
from typing import Dict, Any, Tuple, List, Optional
from enum import IntEnum

# Import the simple navigation agent we created earlier
from src.agents.a_star_navigation_agent import AStarNavigationAgent
from src.agents.base_agent import AgentState
from src.environments.base.constants import Action, TileType


class DoorwayState(IntEnum):
    """Simple state for doorway traversal."""
    FINDING = 0
    APPROACHING = 1
    ENTERING = 2
    COMPLETE = 3


class DoorwayEntryAgent(AStarNavigationAgent):
    """
    Doorway traversal agent that extends the navigation agent.

    Strategy:
    1. Find visible doorways
    2. Navigate to nearest doorway using A*
    3. Step through the doorway
    4. Complete after first doorway traversal

    Tracks execution state for coordination with other agents
    """

    def __init__(self, num_agents: int = 1):
        super().__init__(num_agents)
        self.visited_doorways = set()
        self.current_target = None
        self.state = DoorwayState.FINDING
        self.entry_position = None
        self.no_doorway_counter = 0  # Counter for when no doorways are found
        self.max_no_doorway_steps = 50  # Steps to explore when no doorways found

    def get_actions(self, observations: Dict[str, Any], info: Dict[str, Any]) -> np.ndarray:
        """Main action selection with doorway logic."""
        try:
            # Update execution state when first called
            if self.execution_state == AgentState.NOT_YET_STARTED:
                self.execution_state = AgentState.IN_PROGRESS

            if self.num_agents != 1:
                self.set_error("This agent only supports single agent navigation")
                raise ValueError("This agent only supports single agent navigation")

            # Check if already completed
            if self.state == DoorwayState.COMPLETE:
                self.execution_state = AgentState.COMPLETED
                return np.array([Action.STAY])

            # Extract state
            map_grid = observations['global_map']
            pos = tuple(observations['positions'][0])
            facing = observations['facings'][0]

            # State machine for doorway traversal
            if self.state == DoorwayState.FINDING:
                # Find nearest unvisited doorway
                doorway = self._find_nearest_doorway(pos, map_grid)
                if doorway:
                    self.current_target = doorway
                    self.state = DoorwayState.APPROACHING
                    self.no_doorway_counter = 0
                    # Set goal for parent navigation agent
                    observations['goal_position'] = np.array(doorway)
                else:
                    # No doorways found, explore
                    self.no_doorway_counter += 1
                    if self.no_doorway_counter >= self.max_no_doorway_steps:
                        # No doorways to find, task complete
                        self.execution_state = AgentState.COMPLETED
                        self.state = DoorwayState.COMPLETE
                        return np.array([Action.STAY])
                    return np.array([self._explore(pos, facing, map_grid)])

            elif self.state == DoorwayState.APPROACHING:
                # Check if reached doorway
                if pos == self.current_target:
                    self.entry_position = pos
                    self.state = DoorwayState.ENTERING
                    # Move forward through doorway
                    return np.array([Action.FORWARD])
                # Use parent's A* navigation
                observations['goal_position'] = np.array(self.current_target)

            elif self.state == DoorwayState.ENTERING:
                # Check if we've moved through
                if pos != self.entry_position:
                    # Successfully passed through first doorway - COMPLETE!
                    self.visited_doorways.add(self.current_target)
                    self.current_target = None
                    self.state = DoorwayState.COMPLETE
                    self.execution_state = AgentState.COMPLETED
                    return np.array([Action.STAY])
                # Still in the doorway, continue moving forward
                return np.array([Action.FORWARD])

            # Use parent's navigation if we have a target
            if self.state == DoorwayState.APPROACHING:
                return super().get_actions(observations, info)

            return np.array([Action.STAY])

        except Exception as e:
            self.set_error(str(e))
            return np.array([Action.STAY])

    def _find_nearest_doorway(self, pos: Tuple[int, int],
                              map_grid: np.ndarray) -> Optional[Tuple[int, int]]:
        """
        Find the nearest unvisited doorway.
        A doorway is identified by tile type 3 (door tile).
        """
        height, width = map_grid.shape
        min_dist = float('inf')
        nearest = None

        for y in range(1, height - 1):
            for x in range(1, width - 1):
                # Skip if already visited
                if (x, y) in self.visited_doorways:
                    continue

                # Check if this is a door tile (tile type 3)
                if map_grid[y, x] == 3:  # Door tile from your tile registry
                    dist = abs(pos[0] - x) + abs(pos[1] - y)
                    if dist < min_dist:
                        min_dist = dist
                        nearest = (x, y)

        return nearest

    def _is_doorway(self, x: int, y: int, map_grid: np.ndarray) -> bool:
        """
        Check if a position is a doorway.
        Doorway = free space with walls on opposite sides.
        """
        # Get adjacent tiles
        left = map_grid[y, x - 1]
        right = map_grid[y, x + 1]
        top = map_grid[y - 1, x]
        bottom = map_grid[y + 1, x]

        # Horizontal doorway (walls left/right)
        if left == TileType.WALL and right == TileType.WALL:
            # At least one side should be passable
            if self._is_passable(top) or self._is_passable(bottom):
                return True

        # Vertical doorway (walls top/bottom)
        if top == TileType.WALL and bottom == TileType.WALL:
            # At least one side should be passable
            if self._is_passable(left) or self._is_passable(right):
                return True

        return False

    def reset(self) -> None:
        """Reset agent state."""
        super().reset()  # Reset execution state and parent state
        self.visited_doorways = set()
        self.current_target = None
        self.state = DoorwayState.FINDING
        self.entry_position = None
        self.no_doorway_counter = 0

    def get_metrics(self) -> Dict[str, Any]:
        """Get agent metrics."""
        metrics = super().get_metrics()  # Get base metrics including execution state
        metrics.update({
            'doorway_state': self.state.name,
            'doorways_visited': len(self.visited_doorways),
            'current_target': self.current_target,
            'no_doorway_counter': self.no_doorway_counter
        })
        return metrics