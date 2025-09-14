"""
Simple Wall Following Agent

This agent:
1. Finds the closest wall
2. Approaches until 1 space away
3. Walks along the wall to its end
4. Turns 180° and walks back to the other end
"""

import numpy as np
from typing import Dict, Any, Tuple, Optional
from enum import Enum

from agents.base_agent import BaseSLAMAgent
from environments.base.constants import TileType, Action


class State(Enum):
    """Simple states for wall following"""
    FIND_WALL = 1
    APPROACH_WALL = 2
    FOLLOW_WALL = 3
    TURN_AROUND = 4
    FOLLOW_BACK = 5
    DONE = 6


class WallFollowingAgent(BaseSLAMAgent):
    """Simple wall-following agent."""

    def __init__(self, num_agents: int = 1):
        super().__init__(num_agents)
        self.reset()

    def reset(self):
        """Reset agent state."""
        self.state = State.FIND_WALL
        self.target_wall = None
        self.turn_count = 0
        self.steps_along_wall = 0

    def get_actions(self, observations: Dict[str, Any], info: Dict[str, Any]) -> np.ndarray:
        """Get action based on current state."""
        global_map = observations['global_map']
        pos = tuple(observations['positions'][0])
        facing = observations['facings'][0]

        action = self._execute_state(global_map, pos, facing)
        return np.array([action], dtype=np.int32)

    def _execute_state(self, global_map, pos, facing) -> int:
        """Execute action based on current state."""
        if self.state == State.FIND_WALL:
            return self._find_closest_wall(global_map, pos, facing)

        elif self.state == State.APPROACH_WALL:
            return self._approach_wall(global_map, pos, facing)

        elif self.state == State.FOLLOW_WALL:
            return self._follow_wall_forward(global_map, pos, facing)

        elif self.state == State.TURN_AROUND:
            return self._execute_180_turn()

        elif self.state == State.FOLLOW_BACK:
            return self._follow_wall_back(global_map, pos, facing)

        else:  # DONE
            return Action.STAY

    def _find_closest_wall(self, global_map, pos, facing) -> int:
        """Find and target the closest wall."""
        walls = []
        h, w = global_map.shape

        # Search for visible walls
        for y in range(max(0, pos[1] - 15), min(h, pos[1] + 16)):
            for x in range(max(0, pos[0] - 15), min(w, pos[0] + 16)):
                if global_map[y, x] == TileType.WALL:
                    dist = abs(x - pos[0]) + abs(y - pos[1])
                    walls.append(((x, y), dist))

        if walls:
            # Select closest wall
            walls.sort(key=lambda w: w[1])
            self.target_wall = walls[0][0]
            self.state = State.APPROACH_WALL
            return self._approach_wall(global_map, pos, facing)

        # No wall found, explore
        if self._can_move_forward(global_map, pos, facing):
            return Action.FORWARD
        else:
            return Action.TURN_RIGHT

    def _approach_wall(self, global_map, pos, facing) -> int:
        """Move toward wall until 1 space away."""
        if not self.target_wall:
            self.state = State.FIND_WALL
            return Action.STAY

        dx = self.target_wall[0] - pos[0]
        dy = self.target_wall[1] - pos[1]
        dist = abs(dx) + abs(dy)

        # Check if we're 1 space away
        if dist == 1:
            # We're next to the wall - start following it
            self.state = State.FOLLOW_WALL
            self.steps_along_wall = 0

            # Determine initial direction to follow based on wall orientation
            # Try to move perpendicular to approach direction
            if abs(dx) > 0:  # Approached from east/west
                # Try to go north or south
                if self._can_move_to(global_map, pos, 0, -1):  # North
                    if facing != 0:
                        return self._turn_toward(facing, 0)
                    return Action.FORWARD
                elif self._can_move_to(global_map, pos, 0, 1):  # South
                    if facing != 2:
                        return self._turn_toward(facing, 2)
                    return Action.FORWARD
            else:  # Approached from north/south
                # Try to go east or west
                if self._can_move_to(global_map, pos, 1, 0):  # East
                    if facing != 1:
                        return self._turn_toward(facing, 1)
                    return Action.FORWARD
                elif self._can_move_to(global_map, pos, -1, 0):  # West
                    if facing != 3:
                        return self._turn_toward(facing, 3)
                    return Action.FORWARD

            return Action.STAY

        # Turn toward wall if needed
        desired_facing = self._get_direction_to_target(pos, self.target_wall)
        if facing != desired_facing:
            return self._turn_toward(facing, desired_facing)

        # Move forward if possible
        if self._can_move_forward(global_map, pos, facing):
            return Action.FORWARD
        else:
            return Action.TURN_RIGHT

    def _follow_wall_forward(self, global_map, pos, facing) -> int:
        """Walk along the wall until reaching its end."""
        # Check if we can continue forward
        if self._can_move_forward(global_map, pos, facing):
            # Check if wall is still adjacent
            if self._is_wall_adjacent(global_map, pos):
                self.steps_along_wall += 1
                return Action.FORWARD
            else:
                # Wall ended - start 180° turn
                self.state = State.TURN_AROUND
                self.turn_count = 0
                return self._execute_180_turn()
        else:
            # Hit obstacle - wall continues but we can't go further
            # Start 180° turn
            self.state = State.TURN_AROUND
            self.turn_count = 0
            return self._execute_180_turn()

    def _execute_180_turn(self) -> int:
        """Execute a proper 180° turn (2 x 90° turns)."""
        self.turn_count += 1

        if self.turn_count > 2:  # Completed 180° turn
            self.state = State.FOLLOW_BACK
            self.turn_count = 0
            self.steps_along_wall = 0
            return Action.FORWARD

        # Keep turning right (90° each time)
        return Action.TURN_RIGHT

    def _follow_wall_back(self, global_map, pos, facing) -> int:
        """Walk back along the wall to the other end."""
        # Just go straight until we can't
        if self._can_move_forward(global_map, pos, facing):
            self.steps_along_wall += 1
            # Stop if we've gone too far (safety check)
            if self.steps_along_wall > 100:
                self.state = State.DONE
                return Action.STAY
            return Action.FORWARD
        else:
            # Hit the other end of the wall
            self.state = State.DONE
            return Action.STAY

    # Helper methods
    def _can_move_forward(self, global_map, pos, facing) -> bool:
        """Check if forward movement is possible."""
        deltas = [(0, -1), (1, 0), (0, 1), (-1, 0)]  # N, E, S, W
        dx, dy = deltas[facing]

        new_x = pos[0] + dx
        new_y = pos[1] + dy

        if not (0 <= new_x < global_map.shape[1] and 0 <= new_y < global_map.shape[0]):
            return False

        tile = global_map[new_y, new_x]
        return tile in {TileType.UNKNOWN, TileType.FREE_SPACE,
                       TileType.ENTRY_POINT, TileType.DOOR_OPEN}

    def _can_move_to(self, global_map, pos, dx, dy) -> bool:
        """Check if we can move to a specific offset position."""
        new_x = pos[0] + dx
        new_y = pos[1] + dy

        if not (0 <= new_x < global_map.shape[1] and 0 <= new_y < global_map.shape[0]):
            return False

        tile = global_map[new_y, new_x]
        return tile in {TileType.UNKNOWN, TileType.FREE_SPACE,
                       TileType.ENTRY_POINT, TileType.DOOR_OPEN}

    def _get_direction_to_target(self, pos, target) -> int:
        """Get facing direction toward target."""
        dx = target[0] - pos[0]
        dy = target[1] - pos[1]

        if abs(dx) > abs(dy):
            return 1 if dx > 0 else 3  # East or West
        else:
            return 2 if dy > 0 else 0  # South or North

    def _turn_toward(self, current_facing, desired_facing) -> int:
        """Determine turn action to reach desired facing."""
        diff = (desired_facing - current_facing) % 4
        if diff == 0:
            return Action.FORWARD
        elif diff <= 2:
            return Action.TURN_RIGHT
        else:
            return Action.TURN_LEFT

    def _is_wall_adjacent(self, global_map, pos) -> bool:
        """Check if any wall is adjacent to current position."""
        for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
            check_x = pos[0] + dx
            check_y = pos[1] + dy

            if (0 <= check_x < global_map.shape[1] and
                0 <= check_y < global_map.shape[0]):
                if global_map[check_y, check_x] == TileType.WALL:
                    return True
        return False

    def get_metrics(self) -> Dict[str, Any]:
        """Return agent metrics."""
        return {
            'state': self.state.name,
            'turn_count': self.turn_count,
            'steps_along_wall': self.steps_along_wall,
            'target_wall': self.target_wall
        }