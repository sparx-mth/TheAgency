"""
Room-aware frontier agent that explores within room boundaries.

This agent extends frontier exploration to:
1. Detect doorways/openings in real-time as they are discovered
2. Maintain a forbidden list of doorway positions
3. Never step on detected doorways

ADDED: State tracking for agent execution status
Note: Simplified frontier logic included since base FrontierAgent wasn't provided
"""

import numpy as np
from typing import Dict, Any, Tuple, Set, List, Optional
from heapq import heappush, heappop

from agents.base_agent import BaseSLAMAgent, AgentState
from environments.base.constants import TileType, Action


class RoomFrontierAgent(BaseSLAMAgent):
    """
    Frontier agent that detects and avoids doorways in real-time.
    Tracks execution state for coordination with other agents.
    """

    def __init__(self, num_agents: int = 1, camera_range: int = 10):
        super().__init__(num_agents)
        self.camera_range = camera_range
        self.forbidden_doorways = set()  # Positions we must not step on
        self.last_map_state = None  # To detect map changes

        # Frontier exploration state
        self.goals = {}
        self.paths = {}
        self.wait_counters = {}
        self.stuck_counters = {}
        self.last_positions = {}
        self.random_walk_counter = {}
        self.max_wait = 10

        # Execution tracking
        self.steps_taken = 0
        self.max_steps = 1000
        self.no_frontier_counter = 0
        self.max_no_frontier_steps = 10  # Reduced from 50 for faster completion

    def get_actions(self, observations: Dict[str, Any], info: Dict[str, Any]) -> np.ndarray:
        """Get actions for room exploration with doorway avoidance."""
        try:
            # Update execution state when first called
            if self.execution_state == AgentState.NOT_YET_STARTED:
                self.execution_state = AgentState.IN_PROGRESS

            # Check max steps
            self.steps_taken += 1
            if self.steps_taken >= self.max_steps:
                self.execution_state = AgentState.COMPLETED
                return np.array([Action.STAY] * self.num_agents)

            # For simplicity, handle single agent
            if self.num_agents == 1:
                obs = {
                    'global_map': observations['global_map'],
                    'position': tuple(observations['positions'][0]),
                    'facing': observations['facings'][0]
                }
                action = self._compute_action(0, obs, set(), {})

                # Check if exploration is complete
                if self._is_exploration_complete(observations['global_map']):
                    self.execution_state = AgentState.COMPLETED

                return np.array([action])
            else:
                # Multi-agent case
                actions = []
                assigned = set()
                all_states = {}

                for i in range(self.num_agents):
                    if observations['active'][i]:
                        obs = {
                            'global_map': observations['global_map'],
                            'position': tuple(observations['positions'][i]),
                            'facing': observations['facings'][i]
                        }
                        all_states[i] = {
                            'pos': obs['position'],
                            'active': True
                        }

                for i in range(self.num_agents):
                    if observations['active'][i]:
                        obs = {
                            'global_map': observations['global_map'],
                            'position': tuple(observations['positions'][i]),
                            'facing': observations['facings'][i]
                        }
                        action = self._compute_action(i, obs, assigned, all_states)
                        actions.append(action)
                    else:
                        actions.append(Action.STAY)

                # Check if exploration is complete
                if self._is_exploration_complete(observations['global_map']):
                    self.execution_state = AgentState.COMPLETED

                return np.array(actions)

        except Exception as e:
            self.set_error(str(e))
            return np.array([Action.STAY] * self.num_agents)

    def _is_exploration_complete(self, global_map: np.ndarray) -> bool:
        """
        Enhanced check if room exploration is complete.
        Handles edge cases where unreachable frontiers might exist.
        """
        # Find frontiers that are actually reachable
        frontiers = self._find_safe_frontiers(global_map)

        # Filter out unreachable frontiers
        reachable_frontiers = []
        if frontiers and hasattr(self, 'last_positions') and self.last_positions:
            # Get current position (for single agent, use agent 0)
            current_pos = self.last_positions.get(0)
            if current_pos:
                for frontier in frontiers:
                    # Check if we can actually path to this frontier
                    path = self._plan_safe_path(current_pos, frontier, global_map)
                    if path and len(path) < 50:  # Don't consider very distant frontiers
                        reachable_frontiers.append(frontier)
        else:
            reachable_frontiers = frontiers

        # If no reachable frontiers, increment counter
        if not reachable_frontiers:
            self.no_frontier_counter += 1
            # print(f"No reachable frontiers found. Counter: {self.no_frontier_counter}/{self.max_no_frontier_steps}")

            # Reduced threshold for faster completion
            if self.no_frontier_counter >= self.max_no_frontier_steps:
                # print(f"Room exploration complete - no reachable frontiers for {self.no_frontier_counter} steps")
                return True
        else:
            # Reset counter if we found frontiers
            # if self.no_frontier_counter > 0:
                # print(f"Found {len(reachable_frontiers)} reachable frontiers, resetting counter")
            self.no_frontier_counter = 0

        # Additional check: if we've been exploring for a long time with very few frontiers
        if self.steps_taken > 500 and len(reachable_frontiers) <= 2:
            self.no_frontier_counter += 5  # Accelerate completion for sparse frontiers
            # print(f"Few frontiers remaining ({len(reachable_frontiers)}), accelerating completion")

        return False

    def _detect_new_doorways(self, global_map: np.ndarray) -> Set[Tuple[int, int]]:
        """
        Scan the current visible map for doorway patterns.
        A doorway is a free space with walls on opposite sides.

        This runs every step to detect newly discovered doorways.
        """
        height, width = global_map.shape
        new_doorways = set()

        for y in range(1, height - 1):
            for x in range(1, width - 1):
                # Skip unknown areas
                if global_map[y, x] == TileType.UNKNOWN:
                    continue

                # Skip if already identified as doorway
                if (x, y) in self.forbidden_doorways:
                    continue

                # Must be a traversable space
                center = global_map[y, x]
                if center not in [TileType.FREE_SPACE, TileType.DOOR_OPEN, TileType.ENTRY_POINT]:
                    continue

                # Get adjacent cells (only if they're known)
                left = global_map[y, x-1] if x > 0 else TileType.UNKNOWN
                right = global_map[y, x+1] if x < width-1 else TileType.UNKNOWN
                top = global_map[y-1, x] if y > 0 else TileType.UNKNOWN
                bottom = global_map[y+1, x] if y < height-1 else TileType.UNKNOWN

                # Skip if any adjacent cell is unknown (can't confirm doorway yet)
                if TileType.UNKNOWN in [left, right, top, bottom]:
                    continue

                # Horizontal doorway: wall-space-wall pattern
                is_horizontal_door = (left == TileType.WALL and right == TileType.WALL and
                                     top != TileType.WALL and bottom != TileType.WALL)

                # Vertical doorway: wall-space-wall pattern
                is_vertical_door = (top == TileType.WALL and bottom == TileType.WALL and
                                   left != TileType.WALL and right != TileType.WALL)

                # If it's a doorway, mark it as forbidden
                if is_horizontal_door or is_vertical_door:
                    new_doorways.add((x, y))
                    self.forbidden_doorways.add((x, y))

        return new_doorways

    def _is_safe_position(self, pos: Tuple[int, int]) -> bool:
        """Check if a position is safe to move to (not a doorway)."""
        return pos not in self.forbidden_doorways

    def _compute_action(
        self,
        agent_id: int,
        obs: Dict,
        assigned: Set[Tuple[int, int]],
        all_states: Dict[int, Dict[str, Any]]
    ) -> int:
        """
        Override compute action with better completion detection.
        """
        global_map = obs['global_map']
        pos = tuple(obs['position'])
        facing_idx = obs['facing']

        # REAL-TIME DOORWAY DETECTION
        new_doorways = self._detect_new_doorways(global_map)
        if new_doorways:
            # Found new doorways, update our paths if needed
            if agent_id in self.paths and self.paths[agent_id]:
                for doorway in new_doorways:
                    if doorway in self.paths[agent_id]:
                        self.paths[agent_id] = []
                        self.goals[agent_id] = None
                        break

        # Check if we're facing a doorway
        dx, dy = [(0, -1), (1, 0), (0, 1), (-1, 0)][facing_idx]
        next_pos = (pos[0] + dx, pos[1] + dy)

        if next_pos in self.forbidden_doorways:
            return Action.TURN_RIGHT

        # Initialize counters
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

        # Handle stuck situation
        if self.stuck_counters.get(agent_id, 0) > 5:
            old_goal = self.goals.get(agent_id)
            if old_goal and old_goal in assigned:
                assigned.discard(old_goal)

            self.goals[agent_id] = None
            self.paths[agent_id] = []
            self.stuck_counters[agent_id] = 0
            self.wait_counters[agent_id] = 0
            return self._safe_random_walk(pos, facing_idx, global_map)

        # Continue random walk if in that mode
        if self.random_walk_counter.get(agent_id, 0) > 0:
            self.random_walk_counter[agent_id] -= 1
            return self._safe_random_walk(pos, facing_idx, global_map)

        # Sensor exploration
        action = self._check_sensor_exploration(pos, facing_idx, global_map)
        if action is not None:
            if action == Action.FORWARD:
                if next_pos in self.forbidden_doorways:
                    action = Action.TURN_RIGHT
            return action

        # Find frontiers (excluding doorways)
        all_frontiers = self._find_safe_frontiers(global_map)

        # Filter for reachable frontiers only
        reachable_frontiers = []
        for frontier in all_frontiers:
            path = self._plan_safe_path(pos, frontier, global_map)
            if path and len(path) < 50:  # Don't pursue very distant frontiers
                reachable_frontiers.append(frontier)

        # Use reachable frontiers instead of all frontiers
        all_frontiers = reachable_frontiers

        # Check current goal validity
        current_goal = self.goals.get(agent_id)
        goal_still_valid = False

        if current_goal:
            if current_goal in self.forbidden_doorways:
                goal_still_valid = False
            elif current_goal in all_frontiers or self._manhattan_distance(current_goal, pos) <= 2:
                if self.paths.get(agent_id):
                    path_valid = all(p not in self.forbidden_doorways for p in self.paths[agent_id])
                    goal_still_valid = path_valid
                else:
                    new_path = self._plan_safe_path(pos, current_goal, global_map)
                    if new_path:
                        self.paths[agent_id] = new_path
                        goal_still_valid = True

        # Find new goal if needed
        if not goal_still_valid:
            if current_goal and current_goal in assigned:
                assigned.discard(current_goal)

            available_frontiers = [f for f in all_frontiers if f not in assigned or f == current_goal]

            if available_frontiers:
                best_goal, best_path = self._select_safe_frontier(pos, available_frontiers, global_map)

                if best_goal:
                    self.goals[agent_id] = best_goal
                    self.paths[agent_id] = best_path
                    assigned.add(best_goal)
                    self.wait_counters[agent_id] = 0
                else:
                    self.goals[agent_id] = None
                    self.paths[agent_id] = []
                    # No valid goals, might be complete
                    return Action.STAY
            else:
                # No frontiers available - room likely fully explored
                self.goals[agent_id] = None
                self.paths[agent_id] = []
                # print(f"No available frontiers at step {self.steps_taken}")
                return Action.STAY

        # Follow path
        if self.paths.get(agent_id):
            return self._follow_safe_path(agent_id, pos, facing_idx, all_states, global_map)

        # No path, do a short random walk then check again
        self.random_walk_counter[agent_id] = 3
        return self._safe_random_walk(pos, facing_idx, global_map)

    def _check_sensor_exploration(self, pos: Tuple[int, int], facing_idx: int,
                                 global_map: np.ndarray) -> Optional[int]:
        """Check if we should explore sensor boundaries."""
        # Simple sensor check - look for unknown cells in view
        height, width = global_map.shape
        dx, dy = [(0, -1), (1, 0), (0, 1), (-1, 0)][facing_idx]

        # Check ahead for unknowns
        for dist in range(1, min(self.camera_range, 5)):
            check_x = pos[0] + dx * dist
            check_y = pos[1] + dy * dist

            if 0 <= check_x < width and 0 <= check_y < height:
                if global_map[check_y, check_x] == TileType.UNKNOWN:
                    # Found unknown ahead, move forward if safe
                    next_pos = (pos[0] + dx, pos[1] + dy)
                    if next_pos not in self.forbidden_doorways:
                        if self._is_passable(global_map[pos[1] + dy, pos[0] + dx]):
                            return Action.FORWARD
                    break

        return None

    def _safe_random_walk(self, pos: Tuple[int, int], facing_idx: int, global_map: np.ndarray) -> int:
        """Random walk that avoids doorways."""
        # Check if forward is safe
        dx, dy = [(0, -1), (1, 0), (0, 1), (-1, 0)][facing_idx]
        next_pos = (pos[0] + dx, pos[1] + dy)

        # If facing a doorway, must turn
        if next_pos in self.forbidden_doorways:
            return Action.TURN_RIGHT if np.random.random() > 0.5 else Action.TURN_LEFT

        # Otherwise do normal random walk
        if np.random.random() < 0.25:
            return Action.TURN_RIGHT if np.random.random() > 0.5 else Action.TURN_LEFT

        # Try to move forward if safe
        height, width = global_map.shape
        new_x, new_y = next_pos

        if 0 <= new_x < width and 0 <= new_y < height:
            if self._is_passable(global_map[new_y, new_x]):
                return Action.FORWARD

        return Action.TURN_RIGHT if np.random.random() > 0.5 else Action.TURN_LEFT

    def _is_passable(self, tile: int) -> bool:
        """Check if a tile is passable."""
        passable_tiles = {
            TileType.UNKNOWN,
            TileType.FREE_SPACE,
            TileType.ENTRY_POINT,
            TileType.DOOR_OPEN
        }
        return tile in passable_tiles

    def _find_safe_frontiers(self, global_map: np.ndarray) -> List[Tuple[int, int]]:
        """
        Find frontier cells that are not doorways.
        Enhanced to filter out problematic frontiers.
        """
        frontiers = []
        height, width = global_map.shape

        for y in range(height):
            for x in range(width):
                # Skip doorways
                if (x, y) in self.forbidden_doorways:
                    continue

                # Must be known free space
                if global_map[y, x] == TileType.UNKNOWN:
                    continue
                if global_map[y, x] in {TileType.WALL, TileType.DOOR_CLOSED, TileType.OUT_OF_BOUNDS}:
                    continue

                # Check if adjacent to unknown
                is_frontier = False
                unknown_neighbor_count = 0

                for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < width and 0 <= ny < height:
                        if global_map[ny, nx] == TileType.UNKNOWN:
                            # Additional check: don't consider it a frontier if the unknown
                            # is beyond a doorway
                            path_to_unknown = [(x, y), (nx, ny)]
                            if not any(p in self.forbidden_doorways for p in path_to_unknown):
                                is_frontier = True
                                unknown_neighbor_count += 1

                # Only consider it a frontier if it has enough unknown neighbors
                # This filters out single-pixel artifacts
                if is_frontier and unknown_neighbor_count >= 1:
                    frontiers.append((x, y))

        return frontiers

    def _manhattan_distance(self, pos1: Tuple[int, int], pos2: Tuple[int, int]) -> int:
        """Calculate Manhattan distance between two positions."""
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])

    def _plan_safe_path(
        self,
        start: Tuple[int, int],
        goal: Tuple[int, int],
        global_map: np.ndarray
    ) -> List[Tuple[int, int]]:
        """Plan a path that avoids all forbidden doorways."""
        if start == goal:
            return []

        height, width = global_map.shape

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

                # CRITICAL: Skip forbidden doorways
                if neighbor in self.forbidden_doorways:
                    continue

                # Check traversability
                tile = global_map[neighbor[1], neighbor[0]]

                if not self._is_passable(tile):
                    continue

                if neighbor in closed_set:
                    continue

                tentative_g = g_score[current] + 1

                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f_score[neighbor] = tentative_g + self._manhattan_distance(neighbor, goal)
                    heappush(open_set, (f_score[neighbor], neighbor))

        return []

    def _select_safe_frontier(
        self,
        pos: Tuple[int, int],
        available_frontiers: List[Tuple[int, int]],
        global_map: np.ndarray
    ) -> Tuple[Optional[Tuple[int, int]], List[Tuple[int, int]]]:
        """Select best frontier with safe path."""
        reachable_frontiers = []

        for frontier in available_frontiers:
            # Skip if frontier is a doorway (double-check)
            if frontier in self.forbidden_doorways:
                continue

            path = self._plan_safe_path(pos, frontier, global_map)
            if path:
                reachable_frontiers.append((len(path), frontier, path))

        if not reachable_frontiers:
            return None, []

        # Sort by distance and pick closest
        reachable_frontiers.sort(key=lambda x: x[0])
        _, best_frontier, best_path = reachable_frontiers[0]

        return best_frontier, best_path

    def _follow_safe_path(
        self,
        agent_id: int,
        pos: Tuple[int, int],
        facing_idx: int,
        all_states: Dict[int, Dict[str, Any]],
        global_map: np.ndarray
    ) -> int:
        """Follow path with extra safety checks for doorways."""
        path = self.paths.get(agent_id, [])
        if not path:
            return Action.STAY

        # Clean up path
        while path and pos == path[0]:
            path.pop(0)

        if not path:
            self.goals[agent_id] = None
            return Action.STAY

        next_pos = path[0]

        # CRITICAL: Double-check next position is safe
        if next_pos in self.forbidden_doorways:
            # Path is compromised, replan
            self.goals[agent_id] = None
            self.paths[agent_id] = []
            return Action.TURN_RIGHT

        # Check for other agents blocking
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
                # Try to replan
                goal = self.goals.get(agent_id)
                if goal:
                    new_path = self._plan_safe_path(pos, goal, global_map)
                    if new_path:
                        self.paths[agent_id] = new_path
                        self.wait_counters[agent_id] = 0
                        return self._follow_safe_path(agent_id, pos, facing_idx, all_states, global_map)

                self.goals[agent_id] = None
                self.paths[agent_id] = []
                self.wait_counters[agent_id] = 0
                return self._safe_random_walk(pos, facing_idx, global_map)
            else:
                return Action.STAY

        self.wait_counters[agent_id] = 0

        # Move towards next position
        dx = next_pos[0] - pos[0]
        dy = next_pos[1] - pos[1]

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

        if facing_idx == target_dir:
            return Action.FORWARD

        diff = (target_dir - facing_idx) % 4
        if diff == 1:
            return Action.TURN_RIGHT
        elif diff == 3:
            return Action.TURN_LEFT
        else:
            return Action.TURN_RIGHT

    def reset(self) -> None:
        """Reset agent state."""
        super().reset()  # Reset execution state
        self.forbidden_doorways = set()
        self.last_map_state = None
        self.goals = {}
        self.paths = {}
        self.wait_counters = {}
        self.stuck_counters = {}
        self.last_positions = {}
        self.random_walk_counter = {}
        self.steps_taken = 0
        self.no_frontier_counter = 0

    def get_metrics(self) -> Dict[str, Any]:
        """Get agent metrics."""
        metrics = super().get_metrics()  # Get base metrics including execution state
        metrics.update({
            'detected_doorways': len(self.forbidden_doorways),
            'active_goals': len([g for g in self.goals.values() if g is not None]),
            'steps_taken': self.steps_taken,
            'max_steps': self.max_steps,
            'no_frontier_counter': self.no_frontier_counter
        })
        return metrics