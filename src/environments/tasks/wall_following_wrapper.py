"""
Improved Wall Following Wrapper for single-agent training.

This wrapper trains an agent to:
1. Find the nearest visible wall
2. Approach and stick to it
3. Follow along the entire wall from end to end
"""

from typing import Dict, Tuple, Set, Optional
import numpy as np
from environments.tasks.base_task_wrapper import BaseTaskWrapper, TaskStatus
from environments.base.constants import TileType, DIRECTION_DELTAS


class WallFollowingWrapper(BaseTaskWrapper):
    """
    Environment wrapper for training wall-following behavior.

    The agent must:
    1. Identify the nearest visible wall
    2. Approach it and get adjacent
    3. Follow it to discover the entire wall segment
    """

    def __init__(self, env_config: Dict = None):
        super().__init__(env_config)

        # Wall tracking
        self.target_wall_segment: Set[Tuple[int, int]] = set()
        self.wall_endpoints: Set[Tuple[int, int]] = set()
        self.discovered_cells: Set[Tuple[int, int]] = set()

        # Behavior tracking
        self.phase = 'searching'  # 'searching', 'approaching', 'following'
        self.wall_contact_steps = 0
        self.last_distance = float('inf')
        self.found_endpoints = 0

    def _reset_task(self):
        """Reset wall-following specific state."""
        self.target_wall_segment = set()
        self.wall_endpoints = set()
        self.discovered_cells = set()
        self.phase = 'searching'
        self.wall_contact_steps = 0
        self.last_distance = float('inf')
        self.found_endpoints = 0

    def _find_visible_walls(self, global_map) -> Set[Tuple[int, int]]:
        """Find all currently visible (not unknown) wall cells."""
        visible_walls = set()
        for y in range(global_map.shape[0]):
            for x in range(global_map.shape[1]):
                if global_map[y, x] == TileType.WALL:
                    visible_walls.add((x, y))
        return visible_walls

    def _find_nearest_visible_wall(self, pos, global_map) -> Optional[Tuple[int, int]]:
        """Find the nearest visible wall cell."""
        visible_walls = self._find_visible_walls(global_map)
        if not visible_walls:
            return None

        min_dist = float('inf')
        nearest = None
        for wx, wy in visible_walls:
            dist = abs(wx - pos[0]) + abs(wy - pos[1])
            if dist < min_dist:
                min_dist = dist
                nearest = (wx, wy)
        return nearest

    def _trace_wall_segment(self, start_pos: Tuple[int, int], global_map) -> Set[Tuple[int, int]]:
        """
        Trace a continuous wall segment from a starting position.
        Only follows walls that form a continuous line/curve, not all connected walls.
        """
        segment = {start_pos}
        to_visit = [start_pos]
        visited = set()

        while to_visit:
            x, y = to_visit.pop(0)
            if (x, y) in visited:
                continue
            visited.add((x, y))

            # Count wall neighbors
            wall_neighbors = []
            for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                nx, ny = x + dx, y + dy
                if 0 <= nx < global_map.shape[1] and 0 <= ny < global_map.shape[0]:
                    if global_map[ny, nx] == TileType.WALL:
                        wall_neighbors.append((nx, ny))

            # Add to segment
            segment.add((x, y))

            # Continue tracing if this is part of a continuous wall (1-2 neighbors)
            # This avoids connecting separate walls at corners
            if 1 <= len(wall_neighbors) <= 2:
                for neighbor in wall_neighbors:
                    if neighbor not in visited:
                        to_visit.append(neighbor)

        return segment

    def _find_wall_endpoints(self, wall_segment: Set[Tuple[int, int]], global_map) -> Set[Tuple[int, int]]:
        """Find the endpoints of a wall segment (cells with only 1 wall neighbor)."""
        endpoints = set()

        for x, y in wall_segment:
            wall_neighbor_count = 0
            for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                nx, ny = x + dx, y + dy
                if (nx, ny) in wall_segment:
                    wall_neighbor_count += 1

            # Endpoint has exactly 1 wall neighbor in the segment
            if wall_neighbor_count == 1:
                endpoints.add((x, y))

        return endpoints

    def _is_adjacent_to_wall(self, pos: Tuple[int, int]) -> bool:
        """Check if position is adjacent to the target wall."""
        for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            if (pos[0] + dx, pos[1] + dy) in self.target_wall_segment:
                return True
        return False

    def _distance_to_wall(self, pos: Tuple[int, int]) -> float:
        """Calculate minimum Manhattan distance to target wall."""
        if not self.target_wall_segment:
            return float('inf')
        return min(abs(wx - pos[0]) + abs(wy - pos[1])
                   for wx, wy in self.target_wall_segment)

    def _is_endpoint_discovered(self, endpoint: Tuple[int, int], pos: Tuple[int, int]) -> bool:
        """Check if an endpoint has been discovered (agent was adjacent to it)."""
        # Check if agent is currently adjacent to this endpoint
        dist = abs(endpoint[0] - pos[0]) + abs(endpoint[1] - pos[1])
        return dist <= 1

    def _compute_task_reward(self, obs, action, base_reward) -> float:
        """
        Compute wall-following specific reward.

        Phase-based rewards:
        - Searching: Small penalty to encourage finding walls
        - Approaching: Reward for getting closer to wall
        - Following: Reward for staying adjacent and discovering wall
        """
        reward = -0.01  # Small step penalty

        pos = tuple(obs['positions'][0])
        global_map = obs['global_map']

        # Phase: SEARCHING - Find a wall to follow
        if self.phase == 'searching':
            nearest_wall = self._find_nearest_visible_wall(pos, global_map)

            if nearest_wall:
                # Lock onto this wall segment
                self.target_wall_segment = self._trace_wall_segment(nearest_wall, global_map)
                self.wall_endpoints = self._find_wall_endpoints(self.target_wall_segment, global_map)
                self.phase = 'approaching'
                reward += 2.0  # Bonus for finding a wall

        # Phase: APPROACHING - Get to the wall
        elif self.phase == 'approaching':
            dist = self._distance_to_wall(pos)

            if self._is_adjacent_to_wall(pos):
                # Reached the wall!
                self.phase = 'following'
                self.wall_contact_steps = 1
                reward += 10.0  # Big bonus for reaching wall
            else:
                # Reward getting closer
                if dist < self.last_distance:
                    reward += 1.0
                elif dist > self.last_distance:
                    reward -= 0.5

            self.last_distance = dist

        # Phase: FOLLOWING - Explore the entire wall
        elif self.phase == 'following':
            adjacent = self._is_adjacent_to_wall(pos)

            if adjacent:
                self.wall_contact_steps += 1
                reward += 1.0  # Reward for staying with wall

                # Check for endpoint discovery
                for endpoint in self.wall_endpoints:
                    if self._is_endpoint_discovered(endpoint, pos):
                        if endpoint not in self.discovered_cells:
                            self.discovered_cells.add(endpoint)
                            self.found_endpoints += 1
                            reward += 20.0  # Bonus for finding endpoint

                # Track all discovered wall cells
                for dx, dy in [(0, 0), (0, 1), (0, -1), (1, 0), (-1, 0)]:
                    check_x, check_y = pos[0] + dx, pos[1] + dy
                    if (check_x, check_y) in self.target_wall_segment:
                        if (check_x, check_y) not in self.discovered_cells:
                            self.discovered_cells.add((check_x, check_y))
                            reward += 0.5  # Small bonus for new discovery

            else:
                # Penalty for leaving wall
                reward -= 2.0
                self.wall_contact_steps = 0

        # Check completion
        if self.wall_endpoints and self.found_endpoints >= len(self.wall_endpoints):
            # Found all endpoints - wall fully explored!
            reward += 100.0

        # Collision penalty
        if base_reward < -0.5:  # Collision detected in base env
            reward -= 5.0

        return reward

    def _check_task_status(self, obs, action) -> TaskStatus:
        """Check if wall-following task is complete."""
        # Timeout for finding a wall
        if self.phase == 'searching' and self.task_step > 100:
            return TaskStatus.FAILURE

        # Success: discovered all endpoints
        if self.wall_endpoints and self.found_endpoints >= len(self.wall_endpoints):
            # Additional check: sufficient wall coverage
            coverage = len(self.discovered_cells) / len(self.target_wall_segment) if self.target_wall_segment else 0
            if coverage > 0.8:  # 80% coverage is sufficient
                return TaskStatus.SUCCESS

        # General timeout
        if self.task_step > 400:
            return TaskStatus.FAILURE

        return TaskStatus.IN_PROGRESS

    def get_info(self) -> Dict:
        """Get additional task-specific information."""
        info = {
            'phase': self.phase,
            'wall_contact_steps': self.wall_contact_steps,
            'endpoints_found': self.found_endpoints,
            'total_endpoints': len(self.wall_endpoints),
            'wall_coverage': len(self.discovered_cells) / len(
                self.target_wall_segment) if self.target_wall_segment else 0.0,
        }
        return info