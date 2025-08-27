"""
Further Optimized Room Exploration Environment Wrapper for SLAM

Additional optimizations:
1. Cached room completion checking - only recompute when map changes
2. Numpy vectorized operations for coverage calculation
3. Reduced overhead in doorway checking
"""

import gymnasium as gym
import numpy as np
import pygame
from typing import Dict, Tuple, Optional, Set
from numba import njit

from environments.tasks.base_task_wrapper import BaseTaskWrapper, TaskStatus
from environments.base.constants import TileType, TILE_SIZE, TILE_COLORS


@njit
def fast_coverage_check(global_map: np.ndarray, cells_x: np.ndarray, cells_y: np.ndarray, threshold: float) -> tuple:
    """Fast coverage calculation using numba."""
    discovered = 0
    total = len(cells_x)

    for i in range(total):
        if global_map[cells_y[i], cells_x[i]] != TileType.UNKNOWN:
            discovered += 1

    coverage = discovered / total if total > 0 else 0.0
    return coverage >= threshold, coverage


class RoomExplorationWrapper(BaseTaskWrapper):
    """
    Further optimized environment wrapper for training room exploration behavior.

    Additional optimizations:
    - Cached coverage computation
    - Vectorized operations where possible
    - Minimal map hash computations
    """

    def __init__(
        self,
        env_config: Dict = None,
        # Reward parameters
        exploration_reward: float = 0.1,
        door_penalty: float = -10.0,
        completion_reward: float = 10.0,
        step_penalty: float = -0.001,
        coverage_threshold: float = 1.0,
        max_task_steps: int = 500,
    ):
        """Initialize the optimized room exploration environment."""
        super().__init__(env_config)

        # Reward parameters
        self.exploration_reward = exploration_reward
        self.door_penalty = door_penalty
        self.completion_reward = completion_reward
        self.step_penalty = step_penalty
        self.coverage_threshold = coverage_threshold
        self.max_task_steps = max_task_steps

        # Pre-computed doorway storage
        self.all_doorways_set = set()  # Set of all doorway positions from true map
        self.discovered_doorways_set = set()  # Set for O(1) lookups

        # Cache for room completion checking
        self.cells_to_discover = set()  # Cells that need to be discovered for completion
        self.cells_to_discover_x = None  # Numpy array for fast checking
        self.cells_to_discover_y = None  # Numpy array for fast checking

        # Map change tracking
        self.last_global_map_hash = None
        self.last_coverage = 0.0
        self.last_completion_check = False

        # Task state
        self.passed_through_door = False
        self.completion_achieved = False

    def reset(self, **kwargs):
        """Reset the environment and pre-compute doorways and reachable cells."""
        obs, info = super().reset(**kwargs)

        # Pre-compute ALL doorways from the true map
        self._precompute_all_doorways()

        # Pre-compute reachable cells for this episode
        self._precompute_reachable_cells()

        # Reset discovered doorways
        self.discovered_doorways_set = set()
        self.last_global_map_hash = None
        self.last_coverage = 0.0
        self.last_completion_check = False

        return obs, info

    def _precompute_all_doorways(self):
        """
        Pre-compute all doorways from the true map.
        Optimized to do single pass and avoid redundant checks.
        """
        self.all_doorways_set = set()
        true_map = self.env.true_map
        height, width = true_map.shape

        # Single pass through the map
        for y in range(1, height - 1):
            for x in range(1, width - 1):
                center = true_map[y, x]

                # Skip if center is not passable
                if center not in [TileType.FREE_SPACE, TileType.DOOR_OPEN]:
                    continue

                # Check horizontal doorway
                if (true_map[y, x-1] == TileType.WALL and
                    true_map[y, x+1] == TileType.WALL):
                    self.all_doorways_set.add((x, y))
                # Check vertical doorway
                elif (true_map[y-1, x] == TileType.WALL and
                      true_map[y+1, x] == TileType.WALL):
                    self.all_doorways_set.add((x, y))

    def _precompute_reachable_cells(self):
        """
        Pre-compute cells reachable from starting position without crossing doorways.
        Optimized with numpy arrays for fast checking.
        """
        true_map = self.env.true_map
        start_pos = self.env.drones[0].pos

        # BFS to find reachable cells
        reachable_cells = set()
        queue = [start_pos]
        visited = set()

        while queue:
            x, y = queue.pop(0)
            if (x, y) in visited:
                continue
            visited.add((x, y))

            # Don't cross doorways
            if (x, y) in self.all_doorways_set:
                continue

            # Check if traversable
            if true_map[y, x] in [TileType.FREE_SPACE, TileType.DOOR_OPEN, TileType.ENTRY_POINT]:
                reachable_cells.add((x, y))

                # Add neighbors
                for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < self.env.width and 0 <= ny < self.env.height:
                        if (nx, ny) not in visited:
                            queue.append((nx, ny))

        # Include adjacent walls
        self.cells_to_discover = set()
        for x, y in reachable_cells:
            self.cells_to_discover.add((x, y))
            # Check adjacent cells
            for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                nx, ny = x + dx, y + dy
                if 0 <= nx < self.env.width and 0 <= ny < self.env.height:
                    if true_map[ny, nx] in [TileType.WALL, TileType.DOOR_CLOSED]:
                        self.cells_to_discover.add((nx, ny))

        # Remove doorways
        self.cells_to_discover = self.cells_to_discover - self.all_doorways_set

        # Convert to numpy arrays for fast checking
        if self.cells_to_discover:
            cells_list = list(self.cells_to_discover)
            self.cells_to_discover_x = np.array([c[0] for c in cells_list], dtype=np.int32)
            self.cells_to_discover_y = np.array([c[1] for c in cells_list], dtype=np.int32)
        else:
            self.cells_to_discover_x = np.array([], dtype=np.int32)
            self.cells_to_discover_y = np.array([], dtype=np.int32)

    def _check_doorways_fast(self):
        """
        Optimized doorway checking - only check if map changed.
        """
        global_map = self.env.global_map

        # Quick hash check
        map_hash = hash(global_map.tobytes())
        if map_hash == self.last_global_map_hash:
            return False
        self.last_global_map_hash = map_hash

        # Check only undiscovered doorways
        new_doorways = False
        for x, y in self.all_doorways_set - self.discovered_doorways_set:
            # Check if doorway area is discovered
            if x > 0 and x < self.env.width - 1 and y > 0 and y < self.env.height - 1:
                # Quick check - just see if center is discovered
                if global_map[y, x] != TileType.UNKNOWN:
                    # Verify it's actually a doorway pattern
                    if ((global_map[y, x-1] == TileType.WALL and
                         global_map[y, x+1] == TileType.WALL) or
                        (global_map[y-1, x] == TileType.WALL and
                         global_map[y+1, x] == TileType.WALL)):
                        self.discovered_doorways_set.add((x, y))
                        new_doorways = True

        return new_doorways

    def _reset_task(self):
        """Reset task-specific state."""
        self.passed_through_door = False
        self.completion_achieved = False
        self.last_coverage = 0.0
        self.last_completion_check = False

    def _check_room_completion(self):
        """
        Check if room exploration is complete using cached results.
        """
        # Use cached result if map hasn't changed
        if self.last_global_map_hash == hash(self.env.global_map.tobytes()):
            return self.last_completion_check

        # Use numba-optimized coverage check
        if len(self.cells_to_discover_x) > 0:
            is_complete, coverage = fast_coverage_check(
                self.env.global_map,
                self.cells_to_discover_x,
                self.cells_to_discover_y,
                self.coverage_threshold
            )
            self.last_coverage = coverage
            self.last_completion_check = is_complete
            return is_complete

        return False

    def _compute_task_reward(self, obs, action, base_reward) -> float:
        """Compute room exploration specific reward with optimized logic."""
        drone_pos = tuple(obs['positions'][0])

        # Check for newly discovered doorways (optimized)
        self._check_doorways_fast()

        # Base reward
        reward = base_reward + self.step_penalty

        # Check if we passed through a doorway (O(1) lookup)
        if drone_pos in self.discovered_doorways_set and not self.passed_through_door:
            reward += self.door_penalty
            self.passed_through_door = True

        # Check for room completion (cached)
        if not self.completion_achieved and not self.passed_through_door:
            if self._check_room_completion():
                reward += self.completion_reward
                self.completion_achieved = True

        return reward

    def _check_task_status(self, obs, action) -> TaskStatus:
        """Check if the room exploration task is complete."""
        if self.passed_through_door:
            return TaskStatus.FAILURE

        if self.completion_achieved:
            return TaskStatus.SUCCESS

        if self.task_step >= self.max_task_steps:
            return TaskStatus.FAILURE

        return TaskStatus.IN_PROGRESS

    def render(self) -> Optional[np.ndarray]:
        """Render the environment with doorway visualization."""
        if self.env.render_mode is None:
            return None

        # First render the base environment
        base_render = self.env.render()

        # Add doorway visualization
        if self.env.screen is not None:
            drone_pos = self.env.drones[0].pos

            # Highlight doorways on BOTH maps
            for map_offset in [0, self.env.width * TILE_SIZE + 50]:
                for x, y in self.discovered_doorways_set:
                    color = (255, 0, 0) if self.passed_through_door else (255, 255, 0)

                    # Draw doorway marker
                    rect = pygame.Rect(
                        map_offset + x * TILE_SIZE,
                        y * TILE_SIZE,
                        TILE_SIZE,
                        TILE_SIZE
                    )
                    pygame.draw.rect(self.env.screen, color, rect, 3)

                    # Fill if drone is on doorway
                    if drone_pos == (x, y):
                        s = pygame.Surface((TILE_SIZE, TILE_SIZE))
                        s.set_alpha(50)
                        s.fill(color)
                        self.env.screen.blit(s, (map_offset + x * TILE_SIZE, y * TILE_SIZE))

            # Display task status
            if self.env.font:
                if self.passed_through_door:
                    status_text = f"FAILED: Passed through doorway! Steps: {self.task_step}"
                    color = (255, 0, 0)
                elif self.completion_achieved:
                    status_text = f"SUCCESS: Room explored! Steps: {self.task_step}"
                    color = (0, 255, 0)
                else:
                    coverage_pct = self.last_coverage * 100
                    status_text = f"Exploring - Steps: {self.task_step} | Coverage: {coverage_pct:.1f}% | Doors: {len(self.discovered_doorways_set)}"
                    color = (255, 255, 255)

                text_surface = self.env.font.render(status_text, True, color)
                self.env.screen.blit(text_surface, (10, 10))

            pygame.display.flip()

        return base_render