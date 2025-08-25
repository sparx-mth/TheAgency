"""
Room Exploration Environment Wrapper for SLAM

This wrapper trains an agent to explore a room without passing through doorways.
The agent must scan the current room while avoiding exits through doorways.
"""

import gymnasium as gym
import numpy as np
import pygame
from typing import Dict, Tuple, Optional, Set

from environments.tasks.base_task_wrapper import BaseTaskWrapper, TaskStatus
from environments.base.constants import TileType, TILE_SIZE, TILE_COLORS


class RoomExplorationWrapper(BaseTaskWrapper):
    """
    Environment wrapper for training room exploration behavior.

    The agent must:
    1. Explore the current room
    2. Not pass through doorways (failure condition)
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
        """
        Initialize the room exploration environment.

        Args:
            env_config: Configuration for base environment
            exploration_reward: Reward for discovering new cells
            door_penalty: Large penalty for passing through doorway
            completion_reward: Reward for completing room exploration
            step_penalty: Small penalty per step
            coverage_threshold: Fraction of visible area that must be explored
            max_task_steps: Maximum steps for the task
        """
        super().__init__(env_config)

        # Reward parameters
        self.exploration_reward = exploration_reward
        self.door_penalty = door_penalty
        self.completion_reward = completion_reward
        self.step_penalty = step_penalty
        self.coverage_threshold = coverage_threshold
        self.max_task_steps = max_task_steps

        # Task state
        self.doorways = []  # List of detected doorways
        self.passed_through_door = False
        self.cells_discovered = 0
        self.completion_achieved = False

    def _reset_task(self):
        """Reset task-specific state."""
        self.doorways = []
        self.passed_through_door = False
        self.cells_discovered = 0
        self.completion_achieved = False

    def _find_doorways(self):
        """Find all doorways (1-0-1 patterns) in the discovered map."""
        self.doorways = []
        global_map = self.env.global_map

        # Check horizontal doorways
        for y in range(self.env.height):
            for x in range(1, self.env.width - 1):
                center = global_map[y, x]
                left = global_map[y, x-1]
                right = global_map[y, x+1]

                # All three must be discovered
                if center == TileType.UNKNOWN or left == TileType.UNKNOWN or right == TileType.UNKNOWN:
                    continue

                # Check for doorway pattern: wall-passage-wall
                if (left == TileType.WALL and
                    center in [TileType.FREE_SPACE, TileType.DOOR_OPEN] and
                    right == TileType.WALL):
                    self.doorways.append((x, y))

        # Check vertical doorways
        for x in range(self.env.width):
            for y in range(1, self.env.height - 1):
                center = global_map[y, x]
                top = global_map[y-1, x]
                bottom = global_map[y+1, x]

                # All three must be discovered
                if center == TileType.UNKNOWN or top == TileType.UNKNOWN or bottom == TileType.UNKNOWN:
                    continue

                # Check for doorway pattern: wall-passage-wall
                if (top == TileType.WALL and
                    center in [TileType.FREE_SPACE, TileType.DOOR_OPEN] and
                    bottom == TileType.WALL):
                    # Avoid duplicate
                    if (x, y) not in self.doorways:
                        self.doorways.append((x, y))

    def _check_room_completion(self):
        """Check if all reachable areas (without crossing doorways) are fully explored."""
        global_map = self.env.global_map
        true_map = self.env.true_map

        # Get starting position
        start_pos = self.env.drones[0].pos

        # Find all cells reachable from start without crossing doorways (using TRUE map for ground truth)
        reachable_cells = set()
        queue = [start_pos]
        visited = set()

        while queue:
            x, y = queue.pop(0)
            if (x, y) in visited:
                continue
            visited.add((x, y))

            # Don't cross doorways
            if (x, y) in self.doorways:
                continue

            # Check if this is a traversable cell in the TRUE map
            if true_map[y, x] in [TileType.FREE_SPACE, TileType.DOOR_OPEN, TileType.ENTRY_POINT]:
                reachable_cells.add((x, y))

                # Add neighbors to queue
                for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < self.env.width and 0 <= ny < self.env.height:
                        if (nx, ny) not in visited:
                            queue.append((nx, ny))

        # Also include walls and obstacles adjacent to reachable cells (these should be discovered too)
        cells_to_discover = set()
        for x, y in reachable_cells:
            cells_to_discover.add((x, y))
            # Check all adjacent cells
            for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                nx, ny = x + dx, y + dy
                if 0 <= nx < self.env.width and 0 <= ny < self.env.height:
                    # Add walls and other obstacles that should be visible
                    if true_map[ny, nx] in [TileType.WALL, TileType.DOOR_CLOSED]:
                        cells_to_discover.add((nx, ny))

        # Remove doorways from cells to discover (we see them but don't count them)
        cells_to_discover = cells_to_discover - set(self.doorways)

        # Check how many of these cells have been discovered
        discovered_count = 0
        for x, y in cells_to_discover:
            if global_map[y, x] != TileType.UNKNOWN:
                discovered_count += 1

        total_to_discover = len(cells_to_discover)
        if total_to_discover > 0:
            coverage = discovered_count / total_to_discover
            if coverage >= self.coverage_threshold:
                print(
                    f"Room exploration complete! Coverage: {coverage:.2%} ({discovered_count}/{total_to_discover} cells)")
                return True
            # Debug output every 50 steps
            if self.task_step % 50 == 0:
                print(f"Exploration progress: {coverage:.2%} ({discovered_count}/{total_to_discover} cells)")

        return False

    def _compute_task_reward(self, obs, action, base_reward) -> float:
        """Compute room exploration specific reward."""
        drone_pos = tuple(obs['positions'][0])

        # Update doorway detection
        self._find_doorways()

        # Base reward from discoveries (like base environment)
        reward = base_reward

        # Add step penalty
        reward += self.step_penalty

        # Check if we passed through a doorway
        if drone_pos in self.doorways:
            if not self.passed_through_door:
                reward += self.door_penalty
                self.passed_through_door = True
                print(f"FAILURE: Passed through doorway at {drone_pos}")

        # Check for room completion
        if not self.completion_achieved and not self.passed_through_door:
            if self._check_room_completion():
                reward += self.completion_reward
                self.completion_achieved = True
                print(f"SUCCESS: Room exploration complete!")

        return reward

    def _check_task_status(self, obs, action) -> TaskStatus:
        """Check if the room exploration task is complete."""
        # Failure: passed through doorway
        if self.passed_through_door:
            return TaskStatus.FAILURE

        # Success: explored the room
        if self.completion_achieved:
            return TaskStatus.SUCCESS

        # Failure: exceeded maximum steps
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
                for x, y in self.doorways:
                    color = (255, 0, 0) if self.passed_through_door else (255, 255, 0)

                    # Draw doorway marker with thick border
                    rect = pygame.Rect(
                        map_offset + x * TILE_SIZE,
                        y * TILE_SIZE,
                        TILE_SIZE,
                        TILE_SIZE
                    )
                    pygame.draw.rect(self.env.screen, color, rect, 3)

                    # Fill with transparent color if drone is on doorway
                    if drone_pos == (x, y):
                        s = pygame.Surface((TILE_SIZE, TILE_SIZE))
                        s.set_alpha(50)
                        s.fill(color)
                        self.env.screen.blit(s, (map_offset + x * TILE_SIZE, y * TILE_SIZE))

            # Display task status
            if self.env.font:
                # Status text
                if self.passed_through_door:
                    status_text = f"FAILED: Passed through doorway! Steps: {self.task_step}"
                    color = (255, 0, 0)
                else:
                    status_text = f"Exploring room - Steps: {self.task_step} | Doorways found: {len(self.doorways)}"
                    color = (255, 255, 255)

                text_surface = self.env.font.render(status_text, True, color)
                self.env.screen.blit(text_surface, (10, 10))

            pygame.display.flip()

        return base_render