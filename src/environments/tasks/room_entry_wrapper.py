"""
Optimized Room Entry Environment Wrapper for SLAM

Performance optimizations:
1. Pre-compute all doorways from true map at initialization
2. Only check if newly discovered cells are known doorways (O(1) lookup)
3. No repeated scanning of the map
4. Minimal overhead per step
"""

import gymnasium as gym
import numpy as np
from typing import Dict, Tuple, Optional, List, Set
from numba import njit

from environments.tasks.base_task_wrapper import BaseTaskWrapper, TaskStatus
from environments.base.constants import TileType


# Numba-optimized distance calculation
@njit
def manhattan_distance(x1: int, y1: int, x2: int, y2: int) -> int:
    """Fast Manhattan distance calculation."""
    return abs(x1 - x2) + abs(y1 - y2)


class RoomEntryWrapper(BaseTaskWrapper):
    """
    Optimized environment wrapper for training doorway entry behavior.

    Key optimizations:
    - Pre-compute all doorways from true map
    - O(1) lookups for discovered doorways
    - No map scanning during runtime
    """

    def __init__(
        self,
        env_config: Dict = None,
        # Reward parameters
        entry_reward: float = 20.0,
        approach_reward: float = 1.0,
        wrong_direction_penalty: float = -5.0,
        collision_penalty: float = -1.0,
        step_penalty: float = -0.01,
        max_task_steps: int = 500,
        # Remove auto-explore completely
        auto_explore: bool = False,
        max_exploration_steps: int = 0,
    ):
        """Initialize the optimized doorway entry environment."""
        super().__init__(env_config)

        # Reward parameters
        self.entry_reward = entry_reward
        self.approach_reward = approach_reward
        self.wrong_direction_penalty = wrong_direction_penalty
        self.collision_penalty = collision_penalty
        self.step_penalty = step_penalty
        self.max_task_steps = max_task_steps

        # Pre-computed doorway storage
        self.all_doorways = {}  # {(x,y): orientation} for all doorways in true map
        self.discovered_doorways = []  # List of discovered doorways
        self.discovered_doorway_set = set()  # Set for O(1) lookups

        # Track what we've checked
        self.last_global_map_hash = None

        # Task state
        self.target_doorway = None
        self.initial_distance = None
        self.previous_distance = None
        self.previous_pos = None
        self.has_passed_through = False
        self.doorway_orientation = None
        self.approach_side = None
        self.position_before_doorway = None

    def reset(self, **kwargs):
        """Reset the environment and pre-compute all doorways."""
        obs, info = super().reset(**kwargs)

        # Pre-compute ALL doorways from the true map
        self._precompute_all_doorways()

        # Reset discovered doorways
        self.discovered_doorways = []
        self.discovered_doorway_set = set()
        self.last_global_map_hash = None

        return obs, info

    def _precompute_all_doorways(self):
        """
        Pre-compute all doorways from the true map.
        This is done once at reset, not during runtime.
        """
        self.all_doorways = {}
        true_map = self.env.true_map
        height, width = true_map.shape

        # Check all positions for doorway patterns
        for y in range(height):
            for x in range(width):
                # Check horizontal doorway (wall-passage-wall)
                if 1 <= x < width - 1:
                    center = true_map[y, x]
                    left = true_map[y, x-1]
                    right = true_map[y, x+1]

                    if (left == TileType.WALL and
                        center in [TileType.FREE_SPACE, TileType.DOOR_OPEN] and
                        right == TileType.WALL):
                        # This is a horizontal doorway
                        self.all_doorways[(x, y)] = 'horizontal'

                # Check vertical doorway (wall-passage-wall)
                if 1 <= y < height - 1:
                    center = true_map[y, x]
                    top = true_map[y-1, x]
                    bottom = true_map[y+1, x]

                    if (top == TileType.WALL and
                        center in [TileType.FREE_SPACE, TileType.DOOR_OPEN] and
                        bottom == TileType.WALL):
                        # This is a vertical doorway (only add if not already horizontal)
                        if (x, y) not in self.all_doorways:
                            self.all_doorways[(x, y)] = 'vertical'

        # print(f"Pre-computed {len(self.all_doorways)} doorways from true map")

    def _check_for_new_doorways(self):
        """
        Check if any newly discovered cells reveal doorways.
        Very fast since we only check against pre-computed doorways.
        """
        global_map = self.env.global_map

        # Quick hash check to see if map has changed
        map_hash = hash(global_map.tobytes())
        if map_hash == self.last_global_map_hash:
            return False  # No changes
        self.last_global_map_hash = map_hash

        # Check each pre-computed doorway to see if it's now discovered
        new_doorways_found = False
        for (x, y), orientation in self.all_doorways.items():
            # Skip if already discovered
            if (x, y) in self.discovered_doorway_set:
                continue

            # Check if this doorway position is now discovered
            if global_map[y, x] != TileType.UNKNOWN:
                # For horizontal doorways, check left and right are also discovered
                if orientation == 'horizontal':
                    if (x > 0 and global_map[y, x-1] != TileType.UNKNOWN and
                        x < self.env.width - 1 and global_map[y, x+1] != TileType.UNKNOWN):
                        # Doorway is fully discovered
                        self.discovered_doorways.append(((x, y), orientation))
                        self.discovered_doorway_set.add((x, y))
                        new_doorways_found = True

                # For vertical doorways, check top and bottom are also discovered
                else:  # vertical
                    if (y > 0 and global_map[y-1, x] != TileType.UNKNOWN and
                        y < self.env.height - 1 and global_map[y+1, x] != TileType.UNKNOWN):
                        # Doorway is fully discovered
                        self.discovered_doorways.append(((x, y), orientation))
                        self.discovered_doorway_set.add((x, y))
                        new_doorways_found = True

        return new_doorways_found

    def _reset_task(self):
        """Reset task-specific state."""
        self.target_doorway = None
        self.initial_distance = None
        self.previous_distance = None
        self.previous_pos = None
        self.has_passed_through = False
        self.doorway_orientation = None
        self.approach_side = None
        self.position_before_doorway = None

    def _select_target_doorway(self):
        """Select the nearest discovered doorway as target."""
        if not self.discovered_doorways:
            return

        drone_x, drone_y = self.env.drones[0].pos

        # Find nearest doorway using numba-optimized distance
        min_distance = float('inf')
        best_doorway = None

        for (door_x, door_y), orientation in self.discovered_doorways:
            distance = manhattan_distance(door_x, door_y, drone_x, drone_y)
            if distance < min_distance:
                min_distance = distance
                best_doorway = ((door_x, door_y), orientation)

        if best_doorway:
            self.target_doorway = best_doorway[0]
            self.doorway_orientation = best_doorway[1]
            self.initial_distance = min_distance
            self.previous_distance = min_distance

    def _determine_approach_side(self, drone_pos, doorway_pos):
        """Determine from which side the drone is approaching the doorway."""
        if self.doorway_orientation == 'horizontal':
            return 'above' if drone_pos[1] < doorway_pos[1] else 'below'
        else:
            return 'left' if drone_pos[0] < doorway_pos[0] else 'right'

    def _check_valid_pass_through(self, current_pos):
        """Check if the drone has validly passed through the doorway."""
        if not self.target_doorway or not self.position_before_doorway:
            return False

        dx, dy = self.target_doorway
        cx, cy = current_pos

        if self.doorway_orientation == 'horizontal':
            if self.approach_side == 'above':
                return cy > dy
            else:
                return cy < dy
        else:
            if self.approach_side == 'left':
                return cx > dx
            else:
                return cx < dx

    def _compute_task_reward(self, obs, action, base_reward) -> float:
        """Compute doorway entry specific reward with optimized logic."""
        drone_x, drone_y = obs['positions'][0]
        drone_pos = (drone_x, drone_y)

        # Check for newly discovered doorways (very fast with pre-computed data)
        if self._check_for_new_doorways():
            # New doorway found, maybe update target
            if not self.target_doorway:
                self._select_target_doorway()

        # Select target if we don't have one but have discovered doorways
        if not self.target_doorway and self.discovered_doorways:
            self._select_target_doorway()

        # Stage 1: Searching for door
        if not self.target_doorway:
            reward = 0.0
            # Check collision
            if self.previous_pos and self.previous_pos == drone_pos and action == 0:
                reward += self.collision_penalty

        # Stage 2 & 3: Door found
        else:
            reward = self.step_penalty

            # Check collision
            if self.previous_pos and self.previous_pos == drone_pos and action == 0:
                reward += self.collision_penalty

            # Check if on doorway
            on_doorway = (drone_x == self.target_doorway[0] and
                         drone_y == self.target_doorway[1])

            # Track approach side
            if on_doorway and self.previous_pos and self.previous_pos != self.target_doorway:
                self.position_before_doorway = self.previous_pos
                self.approach_side = self._determine_approach_side(self.previous_pos, self.target_doorway)

            # Check if moved away from doorway
            if (self.position_before_doorway and
                self.previous_pos == self.target_doorway and
                drone_pos != self.target_doorway):

                if self._check_valid_pass_through(drone_pos):
                    if not self.has_passed_through:
                        reward += self.entry_reward
                        self.has_passed_through = True
                else:
                    if drone_pos == self.position_before_doorway:
                        reward += self.wrong_direction_penalty

            # Distance-based reward
            if not self.has_passed_through and self.target_doorway:
                current_distance = manhattan_distance(
                    drone_x, drone_y,
                    self.target_doorway[0], self.target_doorway[1]
                )

                if self.previous_distance is not None:
                    if current_distance < self.previous_distance:
                        reward += self.approach_reward
                    elif current_distance > self.previous_distance:
                        reward -= self.approach_reward * 0.3

                self.previous_distance = current_distance

        self.previous_pos = drone_pos
        return reward

    def _check_task_status(self, obs, action) -> TaskStatus:
        """Check if the doorway entry task is complete."""
        if self.has_passed_through:
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

        # Add doorway specific visualization
        if self.env.screen is not None:
            import pygame

            drone_pos = self.env.drones[0].pos
            TILE_SIZE = 20  # Assuming standard tile size

            # Highlight all detected doorways on BOTH maps
            for map_offset in [0, self.env.width * TILE_SIZE + 50]:  # Left map and right map
                for (x, y), orientation in self.discovered_doorways:
                    # Different colors for different states
                    if (x, y) == self.target_doorway:
                        if self.has_passed_through:
                            color = (0, 255, 0)  # Green - completed
                        elif drone_pos == (x, y):
                            color = (255, 255, 0)  # Yellow - on doorway
                        else:
                            color = (0, 200, 255)  # Cyan - target
                    else:
                        color = (100, 100, 255)  # Blue - other doorways

                    # Draw doorway marker with thick border
                    rect = pygame.Rect(
                        map_offset + x * TILE_SIZE,
                        y * TILE_SIZE,
                        TILE_SIZE,
                        TILE_SIZE
                    )
                    pygame.draw.rect(self.env.screen, color, rect, 3)

                    # Fill the target doorway cell with transparent color
                    if (x, y) == self.target_doorway:
                        s = pygame.Surface((TILE_SIZE, TILE_SIZE))
                        s.set_alpha(50)
                        s.fill(color)
                        self.env.screen.blit(s, (map_offset + x * TILE_SIZE, y * TILE_SIZE))

                    # Draw orientation indicator
                    center_x = map_offset + x * TILE_SIZE + TILE_SIZE // 2
                    center_y = y * TILE_SIZE + TILE_SIZE // 2

                    if orientation == 'horizontal':
                        # Draw horizontal line
                        pygame.draw.line(
                            self.env.screen,
                            color,
                            (map_offset + x * TILE_SIZE + 2, center_y),
                            (map_offset + (x + 1) * TILE_SIZE - 2, center_y),
                            3
                        )
                    else:  # vertical
                        # Draw vertical line
                        pygame.draw.line(
                            self.env.screen,
                            color,
                            (center_x, y * TILE_SIZE + 2),
                            (center_x, (y + 1) * TILE_SIZE - 2),
                            3
                        )

            # Draw path from drone to target doorway (on observed map)
            if self.target_doorway and not self.has_passed_through:
                map_offset = self.env.width * TILE_SIZE + 50
                pygame.draw.line(
                    self.env.screen,
                    (255, 255, 0),
                    (map_offset + drone_pos[0] * TILE_SIZE + TILE_SIZE // 2,
                     drone_pos[1] * TILE_SIZE + TILE_SIZE // 2),
                    (map_offset + self.target_doorway[0] * TILE_SIZE + TILE_SIZE // 2,
                     self.target_doorway[1] * TILE_SIZE + TILE_SIZE // 2),
                    2
                )

            # Display task status
            if self.env.font:
                # Main status line
                if not self.target_doorway:
                    status_text = f"Stage 1: SEARCHING FOR DOOR - Steps: {self.task_step}"
                    color = (255, 255, 255)
                elif not self.has_passed_through:
                    distance = abs(drone_pos[0] - self.target_doorway[0]) + abs(drone_pos[1] - self.target_doorway[1])
                    status_text = f"Stage 2: APPROACHING DOOR - Steps: {self.task_step} | Distance: {distance}"
                    color = (255, 255, 0)
                else:
                    status_text = f"Stage 3: ✓ PASSED THROUGH! - Steps: {self.task_step}"
                    color = (0, 255, 0)

                text_surface = self.env.font.render(status_text, True, color)
                self.env.screen.blit(text_surface, (10, 10))

                # Door count
                door_text = f"Doorways found: {len(self.discovered_doorways)}"
                door_surface = self.env.font.render(door_text, True, (200, 200, 200))
                self.env.screen.blit(door_surface, (10, 30))

                # Show success or hints
                if self.has_passed_through:
                    success_text = "SUCCESS! MOVED THROUGH DOORWAY!"
                    success_surface = self.env.font.render(success_text, True, (0, 255, 0))
                    text_rect = success_surface.get_rect(center=(self.env.screen.get_width() // 2, 60))
                    self.env.screen.blit(success_surface, text_rect)
                elif drone_pos == self.target_doorway:
                    hint_text = f"On doorway! Move {self._get_pass_direction()} to pass through!"
                    hint_surface = self.env.font.render(hint_text, True, (255, 255, 0))
                    text_rect = hint_surface.get_rect(center=(self.env.screen.get_width() // 2, 60))
                    self.env.screen.blit(hint_surface, text_rect)

            pygame.display.flip()

        return base_render

    def _get_pass_direction(self):
        """Get the direction to move to pass through the doorway."""
        if not self.approach_side:
            return "forward"

        if self.doorway_orientation == 'horizontal':
            return "down" if self.approach_side == 'above' else "up"
        else:
            return "right" if self.approach_side == 'left' else "left"