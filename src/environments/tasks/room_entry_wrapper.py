"""
Room Entry Environment Wrapper for SLAM

This wrapper trains an agent to identify and pass through doorways.
A doorway is defined as a passage (0) between two walls (1) in a straight line.
The agent must step ON the doorway and then move THROUGH it to the other side.
"""

import gymnasium as gym
import numpy as np
import pygame
from typing import Dict, Tuple, Optional, List, Set

from environments.tasks.base_task_wrapper import BaseTaskWrapper, TaskStatus
from environments.base.constants import TileType, TILE_SIZE, TILE_COLORS


class RoomEntryWrapper(BaseTaskWrapper):
    """
    Environment wrapper for training doorway entry behavior.

    The agent must:
    1. Identify doorways in the discovered map (1-0-1 patterns)
    2. Navigate to the nearest doorway
    3. Step ON the doorway position
    4. Move THROUGH to the OTHER side (not back where it came from)
    """

    def __init__(
        self,
        env_config: Dict = None,
        # Reward parameters
        entry_reward: float = 10.0,
        approach_reward: float = 0.5,
        wrong_direction_penalty: float = -2.0,
        collision_penalty: float = -1.0,
        step_penalty: float = -0.01,
        max_task_steps: int = 200,
    ):
        """
        Initialize the doorway entry environment.

        Args:
            env_config: Configuration for base environment
            entry_reward: Reward for successfully passing through doorway
            approach_reward: Reward for getting closer to target doorway
            wrong_direction_penalty: Penalty for going back through doorway
            collision_penalty: Penalty for collisions
            step_penalty: Small penalty per step (only after door is found)
            max_task_steps: Maximum steps for the task
        """
        super().__init__(env_config)

        # Reward parameters
        self.entry_reward = entry_reward
        self.approach_reward = approach_reward
        self.wrong_direction_penalty = wrong_direction_penalty
        self.collision_penalty = collision_penalty
        self.step_penalty = step_penalty
        self.max_task_steps = max_task_steps

        # Task state
        self.doorways = []  # List of detected doorways
        self.target_doorway = None  # (x, y) of target doorway center
        self.initial_distance = None
        self.previous_distance = None
        self.previous_pos = None
        self.has_passed_through = False
        self.doorway_orientation = None
        self.approach_side = None  # Side from which we approached the doorway
        self.position_before_doorway = None  # Position just before stepping on doorway

    def _reset_task(self):
        """Reset task-specific state."""
        self.doorways = []
        self.target_doorway = None
        self.initial_distance = None
        self.previous_distance = None
        self.previous_pos = None
        self.has_passed_through = False
        self.doorway_orientation = None
        self.approach_side = None
        self.position_before_doorway = None

    def _find_doorways(self):
        """Find all doorways (1-0-1 patterns) in the discovered map."""
        self.doorways = []
        global_map = self.env.global_map

        # Check horizontal doorways (same Y coordinate)
        for y in range(self.env.height):
            for x in range(1, self.env.width - 1):
                # Check if we have discovered enough to see a doorway pattern
                center = global_map[y, x]
                left = global_map[y, x-1]
                right = global_map[y, x+1]

                # All three must be discovered (not -1)
                if center == TileType.UNKNOWN or left == TileType.UNKNOWN or right == TileType.UNKNOWN:
                    continue

                # Check for doorway pattern: wall-passage-wall
                if (left == TileType.WALL and
                    center in [TileType.FREE_SPACE, TileType.DOOR_OPEN] and
                    right == TileType.WALL):
                    self.doorways.append(((x, y), 'horizontal'))

        # Check vertical doorways (same X coordinate)
        for x in range(self.env.width):
            for y in range(1, self.env.height - 1):
                # Check if we have discovered enough to see a doorway pattern
                center = global_map[y, x]
                top = global_map[y-1, x]
                bottom = global_map[y+1, x]

                # All three must be discovered (not -1)
                if center == TileType.UNKNOWN or top == TileType.UNKNOWN or bottom == TileType.UNKNOWN:
                    continue

                # Check for doorway pattern: wall-passage-wall
                if (top == TileType.WALL and
                    center in [TileType.FREE_SPACE, TileType.DOOR_OPEN] and
                    bottom == TileType.WALL):
                    # Avoid duplicate if already added as horizontal
                    if ((x, y), 'horizontal') not in self.doorways:
                        self.doorways.append(((x, y), 'vertical'))

    def _select_target_doorway(self):
        """Select the nearest doorway as target."""
        if not self.doorways:
            return

        drone_pos = self.env.drones[0].pos

        # Find nearest doorway
        min_distance = float('inf')
        best_doorway = None

        for doorway_pos, orientation in self.doorways:
            distance = abs(doorway_pos[0] - drone_pos[0]) + abs(doorway_pos[1] - drone_pos[1])
            if distance < min_distance:
                min_distance = distance
                best_doorway = (doorway_pos, orientation)

        if best_doorway:
            self.target_doorway = best_doorway[0]
            self.doorway_orientation = best_doorway[1]
            self.initial_distance = min_distance
            self.previous_distance = min_distance
            print(f"Target doorway selected at {self.target_doorway}, orientation: {self.doorway_orientation}")

    def _determine_approach_side(self, drone_pos, doorway_pos):
        """Determine from which side the drone is approaching the doorway."""
        if self.doorway_orientation == 'horizontal':
            # Doorway runs left-right, check if drone is above or below
            return 'above' if drone_pos[1] < doorway_pos[1] else 'below'
        else:  # vertical
            # Doorway runs up-down, check if drone is left or right
            return 'left' if drone_pos[0] < doorway_pos[0] else 'right'

    def _check_valid_pass_through(self, current_pos):
        """Check if the drone has validly passed through the doorway."""
        if not self.target_doorway or not self.position_before_doorway:
            return False

        # Check if we're on the opposite side from where we approached
        if self.doorway_orientation == 'horizontal':
            # For horizontal doorway, check Y position
            if self.approach_side == 'above':
                # Should now be below the doorway
                return current_pos[1] > self.target_doorway[1]
            else:  # approached from below
                # Should now be above the doorway
                return current_pos[1] < self.target_doorway[1]
        else:  # vertical
            # For vertical doorway, check X position
            if self.approach_side == 'left':
                # Should now be right of the doorway
                return current_pos[0] > self.target_doorway[0]
            else:  # approached from right
                # Should now be left of the doorway
                return current_pos[0] < self.target_doorway[0]

    def _compute_task_reward(self, obs, action, base_reward) -> float:
        """Compute doorway entry specific reward."""
        drone_pos = tuple(obs['positions'][0])

        # Update doorway detection every step
        self._find_doorways()
        if not self.target_doorway and self.doorways:
            self._select_target_doorway()

        # Stage 1: Searching for door - no step penalty until door is found
        if not self.target_doorway:
            reward = 0.0  # No reward or penalty while searching

            # Only apply collision penalty
            if self.previous_pos and self.previous_pos == drone_pos and action == 0:  # FORWARD
                reward += self.collision_penalty

        # Stage 2 & 3: Door found - apply rewards/penalties
        else:
            reward = self.step_penalty  # Apply step penalty once door is found

            # Check for collision
            if self.previous_pos and self.previous_pos == drone_pos and action == 0:  # FORWARD
                reward += self.collision_penalty

            # Check if on doorway
            on_doorway = (drone_pos[0] == self.target_doorway[0] and
                         drone_pos[1] == self.target_doorway[1])

            # Track approach side when stepping on doorway
            if on_doorway and self.previous_pos and self.previous_pos != self.target_doorway:
                self.position_before_doorway = self.previous_pos
                self.approach_side = self._determine_approach_side(self.previous_pos, self.target_doorway)
                print(f"Stepped on doorway from {self.approach_side} side")

            # Check if moved away from doorway
            if self.position_before_doorway and self.previous_pos == self.target_doorway and drone_pos != self.target_doorway:
                # We just moved off the doorway
                if self._check_valid_pass_through(drone_pos):
                    # Successfully passed through to the other side
                    if not self.has_passed_through:
                        reward += self.entry_reward
                        self.has_passed_through = True
                        print(f"SUCCESS! Passed through doorway to the other side!")
                else:
                    # Went back where we came from
                    if drone_pos == self.position_before_doorway:
                        reward += self.wrong_direction_penalty
                        print(f"Wrong direction! Went back to where you came from!")

            # Distance-based reward (only if haven't completed task)
            if not self.has_passed_through:
                current_distance = abs(drone_pos[0] - self.target_doorway[0]) + abs(drone_pos[1] - self.target_doorway[1])

                # Reward for getting closer to target
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
        # Success: passed through the doorway
        if self.has_passed_through:
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

        # Add doorway specific visualization
        if self.env.screen is not None:
            drone_pos = self.env.drones[0].pos

            # Highlight all detected doorways on BOTH maps
            for map_offset in [0, self.env.width * TILE_SIZE + 50]:  # Left map and right map
                for (x, y), orientation in self.doorways:
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
                        # Draw horizontal line with arrows
                        pygame.draw.line(
                            self.env.screen,
                            color,
                            (map_offset + x * TILE_SIZE + 2, center_y),
                            (map_offset + (x + 1) * TILE_SIZE - 2, center_y),
                            3
                        )
                    else:  # vertical
                        # Draw vertical line with arrows
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
                door_text = f"Doorways found: {len(self.doorways)}"
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