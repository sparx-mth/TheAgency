"""
Ultra-optimized Room Entry Environment Wrapper for SLAM

Key optimizations:
1. Doorways are pre-computed externally and passed as parameter
2. Use numpy operations for batch checking of doorway visibility
3. Cache global map as numpy array for fast access
4. Minimal object creation and function calls
5. Pre-allocated arrays for common operations
"""

import gymnasium as gym
import numpy as np
from typing import Dict, Tuple, Optional, List, Set
from numba import njit

from environments.tasks.base_task_wrapper import BaseTaskWrapper, TaskStatus
from environments.base.constants import TileType, Action, DIRECTION_DELTAS


# Numba-optimized functions for maximum speed
@njit
def manhattan_distance(x1: int, y1: int, x2: int, y2: int) -> int:
    """Fast Manhattan distance calculation."""
    return abs(x1 - x2) + abs(y1 - y2)


@njit
def find_frontiers_fast(global_map: np.ndarray, height: int, width: int) -> List[Tuple[int, int]]:
    """Fast frontier detection using numba."""
    frontiers = []
    for y in range(height):
        for x in range(width):
            if global_map[y, x] == TileType.UNKNOWN:
                # Check if adjacent to known free space
                if ((x > 0 and global_map[y, x-1] in (TileType.FREE_SPACE, TileType.DOOR_OPEN)) or
                    (x < width-1 and global_map[y, x+1] in (TileType.FREE_SPACE, TileType.DOOR_OPEN)) or
                    (y > 0 and global_map[y-1, x] in (TileType.FREE_SPACE, TileType.DOOR_OPEN)) or
                    (y < height-1 and global_map[y+1, x] in (TileType.FREE_SPACE, TileType.DOOR_OPEN))):
                    frontiers.append((x, y))
    return frontiers


class RoomEntryWrapper(BaseTaskWrapper):
    """
    Ultra-optimized environment wrapper for training doorway entry behavior.

    Doorways are pre-computed externally and passed as a parameter for maximum efficiency.
    """

    def __init__(
        self,
        env_config: Dict = None,
        precomputed_doorways: Dict[Tuple[int, int], str] = None,
        # Simple reward structure - 4 clear signals
        success_reward: float = 10.0,
        progress_reward: float = 0.5,
        collision_penalty: float = -0.5,
        step_penalty: float = -0.01,
        max_task_steps: int = 500,
        # Auto-exploration parameters stay the same
        auto_explore: bool = True,
        max_exploration_steps: int = 1000,
        min_doorways_to_discover: int = 1,
        exploration_strategy: str = "frontier",
    ):
        """Initialize the optimized doorway entry environment."""
        super().__init__(env_config)

        # Store pre-computed doorways
        if precomputed_doorways is None:
            raise ValueError("precomputed_doorways is required for optimized performance!")

        self.all_doorways = precomputed_doorways
        self.all_doorways_array = np.array(list(self.all_doorways.keys()), dtype=np.int32)  # For fast numpy ops
        self.doorway_orientations = [self.all_doorways[tuple(pos)] for pos in self.all_doorways_array]

        # Pre-allocate arrays for doorway checking
        self.doorway_visible = np.zeros(len(self.all_doorways), dtype=bool)
        self.discovered_doorway_indices = []

        # Reward parameters
        self.success_reward = success_reward
        self.progress_reward = progress_reward
        self.collision_penalty = collision_penalty
        self.step_penalty = step_penalty
        self.max_task_steps = max_task_steps

        # Auto-exploration parameters
        self.auto_explore = auto_explore
        self.max_exploration_steps = max_exploration_steps
        self.min_doorways_to_discover = min_doorways_to_discover
        self.exploration_strategy = exploration_strategy

        # Task state
        self.target_doorway = None
        self.target_doorway_idx = None
        self.initial_distance = None
        self.previous_distance = None
        self.previous_pos = None
        self.has_passed_through = False
        self.doorway_orientation = None
        self.approach_side = None
        self.position_before_doorway = None

        # Exploration state
        self.is_exploring = False
        self.exploration_steps = 0

        # Cache for fast access
        self.cached_global_map = None
        self.last_map_update_step = -1

    def reset(self, **kwargs):
        """Reset the environment."""
        obs, info = super().reset(**kwargs)

        # Reset discovery tracking
        self.doorway_visible.fill(False)
        self.discovered_doorway_indices = []
        self.cached_global_map = None
        self.last_map_update_step = -1

        # Reset exploration state
        self.is_exploring = self.auto_explore
        self.exploration_steps = 0

        # Run auto-exploration if enabled
        if self.auto_explore:
            obs, info = self._run_auto_exploration()

        return obs, info

    def _run_auto_exploration(self):
        """Run automatic exploration until enough doorways are discovered."""
        # print(f"Starting auto-exploration (max {self.max_exploration_steps} steps)...")

        for step in range(self.max_exploration_steps):
            # Fast doorway check
            self._fast_check_doorways()

            # Check if we've found enough doorways
            if len(self.discovered_doorway_indices) >= self.min_doorways_to_discover:
                # print(f"Found {len(self.discovered_doorway_indices)} doorways in {step} steps")
                self.is_exploring = False
                break

            # Choose exploration action
            if self.exploration_strategy == "frontier":
                action = self._frontier_exploration_action()
            else:
                action = self._random_walk_action()

            # Execute action
            actions = np.array([action], dtype=np.int32)
            obs, _, _, _, _ = self.env.step(actions)
            self.exploration_steps = step + 1

        # Final check
        self._fast_check_doorways()

        if len(self.discovered_doorway_indices) != 0:
            self._select_target_doorway_fast()

        self.is_exploring = False
        return self._get_observations(), self._get_info()

    def _fast_check_doorways(self):
        """Ultra-fast doorway discovery check using numpy operations."""
        # Cache the global map
        if self.cached_global_map is None or self.env.current_step > self.last_map_update_step:
            self.cached_global_map = self.env.global_map
            self.last_map_update_step = self.env.current_step

        global_map = self.cached_global_map

        # Check each doorway for visibility using vectorized operations where possible
        for i, ((x, y), orientation) in enumerate(zip(self.all_doorways_array, self.doorway_orientations)):
            # Skip if already discovered
            if self.doorway_visible[i]:
                continue

            # Fast visibility check
            if global_map[y, x] != TileType.UNKNOWN:
                visible = False

                if orientation == 'horizontal':
                    # Check left and right
                    if (x > 0 and global_map[y, x-1] != TileType.UNKNOWN and
                        x < self.env.width - 1 and global_map[y, x+1] != TileType.UNKNOWN):
                        visible = True
                else:  # vertical
                    # Check top and bottom
                    if (y > 0 and global_map[y-1, x] != TileType.UNKNOWN and
                        y < self.env.height - 1 and global_map[y+1, x] != TileType.UNKNOWN):
                        visible = True

                if visible:
                    self.doorway_visible[i] = True
                    self.discovered_doorway_indices.append(i)

    def _select_target_doorway_fast(self):
        """Select nearest doorway using pre-computed indices."""
        if not self.discovered_doorway_indices:
            return

        drone_x, drone_y = self.env.drones[0].pos

        min_distance = float('inf')
        best_idx = None

        for idx in self.discovered_doorway_indices:
            door_x, door_y = self.all_doorways_array[idx]
            distance = manhattan_distance(door_x, door_y, drone_x, drone_y)
            if distance < min_distance:
                min_distance = distance
                best_idx = idx

        if best_idx is not None:
            self.target_doorway_idx = best_idx
            self.target_doorway = tuple(self.all_doorways_array[best_idx])
            self.doorway_orientation = self.doorway_orientations[best_idx]
            self.initial_distance = min_distance
            self.previous_distance = min_distance

    def _frontier_exploration_action(self):
        """Optimized frontier exploration using numba."""
        drone = self.env.drones[0]
        drone_x, drone_y = drone.pos

        # Use numba-optimized frontier detection
        frontiers = find_frontiers_fast(self.cached_global_map, self.env.height, self.env.width)

        if not frontiers:
            return self._random_walk_action()

        # Find nearest frontier
        min_dist = float('inf')
        target = None
        for fx, fy in frontiers:
            dist = manhattan_distance(drone_x, drone_y, fx, fy)
            if dist < min_dist:
                min_dist = dist
                target = (fx, fy)

        if target:
            return self._move_towards_target(drone, target)

        return self._random_walk_action()

    def _move_towards_target(self, drone, target):
        """Choose action to move towards target."""
        dx = target[0] - drone.pos[0]
        dy = target[1] - drone.pos[1]

        facing_idx = drone.get_facing_idx()
        facing_deltas = [(0, -1), (1, 0), (0, 1), (-1, 0)]  # N, E, S, W
        current_dx, current_dy = facing_deltas[facing_idx]

        # Determine desired direction
        if abs(dx) > abs(dy):
            desired_dx = 1 if dx > 0 else -1
            desired_dy = 0
        else:
            desired_dx = 0
            desired_dy = 1 if dy > 0 else -1

        # If facing right direction, move forward
        if (current_dx, current_dy) == (desired_dx, desired_dy):
            return Action.FORWARD

        # Calculate turns
        desired_facing = facing_deltas.index((desired_dx, desired_dy))
        turns_right = (desired_facing - facing_idx) % 4

        # Choose shortest turn
        if turns_right <= 2:
            return Action.TURN_RIGHT
        else:
            return Action.TURN_LEFT

    def _random_walk_action(self):
        """Simple random walk with forward bias."""
        # Pre-computed probabilities
        return np.random.choice(4, p=[0.6, 0.2, 0.2, 0.0])

    def _reset_task(self):
        """Reset task-specific state."""
        self.target_doorway = None
        self.target_doorway_idx = None
        self.initial_distance = None
        self.previous_distance = None
        self.previous_pos = None
        self.has_passed_through = False
        self.doorway_orientation = None
        self.approach_side = None
        self.position_before_doorway = None

    def _determine_approach_side(self, drone_pos, doorway_pos):
        """Determine approach side."""
        if self.doorway_orientation == 'horizontal':
            return 'above' if drone_pos[1] < doorway_pos[1] else 'below'
        else:
            return 'left' if drone_pos[0] < doorway_pos[0] else 'right'

    def _check_valid_pass_through(self, current_pos):
        """Check valid pass through."""
        if not self.target_doorway or not self.position_before_doorway:
            return False

        dx, dy = self.target_doorway
        cx, cy = current_pos

        if self.doorway_orientation == 'horizontal':
            return (cy > dy) if self.approach_side == 'above' else (cy < dy)
        else:
            return (cx > dx) if self.approach_side == 'left' else (cx < dx)

    def _compute_task_reward(self, obs, action, base_reward) -> float:
        """Balanced reward with guidance but no exploitation."""
        drone_x, drone_y = obs['positions'][0]
        drone_pos = (drone_x, drone_y)

        # Base penalty for efficiency
        reward = self.step_penalty

        # Collision penalty
        if action == Action.FORWARD and self.previous_pos == drone_pos:
            reward += self.collision_penalty

        if self.target_doorway and not self.has_passed_through:
            current_distance = abs(drone_x - self.target_doorway[0]) + abs(drone_y - self.target_doorway[1])

            # Small progress reward, but only when getting closer (not for staying close)
            if self.previous_distance is not None:
                if current_distance < self.previous_distance:
                    reward += self.progress_reward
                # Optional: small penalty for moving away
                elif current_distance > self.previous_distance:
                    reward -= self.progress_reward * 0.5

            # Track doorway entry
            if drone_pos == self.target_doorway and self.previous_pos != self.target_doorway:
                self.position_before_doorway = self.previous_pos
                # Small one-time bonus for reaching doorway (not repeatable)
                if not hasattr(self, 'reached_doorway'):
                    reward += 1.0  # Small bonus for first contact
                    self.reached_doorway = True

            # Success check
            elif (self.position_before_doorway and
                  self.previous_pos == self.target_doorway and
                  drone_pos != self.target_doorway and
                  drone_pos != self.position_before_doorway):

                self.has_passed_through = True
                reward += self.success_reward

            self.previous_distance = current_distance

        self.previous_pos = drone_pos
        return reward

    def _check_task_status(self, obs, action) -> TaskStatus:
        """Check task completion status."""
        if self.has_passed_through:
            return TaskStatus.SUCCESS
        if self.task_step >= self.max_task_steps:
            return TaskStatus.FAILURE
        return TaskStatus.IN_PROGRESS

    def _get_observations(self):
        """Get observations from base environment."""
        return self.env._get_observations()

    def _get_info(self):
        """Get info with task details."""
        info = self.env._get_info()
        info.update({
            'discovered_doorways': len(self.discovered_doorway_indices),
            'target_doorway': self.target_doorway,
            'has_passed_through': self.has_passed_through,
            'exploration_steps': self.exploration_steps,
            'is_exploring': self.is_exploring,
        })
        return info

    def render(self) -> Optional[np.ndarray]:
        """Render the environment with doorway visualization."""
        if self.env.render_mode is None:
            return None

        base_render = self.env.render()

        if self.env.screen is not None:
            import pygame

            drone_pos = self.env.drones[0].pos
            TILE_SIZE = 20

            # Draw discovered doorways
            for map_offset in [0, self.env.width * TILE_SIZE + 50]:
                for idx in self.discovered_doorway_indices:
                    x, y = self.all_doorways_array[idx]
                    orientation = self.doorway_orientations[idx]

                    # Color based on state
                    if (x, y) == self.target_doorway:
                        if self.has_passed_through:
                            color = (0, 255, 0)  # Green
                        elif drone_pos == (x, y):
                            color = (255, 255, 0)  # Yellow
                        else:
                            color = (0, 200, 255)  # Cyan
                    else:
                        color = (100, 100, 255)  # Blue

                    # Draw doorway marker
                    rect = pygame.Rect(
                        map_offset + x * TILE_SIZE,
                        y * TILE_SIZE,
                        TILE_SIZE,
                        TILE_SIZE
                    )
                    pygame.draw.rect(self.env.screen, color, rect, 3)

                    # Fill target doorway
                    if (x, y) == self.target_doorway:
                        s = pygame.Surface((TILE_SIZE, TILE_SIZE))
                        s.set_alpha(50)
                        s.fill(color)
                        self.env.screen.blit(s, (map_offset + x * TILE_SIZE, y * TILE_SIZE))

                    # Draw orientation indicator
                    center_x = map_offset + x * TILE_SIZE + TILE_SIZE // 2
                    center_y = y * TILE_SIZE + TILE_SIZE // 2

                    if orientation == 'horizontal':
                        pygame.draw.line(
                            self.env.screen, color,
                            (map_offset + x * TILE_SIZE + 2, center_y),
                            (map_offset + (x + 1) * TILE_SIZE - 2, center_y),
                            3
                        )
                    else:
                        pygame.draw.line(
                            self.env.screen, color,
                            (center_x, y * TILE_SIZE + 2),
                            (center_x, (y + 1) * TILE_SIZE - 2),
                            3
                        )

            # Draw path to target
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

            # Display status
            if self.env.font:
                if self.is_exploring:
                    text = f"AUTO-EXPLORING... Steps: {self.exploration_steps}/{self.max_exploration_steps}"
                    surface = self.env.font.render(text, True, (255, 128, 0))
                    self.env.screen.blit(surface, (10, 10))
                else:
                    if not self.target_doorway:
                        status = f"Stage 1: SEARCHING - Steps: {self.task_step}"
                        color = (255, 255, 255)
                    elif not self.has_passed_through:
                        dist = abs(drone_pos[0] - self.target_doorway[0]) + abs(drone_pos[1] - self.target_doorway[1])
                        status = f"Stage 2: APPROACHING - Steps: {self.task_step} | Dist: {dist}"
                        color = (255, 255, 0)
                    else:
                        status = f"Stage 3: ✓ COMPLETE - Steps: {self.task_step}"
                        color = (0, 255, 0)

                    surface = self.env.font.render(status, True, color)
                    self.env.screen.blit(surface, (10, 10))

                door_text = f"Doorways: {len(self.discovered_doorway_indices)} (Explored: {self.exploration_steps})"
                door_surface = self.env.font.render(door_text, True, (200, 200, 200))
                self.env.screen.blit(door_surface, (10, 30))

            pygame.display.flip()

        return base_render

    def _get_pass_direction(self):
        """Get pass direction hint."""
        if not self.approach_side:
            return "forward"
        if self.doorway_orientation == 'horizontal':
            return "down" if self.approach_side == 'above' else "up"
        else:
            return "right" if self.approach_side == 'left' else "left"