"""
Simplified Navigation Wrapper with Random Walk Exploration

Key changes:
1. Simple random walk for 20 steps (no frontier computation)
2. Select goal from discovered free spaces
3. Minimal overhead for maximum speed
4. GOAL IS NOW EXPLICITLY INCLUDED IN OBSERVATIONS
"""

import numpy as np
from typing import Optional, Tuple, Set, Dict
import pygame
from numba import njit

from environments.tasks.base_task_wrapper import BaseTaskWrapper, TaskStatus
from environments.base.constants import TileType, TILE_SIZE, Action


@njit
def manhattan_distance(x1: int, y1: int, x2: int, y2: int) -> int:
    """Fast Manhattan distance calculation."""
    return abs(x1 - x2) + abs(y1 - y2)


class NavigationWrapper(BaseTaskWrapper):
    """
    Simplified navigation wrapper with random walk exploration.

    Task flow:
    1. First 20 steps: Random walk exploration
    2. Goal assignment from discovered locations
    3. Agent navigates to goal
    """

    def __init__(
        self,
        env_config: dict = None,
        # Exploration parameters
        exploration_steps: int = 20,
        # Task parameters
        max_steps_to_goal: int = 200,
        # Reward parameters
        goal_reached_reward: float = 200.0,
        closer_reward_scale: float = 1.0,
        farther_penalty_scale: float = 0.5,
        time_penalty: float = 0.01,
        collision_penalty: float = -1.0,
    ):
        super().__init__(env_config)

        # Exploration
        self.exploration_steps = exploration_steps
        self.is_exploring = True

        # Task parameters
        self.max_steps_to_goal = max_steps_to_goal

        # Reward parameters
        self.goal_reached_reward = goal_reached_reward
        self.closer_reward_scale = closer_reward_scale
        self.farther_penalty_scale = farther_penalty_scale
        self.time_penalty = time_penalty
        self.collision_penalty = collision_penalty

        # Task state
        self.goal_position: Optional[Tuple[int, int]] = None
        self.discovered_free_spaces = set()
        self.steps_since_goal = 0
        self.prev_distance_to_goal = None
        self.previous_pos = None

    def reset(self, **kwargs):
        """Reset environment and run random exploration."""
        obs, info = super().reset(**kwargs)

        # Reset task state
        self._reset_task()

        # Run random walk exploration
        obs, info = self._run_random_exploration()

        # Assign goal after exploration
        self._assign_goal()

        return obs, info

    def _reset_task(self):
        """Reset navigation task state."""
        self.goal_position = None
        self.discovered_free_spaces = set()
        self.steps_since_goal = 0
        self.prev_distance_to_goal = None
        self.previous_pos = None
        self.is_exploring = True

    def _run_random_exploration(self):
        """Run simple random walk for initial exploration."""
        # Random walk with forward bias for efficiency
        action_probs = [0.6, 0.2, 0.2, 0.0]  # Forward, Left, Right, Stay

        for step in range(self.exploration_steps):
            # Update discovered positions
            global_map = self.env.global_map
            drone_pos = self.env.drones[0].pos

            # Quick scan around drone position (more efficient than full map scan)
            view_range = 10  # Only check nearby area
            min_x = max(0, drone_pos[0] - view_range)
            max_x = min(self.env.width, drone_pos[0] + view_range + 1)
            min_y = max(0, drone_pos[1] - view_range)
            max_y = min(self.env.height, drone_pos[1] + view_range + 1)

            for y in range(min_y, max_y):
                for x in range(min_x, max_x):
                    if global_map[y, x] in [TileType.FREE_SPACE, TileType.DOOR_OPEN]:
                        self.discovered_free_spaces.add((x, y))

            # Random action
            action = np.random.choice(4, p=action_probs)
            actions = np.array([action], dtype=np.int32)
            obs, _, _, _, _ = self.env.step(actions)

        # Final scan
        global_map = self.env.global_map
        drone_pos = self.env.drones[0].pos
        for y in range(min_y, max_y):
            for x in range(min_x, max_x):
                if global_map[y, x] in [TileType.FREE_SPACE, TileType.DOOR_OPEN]:
                    self.discovered_free_spaces.add((x, y))

        self.is_exploring = False

        return self._get_observations(), self._get_info()

    def _assign_goal(self):
        """Assign goal from discovered positions."""
        if not self.discovered_free_spaces:
            # Fallback: if nothing discovered, scan entire visible map
            global_map = self.env.global_map
            for y in range(global_map.shape[0]):
                for x in range(global_map.shape[1]):
                    if global_map[y, x] in [TileType.FREE_SPACE, TileType.DOOR_OPEN]:
                        self.discovered_free_spaces.add((x, y))

        if not self.discovered_free_spaces:
            print("Warning: No free spaces discovered")
            return

        # Remove current position from candidates
        current_pos = tuple(self.env.drones[0].pos)
        candidates = list(self.discovered_free_spaces - {current_pos})

        if not candidates:
            candidates = list(self.discovered_free_spaces)

        if not candidates:
            return

        self.goal_position = candidates[np.random.randint(len(candidates))]

        # Initialize distance tracking
        if self.goal_position:
            self.prev_distance_to_goal = manhattan_distance(
                current_pos[0], current_pos[1],
                self.goal_position[0], self.goal_position[1]
            )

    def step(self, action):
        """Execute action with navigation logic."""
        # Skip if still exploring (shouldn't happen as exploration is in reset)
        if self.is_exploring:
            return super().step(action)

        # Store previous position for collision detection
        self.previous_pos = tuple(self.env.drones[0].pos)

        # Normal step
        obs, reward, terminated, truncated, info = super().step(action)

        # Add goal info
        info['goal_position'] = self.goal_position
        info['steps_to_goal'] = self.steps_since_goal
        if self.prev_distance_to_goal:
            info['distance_to_goal'] = self.prev_distance_to_goal

        return obs, reward, terminated, truncated, info

    def _compute_task_reward(self, obs, action, base_reward) -> float:
        """Compute navigation-specific reward."""
        # No reward during exploration
        if self.is_exploring or self.goal_position is None:
            return 0.0

        self.steps_since_goal += 1
        current_pos = tuple(obs['positions'][0])

        # Check if goal reached
        if current_pos == self.goal_position:
            return self.goal_reached_reward

        # Calculate current distance
        current_distance = manhattan_distance(
            current_pos[0], current_pos[1],
            self.goal_position[0], self.goal_position[1]
        )

        # Distance-based reward shaping
        reward = 0.0

        if self.prev_distance_to_goal is not None:
            distance_change = self.prev_distance_to_goal - current_distance

            if distance_change > 0:
                # Got closer
                reward = self.closer_reward_scale * distance_change
            elif distance_change < 0:
                # Got farther
                reward = -self.farther_penalty_scale * abs(distance_change)
            else:
                # Same distance - check if collision
                if self.previous_pos and current_pos == self.previous_pos and action == 0:
                    reward = self.collision_penalty
                else:
                    reward = -0.1  # Small penalty for turning

        # Update distance
        self.prev_distance_to_goal = current_distance

        # Time penalty
        reward -= self.time_penalty

        return reward

    def _check_task_status(self, obs, action) -> TaskStatus:
        """Check navigation task status."""
        if self.is_exploring or self.goal_position is None:
            return TaskStatus.IN_PROGRESS

        current_pos = tuple(obs['positions'][0])

        # Success: reached goal
        if current_pos == self.goal_position:
            return TaskStatus.SUCCESS

        # Failure: timeout
        if self.steps_since_goal > self.max_steps_to_goal:
            return TaskStatus.FAILURE

        return TaskStatus.IN_PROGRESS

    def _get_observations(self):
        """Get observations from base environment WITH GOAL INFORMATION."""
        obs = self.env._get_observations()

        # ADD GOAL POSITION TO OBSERVATIONS
        if self.goal_position is not None and not self.is_exploring:
            # Add absolute goal position
            obs['goal_position'] = np.array(self.goal_position, dtype=np.int32)

            # Add relative goal position (often more useful for RL)
            drone_pos = self.env.drones[0].pos
            obs['goal_relative'] = np.array([
                self.goal_position[0] - drone_pos[0],
                self.goal_position[1] - drone_pos[1]
            ], dtype=np.float32)

            # Add distance and angle to goal (polar coordinates)
            dx = self.goal_position[0] - drone_pos[0]
            dy = self.goal_position[1] - drone_pos[1]
            distance = np.sqrt(dx**2 + dy**2)
            angle = np.arctan2(dy, dx)
            obs['goal_distance'] = np.float32(distance)
            obs['goal_angle'] = np.float32(angle)

            # Add goal visibility flag (1 if goal is in the current view, 0 otherwise)
            if 'global_map' in obs:
                # Check if goal position is visible in the global map
                if self.env.global_map[self.goal_position[1], self.goal_position[0]] != TileType.UNKNOWN:
                    obs['goal_visible'] = np.int8(1)
                else:
                    obs['goal_visible'] = np.int8(0)
        else:
            # During exploration or no goal set
            obs['goal_position'] = np.array([-1, -1], dtype=np.int32)
            obs['goal_relative'] = np.array([0.0, 0.0], dtype=np.float32)
            obs['goal_distance'] = np.float32(-1.0)
            obs['goal_angle'] = np.float32(0.0)
            obs['goal_visible'] = np.int8(0)

        return obs

    def _get_info(self):
        """Get info with navigation details."""
        info = self.env._get_info()
        info.update({
            'goal_position': self.goal_position,
            'steps_to_goal': self.steps_since_goal,
            'distance_to_goal': self.prev_distance_to_goal if self.prev_distance_to_goal else -1,
            'discovered_positions': len(self.discovered_free_spaces),
            'is_exploring': self.is_exploring,
        })
        return info

    def render(self) -> Optional[np.ndarray]:
        """Render with goal visualization."""
        rgb_array = self.env.render()

        if self.goal_position and self.env.screen:
            # Draw goal on both maps
            for offset_x in [0, self.env.width * TILE_SIZE + 50]:
                goal_x = offset_x + self.goal_position[0] * TILE_SIZE + TILE_SIZE // 2
                goal_y = self.goal_position[1] * TILE_SIZE + TILE_SIZE // 2

                # Goal marker
                pygame.draw.circle(self.env.screen, (255, 0, 0), (goal_x, goal_y), 8, 2)
                pygame.draw.circle(self.env.screen, (255, 255, 255), (goal_x, goal_y), 4)

                # Goal text
                if self.env.font:
                    goal_text = self.env.font.render("GOAL", True, (255, 0, 0))
                    self.env.screen.blit(goal_text, (goal_x - 15, goal_y - 20))

            # Status text
            if self.env.font:
                if self.is_exploring:
                    status = f"EXPLORING... Step {self.task_step}/{self.exploration_steps}"
                    color = (255, 128, 0)
                else:
                    dist = self.prev_distance_to_goal if self.prev_distance_to_goal else 0
                    status = f"NAVIGATING - Steps: {self.steps_since_goal} | Distance: {dist}"
                    color = (255, 255, 255)

                text_surface = self.env.font.render(status, True, color)
                self.env.screen.blit(text_surface, (10, 10))

            pygame.display.flip()

            if self.env.render_mode == 'rgb_array':
                return np.transpose(
                    np.array(pygame.surfarray.pixels3d(self.env.screen)),
                    axes=(1, 0, 2)
                )

        return rgb_array