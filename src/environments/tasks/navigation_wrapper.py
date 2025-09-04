"""
navigation_wrapper.py - Fixed NavigationWrapper with proper observation handling
"""

import numpy as np
from typing import Optional, Tuple, Set, Dict, Any
import pygame
from numba import njit
from gymnasium import spaces

from environments.tasks.base_task_wrapper import BaseTaskWrapper, TaskStatus
from environments.base.constants import TileType, TILE_SIZE, Action


@njit
def manhattan_distance(x1: int, y1: int, x2: int, y2: int) -> int:
    """Fast Manhattan distance calculation."""
    return abs(x1 - x2) + abs(y1 - y2)


class NavigationWrapper(BaseTaskWrapper):
    """
    Simplified navigation wrapper with only goal position in observations.
    Ensures goal_position is always present in observations.
    """

    def __init__(
        self,
        env_config: dict = None,
        # Exploration parameters
        exploration_steps: int = 20,
        # Task parameters
        max_steps_to_goal: int = 200,
        # Reward parameters
        goal_reached_reward: float = 100.0,
        time_penalty: float = 0.01,
    ):
        super().__init__(env_config)

        # Exploration
        self.exploration_steps = exploration_steps
        self.is_exploring = True

        # Task parameters
        self.max_steps_to_goal = max_steps_to_goal

        # Reward parameters
        self.goal_reached_reward = goal_reached_reward
        self.time_penalty = time_penalty

        # Task state
        self.goal_position: Optional[Tuple[int, int]] = None
        self.discovered_free_spaces = set()
        self.steps_since_goal = 0

        # Update observation space - ONLY add goal_position
        base_obs_space = self.env.observation_space

        self.observation_space = spaces.Dict({
            # Keep all existing spaces from base environment
            **base_obs_space.spaces,

            # Add ONLY goal position
            'goal_position': spaces.Box(
                low=-1,
                high=max(self.env.width, self.env.height),
                shape=(2,),
                dtype=np.int32
            ),
        })

    def reset(self, **kwargs):
        """Reset environment and run random exploration."""
        # Reset base environment
        obs, info = self.env.reset(**kwargs)

        # Reset task-specific state
        self.task_status = TaskStatus.IN_PROGRESS
        self.task_step = 0
        self._reset_task()

        # Run exploration and assign goal
        self._run_random_exploration()
        self._assign_goal()

        # IMPORTANT: Reset episode tracking after exploration
        # This makes the exploration not count towards episode length/collisions
        self.env.current_step = 0  # Reset step counter in base environment
        if hasattr(self.env, 'drones'):
            for drone in self.env.drones:
                drone.collision_count = 0  # Reset collision counter

        # Get observations with goal_position included
        obs = self._get_observations_with_goal(obs)

        return obs, info

    def _reset_task(self):
        """Reset navigation task state."""
        self.goal_position = None
        self.discovered_free_spaces = set()
        self.steps_since_goal = 0
        self.is_exploring = True

    def _run_random_exploration(self):
        """Run simple random walk for initial exploration."""
        # Random walk with forward bias
        action_probs = [0.6, 0.2, 0.2, 0.0]  # Forward, Left, Right, Stay

        for step in range(self.exploration_steps):
            # Update discovered positions
            global_map = self.env.global_map
            drone_pos = self.env.drones[0].pos

            # Quick scan around drone position
            view_range = 10
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
            self.env.step(actions)

        self.is_exploring = False

    def _assign_goal(self):
        """Assign goal from discovered positions."""
        if not self.discovered_free_spaces:
            # Fallback: scan entire visible map
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

        if candidates:
            self.goal_position = candidates[np.random.randint(len(candidates))]

    def step(self, action):
        """Execute action with navigation logic."""
        # Convert single action to multi-agent format
        if isinstance(action, (int, np.integer)):
            actions = np.array([action], dtype=np.int32)
        else:
            actions = np.array([action], dtype=np.int32)

        # Step base environment
        obs, base_reward, terminated, truncated, info = self.env.step(actions)

        # Add goal to observations
        obs = self._get_observations_with_goal(obs)

        # Skip task logic during exploration
        if self.is_exploring:
            return obs, 0.0, False, False, info

        # Compute task-specific reward
        task_reward = self._compute_task_reward(obs, action, base_reward)

        # Check task completion
        self.task_status = self._check_task_status(obs, action)

        # Update termination based on task
        if self.task_status == TaskStatus.SUCCESS:
            terminated = True
            info['task_success'] = True
        elif self.task_status == TaskStatus.FAILURE:
            truncated = True
            info['task_success'] = False

        self.task_step += 1
        self.steps_since_goal += 1

        # Add task info
        info['task_status'] = self.task_status.value
        info['task_step'] = self.task_step
        info['goal_position'] = self.goal_position
        info['steps_to_goal'] = self.steps_since_goal

        return obs, task_reward, terminated, truncated, info

    def _compute_task_reward(self, obs, action, base_reward) -> float:
        """
        Compute task-specific reward.

        Args:
            obs: Current observations (with goal_position)
            action: Action taken
            base_reward: Reward from base environment

        Returns:
            Task-specific reward
        """
        if self.is_exploring or self.goal_position is None:
            return 0.0

        current_pos = tuple(obs['positions'][0])

        # Goal reached
        if current_pos == self.goal_position:
            return self.goal_reached_reward

        # Otherwise just time penalty
        return -self.time_penalty

    def _check_task_status(self, obs, action) -> TaskStatus:
        """
        Check navigation task status.

        Args:
            obs: Current observations (with goal_position)
            action: Action taken

        Returns:
            Current task status
        """
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

    def _get_observations_with_goal(self, obs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Add goal position to observations.

        Args:
            obs: Base environment observations

        Returns:
            Observations with goal_position added
        """
        # Add goal position to the observations
        if self.goal_position is not None and not self.is_exploring:
            obs['goal_position'] = np.array(self.goal_position, dtype=np.int32)
        else:
            # During exploration or no goal set
            obs['goal_position'] = np.array([-1, -1], dtype=np.int32)

        return obs

    def _get_observations(self):
        """Get current observations with goal position."""
        obs = self.env._get_observations()
        return self._get_observations_with_goal(obs)

    def _get_info(self):
        """Get info with navigation details."""
        info = self.env._get_info()
        info.update({
            'goal_position': self.goal_position,
            'steps_to_goal': self.steps_since_goal,
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

            pygame.display.flip()

            if self.env.render_mode == 'rgb_array':
                return np.transpose(
                    np.array(pygame.surfarray.pixels3d(self.env.screen)),
                    axes=(1, 0, 2)
                )

        return rgb_array