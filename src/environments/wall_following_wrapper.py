"""
wall_following_wrapper.py

Environment wrapper for training a Wall-Following agent.
This wrapper modifies the base SLAM environment to focus on wall-following behavior.
"""

import gymnasium as gym
import numpy as np
from typing import Dict, Tuple, Optional, Any
from enum import Enum

from gymnasium import spaces

from environments.constants import TileType, DIRECTION_DELTAS


class WallFollowingState(Enum):
    """States for wall-following task"""
    SEARCHING_WALL = 0  # Looking for a wall to follow
    FOLLOWING_WALL = 1  # Currently following a wall
    TASK_COMPLETE = 2  # Reached corner/gap/end


class WallFollowingWrapper(gym.Wrapper):
    """
    Wrapper that transforms the SLAM environment for wall-following training.

    The agent succeeds when it:
    - Finds a wall and follows it
    - Reaches a corner, gap, or end of wall

    Rewards are structured to encourage:
    1. Finding a wall quickly
    2. Maintaining contact with the wall
    3. Following the wall smoothly
    4. Successfully detecting termination conditions
    """

    def __init__(
            self,
            env,
            # Reward parameters
            wall_found_reward: float = 1.0,
            wall_following_reward: float = 0.1,
            wall_lost_penalty: float = -0.5,
            corner_detected_reward: float = 5.0,
            gap_detected_reward: float = 5.0,
            wall_end_reward: float = 5.0,
            step_penalty: float = -0.01,
            collision_penalty: float = -0.5,
            # Task parameters
            max_steps_without_wall: int = 50,
            min_follow_steps: int = 5,
            wall_detection_range: int = 1,  # How close to wall to consider "following"
    ):
        super().__init__(env)

        # Use single agent mode
        if env.num_agents != 1:
            raise ValueError("WallFollowingWrapper requires single agent environment")

        # Reward parameters
        self.wall_found_reward = wall_found_reward
        self.wall_following_reward = wall_following_reward
        self.wall_lost_penalty = wall_lost_penalty
        self.corner_detected_reward = corner_detected_reward
        self.gap_detected_reward = gap_detected_reward
        self.wall_end_reward = wall_end_reward
        self.step_penalty = step_penalty
        self.collision_penalty = collision_penalty

        # Task parameters
        self.max_steps_without_wall = max_steps_without_wall
        self.min_follow_steps = min_follow_steps
        self.wall_detection_range = wall_detection_range

        # State tracking
        self.state = WallFollowingState.SEARCHING_WALL
        self.steps_without_wall = 0
        self.steps_following_wall = 0
        self.wall_side = None  # 'left' or 'right'
        self.last_wall_relative_pos = None
        self.task_completed = False
        self.last_position = None
        self.collision_occurred = False

        # Modify observation space to include wall-following specific features
        self.observation_space = spaces.Dict({
            'global_map': env.observation_space['global_map'],
            'position': spaces.Box(low=0, high=max(env.width, env.height), shape=(2,), dtype=np.int32),
            'facing': spaces.Box(low=0, high=3, shape=(1,), dtype=np.int32),
            # Wall-following specific observations
            'wall_left': spaces.Box(low=0, high=1, shape=(1,), dtype=np.int8),
            'wall_right': spaces.Box(low=0, high=1, shape=(1,), dtype=np.int8),
            'wall_front': spaces.Box(low=0, high=1, shape=(1,), dtype=np.int8),
            'wall_detected': spaces.Box(low=0, high=1, shape=(1,), dtype=np.int8),
            'following_wall': spaces.Box(low=0, high=1, shape=(1,), dtype=np.int8),
            'steps_following': spaces.Box(low=0, high=1000, shape=(1,), dtype=np.int32),
        })

    def reset(self, **kwargs) -> Tuple[Dict, Dict]:
        """Reset the environment and wall-following state"""
        obs, info = self.env.reset(**kwargs)

        # Reset task state
        self.state = WallFollowingState.SEARCHING_WALL
        self.steps_without_wall = 0
        self.steps_following_wall = 0
        self.wall_side = None
        self.last_wall_relative_pos = None
        self.task_completed = False
        self.last_position = None
        self.collision_occurred = False

        # Convert observation
        wall_obs = self._get_wall_following_observation(obs)

        return wall_obs, info

    def step(self, action: int) -> Tuple[Dict, float, bool, bool, Dict]:
        """Execute action and compute wall-following specific rewards"""
        # Store previous position
        drone = self.env.drones[0]
        self.last_position = drone.pos
        prev_collision_count = drone.collision_count

        # Execute action in base environment
        obs, base_reward, terminated, truncated, info = self.env.step(np.array([action]))

        # Check if collision occurred
        self.collision_occurred = drone.collision_count > prev_collision_count

        # Get wall detection info
        wall_info = self._detect_walls(obs)

        # Compute wall-following reward
        wall_reward = self._compute_wall_following_reward(wall_info)

        # Check task completion
        if self.state == WallFollowingState.FOLLOWING_WALL:
            termination_type = self._check_termination_conditions(wall_info)
            if termination_type:
                self.task_completed = True
                terminated = True
                # Add completion bonus
                if termination_type == 'corner':
                    wall_reward += self.corner_detected_reward
                elif termination_type == 'gap':
                    wall_reward += self.gap_detected_reward
                elif termination_type == 'end':
                    wall_reward += self.wall_end_reward

                info['termination_type'] = termination_type

        # Check failure conditions
        if self.state == WallFollowingState.SEARCHING_WALL:
            self.steps_without_wall += 1
            if self.steps_without_wall >= self.max_steps_without_wall:
                truncated = True
                info['failure_reason'] = 'timeout_finding_wall'

        # Combine rewards (ignore base environment's discovery rewards for this task)
        total_reward = wall_reward + self.step_penalty
        if self.collision_occurred:
            total_reward += self.collision_penalty

        # Get wall-following observation
        wall_obs = self._get_wall_following_observation(obs)

        # Update info
        info.update({
            'wall_following_state': self.state.name,
            'steps_following_wall': self.steps_following_wall,
            'wall_side': self.wall_side,
            'task_completed': self.task_completed,
        })

        return wall_obs, total_reward, terminated, truncated, info

    def _detect_walls(self, obs: Dict) -> Dict:
        """Detect walls around the drone"""
        global_map = obs['global_map']
        pos = obs['positions'][0]
        facing_idx = obs['facings'][0]

        x, y = pos

        # Get relative directions
        directions = ['NORTH', 'EAST', 'SOUTH', 'WEST']
        facing = directions[facing_idx]

        # Check walls in relative directions
        left_dir = directions[(facing_idx - 1) % 4]
        right_dir = directions[(facing_idx + 1) % 4]

        wall_info = {
            'left': self._is_wall_in_direction(x, y, left_dir, global_map),
            'right': self._is_wall_in_direction(x, y, right_dir, global_map),
            'front': self._is_wall_in_direction(x, y, facing, global_map),
            'behind': self._is_wall_in_direction(x, y, directions[(facing_idx + 2) % 4], global_map),
        }

        # Check for corners (wall on two adjacent sides)
        wall_info['corner'] = (
                (wall_info['left'] and wall_info['front']) or
                (wall_info['right'] and wall_info['front']) or
                (wall_info['left'] and wall_info['behind']) or
                (wall_info['right'] and wall_info['behind'])
        )

        # Check for gaps (wall suddenly disappears while following)
        wall_info['gap'] = False
        if self.state == WallFollowingState.FOLLOWING_WALL and self.wall_side:
            if self.wall_side == 'left' and not wall_info['left']:
                wall_info['gap'] = True
            elif self.wall_side == 'right' and not wall_info['right']:
                wall_info['gap'] = True

        return wall_info

    def _is_wall_in_direction(self, x: int, y: int, direction: str, global_map: np.ndarray) -> bool:
        """Check if there's a wall in the given direction within detection range"""
        dx, dy = DIRECTION_DELTAS[direction]

        for dist in range(1, self.wall_detection_range + 1):
            check_x = x + dx * dist
            check_y = y + dy * dist

            # Check bounds
            if not (0 <= check_x < global_map.shape[1] and 0 <= check_y < global_map.shape[0]):
                return True  # Out of bounds counts as wall

            # Check if it's a wall
            tile = global_map[check_y, check_x]
            if tile in {TileType.WALL, TileType.DOOR_CLOSED}:
                return True
            elif tile != TileType.UNKNOWN:
                # If we can see through, no wall in this direction
                return False

        return False

    def _compute_wall_following_reward(self, wall_info: Dict) -> float:
        """Compute reward based on wall-following behavior"""
        reward = 0.0

        if self.state == WallFollowingState.SEARCHING_WALL:
            # Looking for a wall
            if wall_info['left'] or wall_info['right']:
                # Found a wall!
                self.state = WallFollowingState.FOLLOWING_WALL
                self.wall_side = 'left' if wall_info['left'] else 'right'
                self.steps_following_wall = 0
                reward += self.wall_found_reward

        elif self.state == WallFollowingState.FOLLOWING_WALL:
            # Following a wall
            self.steps_following_wall += 1

            # Check if still following the wall
            if self.wall_side == 'left' and wall_info['left']:
                reward += self.wall_following_reward
            elif self.wall_side == 'right' and wall_info['right']:
                reward += self.wall_following_reward
            else:
                # Lost the wall (but might be a gap/corner)
                if not wall_info['gap'] and not wall_info['corner']:
                    reward += self.wall_lost_penalty

        return reward

    def _check_termination_conditions(self, wall_info: Dict) -> Optional[str]:
        """Check if task completion conditions are met"""
        # Need minimum steps following before can complete
        if self.steps_following_wall < self.min_follow_steps:
            return None

        # Check for corner
        if wall_info['corner']:
            return 'corner'

        # Check for gap/opening
        if wall_info['gap']:
            return 'gap'

        # Check for wall end (wall in front while following)
        if self.wall_side and wall_info['front']:
            # Wall blocks the path while following
            return 'end'

        return None

    def _get_wall_following_observation(self, base_obs: Dict) -> Dict:
        """Convert base observation to wall-following specific observation"""
        wall_info = self._detect_walls(base_obs)

        return {
            'global_map': base_obs['global_map'],
            'position': base_obs['positions'][0],
            'facing': base_obs['facings'][0:1],
            'wall_left': np.array([1 if wall_info['left'] else 0], dtype=np.int8),
            'wall_right': np.array([1 if wall_info['right'] else 0], dtype=np.int8),
            'wall_front': np.array([1 if wall_info['front'] else 0], dtype=np.int8),
            'wall_detected': np.array([1 if (wall_info['left'] or wall_info['right']) else 0], dtype=np.int8),
            'following_wall': np.array([1 if self.state == WallFollowingState.FOLLOWING_WALL else 0], dtype=np.int8),
            'steps_following': np.array([self.steps_following_wall], dtype=np.int32),
        }