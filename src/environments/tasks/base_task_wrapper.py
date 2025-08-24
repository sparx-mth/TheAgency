"""
Task-specific environment wrappers for hierarchical agent training.

This module provides:
1. BaseTaskWrapper - Abstract base class for all task-specific wrappers
2. WallFollowingWrapper - Environment for training wall-following behavior
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import Dict, Tuple, Optional, Set
from abc import ABC, abstractmethod
from enum import IntEnum

from environments.base.slam_env import MultiAgentSLAMEnv
from environments.base.constants import Action


class TaskStatus(IntEnum):
    """Status codes for task completion."""
    IN_PROGRESS = 0
    SUCCESS = 1
    FAILURE = 2


class BaseTaskWrapper(gym.Wrapper, ABC):
    """
    Abstract base class for task-specific environment wrappers.

    All task wrappers maintain the same observation space but modify:
    - Reward function
    - Success/failure conditions
    - Episode termination criteria
    """

    def __init__(self, env_config: Dict = None):
        """
        Initialize the task wrapper.

        Args:
            env_config: Configuration for the base environment
        """
        # Default single-agent configuration
        default_config = {
            'width': 32,
            'height': 32,
            'num_agents': 1,  # Single agent
            'max_steps': 500,
            'randomize': True,
        }

        if env_config:
            default_config.update(env_config)

        # Create base environment
        base_env = MultiAgentSLAMEnv(**default_config)
        super().__init__(base_env)

        # Convert to single-agent action space
        self.action_space = spaces.Discrete(len(Action))

        # Task-specific state
        self.task_status = TaskStatus.IN_PROGRESS
        self.task_step = 0

    def reset(self, **kwargs):
        """Reset environment and task-specific state."""
        obs, info = self.env.reset(**kwargs)
        self.task_status = TaskStatus.IN_PROGRESS
        self.task_step = 0
        self._reset_task()
        return self._process_observation(obs), info

    def step(self, action):
        """Execute action and compute task-specific rewards."""
        # Convert single action to multi-agent format
        actions = np.array([action])

        # Step base environment
        obs, base_reward, terminated, truncated, info = self.env.step(actions)

        # Compute task-specific reward
        task_reward = self._compute_task_reward(obs, action, base_reward)

        # Check task completion
        self.task_status = self._check_task_status(obs, action)

        # Update termination based on task
        if self.task_status != TaskStatus.IN_PROGRESS:
            terminated = True

        self.task_step += 1

        # Add task info
        info['task_status'] = self.task_status
        info['task_step'] = self.task_step

        return self._process_observation(obs), task_reward, terminated, truncated, info

    def _process_observation(self, obs):
        """Process multi-agent observation to single-agent format."""
        # Extract single drone's observation
        return {
            'global_map': obs['global_map'],
            'position': obs['positions'][0],
            'facing': obs['facings'][0],
        }

    @abstractmethod
    def _reset_task(self):
        """Reset task-specific state."""
        pass

    @abstractmethod
    def _compute_task_reward(self, obs, action, base_reward) -> float:
        """Compute task-specific reward."""
        pass

    @abstractmethod
    def _check_task_status(self, obs, action) -> TaskStatus:
        """Check if task is complete."""
        pass