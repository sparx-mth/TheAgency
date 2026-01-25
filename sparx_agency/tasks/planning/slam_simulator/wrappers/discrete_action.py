"""
Converts MultiDiscrete action space to Discrete for DQN compatibility.

Example:
    env = SLAMEnv(num_agents=3)  # MultiDiscrete([4, 4, 4])
    env = DiscreteActionWrapper(env)  # Discrete(64)
"""

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from typing import Tuple, Dict, Any, Optional


class DiscreteActionWrapper(gym.Wrapper):
    """
    Converts MultiDiscrete([n, n, ..., n]) to Discrete(n^k).

    Each discrete action index maps to a unique combination of agent actions.
    Action 0 = [0,0,0], Action 1 = [0,0,1], ..., Action 63 = [3,3,3]
    """

    def __init__(self, env: gym.Env):
        super().__init__(env)

        if not isinstance(env.action_space, spaces.MultiDiscrete):
            raise ValueError(f"Expected MultiDiscrete action space, got {type(env.action_space)}")

        self.original_nvec = env.action_space.nvec
        self.num_agents = len(self.original_nvec)

        if not all(n == self.original_nvec[0] for n in self.original_nvec):
            raise ValueError(f"All agents must have same action count. Got: {self.original_nvec}")

        self.actions_per_agent = int(self.original_nvec[0])
        self.total_actions = self.actions_per_agent ** self.num_agents

        self.action_space = spaces.Discrete(self.total_actions)

        # Pre-compute all action combinations
        self._combinations = self._build_combinations()

    def _build_combinations(self) -> np.ndarray:
        """Pre-compute all action combinations for fast lookup."""
        combos = np.zeros((self.total_actions, self.num_agents), dtype=np.int32)
        for i in range(self.total_actions):
            remaining = i
            for j in range(self.num_agents):
                power = self.actions_per_agent ** (self.num_agents - 1 - j)
                combos[i, j] = remaining // power
                remaining %= power
        return combos

    def decode(self, action: int) -> np.ndarray:
        """Convert discrete action to multi-agent actions."""
        return self._combinations[action].copy()

    def encode(self, multi_action: np.ndarray) -> int:
        """Convert multi-agent actions to discrete action."""
        action = 0
        for i, a in enumerate(multi_action):
            power = self.actions_per_agent ** (self.num_agents - 1 - i)
            action += int(a) * power
        return action

    def step(self, action: int) -> Tuple[Any, float, bool, bool, Dict]:
        multi_action = self._combinations[action]
        return self.env.step(multi_action)

    def reset(self, **kwargs) -> Tuple[Any, Dict]:
        return self.env.reset(**kwargs)

    def get_action_meanings(self) -> Dict[int, str]:
        """Get human-readable action descriptions."""
        names = ["FWD", "L", "R", "STAY"]
        meanings = {}
        for idx in range(self.total_actions):
            parts = [f"A{i}:{names[a]}" for i, a in enumerate(self._combinations[idx])]
            meanings[idx] = "|".join(parts)
        return meanings