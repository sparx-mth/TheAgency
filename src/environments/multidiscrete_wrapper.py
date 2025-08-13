"""
multidiscrete_wrapper.py

A gymnasium wrapper that converts MultiDiscrete action spaces to a single Discrete space
for compatibility with DQN algorithms that don't support MultiDiscrete actions.

Usage:
    env = MultiAgentSLAMEnv(...)
    env = MultiDiscreteToDiscreteWrapper(env)
    # Now env.action_space is Discrete instead of MultiDiscrete
"""

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from typing import Any, Dict, Tuple


class MultiDiscreteToDiscreteWrapper(gym.Wrapper):
    """
    Wrapper that converts MultiDiscrete action space to single Discrete space.

    For a MultiDiscrete([n, n, ..., n]) action space with k agents,
    this creates a Discrete(n^k) space where each action index represents
    a unique combination of actions for all agents.

    Example:
        - Original: MultiDiscrete([4, 4, 4]) for 3 agents with 4 actions each
        - Wrapped: Discrete(64) representing all 4^3 = 64 combinations
        - Action 0 = [0, 0, 0], Action 1 = [0, 0, 1], ..., Action 63 = [3, 3, 3]
    """

    def __init__(self, env):
        """
        Initialize the wrapper.

        Args:
            env: Environment with MultiDiscrete action space

        Raises:
            ValueError: If action space is not MultiDiscrete or has non-uniform dimensions
        """
        super().__init__(env)

        # Verify that we have a MultiDiscrete action space
        if not isinstance(env.action_space, spaces.MultiDiscrete):
            raise ValueError(f"Expected MultiDiscrete action space, got {type(env.action_space)}")

        # Get the original MultiDiscrete parameters
        self.original_nvec = env.action_space.nvec
        self.num_agents = len(self.original_nvec)

        # Check if all agents have the same number of actions (uniform)
        if not all(n == self.original_nvec[0] for n in self.original_nvec):
            raise ValueError(f"All agents must have the same number of actions. Got: {self.original_nvec}")

        self.actions_per_agent = self.original_nvec[0]

        # Calculate total number of action combinations
        self.total_actions = self.actions_per_agent ** self.num_agents

        # Create new Discrete action space
        self.action_space = spaces.Discrete(self.total_actions)

        # Pre-compute all possible action combinations for efficient decoding
        self._action_combinations = self._generate_action_combinations()

        print(f"MultiDiscreteToDiscreteWrapper initialized:")
        print(f"  Original space: MultiDiscrete({list(self.original_nvec)})")
        print(f"  New space: Discrete({self.total_actions})")
        print(f"  {self.num_agents} agents with {self.actions_per_agent} actions each")

    def _generate_action_combinations(self) -> np.ndarray:
        """
        Pre-generate all possible action combinations.

        Returns:
            Array of shape (total_actions, num_agents) containing all combinations
        """
        combinations = np.zeros((self.total_actions, self.num_agents), dtype=np.int32)

        for i in range(self.total_actions):
            # Convert single action index to multi-agent action array
            combinations[i] = self._decode_action(i)

        return combinations

    def _decode_action(self, action: int) -> np.ndarray:
        """
        Convert single discrete action to MultiDiscrete action array.

        Uses base conversion: for n actions per agent and k agents,
        action i maps to [i//(n^(k-1)) % n, i//(n^(k-2)) % n, ..., i % n]

        Args:
            action: Single integer action from Discrete space

        Returns:
            Array of actions for each agent
        """
        multi_action = np.zeros(self.num_agents, dtype=np.int32)

        remaining = action
        for i in reversed(range(self.num_agents)):
            # Calculate the power of actions_per_agent for this position
            power = self.actions_per_agent ** i
            multi_action[self.num_agents - 1 - i] = remaining // power
            remaining = remaining % power

        return multi_action

    def _encode_action(self, multi_action: np.ndarray) -> int:
        """
        Convert MultiDiscrete action array to single discrete action.

        This is the inverse of _decode_action.

        Args:
            multi_action: Array of actions for each agent

        Returns:
            Single integer action for Discrete space
        """
        action = 0
        for i, agent_action in enumerate(multi_action):
            power = self.actions_per_agent ** (self.num_agents - 1 - i)
            action += agent_action * power

        return action

    def step(self, action: int) -> Tuple[Any, float, bool, bool, Dict]:
        """
        Execute a step in the environment.

        Args:
            action: Single discrete action to execute

        Returns:
            Standard gymnasium step return tuple
        """
        # Validate action
        if not self.action_space.contains(action):
            raise ValueError(f"Action {action} not in action space {self.action_space}")

        # Convert single action to multi-agent actions using pre-computed combinations
        multi_action = self._action_combinations[action]

        # Execute in original environment
        return self.env.step(multi_action)

    def get_action_meanings(self) -> Dict[int, str]:
        """
        Get human-readable meanings for all actions.

        Returns:
            Dictionary mapping action indices to their meanings
        """
        action_names = ["TURN_LEFT", "TURN_RIGHT", "FORWARD", "STAY"]  # Assuming these are the actions

        meanings = {}
        for action_idx in range(self.total_actions):
            multi_action = self._action_combinations[action_idx]
            meaning_parts = []
            for agent_idx, agent_action in enumerate(multi_action):
                agent_action_name = action_names[agent_action] if agent_action < len(
                    action_names) else f"ACTION_{agent_action}"
                meaning_parts.append(f"Agent{agent_idx}:{agent_action_name}")
            meanings[action_idx] = " | ".join(meaning_parts)

        return meanings

    def sample_random_action(self) -> int:
        """
        Sample a random action from the discrete space.

        Returns:
            Random action index
        """
        return self.action_space.sample()

    def get_original_action_space(self) -> spaces.MultiDiscrete:
        """
        Get the original MultiDiscrete action space.

        Returns:
            Original MultiDiscrete action space
        """
        return spaces.MultiDiscrete(self.original_nvec)


def test_wrapper():
    """Test the wrapper with a simple example."""

    # Create a simple test environment
    class TestEnv(gym.Env):
        def __init__(self):
            self.action_space = spaces.MultiDiscrete([4, 4, 4])  # 3 agents, 4 actions each
            self.observation_space = spaces.Box(low=0, high=1, shape=(10, 10))

        def step(self, action):
            print(f"Environment received action: {action}")
            return np.zeros((10, 10)), 0.0, False, False, {}

        def reset(self, **kwargs):
            return np.zeros((10, 10)), {}

    # Test the wrapper
    print("Testing MultiDiscreteToDiscreteWrapper:")
    print("-" * 50)

    env = TestEnv()
    wrapped_env = MultiDiscreteToDiscreteWrapper(env)

    print(f"Original action space: {env.action_space}")
    print(f"Wrapped action space: {wrapped_env.action_space}")
    print()

    # Test a few action conversions
    test_actions = [0, 1, 5, 21, 63]  # Some example actions
    print("Action conversion examples:")
    for action in test_actions:
        if action < wrapped_env.total_actions:
            multi_action = wrapped_env._action_combinations[action]
            print(f"  Action {action:2d} -> {multi_action}")

    print()

    # Test round-trip conversion
    print("Round-trip conversion test:")
    original_multi = np.array([2, 1, 3])
    encoded = wrapped_env._encode_action(original_multi)
    decoded = wrapped_env._decode_action(encoded)
    print(f"  Original: {original_multi}")
    print(f"  Encoded:  {encoded}")
    print(f"  Decoded:  {decoded}")
    print(f"  Match:    {np.array_equal(original_multi, decoded)}")

    print()

    # Show some action meanings
    meanings = wrapped_env.get_action_meanings()
    print("First 10 action meanings:")
    for i in range(min(10, len(meanings))):
        print(f"  {i:2d}: {meanings[i]}")


if __name__ == "__main__":
    test_wrapper()