"""
Random SLAM Agent - Simple random exploration strategy

This agent implements a basic random walk strategy with a bias towards
forward movement. It serves as a baseline for comparison with more
sophisticated agents.
"""

import random
from typing import Dict, Any
from .base_slam_agent import BaseSLAMAgent


class RandomAgent(BaseSLAMAgent):
    """
    Simple random agent for baseline comparison.

    This agent chooses actions randomly with a preference for forward movement.
    It's useful as a baseline to compare against more intelligent agents.
    """

    def __init__(self, num_agents: int, forward_bias: float = 0.6):
        """
        Initialize the random agent.

        Args:
            num_agents: Number of agents in the environment
            forward_bias: Probability of choosing forward movement (0-1)
        """
        super().__init__(num_agents)
        self.forward_bias = forward_bias

    def get_actions(self, observations: Dict[int, Any], info: Dict[str, Any]) -> Dict[int, int]:
        """
        Get random actions for all agents.

        The agent has a bias towards forward movement to encourage exploration
        rather than just spinning in place.

        Action indices:
        - 0: FORWARD
        - 1: TURN_LEFT
        - 2: TURN_RIGHT
        - 3: STAY
        """
        actions = {}

        for agent_id, obs in observations.items():
            if obs['active']:
                # Bias towards forward movement
                if random.random() < self.forward_bias:
                    actions[agent_id] = 0  # FORWARD
                else:
                    # Randomly choose between turning and staying
                    actions[agent_id] = random.choice([1, 2, 3])  # TURN_LEFT, TURN_RIGHT, STAY
            else:
                # Inactive agents must stay
                actions[agent_id] = 3  # STAY

        return actions