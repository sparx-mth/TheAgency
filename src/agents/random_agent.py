"""
agents/random_agent.py

This file implements a simple random exploration agent for the SLAM environment.
The random agent serves as a baseline for comparison and testing, making decisions
through random action selection with a configurable bias toward forward movement.

Updated to work with the new unified state format.
"""

import random
from typing import Any, Dict
import numpy as np

from .base_agent import BaseSLAMAgent
from environments.base.constants import Action


class RandomAgent(BaseSLAMAgent):
    """
    Random exploration agent with configurable forward bias.

    This agent selects actions randomly but with a preference for forward
    movement to encourage exploration rather than spinning in place.

    Attributes:
        forward_bias: Probability of choosing forward movement (0-1)
        action_history: List tracking recent actions for analysis
    """

    def __init__(
        self,
        num_agents: int = 1,
        forward_bias: float = 0.6,
        seed: int = None
    ):
        """
        Initialize the random agent.

        Args:
            num_agents: Number of agents to control
            forward_bias: Probability of moving forward (0-1)
                         Higher values lead to more exploration
                         Lower values lead to more turning
            seed: Random seed for reproducibility
        """
        super().__init__(num_agents)
        self.forward_bias = forward_bias
        self.action_history = []

        if seed is not None:
            random.seed(seed)

    def get_actions(
        self,
        observations: Dict[str, Any],
        info: Dict[str, Any]
    ) -> np.ndarray:
        """
        Get random actions with forward bias.

        Now works with unified state format where all agent data is in arrays.

        Args:
            observations: Current observations with unified format:
                - global_map: Shared map
                - positions: Array of positions [num_agents, 2]
                - facings: Array of facing directions [num_agents]
                - active: Array of active states [num_agents]
            info: Additional environment information

        Returns:
            Array of random actions for all controlled agents
        """
        # Extract active states from unified format
        active = observations['active']

        # Create actions array
        actions = np.zeros(self.num_agents, dtype=np.int32)

        # Generate random action for each agent
        for agent_id in range(self.num_agents):
            if not active[agent_id]:
                action = Action.STAY
            else:
                # Random action with forward bias
                if random.random() < self.forward_bias:
                    action = Action.FORWARD
                else:
                    # Randomly choose between turning and staying
                    action = random.choice([
                        Action.TURN_LEFT,
                        Action.TURN_RIGHT,
                        Action.STAY
                    ])

            actions[agent_id] = action
            self.action_history.append((agent_id, action))

        return actions

    def reset(self) -> None:
        """Reset the agent's internal state."""
        self.action_history.clear()

    def get_metrics(self) -> Dict[str, Any]:
        """
        Get agent metrics.

        Returns:
            Dictionary with action distribution statistics
        """
        if not self.action_history:
            return {}

        # Calculate action distribution
        action_counts = {
            'forward': 0,
            'turn_left': 0,
            'turn_right': 0,
            'stay': 0
        }

        for agent_id, action in self.action_history:
            if action == Action.FORWARD:
                action_counts['forward'] += 1
            elif action == Action.TURN_LEFT:
                action_counts['turn_left'] += 1
            elif action == Action.TURN_RIGHT:
                action_counts['turn_right'] += 1
            elif action == Action.STAY:
                action_counts['stay'] += 1

        total_actions = len(self.action_history)

        return {
            'total_actions': total_actions,
            'forward_ratio': action_counts['forward'] / total_actions if total_actions > 0 else 0,
            'turn_ratio': (action_counts['turn_left'] + action_counts['turn_right']) / total_actions if total_actions > 0 else 0,
            'stay_ratio': action_counts['stay'] / total_actions if total_actions > 0 else 0,
        }