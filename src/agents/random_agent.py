"""
agents/random_agent.py

This file implements a simple random exploration agent for the SLAM environment.
The random agent serves as a baseline for comparison and testing, making decisions
through random action selection with a configurable bias toward forward movement.

This agent is useful for:
- Baseline performance comparison
- Testing environment stability
- Generating diverse exploration patterns
- Simple exploration when computation is limited
"""

import random
from typing import Any, Dict, Union

from .base_agent import BaseSLAMAgent
from core.constants import Action


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
        observations: Union[Dict, Any],
        info: Dict[str, Any]
    ) -> Union[int, Dict[int, int]]:
        """
        Get random actions with forward bias.

        Args:
            observations: Current observations from environment
            info: Additional environment information

        Returns:
            Random actions for all controlled agents
        """
        if self.is_single_agent:
            # Single agent case
            action = self._get_single_action(observations)
            self.action_history.append(action)
            return action
        else:
            # Multi-agent case
            actions = {}
            for agent_id in range(self.num_agents):
                obs = observations[agent_id]
                action = self._get_single_action(obs)
                actions[agent_id] = action
                self.action_history.append((agent_id, action))
            return actions

    def _get_single_action(self, obs: Dict) -> int:
        """
        Get a random action for a single agent.

        Args:
            obs: Observation dictionary for one agent

        Returns:
            Random action integer
        """
        # Check if agent is active
        if not obs.get('active', 1):  # Default to active if not present
            return Action.STAY

        # Random action with forward bias
        if random.random() < self.forward_bias:
            return Action.FORWARD
        else:
            # Randomly choose between turning and staying
            return random.choice([
                Action.TURN_LEFT,
                Action.TURN_RIGHT,
                Action.STAY
            ])

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

        for action in self.action_history:
            if isinstance(action, tuple):
                action = action[1]  # Multi-agent case

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