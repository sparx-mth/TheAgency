"""
Base SLAM Agent - Abstract base class for all SLAM agents

This module provides the abstract base class that all SLAM agents must inherit from.
It defines the interface for interacting with the MultiAgentSLAMGymEnv.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any


class BaseSLAMAgent(ABC):
    """Abstract base class for SLAM agents."""

    def __init__(self, num_agents: int):
        """
        Initialize the base agent.

        Args:
            num_agents: Number of agents in the environment
        """
        self.num_agents = num_agents

    @abstractmethod
    def get_actions(self, observations: Dict[int, Any], info: Dict[str, Any]) -> Dict[int, int]:
        """
        Get actions for all agents based on observations.

        Args:
            observations: Dict mapping agent_id to observation dict
            info: Additional environment information

        Returns:
            Dict mapping agent_id to action index
        """
        pass

    def reset(self):
        """Reset agent state for new episode."""
        pass