"""
agents/base_agent.py

This file defines the abstract base class for all SLAM agents. The base agent
interface ensures clean separation between agent logic and environment implementation.

Updated to work with the new unified state format where observations contain:
- global_map: Single shared map
- positions: Array of all agent positions
- facings: Array of all agent facing directions
- active: Array of agent active states
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Union
import numpy as np


class BaseSLAMAgent(ABC):
    """
    Abstract base class for SLAM agents.

    This interface defines the contract that all agent implementations must follow.
    Agents are responsible for decision-making based on observations, but have no
    direct access to environment internals.

    Key principles:
    - Agents only receive observations and return actions
    - No direct environment manipulation
    - Support for both single and multi-agent scenarios
    - Stateful agents should implement reset() properly

    With the new unified state format, all agents work with the same observation
    structure regardless of the number of agents.

    Attributes:
        num_agents: Number of agents this controller manages
    """

    def __init__(self, num_agents: int = 1):
        """
        Initialize the base agent.

        Args:
            num_agents: Number of agents to control (1 for single-agent)
        """
        self.num_agents = num_agents

    @abstractmethod
    def get_actions(
        self,
        observations: Dict[str, Any],
        info: Dict[str, Any]
    ) -> np.ndarray:
        """
        Get actions based on current observations.

        This is the main decision-making method that all agents must implement.
        It receives observations from the environment and returns actions to execute.

        Args:
            observations: Dictionary with unified state format:
                - 'global_map': np.ndarray of shape (height, width) - shared map
                - 'positions': np.ndarray of shape (num_agents, 2) - agent positions
                - 'facings': np.ndarray of shape (num_agents,) - facing directions (0-3)
                - 'active': np.ndarray of shape (num_agents,) - active states (0 or 1)
            info: Additional environment information (e.g., step count, progress)

        Returns:
            np.ndarray of shape (num_agents,) containing integer actions:
                0=FORWARD, 1=TURN_LEFT, 2=TURN_RIGHT, 3=STAY
        """
        pass

    def reset(self) -> None:
        """
        Reset the agent's internal state.

        Called at the beginning of each episode. Agents should reset any
        internal state, memory, or learned parameters here.

        Default implementation does nothing, but stateful agents should
        override this method.
        """
        pass

    def save(self, path: str) -> None:
        """
        Save agent state to disk.

        Optional method for agents that maintain learned parameters or
        important state that should be persisted.

        Args:
            path: Path to save the agent state
        """
        pass

    def load(self, path: str) -> None:
        """
        Load agent state from disk.

        Optional method for agents that can restore previously saved state.

        Args:
            path: Path to load the agent state from
        """
        pass

    def get_metrics(self) -> Dict[str, Any]:
        """
        Get agent-specific metrics.

        Optional method for agents to report custom metrics for evaluation.

        Returns:
            Dictionary of metric names to values
        """
        return {}

    @property
    def is_single_agent(self) -> bool:
        """Check if this is a single-agent controller."""
        return self.num_agents == 1

    @property
    def name(self) -> str:
        """Get agent name for logging and identification."""
        return self.__class__.__name__