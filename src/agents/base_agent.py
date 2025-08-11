"""
agents/base_agent.py

This file defines the abstract base class for all SLAM agents. The base agent
interface ensures clean separation between agent logic and environment implementation.

Agents receive observations from the environment and return actions without any
direct access to environment internals, maintaining proper abstraction boundaries.
This design allows for easy swapping of different agent strategies and supports
both single-agent and multi-agent scenarios.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Union


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
        observations: Union[Dict, Any],
        info: Dict[str, Any]
    ) -> Union[int, Dict[int, int]]:
        """
        Get actions based on current observations.

        This is the main decision-making method that all agents must implement.
        It receives observations from the environment and returns actions to execute.

        Args:
            observations: For single-agent: single observation dictionary
                         For multi-agent: dictionary mapping agent_id to observations
            info: Additional environment information (e.g., step count, progress)

        Returns:
            For single-agent: integer action (0=FORWARD, 1=TURN_LEFT, 2=TURN_RIGHT, 3=STAY)
            For multi-agent: dictionary mapping agent_id to integer action
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