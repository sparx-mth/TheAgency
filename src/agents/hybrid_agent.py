"""
Hybrid SLAM Agent - Mixed exploration strategies

This agent combines different exploration strategies by assigning some drones
to use frontier-based exploration while others use random walk. This can be
useful for balancing efficiency with exploration diversity.
"""

from typing import Dict, Any
from .base_slam_agent import BaseSLAMAgent
from .random_agent import RandomAgent
from .frontier_agent import FrontierAgent


class HybridAgent(BaseSLAMAgent):
    """
    Hybrid agent that uses different strategies for different drones.

    This agent assigns a portion of drones to use intelligent frontier-based
    exploration while the remaining drones use random walk. This can help
    ensure thorough coverage while still maintaining efficiency.
    """

    def __init__(self, num_agents: int, camera_range: int = 10, frontier_ratio: float = 0.5):
        """
        Initialize the hybrid agent.

        Args:
            num_agents: Number of agents in the environment
            camera_range: Maximum sensing range of the drones
            frontier_ratio: Ratio of drones that should use frontier exploration (0-1)
        """
        super().__init__(num_agents)
        self.frontier_agent = FrontierAgent(num_agents, camera_range)
        self.random_agent = RandomAgent(num_agents)

        # Determine which agents use which strategy
        num_frontier = int(num_agents * frontier_ratio)
        self.frontier_agents = set(range(num_frontier))

        # Store configuration for display
        self.frontier_ratio = frontier_ratio
        self.camera_range = camera_range

    def reset(self):
        """Reset both sub-agents for new episode."""
        self.frontier_agent.reset()
        self.random_agent.reset()

    def get_actions(self, observations: Dict[int, Any], info: Dict[str, Any]) -> Dict[int, int]:
        """
        Get actions by combining strategies from both agents.

        Each drone is assigned to either frontier or random strategy based on
        the frontier_ratio specified during initialization.
        """
        # Get actions from both agents
        frontier_actions = self.frontier_agent.get_actions(observations, info)
        random_actions = self.random_agent.get_actions(observations, info)

        # Mix strategies based on agent assignment
        actions = {}
        for agent_id in observations:
            if agent_id in self.frontier_agents:
                actions[agent_id] = frontier_actions[agent_id]
            else:
                actions[agent_id] = random_actions[agent_id]

        return actions

    def get_strategy_assignment(self) -> Dict[int, str]:
        """
        Get the strategy assignment for each drone.

        Returns:
            Dict mapping agent_id to strategy name ('frontier' or 'random')
        """
        assignment = {}
        for agent_id in range(self.num_agents):
            if agent_id in self.frontier_agents:
                assignment[agent_id] = 'frontier'
            else:
                assignment[agent_id] = 'random'
        return assignment