"""
SLAM Agents Package

This package provides various agent implementations for the Multi-Agent SLAM environment.
All agents are completely separate from the environment and interact through the standard
Gym interface.

Available agents:
- BaseSLAMAgent: Abstract base class for all agents
- RandomAgent: Simple random exploration strategy
- FrontierAgent: Intelligent frontier-based exploration
- HybridAgent: Mixed strategy combining frontier and random exploration
"""

from .base_slam_agent import BaseSLAMAgent
from .random_agent import RandomAgent
from .frontier_agent import FrontierAgent
from .hybrid_agent import HybridAgent
from .dqn_slam_agent import CustomDQNAgent

__all__ = [
    'BaseSLAMAgent',
    'RandomAgent',
    'FrontierAgent',
    'HybridAgent',
    'CustomDQNAgent'
]

# Version information
__version__ = '1.0.0'