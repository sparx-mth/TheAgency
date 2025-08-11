"""
agents/__init__.py

Agents package initialization. Exports all agent classes.
"""

from .base_agent import BaseSLAMAgent
from .random_agent import RandomAgent
from .frontier_agent import FrontierAgent
# Note: spiral_agent and hybrid_agent can be added when implemented

__all__ = ['BaseSLAMAgent', 'RandomAgent', 'FrontierAgent']