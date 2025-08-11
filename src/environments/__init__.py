"""
environments/__init__.py

Environments package initialization. Exports environment classes.
"""

from .slam_env import MultiAgentSLAMEnv
from .single_agent_wrapper import SingleAgentSLAMEnv

__all__ = ['MultiAgentSLAMEnv', 'SingleAgentSLAMEnv']