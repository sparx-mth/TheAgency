# src/environments/registration.py
"""
Central Gymnasium registration for SLAM envs and variants.

Usage:
    import gymnasium as gym
    from environments.registration import register_all

    register_all()  # idempotent; safe to call multiple times

    env = gym.make("House-SLAM-v0", width=32, height=32, num_agents=3)
    env = gym.make("House-SLAM-Discrete-v0", num_agents=3)
    env = gym.make(
        "House-SLAM-Curriculum-v0",
        curriculum=dict(hidden_size=8, random_position=True),
        num_agents=1,  # curriculum wrapper assumes single-agent by design
    )
"""

from __future__ import annotations
from typing import Any, Dict

from gymnasium.envs.registration import register, registry

# Import the actual classes (logic unchanged)
from environments.base.slam_env import MultiAgentSLAMEnv
from environments.wrappers.multidiscrete_wrapper import MultiDiscreteToDiscreteWrapper
from environments.wrappers.curriculum_wrapper import CurriculumWrapper


# ---------------------------
# Internal factory functions
# ---------------------------

def _make_slam_env(**kwargs) -> MultiAgentSLAMEnv:
    """Base SLAM environment (no wrappers)."""
    return MultiAgentSLAMEnv(**kwargs)


def _make_slam_env_discrete(**kwargs):
    """
    Base SLAM env wrapped with MultiDiscrete->Discrete wrapper.
    Observation space unchanged; only action interface differs.
    """
    env = MultiAgentSLAMEnv(**kwargs)
    return MultiDiscreteToDiscreteWrapper(env)


def _make_slam_env_curriculum(**kwargs):
    """
    Base SLAM env wrapped with CurriculumWrapper.
    Pass curriculum parameters via kwarg 'curriculum' (dict),
    e.g., curriculum={'hidden_size': 8, 'random_position': True}
    """
    curriculum_cfg: Dict[str, Any] = kwargs.pop("curriculum", {}) or {}
    env = MultiAgentSLAMEnv(**kwargs)
    return CurriculumWrapper(env, **curriculum_cfg)


# ---------------------------
# Public registration helpers
# ---------------------------

BASE_ID = "House-SLAM-v0"
DISCRETE_ID = "House-SLAM-Discrete-v0"
CURRICULUM_ID = "House-SLAM-Curriculum-v0"

def _already_registered(env_id: str) -> bool:
    """Check if an ID is already in the Gymnasium registry."""
    try:
        # Gymnasium's registry acts like a mapping
        _ = registry[env_id]
        return True
    except Exception:
        return False


def register_all() -> None:
    """
    Register all environment IDs (idempotent).
    Safe to call multiple times (won't re-register).
    """
    if not _already_registered(BASE_ID):
        register(
            id=BASE_ID,
            entry_point="environments.registration:_make_slam_env",
        )

    if not _already_registered(DISCRETE_ID):
        register(
            id=DISCRETE_ID,
            entry_point="environments.registration:_make_slam_env_discrete",
        )

    if not _already_registered(CURRICULUM_ID):
        register(
            id=CURRICULUM_ID,
            entry_point="environments.registration:_make_slam_env_curriculum",
        )


def get_registered_ids() -> list[str]:
    """Return the list of IDs this module manages (not the entire Gym registry)."""
    return [BASE_ID, DISCRETE_ID, CURRICULUM_ID]
