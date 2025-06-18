"""
Abstract reward-function interface.

Any concrete reward must inherit :class:`RewardFunction` and implement
`reset(env)` (optional) and `step(env, info)`.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, TYPE_CHECKING
if TYPE_CHECKING:
    from src.planner.simulation.grid_map_env import GridMapEnv


class RewardFunction(ABC):
    """Base class for all reward functions."""

    def __init__(self, **params: Any):
        self.params: Dict[str, Any] = params

    # ------------------------------------------------------------------ #
    # Lifecycle hooks
    # ------------------------------------------------------------------ #
    def reset(self, env: "GridMapEnv") -> None:  # noqa: F821
        """Called once at the start of every episode."""
        pass

    @abstractmethod
    def step(self, env: "GridMapEnv", info: Dict[str, Any]) -> float:  # noqa: F821
        """
        Compute reward for the *just-finished* environment step.

        `info` is a diagnostics dict produced by the controller:
            {
                "new_cells":     int,
                "redundant":     int,
                "collisions":    int,
                "avg_distance":  float,
                "done":          bool
            }
        """
