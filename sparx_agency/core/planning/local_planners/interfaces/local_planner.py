"""
LocalPlanner interface.

A LocalPlanner produces a short-horizon safe trajectory that tracks a reference
(global/smoothed path or trajectory) under a local collision model.

Local planners differ from global planners:
- short horizon (meters/seconds), high-frequency execution
- reactive to newly observed static/dynamic obstacles
- aims to stay close to the reference, not to optimize globally
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from .types import LocalPlanInput, LocalPlanOutput


class LocalPlanner(ABC):
    """Abstract base class for local replanners."""

    @abstractmethod
    def plan(self, inp: LocalPlanInput) -> LocalPlanOutput:
        """
        Compute a short-horizon trajectory (or fallback) for the current tick.

        Requirements:
            - Must be fast under tight deadlines.
            - Must never report SUCCESS without collision-checked output.
        """
        raise NotImplementedError
