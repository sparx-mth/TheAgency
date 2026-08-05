"""The contract every VLA policy implements, and the goal types it accepts.

Mirrors the shape the rest of ``core/planning`` already uses for a pluggable
stage -- a frozen ``*Request`` dataclass in, a frozen ``*Result`` dataclass out,
both carrying a free-form extension dict (see ``interfaces/planner.py``,
``interfaces/tracker.py``, ``interfaces/smoother.py``).

A VLA is a *swappable backend*, not a pipeline stage, so :class:`NavigationPolicy`
is an ``abc.ABC`` with ``raise NotImplementedError`` bodies -- matching
``mapping/interfaces/depth_model.py`` and ``safety/path_correction/base.py`` --
rather than a ``typing.Protocol``.
"""
from sparx_agency.core.planning.vlas.interfaces.goals import (
    Goal,
    ImageGoal,
    LanguageGoal,
    PointGoal,
    PoseGoal,
)
from sparx_agency.core.planning.vlas.interfaces.policy import (
    NavigationPolicy,
    PolicyObservation,
    PolicyResult,
)

__all__ = [
    "Goal",
    "PointGoal",
    "ImageGoal",
    "LanguageGoal",
    "PoseGoal",
    "PolicyObservation",
    "PolicyResult",
    "NavigationPolicy",
]
