"""
Behavior interface definitions.

This subpackage contains the core abstractions for the behavior system:
- Behavior protocol and BehaviorDecision
- BehaviorContext for input encapsulation
- BehaviorOutput and BehaviorStatus for outputs
- Semantic features (Portal2D)
"""

from .behavior import Behavior, BehaviorDecision
from .context import BehaviorContext
from .features import Portal2D
from .output import BehaviorOutput, BehaviorStatus

__all__ = [
    "Behavior",
    "BehaviorContext",
    "BehaviorDecision",
    "BehaviorOutput",
    "BehaviorStatus",
    "Portal2D",
]