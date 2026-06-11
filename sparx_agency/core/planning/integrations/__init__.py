"""
Integration layers for planning.

This package contains glue code that connects planning logic with concrete
external or domain-specific representations (maps, safety checks, libraries).

Integrations are intentionally thin and stateless.
They should not contain core planning logic or data models.
"""

from . import maps
from . import safety_maps
from . import collision_maps

__all__ = [
    "maps",
    "safety_maps",
    "collision_maps",
]
