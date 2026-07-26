"""
Mission module.

Mission-level target selection: turning a named catalog of world-placed objects
into a concrete flight target (a detector prompt + a coordinate goal).

Provides:
- ObjectGoal: a named object at a fixed world position (metres).
- ObjectCatalog: an ordered, immutable collection loaded from an objects JSON file,
  with random selection and label lookup.
"""

from .object_catalog import ObjectCatalog, ObjectGoal

__all__ = [
    "ObjectCatalog",
    "ObjectGoal",
]
