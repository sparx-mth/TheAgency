"""
Mission module.

Mission-level target selection: turning either a named catalog of world-placed
objects, or a surveyed map of free space, into a concrete flight target.

Provides:
- ObjectGoal: a named object at a fixed world position (metres).
- ObjectCatalog: an ordered, immutable collection loaded from an objects JSON file,
  with random selection and label lookup.
- free_space_sampler: random, reachable, worth-flying start/goal pairs drawn from
  an OccupancyGrid2D -- the mission generator behind autonomous data collection.
"""

from .free_space_sampler import (
    StartGoal, connected_regions, largest_region, sample_goal_from,
    sample_start_goal, snap_to_region, traversable_mask,
)
from .object_catalog import ObjectCatalog, ObjectGoal

__all__ = [
    "ObjectCatalog",
    "ObjectGoal",
    "StartGoal",
    "connected_regions",
    "largest_region",
    "sample_goal_from",
    "sample_start_goal",
    "snap_to_region",
    "traversable_mask",
]
