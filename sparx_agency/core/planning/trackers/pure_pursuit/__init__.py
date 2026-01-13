"""
Pure Pursuit tracker package.

Exports:
- PurePursuitParams: algorithm configuration
- PurePursuitTracker: tracker implementation (returns core planning TrackerResult)
"""

from .params import PurePursuitParams
from .tracker import PurePursuitTracker

__all__ = [
    "PurePursuitParams",
    "PurePursuitTracker",
]