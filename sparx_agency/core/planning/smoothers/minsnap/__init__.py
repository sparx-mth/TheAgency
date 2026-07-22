"""
Minimum-snap smoother.

Exposes:
- MinSnapParams: parameter defaults + validation
- MinSnapSmoother: BaseSmoother implementation

Implementation details live in:
- algorithm.py: generates sampled TrajectoryPoint list using minsnap_trajectories
- adapter.py: wraps samples into a Trajectory implementation
"""
from .params import MinSnapParams
from .smoother import MinSnapSmoother

__all__ = ["MinSnapParams", "MinSnapSmoother"]
