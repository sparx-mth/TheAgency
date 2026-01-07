"""
Trajectory smoothers.

A smoother converts a geometric path (Path2D) into a time-parameterized trajectory (Trajectory).

Smoothers are meant to be swappable components in the planning pipeline:
    Planner -> Path2D -> Smoother -> Trajectory -> Tracker -> ControlCommand

This package contains:
- a small registry for name-based construction (optional)
- concrete smoothers under subpackages (e.g., minsnap/)
"""
from .registry import SmootherRegistry
from .adapter import DiscreteTrajectory

__all__ = ["SmootherRegistry", "DiscreteTrajectory"]