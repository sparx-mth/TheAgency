"""
Tracker interface.

A Tracker consumes the current robot state and a reference trajectory, and produces
a control command (e.g., velocity commands) and status.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Protocol, runtime_checkable

from sparx_agency.core.common.types import (
    State3D,
    Trajectory,
    TrajectoryPoint,
    ControlCommand,
    KinematicLimits,
)


@dataclass(frozen=True)
class TrackerRequest:
    """
    Inputs to a tracking computation.

    - `state`: current measured/estimated robot state
    - `trajectory`: reference trajectory
    - `t`: time since trajectory start (seconds)
    - `limits`: optional command limits; if None tracker may use internal defaults
    - `options`: algorithm-specific knobs (lookahead, gains, etc.)
    """
    state: State3D
    trajectory: Trajectory
    t: float
    limits: Optional[KinematicLimits] = None
    options: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class TrackerResult:
    """
    Output of a tracker step.
    """
    command: ControlCommand
    reference: Optional[TrajectoryPoint] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class BaseTracker(Protocol):
    """Tracker contract."""
    name: str

    def reset(self) -> None:
        """Reset any internal tracker state (filters, integrators, etc.)."""
        ...

    def step(self, request: TrackerRequest) -> TrackerResult:
        """
        Compute a control command for the current timestep.

        Returns:
            TrackerResult with command + optional sampled reference point + metadata.
        """
        ...
