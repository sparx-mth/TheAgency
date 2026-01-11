"""
Drone simulation package.

Provides a simple drone simulator with noise model
for testing trajectory trackers.
"""
from .drone_sim import DroneSimulator, DroneSimParams, DroneState, SimulationResult


__all__ = [
    "DroneSimulator",
    "DroneSimParams",
    "DroneState",
    "SimulationResult",
]