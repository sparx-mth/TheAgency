"""
Drone simulation package with pygame visualization.

This package provides a complete drone simulation environment with:
- Map loading utilities (PGM maps and obstacle definitions)
- Pygame-based real-time visualization
- Scenario configurations for various test environments
- Main simulation loop integrating planning, tracking, and physics

Usage:
    python run_pygame_sim.py --scenario 1
"""

from map_loading import ObstacleMap, load_pgm_map
from visualization import DroneVisualizer, ViewSettings
from scenarios import (
    create_scenario_1,
    create_scenario_2,
    create_scenario_3,
    create_scenario_4,
)
from simulation import run_simulation

__all__ = [
    # Map loading
    "ObstacleMap",
    "load_pgm_map",
    # Visualization
    "DroneVisualizer",
    "ViewSettings",
    # Scenarios
    "create_scenario_1",
    "create_scenario_2",
    "create_scenario_3",
    "create_scenario_4",
    # Simulation
    "run_simulation",
]