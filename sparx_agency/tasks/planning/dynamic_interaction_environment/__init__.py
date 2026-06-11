"""
Trajectory Tracking Simulation Environment - CLEAN VERSION.

This package provides a SIMPLE simulation environment for testing
drone trajectory tracking from the core modules.

Key design principles:
1. NO algorithmic code - all algorithms come from core modules
2. PURE orchestration - just connects components together
3. SIMPLE click-to-place obstacles (static, no movement)

Architecture:
    Config → [RRT* Planner] → [Smoother] → [Pure Pursuit Tracker] → Visualization
                   ↑               ↑               ↑
                All from core modules

Components:
- config.py: Configuration dataclasses (no logic)
- obstacle_map.py: Static + click-placed obstacles (no algorithms)
- drone_sim.py: Simple drone physics (environment only)
- simulation.py: Main loop orchestration (no algorithms)
- visualization.py: Pygame display (no algorithms)

Usage:
    from simulator_clean import run_simulation, SCENARIOS

    cfg = SCENARIOS[1]()  # Get scenario config
    run_simulation(cfg)   # Run simulation
"""

from .config import (
    ScenarioConfig,
    SimulatorConfig,
    PlannerConfig,
    SmootherConfig,
    TrackerConfig,
    MapConfig,
    ClickObstacleConfig,
    Obstacle,
    SCENARIOS,
)
from .simulation import run_simulation
from .drone_sim import DroneSimulator, DroneSimParams, DroneState
from .obstacle_map import ObstacleMap, PlacedObstacle
from .visualization import Visualizer, ViewSettings

__all__ = [
    # Config
    "ScenarioConfig",
    "SimulatorConfig",
    "PlannerConfig",
    "SmootherConfig",
    "TrackerConfig",
    "MapConfig",
    "ClickObstacleConfig",
    "Obstacle",
    "SCENARIOS",
    # Simulation
    "run_simulation",
    # Drone
    "DroneSimulator",
    "DroneSimParams",
    "DroneState",
    # Obstacles
    "ObstacleMap",
    "PlacedObstacle",
    # Visualization
    "Visualizer",
    "ViewSettings",
]