"""
Centralized configuration for the dynamic tracking environment.

This module intentionally contains *no algorithmic logic*.
Only parameters for environment, dynamics, obstacles, and visualization flags.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Tuple, Optional


# -----------------------------
# Simulator / tracker parameters
# -----------------------------

@dataclass
class SimulatorConfig:
    dt: float = 0.02
    tau_velocity: float = 0.15
    tau_yaw: float = 0.1

    max_speed_xy: float = 5.0
    max_speed_z: float = 2.0
    max_yaw_rate: float = 2.0

    collision_radius: float = 0.10

    wind_enabled: bool = True
    wind_mean: Tuple[float, float, float] = (0.03, 0.01, 0.0)
    wind_std: float = 0.3
    wind_tau: float = 2.0

    gust_enabled: bool = True
    gust_probability: float = 0.003
    gust_magnitude: float = 0.15
    gust_duration: float = 0.5

    process_noise_std: float = 0.01
    position_noise_std: float = 0.005
    velocity_noise_std: float = 0.02
    yaw_noise_std: float = 0.01


@dataclass
class PlannerConfig:
    """Used only for initial global path creation (outside environment logic)."""
    timeout: float = 3.0
    use_clearance: bool = True
    clearance_weight: float = 20.0
    interpolation_spacing: float = 2.0


@dataclass
class SmootherConfig:
    type: str = "hermite"    # "hermite" or "minsnap"
    dt: float = 0.02
    nominal_speed: float = 0.5
    tangent_scale: float = 1.0


@dataclass
class TrackerConfig:
    holonomic: bool = True
    base_lookahead: float = 0.5
    min_lookahead: float = 0.3
    max_lookahead: float = 1.5
    min_speed: float = 0.2
    cruise_speed: float = 0.5
    max_speed: float = 1.0
    curvature_speed_factor: float = 0.5
    curvature_lookahead_factor: float = 0.8
    goal_tolerance: float = 0.2
    path_tolerance: float = 1.0


# -----------------------------
# Obstacles / map
# -----------------------------

@dataclass
class Obstacle:
    """Static obstacle: type="rect" (x,y,w,h) or "circle" (x,y,r)."""
    type: str
    x: float
    y: float
    w: float = 0.0
    h: float = 0.0
    r: float = 0.0


@dataclass
class MapConfig:
    width: float = 10.0
    height: float = 10.0
    origin_x: float = 0.0
    origin_y: float = 0.0
    resolution: float = 0.02
    inflate_radius: float = 0.15
    obstacles: List[Obstacle] = field(default_factory=list)


# -----------------------------
# Dynamic objects (environment-only)
# -----------------------------

@dataclass
class DynamicObstaclesConfig:
    enabled: bool = True
    default_circle_radius: float = 0.25
    default_speed: float = 0.6               # m/s
    bounce_on_walls: bool = True
    max_count: int = 50


@dataclass
class LocalInteractionConfig:
    """
    Environment-only local interaction zone:
    - Draw radius around drone
    - Compute simple geometric hazard flags
    """
    enabled: bool = True
    radius_m: float = 1.5

    # Future trajectory samples window (for a "path-related" hazard flag)
    horizon_s: float = 2.0
    sample_dt_s: float = 0.10

    # Distance threshold (in meters) from trajectory points to obstacle boundary
    path_proximity_m: float = 0.35


# -----------------------------
# Scenario
# -----------------------------

@dataclass
class ScenarioConfig:
    name: str = "Default (Dynamic Environment)"
    start: Tuple[float, float] = (0.0, 0.0)
    goal: Tuple[float, float] = (5.0, 5.0)

    map: MapConfig = field(default_factory=MapConfig)
    simulator: SimulatorConfig = field(default_factory=SimulatorConfig)

    # These are used by the driver script to generate a path/trajectory once.
    planner: PlannerConfig = field(default_factory=PlannerConfig)
    smoother: SmootherConfig = field(default_factory=SmootherConfig)
    tracker: TrackerConfig = field(default_factory=TrackerConfig)

    # New environment features
    dynamic: DynamicObstaclesConfig = field(default_factory=DynamicObstaclesConfig)
    local_interaction: LocalInteractionConfig = field(default_factory=LocalInteractionConfig)

    max_time: float = 100.0
    seed: int = 42


# -----------------------------
# Predefined scenarios
# -----------------------------

def scenario_1() -> ScenarioConfig:
    obstacles = [
        Obstacle("rect", -0.1, 0.0, 0.6, 10.0), Obstacle("rect", 9.5, 0.0, 0.6, 10.0),
        Obstacle("circle", 2.0, 1.5, r=0.5), Obstacle("circle", 4.5, 1.8, r=0.6), Obstacle("circle", 7.0, 1.3, r=0.5),
        Obstacle("rect", 1.0, 3.0, 1.2, 0.5), Obstacle("circle", 3.5, 3.5, r=0.55),
        Obstacle("rect", 5.5, 3.2, 1.0, 0.6), Obstacle("circle", 8.0, 3.8, r=0.5),
        Obstacle("circle", 1.5, 5.5, r=0.6), Obstacle("rect", 3.0, 5.0, 0.5, 1.2),
        Obstacle("circle", 5.0, 5.8, r=0.55), Obstacle("rect", 6.5, 5.2, 1.0, 0.6), Obstacle("circle", 8.5, 5.5, r=0.5),
        Obstacle("circle", 2.5, 7.5, r=0.5), Obstacle("rect", 4.0, 7.0, 1.2, 0.5),
        Obstacle("circle", 6.5, 7.8, r=0.6), Obstacle("circle", 8.0, 7.2, r=0.45),
    ]
    return ScenarioConfig(
        name="Dense Obstacles + Dynamic Interactions",
        start=(5.0, 0.3),
        goal=(5.0, 9.5),
        map=MapConfig(width=10.0, height=10.0, obstacles=obstacles),
    )


def scenario_2() -> ScenarioConfig:
    obstacles = [
        Obstacle("rect", 0.0, -0.1, 12.0, 0.6), Obstacle("rect", 0.0, 9.5, 12.0, 0.6),
        Obstacle("rect", 2.0, 0.5, 0.3, 3.5), Obstacle("rect", 2.0, 6.0, 0.3, 3.5),
        Obstacle("rect", 4.5, 0.5, 0.3, 5.0), Obstacle("rect", 4.5, 7.0, 0.3, 2.5),
        Obstacle("rect", 7.0, 0.5, 0.3, 2.5), Obstacle("rect", 7.0, 5.0, 0.3, 4.5),
        Obstacle("rect", 9.5, 2.0, 0.3, 4.0), Obstacle("rect", 9.5, 8.0, 0.3, 1.5),
        Obstacle("rect", 0.5, 4.0, 1.2, 0.3), Obstacle("rect", 2.5, 2.5, 1.8, 0.3),
        Obstacle("rect", 2.5, 7.0, 1.8, 0.3), Obstacle("rect", 5.0, 3.5, 1.8, 0.3),
        Obstacle("rect", 5.0, 6.5, 1.8, 0.3), Obstacle("rect", 7.5, 2.0, 1.8, 0.3), Obstacle("rect", 7.5, 8.0, 2.0, 0.3),
        Obstacle("circle", 1.0, 2.0, r=0.3), Obstacle("circle", 1.0, 7.5, r=0.3), Obstacle("circle", 3.5, 5.0, r=0.35),
        Obstacle("circle", 6.0, 1.5, r=0.3), Obstacle("circle", 6.0, 8.0, r=0.3),
        Obstacle("circle", 8.5, 4.0, r=0.35), Obstacle("circle", 8.5, 6.5, r=0.3),
    ]
    return ScenarioConfig(
        name="Maze + Dynamic Obstacles",
        start=(0.8, 5.0),
        goal=(11.0, 5.0),
        map=MapConfig(width=12.0, height=10.0, obstacles=obstacles),
    )


def scenario_3() -> ScenarioConfig:
    obstacles = [
        Obstacle("rect", 0.0, -0.1, 14.0, 0.5), Obstacle("rect", 0.0, 7.6, 14.0, 0.5),
        Obstacle("circle", 1.5, 1.5, r=0.6), Obstacle("circle", 1.5, 6.5, r=0.6), Obstacle("circle", 3.0, 4.0, r=0.7),
        Obstacle("circle", 4.5, 1.2, r=0.55), Obstacle("circle", 4.5, 6.8, r=0.55),
        Obstacle("circle", 6.0, 3.5, r=0.65), Obstacle("circle", 6.0, 5.5, r=0.5),
        Obstacle("circle", 7.5, 1.5, r=0.6), Obstacle("circle", 7.5, 7.0, r=0.5), Obstacle("circle", 9.0, 4.0, r=0.7),
        Obstacle("circle", 10.5, 1.3, r=0.55), Obstacle("circle", 10.5, 6.7, r=0.55),
        Obstacle("circle", 12.0, 3.0, r=0.5), Obstacle("circle", 12.0, 5.0, r=0.5),
        Obstacle("rect", 2.0, 3.0, 0.4, 1.5), Obstacle("rect", 5.0, 0.8, 0.5, 1.0),
        Obstacle("rect", 8.0, 5.5, 0.5, 1.2), Obstacle("rect", 11.0, 3.5, 0.4, 1.0),
    ]
    return ScenarioConfig(
        name="Slalom + Frequent Disturbances + Dynamic Obstacles",
        start=(0.5, 4.0),
        goal=(13.5, 4.0),
        map=MapConfig(width=14.0, height=8.0, obstacles=obstacles),
    )


SCENARIOS = {1: scenario_1, 2: scenario_2, 3: scenario_3}
