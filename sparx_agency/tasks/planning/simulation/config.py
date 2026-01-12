"""
Drone Simulation Configuration - All parameters in one place.
"""
from dataclasses import dataclass, field
from typing import Tuple, List


@dataclass
class SimulatorConfig:
    """Drone physics and noise parameters."""
    dt: float = 0.02                    # Time step (s), 50 Hz
    tau_velocity: float = 0.15          # Velocity response time constant (s)
    tau_yaw: float = 0.1                # Yaw rate response time constant (s)
    max_speed_xy: float = 5.0           # Max horizontal speed (m/s)
    max_speed_z: float = 2.0            # Max vertical speed (m/s)
    max_yaw_rate: float = 2.0           # Max yaw rate (rad/s)
    collision_radius: float = 0.10      # Drone collision radius (m)
    # Wind model (Ornstein-Uhlenbeck process)
    wind_enabled: bool = True
    wind_mean: Tuple[float, float, float] = (0.03, 0.01, 0.0)  # Mean wind (m/s)
    wind_std: float = 0.3              # Wind variation (m/s)
    wind_tau: float = 2.0               # Wind correlation time (s)
    # Gust model
    gust_enabled: bool = True
    gust_probability: float = 0.003     # Chance of gust per timestep
    gust_magnitude: float = 0.15        # Gust strength (m/s)
    gust_duration: float = 0.5          # Gust duration (s)
    # Noise
    process_noise_std: float = 0.01     # Random acceleration noise (m/s²)
    position_noise_std: float = 0.005   # Position sensor noise (m)
    velocity_noise_std: float = 0.02    # Velocity sensor noise (m/s)
    yaw_noise_std: float = 0.01         # Yaw sensor noise (rad)


@dataclass
class PlannerConfig:
    """RRT* path planner parameters."""
    timeout: float = 3.0                # Planning timeout (s)
    use_clearance: bool = True          # Prefer paths away from obstacles
    clearance_weight: float = 20.0      # Weight for clearance objective
    interpolation_spacing: float = 2.0  # Path point spacing


@dataclass
class SmootherConfig:
    """Trajectory smoother parameters."""
    type: str = "hermite"               # "hermite" or "minsnap"
    dt: float = 0.02                    # Trajectory time step (s)
    nominal_speed: float = 0.5          # Target speed (m/s) - should match cruise_speed
    tangent_scale: float = 1.0          # Hermite tangent scaling


@dataclass
class TrackerConfig:
    """Pure pursuit tracker parameters."""
    holonomic: bool = True              # Holonomic (omnidirectional) motion
    base_lookahead: float = 0.5         # Base lookahead distance (m)
    min_lookahead: float = 0.3          # Minimum lookahead (m)
    max_lookahead: float = 1.5          # Maximum lookahead (m)
    min_speed: float = 0.2              # Minimum speed (m/s)
    cruise_speed: float = 0.5           # Target cruise speed (m/s)
    max_speed: float = 1.0              # Maximum speed (m/s)
    # Curvature adaptation (higher = more reduction on curves)
    curvature_speed_factor: float = 0.5     # Speed reduction on curves
    curvature_lookahead_factor: float = 0.8 # Lookahead reduction on curves
    goal_tolerance: float = 0.2         # Goal reached threshold (m)
    path_tolerance: float = 1.0         # Max deviation before failure (m)


@dataclass
class Obstacle:
    """Single obstacle: type="rect" (x,y,w,h) or "circle" (x,y,r)."""
    type: str
    x: float
    y: float
    w: float = 0.0
    h: float = 0.0
    r: float = 0.0


@dataclass
class MapConfig:
    """Map and obstacle configuration."""
    width: float = 10.0
    height: float = 10.0
    origin_x: float = 0.0
    origin_y: float = 0.0
    resolution: float = 0.02            # Grid resolution (m)
    inflate_radius: float = 0.15        # Obstacle inflation for planning (m)
    obstacles: List[Obstacle] = field(default_factory=list)


@dataclass
class ScenarioConfig:
    """Complete scenario configuration."""
    name: str = "Default"
    start: Tuple[float, float] = (0.0, 0.0)
    goal: Tuple[float, float] = (5.0, 5.0)
    map: MapConfig = field(default_factory=MapConfig)
    simulator: SimulatorConfig = field(default_factory=SimulatorConfig)
    planner: PlannerConfig = field(default_factory=PlannerConfig)
    smoother: SmootherConfig = field(default_factory=SmootherConfig)
    tracker: TrackerConfig = field(default_factory=TrackerConfig)
    max_time: float = 100.0
    seed: int = 42


# ============================================================================
# PREDEFINED SCENARIOS
# ============================================================================

def scenario_1() -> ScenarioConfig:
    """Dense Obstacle Field with Wind."""
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
        name="Dense Obstacle Field", start=(5.0, 0.3), goal=(5.0, 9.5),
        map=MapConfig(width=10.0, height=10.0, obstacles=obstacles),
        simulator=SimulatorConfig(),
        tracker=TrackerConfig(),
    )


def scenario_2() -> ScenarioConfig:
    """Maze-like Corridors + Strong Wind."""
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
        name="Maze Corridors", start=(0.8, 5.0), goal=(11.0, 5.0),
        map=MapConfig(width=12.0, height=10.0, obstacles=obstacles),
        simulator=SimulatorConfig(),
        planner=PlannerConfig(),
        smoother=SmootherConfig(),
        tracker=TrackerConfig(),
    )


def scenario_3() -> ScenarioConfig:
    """Obstacle Slalom with Frequent Gusts."""
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
        name="Obstacle Slalom", start=(0.5, 4.0), goal=(13.5, 4.0),
        map=MapConfig(width=14.0, height=8.0, obstacles=obstacles),
        simulator=SimulatorConfig(),
        planner=PlannerConfig(),
        smoother=SmootherConfig(),
        tracker=TrackerConfig(),
    )


SCENARIOS = {1: scenario_1, 2: scenario_2, 3: scenario_3}