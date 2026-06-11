"""
Scenario configurations for drone simulation.

Each scenario defines the environment, start/goal positions, and algorithm parameters.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

# Types
from sparx_agency.core.common.types import Pose2D

# Planner parameters
from sparx_agency.core.planning.planners.rrtstar import RRTStarOmplParams

# Smoother parameters
from sparx_agency.core.planning.smoothers.hermite import HermiteParams
from sparx_agency.core.planning.smoothers.minsnap import MinSnapParams

# Tracker parameters
from sparx_agency.core.planning.trackers.pure_pursuit import PurePursuitParams

# Simulation parameters
from sparx_agency.tasks.planning.simulation.drone_sim import DroneSimParams

# Local imports
from map_loading import ObstacleMap, load_pgm_map


def create_scenario_1(smoother_type: str = "hermite"):
    """Scenario 1: Dense Obstacle Field with Wind."""
    print("\n" + "=" * 60)
    print("SCENARIO 1: Dense Obstacle Field with Wind")
    print("=" * 60)

    # Narrower map - obstacles fill the space
    obs_map = ObstacleMap(width=10.0, height=10.0, origin_x=0.0, origin_y=0.0)

    # Left wall - blocks going around left
    obs_map.add_rectangle(-0.1, 0.0, 0.6, 10.0)

    # Right wall - blocks going around right
    obs_map.add_rectangle(9.5, 0.0, 0.6, 10.0)

    # Row 1 - obstacles at y=1-2
    obs_map.add_circle(2.0, 1.5, 0.5)
    obs_map.add_circle(4.5, 1.8, 0.6)
    obs_map.add_circle(7.0, 1.3, 0.5)

    # Row 2 - obstacles at y=3-4
    obs_map.add_rectangle(1.0, 3.0, 1.2, 0.5)
    obs_map.add_circle(3.5, 3.5, 0.55)
    obs_map.add_rectangle(5.5, 3.2, 1.0, 0.6)
    obs_map.add_circle(8.0, 3.8, 0.5)

    # Row 3 - obstacles at y=5-6
    obs_map.add_circle(1.5, 5.5, 0.6)
    obs_map.add_rectangle(3.0, 5.0, 0.5, 1.2)
    obs_map.add_circle(5.0, 5.8, 0.55)
    obs_map.add_rectangle(6.5, 5.2, 1.0, 0.6)
    obs_map.add_circle(8.5, 5.5, 0.5)

    # Row 4 - obstacles at y=7-8
    obs_map.add_circle(2.5, 7.5, 0.5)
    obs_map.add_rectangle(4.0, 7.0, 1.2, 0.5)
    obs_map.add_circle(6.5, 7.8, 0.6)
    obs_map.add_circle(8.0, 7.2, 0.45)

    # Start bottom-center, goal top-center - must go through
    start = Pose2D(x=5.0, y=0.3)
    goal = Pose2D(x=5.0, y=9.5)

    # RRT* parameters (core)
    planner_params = RRTStarOmplParams(
        timeout=3.0,
        use_clearance_objective=True,
        clearance_weight=20.0,
        interpolation_spacing=2.0,
    )

    # Smoother parameters (core)
    if smoother_type == "hermite":
        smoother_params = HermiteParams(dt=0.02, nominal_speed_xy=0.4, tangent_scale=0.5)
    else:
        smoother_params = MinSnapParams(dt=0.02, nominal_speed_xy=0.4)

    # Pure Pursuit parameters (core)
    tracker_params = PurePursuitParams(
        holonomic=True,
        base_lookahead=0.5,
        min_lookahead=0.3,
        max_lookahead=1.2,
        cruise_speed=0.4,
        goal_tolerance=0.2,
        path_tolerance=1.0,
    )

    # Simulator parameters (physics)
    sim_params = DroneSimParams(
        dt=0.02,
        wind_enabled=True,
        wind_mean=(0.03, 0.01, 0.0),
        wind_std=0.05,
        gust_enabled=True,
        gust_probability=0.003,
        gust_magnitude=0.15,
        process_noise_std=0.01,
        position_noise_std=0.005,
    )

    return obs_map, start, goal, planner_params, smoother_params, tracker_params, sim_params


def create_scenario_2(smoother_type: str = "hermite"):
    """Scenario 2: Maze-like Corridors + Strong Wind."""
    print("\n" + "=" * 60)
    print("SCENARIO 2: Maze-like Corridors + Strong Wind")
    print("=" * 60)

    obs_map = ObstacleMap(width=12.0, height=10.0, origin_x=0.0, origin_y=0.0)

    # Top and bottom walls
    obs_map.add_rectangle(0.0, -0.1, 12.0, 0.6)
    obs_map.add_rectangle(0.0, 9.5, 12.0, 0.6)

    # Internal maze walls - vertical
    obs_map.add_rectangle(2.0, 0.5, 0.3, 3.5)
    obs_map.add_rectangle(2.0, 6.0, 0.3, 3.5)

    obs_map.add_rectangle(4.5, 0.5, 0.3, 5.0)
    obs_map.add_rectangle(4.5, 7.0, 0.3, 2.5)

    obs_map.add_rectangle(7.0, 0.5, 0.3, 2.5)
    obs_map.add_rectangle(7.0, 5.0, 0.3, 4.5)

    obs_map.add_rectangle(9.5, 2.0, 0.3, 4.0)
    obs_map.add_rectangle(9.5, 8.0, 0.3, 1.5)

    # Internal maze walls - horizontal
    obs_map.add_rectangle(0.5, 4.0, 1.2, 0.3)
    obs_map.add_rectangle(2.5, 2.5, 1.8, 0.3)
    obs_map.add_rectangle(2.5, 7.0, 1.8, 0.3)
    obs_map.add_rectangle(5.0, 3.5, 1.8, 0.3)
    obs_map.add_rectangle(5.0, 6.5, 1.8, 0.3)
    obs_map.add_rectangle(7.5, 2.0, 1.8, 0.3)
    obs_map.add_rectangle(7.5, 8.0, 2.0, 0.3)

    # Scattered obstacles in passages
    obs_map.add_circle(1.0, 2.0, 0.3)
    obs_map.add_circle(1.0, 7.5, 0.3)
    obs_map.add_circle(3.5, 5.0, 0.35)
    obs_map.add_circle(6.0, 1.5, 0.3)
    obs_map.add_circle(6.0, 8.0, 0.3)
    obs_map.add_circle(8.5, 4.0, 0.35)
    obs_map.add_circle(8.5, 6.5, 0.3)

    # Start left side, goal right side - must navigate maze
    start = Pose2D(x=0.8, y=5.0)
    goal = Pose2D(x=11.0, y=5.0)

    planner_params = RRTStarOmplParams(
        timeout=5.0,
        use_clearance_objective=True,
        clearance_weight=20.0,
        interpolation_spacing=2.0,
    )

    if smoother_type == "hermite":
        smoother_params = HermiteParams(dt=0.02, nominal_speed_xy=0.35, tangent_scale=0.4)
    else:
        smoother_params = MinSnapParams(dt=0.02, nominal_speed_xy=0.35)

    tracker_params = PurePursuitParams(
        holonomic=True,
        base_lookahead=0.4,
        min_lookahead=0.25,
        max_lookahead=0.8,
        cruise_speed=0.3,
        max_speed=0.4,
        goal_tolerance=0.2,
        path_tolerance=0.6,
    )

    sim_params = DroneSimParams(
        dt=0.02,
        wind_enabled=True,
        wind_mean=(0.06, 0.03, 0.0),
        wind_std=0.08,
        wind_tau=1.5,
        gust_enabled=True,
        gust_probability=0.005,
        gust_magnitude=0.2,
        gust_duration=0.5,
        process_noise_std=0.015,
    )

    return obs_map, start, goal, planner_params, smoother_params, tracker_params, sim_params


def create_scenario_3(smoother_type: str = "hermite"):
    """Scenario 3: Obstacle Slalom with Frequent Gusts."""
    print("\n" + "=" * 60)
    print("SCENARIO 3: Obstacle Slalom with Frequent Gusts")
    print("=" * 60)

    obs_map = ObstacleMap(width=14.0, height=8.0, origin_x=0.0, origin_y=0.0)

    # Top and bottom walls - narrow corridor
    obs_map.add_rectangle(0.0, -0.1, 14.0, 0.5)
    obs_map.add_rectangle(0.0, 7.6, 14.0, 0.5)

    # Slalom obstacles - alternating top and bottom
    # Column 1
    obs_map.add_circle(1.5, 1.5, 0.6)
    obs_map.add_circle(1.5, 6.5, 0.6)

    # Column 2 - center
    obs_map.add_circle(3.0, 4.0, 0.7)

    # Column 3
    obs_map.add_circle(4.5, 1.2, 0.55)
    obs_map.add_circle(4.5, 6.8, 0.55)

    # Column 4 - center
    obs_map.add_circle(6.0, 3.5, 0.65)
    obs_map.add_circle(6.0, 5.5, 0.5)

    # Column 5
    obs_map.add_circle(7.5, 1.5, 0.6)
    obs_map.add_circle(7.5, 7.0, 0.5)

    # Column 6 - center
    obs_map.add_circle(9.0, 4.0, 0.7)

    # Column 7
    obs_map.add_circle(10.5, 1.3, 0.55)
    obs_map.add_circle(10.5, 6.7, 0.55)

    # Column 8 - center
    obs_map.add_circle(12.0, 3.0, 0.5)
    obs_map.add_circle(12.0, 5.0, 0.5)

    # Some rectangles for variety
    obs_map.add_rectangle(2.0, 3.0, 0.4, 1.5)
    obs_map.add_rectangle(5.0, 0.8, 0.5, 1.0)
    obs_map.add_rectangle(8.0, 5.5, 0.5, 1.2)
    obs_map.add_rectangle(11.0, 3.5, 0.4, 1.0)

    # Start left, goal right - horizontal slalom
    start = Pose2D(x=0.5, y=4.0)
    goal = Pose2D(x=13.5, y=4.0)

    planner_params = RRTStarOmplParams(
        timeout=2.0,
        use_clearance_objective=True,
        clearance_weight=20.0,
        interpolation_spacing=2.0,
    )

    if smoother_type == "hermite":
        smoother_params = HermiteParams(dt=0.02, nominal_speed_xy=0.5, tangent_scale=0.5)
    else:
        smoother_params = MinSnapParams(dt=0.02, nominal_speed_xy=0.5)

    tracker_params = PurePursuitParams(
        holonomic=True,
        base_lookahead=0.7,
        min_lookahead=0.4,
        max_lookahead=1.5,
        cruise_speed=0.5,
        max_speed=0.6,
        goal_tolerance=0.2,
        path_tolerance=1.2,
    )

    sim_params = DroneSimParams(
        dt=0.02,
        wind_enabled=True,
        wind_mean=(0.0, 0.0, 0.0),
        wind_std=0.03,
        gust_enabled=True,
        gust_probability=0.01,
        gust_magnitude=0.25,
        gust_duration=0.6,
        process_noise_std=0.01,
    )

    return obs_map, start, goal, planner_params, smoother_params, tracker_params, sim_params


def create_scenario_4(smoother_type: str = "hermite", map_dir: Optional[str] = None):
    """
    Scenario 4: Hospital Map (real-world map from run_pipeline.py).

    Uses PGM map file instead of synthetic obstacles.
    """
    print("\n" + "=" * 60)
    print("SCENARIO 4: Hospital Map (Real-World)")
    print("=" * 60)

    # Find map files
    if map_dir is None:
        # Try common locations
        possible_dirs = [
            Path(__file__).parent / "maps",
            Path(__file__).parent.parent / "maps",
            Path.cwd() / "maps",
            Path.cwd(),
        ]
        for d in possible_dirs:
            if (d / "hospital_map_cropped.pgm").exists():
                map_dir = d
                break

    if map_dir is None:
        print("ERROR: Could not find hospital_map_cropped.pgm")
        print("Please provide --map-dir argument or place maps in ./maps/")
        return None

    map_dir = Path(map_dir)
    pgm_file = map_dir / "hospital_map_cropped.pgm"
    yaml_file = map_dir / "hospital_map_cropped.yaml"

    if not pgm_file.exists() or not yaml_file.exists():
        print(f"ERROR: Map files not found in {map_dir}")
        return None

    print(f"Loading map: {pgm_file}")

    # Load the costmap using core functions
    costmap = load_pgm_map(str(pgm_file), str(yaml_file), inflate_radius=0.1)
    print(f"  Costmap: {costmap.width}x{costmap.height}, res={costmap.resolution}m")

    # Start and goal (same as run_pipeline.py)
    start = Pose2D(x=-2.0, y=-2.5)
    goal = Pose2D(x=5.0, y=5.0)

    # RRT* parameters (same as run_pipeline.py)
    planner_params = RRTStarOmplParams(
        timeout=3.0,
        use_clearance_objective=True,
        clearance_weight=20.0,
        interpolation_spacing=2.0,
    )

    # Smoother parameters
    if smoother_type == "hermite":
        smoother_params = HermiteParams(dt=0.02, nominal_speed_xy=0.5, tangent_scale=0.5)
    else:
        smoother_params = MinSnapParams(dt=0.02, nominal_speed_xy=0.5)

    # Pure Pursuit parameters
    tracker_params = PurePursuitParams(
        holonomic=True,
        base_lookahead=0.6,
        min_lookahead=0.3,
        max_lookahead=1.5,
        cruise_speed=0.4,
        goal_tolerance=0.2,
        path_tolerance=1.0,
    )

    # Simulator parameters
    sim_params = DroneSimParams(
        dt=0.02,
        wind_enabled=True,
        wind_mean=(0.02, 0.01, 0.0),
        wind_std=0.04,
        gust_enabled=True,
        gust_probability=0.002,
        gust_magnitude=0.12,
        process_noise_std=0.01,
        position_noise_std=0.005,
    )

    # Return costmap directly (scenario 4 uses costmap, not ObstacleMap)
    return costmap, start, goal, planner_params, smoother_params, tracker_params, sim_params