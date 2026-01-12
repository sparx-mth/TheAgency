#!/usr/bin/env python3
"""
Pygame-based Drone Tracking Simulation.

Uses ALL algorithms from sparx_agency.core:
- RRTStarOmplPlanner (path planning)
- HermiteSmoother / MinSnapSmoother (trajectory smoothing)
- PurePursuitTracker (trajectory tracking)

Only the drone physics simulator comes from tasks (it's not an algorithm).

Usage:
    python run_pygame_sim.py                    # Run scenario 1
    python run_pygame_sim.py --scenario 2       # Run scenario 2
    python run_pygame_sim.py --scenario 5       # Run ALL scenarios (1-4) sequentially
    python run_pygame_sim.py --smoother minsnap # Use MinSnap instead of Hermite
    python run_pygame_sim.py --no-wind          # Disable wind/gusts
    python run_pygame_sim.py --wind-strength 0.2 --gust-strength 0.4  # Custom wind

Controls:
    SPACE   - Pause/Resume
    Q/ESC   - Quit
"""
from __future__ import annotations

import argparse

# Simulation parameters
from sparx_agency.tasks.planning.simulation.drone_sim import DroneSimParams

# Local imports
from scenarios import (
    create_scenario_1,
    create_scenario_2,
    create_scenario_3,
    create_scenario_4,
)
from simulation import run_simulation


def main():
    parser = argparse.ArgumentParser(
        description="Drone Simulation using sparx_agency.core algorithms"
    )
    parser.add_argument('--scenario', '-s', type=int, choices=[1, 2, 3, 4, 5], default=5,
                        help='Scenario to run (1-4), or 5 to run all scenarios sequentially')
    parser.add_argument('--smoother', type=str, choices=['hermite', 'minsnap'], default='hermite',
                        help='Trajectory smoother: hermite or minsnap')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--max-time', type=float, default=100.0, help='Max simulation time')
    parser.add_argument('--map-dir', type=str,
                        default="/home/nadavc/PycharmProjects/TheAgency_workspace/sparx_agency/tasks/planning/rrt_smoothing_check/maps/",
                        help='Directory containing map files (for scenario 4)')

    # Wind control arguments
    wind_group = parser.add_argument_group('Wind Settings')
    wind_group.add_argument('--no-wind', action='store_true',
                            help='Disable all wind and gusts')
    wind_group.add_argument('--wind-strength', type=float, default=0.3,
                            help='Wind standard deviation in m/s (default: scenario-specific, typically 0.03-0.8)')
    wind_group.add_argument('--wind-mean', type=float, nargs=2, default=None, metavar=('X', 'Y'),
                            help='Mean wind velocity (vx, vy) in m/s (default: scenario-specific)')
    wind_group.add_argument('--gust-strength', type=float, default=None,
                            help='Gust magnitude in m/s (default: scenario-specific, typically 0.15-0.25)')
    wind_group.add_argument('--gust-probability', type=float, default=None,
                            help='Probability of gust per timestep (default: scenario-specific, typically 0.002-0.01)')
    wind_group.add_argument('--no-gusts', action='store_true',
                            help='Disable gusts only (keep steady wind)')

    args = parser.parse_args()

    print("\n" + "#" * 60)
    print("# DRONE TRACKING SIMULATION")
    print("# All algorithms from sparx_agency.core:")
    print("#   - RRTStarOmplPlanner")
    print(f"#   - {args.smoother.capitalize()}Smoother")
    print("#   - PurePursuitTracker")
    print("#" * 60)

    # Determine which scenarios to run
    if args.scenario == 5:
        scenarios_to_run = [1, 2, 3, 4]
        print("\n*** RUNNING ALL SCENARIOS SEQUENTIALLY ***\n")
    else:
        scenarios_to_run = [args.scenario]

    results = []

    for scenario_num in scenarios_to_run:
        if len(scenarios_to_run) > 1:
            print(f"\n{'=' * 60}")
            print(f"  STARTING SCENARIO {scenario_num} of {len(scenarios_to_run)}")
            print(f"{'=' * 60}")

        # Create scenario
        if scenario_num == 4:
            result = create_scenario_4(args.smoother, args.map_dir)
            if result is None:
                results.append((scenario_num, False))
                continue
            costmap, start, goal, planner_params, smoother_params, tracker_params, sim_params = result
            obs_map = None  # Scenario 4 uses costmap directly
        else:
            scenario_fns = {1: create_scenario_1, 2: create_scenario_2, 3: create_scenario_3}
            obs_map, start, goal, planner_params, smoother_params, tracker_params, sim_params = \
                scenario_fns[scenario_num](args.smoother)
            costmap = None

        # Apply wind options
        wind_updates = {}

        if args.no_wind:
            wind_updates['wind_enabled'] = False
            wind_updates['gust_enabled'] = False
            if scenario_num == scenarios_to_run[0]:
                print("\nWind and gusts DISABLED")
        else:
            if args.wind_strength is not None:
                wind_updates['wind_std'] = args.wind_strength
            if args.wind_mean is not None:
                wind_updates['wind_mean'] = (args.wind_mean[0], args.wind_mean[1], 0.0)
            if args.gust_strength is not None:
                wind_updates['gust_magnitude'] = args.gust_strength
            if args.gust_probability is not None:
                wind_updates['gust_probability'] = args.gust_probability
            if args.no_gusts:
                wind_updates['gust_enabled'] = False

        if wind_updates:
            # Create new sim_params with updates
            current_params = sim_params.__dict__.copy()
            current_params.update(wind_updates)
            sim_params = DroneSimParams(**current_params)

            if not args.no_wind and scenario_num == scenarios_to_run[0]:
                print(f"\nWind settings:")
                print(f"  Wind std: {sim_params.wind_std} m/s")
                print(f"  Wind mean: {sim_params.wind_mean}")
                print(f"  Gusts enabled: {sim_params.gust_enabled}")
                if sim_params.gust_enabled:
                    print(f"  Gust magnitude: {sim_params.gust_magnitude} m/s")
                    print(f"  Gust probability: {sim_params.gust_probability}")

        # Run simulation
        success = run_simulation(
            obstacle_map=obs_map,
            costmap=costmap,
            start=start,
            goal=goal,
            planner_params=planner_params,
            smoother_type=args.smoother,
            smoother_params=smoother_params,
            tracker_params=tracker_params,
            sim_params=sim_params,
            max_time=args.max_time,
            seed=args.seed,
        )

        results.append((scenario_num, success))

        print("\n" + "=" * 60)
        print(f"SCENARIO {scenario_num}: {'COMPLETED!' if success else 'ENDED'}")
        print("=" * 60)

    # Print summary if running all scenarios
    if len(scenarios_to_run) > 1:
        print("\n" + "#" * 60)
        print("# SUMMARY - ALL SCENARIOS")
        print("#" * 60)
        for scenario_num, success in results:
            status = "✓ SUCCESS" if success else "✗ ENDED"
            print(f"  Scenario {scenario_num}: {status}")
        print("#" * 60)


if __name__ == "__main__":
    main()