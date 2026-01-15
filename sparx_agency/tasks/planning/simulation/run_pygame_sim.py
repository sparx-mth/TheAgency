#!/usr/bin/env python3
"""
Drone Tracking Simulation.

Usage:
    python run_pygame_sim.py              # Run all scenarios
    python run_pygame_sim.py -s 1         # Run scenario 1
    python run_pygame_sim.py --no-wind    # Disable wind

Controls: SPACE=pause, Q=quit
"""
import argparse
from config import SCENARIOS, ScenarioConfig
from simulation import run_simulation


def apply_overrides(cfg: ScenarioConfig, args) -> ScenarioConfig:
    """Apply command-line overrides."""
    if args.smoother:
        cfg.smoother.type = args.smoother
    if args.seed:
        cfg.seed = args.seed
    if args.max_time:
        cfg.max_time = args.max_time
    if args.no_wind:
        cfg.simulator.wind_enabled = False
        cfg.simulator.gust_enabled = False
    else:
        if args.wind_std is not None:
            cfg.simulator.wind_std = args.wind_std
        if args.gust_mag is not None:
            cfg.simulator.gust_magnitude = args.gust_mag
        if args.no_gusts:
            cfg.simulator.gust_enabled = False
    return cfg


def main():
    parser = argparse.ArgumentParser(description="Drone Simulation")
    parser.add_argument('-s', '--scenario', type=int, choices=[1, 2, 3, 4], default=4,
                        help='Scenario (1-3) or 4 for all')
    parser.add_argument('--smoother', choices=['hermite', 'minsnap'], help='Smoother type')
    parser.add_argument('--seed', type=int, help='Random seed')
    parser.add_argument('--max-time', type=float, help='Max simulation time (s)')
    parser.add_argument('--no-wind', action='store_true', help='Disable wind and gusts')
    parser.add_argument('--wind-std', type=float, help='Wind strength (m/s)')
    parser.add_argument('--gust-mag', type=float, help='Gust magnitude (m/s)')
    parser.add_argument('--no-gusts', action='store_true', help='Disable gusts only')
    args = parser.parse_args()

    print("\n" + "#" * 50 + "\n# DRONE TRACKING SIMULATION\n" + "#" * 50)

    scenarios = [1, 2, 3] if args.scenario == 4 else [args.scenario]
    results = []

    for num in scenarios:
        cfg = apply_overrides(SCENARIOS[num](), args)
        success = run_simulation(cfg)
        results.append((num, success))
        print(f"\nScenario {num}: {'✓ SUCCESS' if success else '✗ ENDED'}")

    if len(scenarios) > 1:
        print("\n" + "#" * 50 + "\n# SUMMARY")
        for num, success in results:
            print(f"  Scenario {num}: {'✓' if success else '✗'}")
        print("#" * 50)


if __name__ == "__main__":
    main()