#!/usr/bin/env python3
"""
Run Trajectory Tracking Simulation - CLEAN VERSION.

Usage:
    python run_sim.py              # Run scenario 1
    python run_sim.py -s 2         # Run scenario 2
    python run_sim.py -s 3         # Run scenario 3
    python run_sim.py --all        # Run all scenarios

Controls:
    Left Click  : Place obstacle
    Right Click : Remove obstacle
    C           : Clear all placed obstacles
    SPACE       : Pause/resume
    Q/ESC       : Quit
"""
import argparse

from config import SCENARIOS, ScenarioConfig


def apply_overrides(cfg: ScenarioConfig, args) -> ScenarioConfig:
    """Apply command-line overrides to config."""
    if args.smoother:
        cfg.smoother.type = args.smoother
    if args.seed is not None:
        cfg.seed = args.seed
    if args.max_time is not None:
        cfg.max_time = args.max_time
    if args.no_wind:
        cfg.simulator.wind_enabled = False
        cfg.simulator.gust_enabled = False
    if args.obstacle_radius is not None:
        cfg.click_obstacles.default_radius = args.obstacle_radius
    return cfg


def main() -> None:
    parser = argparse.ArgumentParser(description="Trajectory Tracking Simulation")

    # Scenario selection
    parser.add_argument("-s", "--scenario", type=int, choices=[1, 2, 3], default=1,
                        help="Scenario number (1-3)")
    parser.add_argument("--all", action="store_true",default=True,
                        help="Run all scenarios")

    # General options
    parser.add_argument("--smoother", choices=["hermite", "minsnap"],
                        help="Smoother type")
    parser.add_argument("--seed", type=int,
                        help="Random seed")
    parser.add_argument("--max-time", type=float,
                        help="Max simulation time (s)")
    parser.add_argument("--no-wind", action="store_true",
                        help="Disable wind and gusts")
    parser.add_argument("--obstacle-radius", type=float,
                        help="Radius for click-placed obstacles (m)")

    args = parser.parse_args()

    # Import here to avoid issues if pygame not installed
    from simulation import run_simulation

    scenarios = [1, 2, 3] if args.all else [args.scenario]
    results = []

    print("\n" + "#" * 50)
    print("# TRAJECTORY TRACKING SIMULATION")
    print("#" * 50)

    for num in scenarios:
        cfg = apply_overrides(SCENARIOS[num](), args)
        ok = run_simulation(cfg)
        results.append((num, ok))
        print(f"\nScenario {num}: {'✓ SUCCESS' if ok else '✗ ENDED'}")

    if len(scenarios) > 1:
        print("\n" + "#" * 50)
        print("# SUMMARY")
        for num, ok in results:
            print(f"  Scenario {num}: {'✓' if ok else '✗'}")
        print("#" * 50)


if __name__ == "__main__":
    main()