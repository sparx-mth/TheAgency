#!/usr/bin/env python3
"""
Run Trajectory Tracking Dynamic Environment.

Usage:
    python run_pygame_sim.py              # Run all scenarios
    python run_pygame_sim.py -s 1         # Run scenario 1

Controls:
    SPACE  pause
    Q/ESC  quit
    Left Click  spawn dynamic obstacle (circle)
    C      clear dynamic obstacles
    R      toggle local radius overlay
"""
import argparse

from sparx_agency.tasks.planning.dynamic_interaction_environment.config import SCENARIOS, ScenarioConfig
from sparx_agency.tasks.planning.dynamic_interaction_environment.simulation import run_simulation


def apply_overrides(cfg: ScenarioConfig, args) -> ScenarioConfig:
    if args.smoother:
        cfg.smoother.type = args.smoother
    if args.seed is not None:
        cfg.seed = args.seed
    if args.max_time is not None:
        cfg.max_time = args.max_time
    if args.no_wind:
        cfg.simulator.wind_enabled = False
        cfg.simulator.gust_enabled = False
    if args.no_dynamic:
        cfg.dynamic.enabled = False
    if args.local_radius is not None:
        cfg.local_interaction.radius_m = args.local_radius
        cfg.local_interaction.enabled = cfg.local_interaction.radius_m > 0.0
    return cfg


def main() -> None:
    parser = argparse.ArgumentParser(description="Trajectory Tracking Dynamic Environment")
    parser.add_argument("-s", "--scenario", type=int, choices=[1, 2, 3, 4], default=4, help="Scenario (1-3) or 4 for all")
    parser.add_argument("--smoother", choices=["hermite", "minsnap"], help="Smoother type")
    parser.add_argument("--seed", type=int, help="Random seed")
    parser.add_argument("--max-time", type=float, help="Max simulation time (s)")
    parser.add_argument("--no-wind", action="store_true", help="Disable wind and gusts")
    parser.add_argument("--no-dynamic", action="store_true", help="Disable dynamic obstacles update/spawn")
    parser.add_argument("--local-radius", type=float, help="Local interaction radius (m). 0 disables.")
    args = parser.parse_args()

    scenarios = [1, 2, 3] if args.scenario == 4 else [args.scenario]
    results = []

    print("\n" + "#" * 60)
    print("# TRAJECTORY TRACKING DYNAMIC ENVIRONMENT")
    print("#" * 60)

    for num in scenarios:
        cfg = apply_overrides(SCENARIOS[num](), args)
        ok = run_simulation(cfg)
        results.append((num, ok))
        print(f"\nScenario {num}: {'✓ SUCCESS' if ok else '✗ ENDED'}")

    if len(scenarios) > 1:
        print("\n" + "#" * 60)
        print("# SUMMARY")
        for num, ok in results:
            print(f"  Scenario {num}: {'✓' if ok else '✗'}")
        print("#" * 60)


if __name__ == "__main__":
    main()
