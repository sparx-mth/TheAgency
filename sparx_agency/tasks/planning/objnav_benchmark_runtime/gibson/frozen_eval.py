"""Freeze or execute the full published Gibson validation selection explicitly.

Usage: python -m ...gibson.frozen_eval freeze|run --lock PATH -- [gibson.run args]
This is a configuration commitment, not proof that earlier development avoided
validation; the previously inspected five starts are explicitly disclosed.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from sparx_agency.tasks.planning.objnav_benchmark_runtime.gibson.evaluation_lock import freeze_configuration, check_frozen_configuration
from sparx_agency.tasks.planning.objnav_benchmark_runtime.gibson.run import parser as run_parser, prepare, main as run_main


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("freeze", "run"))
    parser.add_argument("--lock", required=True, type=Path)
    known, remainder = parser.parse_known_args(argv)
    if remainder[:1] == ["--"]:
        remainder = remainder[1:]
    args = run_parser().parse_args(remainder)
    env, _, config, issues = prepare(args)
    try:
        if issues:
            raise ValueError("; ".join(issues))
        if not config["full_split"] or len(config["selected_episode_ids"]) != 1000:
            raise ValueError("Freeze/run requires all 1,000 validation episodes, no scene/limit/sharding")
        if known.action == "freeze":
            freeze_configuration(config, known.lock)
            print("Frozen before execution:", known.lock)
            return 0
        check_frozen_configuration(config, known.lock)
    finally:
        if env is not None:
            env.close()
    if args.output is None:
        raise ValueError("Frozen execution requires an explicit --output directory")
    result = run_main(remainder, configuration_guard=lambda current: check_frozen_configuration(current, known.lock))
    if result == 0:
        (args.output / "evaluation_role.json").write_text(json.dumps({
            "role": "frozen_published_validation", "lock": str(known.lock.resolve()),
            "previous_validation_development": True, "held_out_claim": False}, indent=2))
    return result


if __name__ == "__main__":
    raise SystemExit(main())

