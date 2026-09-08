"""Run every scenario through every planner and write the results down.

    python -m sparx_agency.tasks.planning.routing_benchmark.run_benchmark

Writes one JSON file holding a row per (scenario, planner). Everything
downstream -- the tables, the charts -- reads that file and never re-runs the
sweep, so a plot can be redrawn in a second without spending ten minutes of
search again.

Two things this deliberately does *not* do. It does not average over repeated
trials, because the metrics are exact expectations and there is nothing to
average. And it does not stop a planner from losing: the fallback route that
RPT* returns when its budget expires is scored exactly like any other answer,
because that is what would actually be flown.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time
from typing import Dict, List

from sparx_agency.tasks.planning.routing_benchmark import scenarios as design
from sparx_agency.tasks.planning.routing_benchmark.metrics import (
    add_comparisons,
    clairvoyant_distance,
    evaluate,
)
from sparx_agency.tasks.planning.routing_benchmark.planners import PLANNERS

#: Where results go by default -- outside the repository, because a results
#: file is data, not source.
DEFAULT_OUT = pathlib.Path.home() / "rpt_star_benchmark" / "results.json"


def run(sizes=design.SIZES, seeds=design.SEEDS, progress=True):
    # type: (tuple, tuple, bool) -> Dict
    """Sweep the whole design.

    Args:
        sizes: Room counts to include.
        seeds: Repeats per cell.
        progress: Whether to print a running line to stderr.

    Returns:
        A dict with the run's metadata and one row per scenario and planner.
    """
    rows = []                                   # type: List[Dict]
    total = design.count(sizes=sizes, seeds=seeds)
    started = time.time()
    for index, scenario in enumerate(
            design.all_scenarios(sizes=sizes, seeds=seeds), start=1):
        rows.extend(_one(scenario))
        if progress and (index % 10 == 0 or index == total):
            elapsed = time.time() - started
            sys.stderr.write(
                "\r  %d/%d scenarios  %.0f s elapsed, ~%.0f s left      "
                % (index, total, elapsed, elapsed / index * (total - index)))
            sys.stderr.flush()
    if progress:
        sys.stderr.write("\n")
    return {
        "scenarios": total,
        "rows": rows,
        "speed_mps": design.SPEED_MPS,
        "dwell_s": design.DWELL_S,
        "seconds": time.time() - started,
    }


def _one(scenario):
    # type: (design.Scenario) -> List[Dict]
    """Every planner on one scenario, scored against the truth."""
    belief = scenario.belief.belief
    truth = scenario.belief.truth
    distance = scenario.distance

    plans = {}
    for name, planner in PLANNERS:
        plans[name] = planner(belief, distance)

    outcomes = {
        name: evaluate(plan.order, truth, distance,
                       speed_mps=design.SPEED_MPS, dwell_s=design.DWELL_S)
        for name, plan in plans.items()
    }
    best_possible = clairvoyant_distance(truth, distance, scenario.entrance)
    outcomes = add_comparisons(outcomes, best_possible)

    rows = []
    for name, plan in plans.items():
        outcome = outcomes[name]
        rows.append({
            "scenario": scenario.key,
            "topology": scenario.topology,
            "n_rooms": scenario.n_rooms,
            "oracle": scenario.oracle,
            "seed": scenario.seed,
            "agreement": scenario.belief.agreement(),
            "planner": name,
            "distance_m": outcome.distance,
            "rooms_searched": outcome.rooms_searched,
            "mission_time_s": outcome.mission_time,
            "regret_m": outcome.regret,
            "ratio_to_best": outcome.ratio_to_best,
            "clairvoyant_m": best_possible,
            "plan_seconds": plan.seconds,
            "solved": plan.solved,
            "guarantee": plan.guarantee,
            "expansions": plan.expansions,
        })
    return rows


def main(argv=None):
    # type: (list) -> int
    """Command line entry point."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=pathlib.Path, default=DEFAULT_OUT,
                        help="where to write results.json")
    parser.add_argument("--quick", action="store_true",
                        help="a small sweep, for checking the plumbing")
    args = parser.parse_args(argv)

    if args.quick:
        results = run(sizes=(8, 12), seeds=(0, 1))
    else:
        results = run()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=1), encoding="utf-8")
    print("%d scenarios, %d rows, %.0f s -> %s"
          % (results["scenarios"], len(results["rows"]), results["seconds"],
             args.out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
