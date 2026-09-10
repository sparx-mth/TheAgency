"""Run the studies and print the tables.

::

    # everything, at the default sizes -- a few minutes
    .venv/bin/python -m sparx_agency.tasks.planning.rpt_star_paper.run

    # one study
    .venv/bin/python -m sparx_agency.tasks.planning.rpt_star_paper.run reliability

    # bigger, slower, closer to the paper's own sizes
    .venv/bin/python -m sparx_agency.tasks.planning.rpt_star_paper.run --full

Raw rows are written as JSON to ``~/rpt_star_paper/`` -- outside the repository,
because measurements are not source.

Python 3.8 syntax, standard library only.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time
from typing import Dict, List, Optional, Sequence

from sparx_agency.tasks.planning.rpt_star_paper import experiments, report

#: Where raw rows land.
OUTPUT_DIR = pathlib.Path.home() / "rpt_star_paper"

#: The studies, in the order a reader should meet them.
STUDIES = ("fidelity", "objective", "ablation", "baselines", "sharpness",
           "reliability", "tsplib")


def _settings(full):
    # type: (bool) -> Dict[str, Dict]
    """Per-study arguments for the quick and the full sweep.

    The paper runs 20 instances per size up to 40 vertices, in C++. This is
    pure Python against an exponential search, so ``--full`` reaches for the
    same shape at smaller sizes rather than pretending to match the absolute
    numbers.

    ``tsplib`` is the exception that needs its own budget. Its instances are
    17-29 vertices, and at the paper's 60 s limit only 6 of 15 exact runs
    finish here -- which takes half an hour and tells you nothing the shorter
    budget does not, because the finding is about the triangle inequality
    rather than about optimality. ``--full`` restores the paper's limit.
    """
    if full:
        return {
            "fidelity": dict(sizes=(6, 7, 8, 9, 10), seeds=range(20)),
            "objective": dict(sizes=(7, 8, 9), seeds=range(30)),
            "ablation": dict(sizes=(8, 10, 12, 14, 16, 18), seeds=range(10)),
            "baselines": dict(sizes=(8, 10, 12, 14, 16, 18), seeds=range(10)),
            "sharpness": dict(n=14, seeds=range(40)),
            "reliability": dict(n=14, seeds=range(40)),
            "tsplib": dict(seeds=range(3)),
        }
    return {
        "fidelity": dict(sizes=(6, 7, 8), seeds=range(8)),
        "objective": dict(sizes=(7, 8), seeds=range(15)),
        "ablation": dict(sizes=(8, 10, 12, 14), seeds=range(5)),
        "baselines": dict(sizes=(8, 10, 12, 14), seeds=range(5)),
        "sharpness": dict(n=12, seeds=range(20)),
        "reliability": dict(n=12, seeds=range(20)),
        "tsplib": dict(seeds=range(2), time_limit_s=5.0),
    }


def run_study(name, settings):
    # type: (str, Dict) -> List[Dict]
    """Dispatch one study by name."""
    runner = {
        "fidelity": experiments.run_fidelity,
        "objective": experiments.run_objective,
        "ablation": experiments.run_ablation,
        "baselines": experiments.run_baselines,
        "sharpness": experiments.run_sharpness,
        "reliability": experiments.run_reliability,
        "tsplib": experiments.run_tsplib,
    }[name]
    return runner(**settings)


def summarise(name, rows):
    # type: (str, Sequence[Dict]) -> str
    """Render one study's rows."""
    if name == "fidelity":
        return report.summarise_fidelity(rows)
    if name == "objective":
        return report.summarise_objective(rows)
    if name == "ablation":
        return report.summarise_ablation(rows)
    if name == "baselines":
        return report.summarise_baselines(rows)
    if name == "sharpness":
        return report.summarise_sharpness(rows)
    if name == "tsplib":
        return report.summarise_tsplib(rows)

    parts = []
    for column, title in (("distance", "expected flight distance (m)"),
                          ("flight_time_s", "expected flight time (s)"),
                          ("planning_s", "planning time (s)"),
                          ("places_searched", "places opened")):
        parts.append("--- %s ---" % title)
        parts.append(report.summarise_reliability(rows, column))
        parts.append("")
    return "\n".join(parts)


def main(argv=None):
    # type: (Optional[Sequence[str]]) -> int
    """Entry point."""
    parser = argparse.ArgumentParser(
        description="Replicate the RPT* paper's experiments in isolation.")
    parser.add_argument("studies", nargs="*", default=None,
                        help="which studies to run; default is all of %s"
                             % (", ".join(STUDIES),))
    parser.add_argument("--full", action="store_true",
                        help="larger sizes and more instances; much slower")
    parser.add_argument("--output-dir", default=OUTPUT_DIR,
                        help="where to write raw JSON rows")
    args = parser.parse_args(argv)

    chosen = args.studies or list(STUDIES)
    unknown = [s for s in chosen if s not in STUDIES]
    if unknown:
        parser.error("unknown study %r; choose from %s"
                     % (unknown[0], ", ".join(STUDIES)))

    settings = _settings(args.full)
    output_dir = pathlib.Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for name in chosen:
        print("\n" + "=" * 78)
        print("%s" % name.upper())
        print("=" * 78)
        started = time.monotonic()
        rows = run_study(name, settings[name])
        elapsed = time.monotonic() - started

        path = output_dir / ("%s.json" % name)
        with path.open("w") as handle:
            json.dump(rows, handle, indent=1, default=str)

        print(summarise(name, rows))
        print("\n%d rows in %.1fs -> %s" % (len(rows), elapsed, path))
    return 0


if __name__ == "__main__":
    sys.exit(main())

