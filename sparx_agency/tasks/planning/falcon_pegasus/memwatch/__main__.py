"""Watch what a FALCON run actually holds in memory, and say what it means.

Start the run in one terminal and this in another:

    ./run_falcon_pegasus.sh 6_whole_office
    python -m sparx_agency.tasks.planning.falcon_pegasus.memwatch \
        --run 6_whole_office --out /tmp/office_mem.csv

It waits for the container, samples until the run ends or ``--duration`` expires,
writes a CSV and prints the summary. Pass ``--run`` and it also compares the
startup step against what ``mapsize`` predicted the voxel grid would cost, which
is the check that says whether the map is the thing using the memory.

To look again at a finished run:

    python -m sparx_agency.tasks.planning.falcon_pegasus.memwatch \
        --summarise /tmp/office_mem.csv
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import List, Optional

from sparx_agency.tasks.planning.falcon_pegasus.memwatch.sample import (
    CSV_HEADER,
    DEFAULT_CONTAINER,
    DEFAULT_PROCESS,
    Sample,
    container_is_running,
    read_csv,
    sample_once,
)
from sparx_agency.tasks.planning.falcon_pegasus.memwatch.summary import (
    DEFAULT_SETTLE_S,
    format_summary,
    summarise,
)

RUNS_DIR = Path(__file__).resolve().parent.parent / "runs"


def build_parser() -> argparse.ArgumentParser:
    """The command line."""
    parser = argparse.ArgumentParser(
        prog="python -m sparx_agency.tasks.planning.falcon_pegasus.memwatch",
        description="Sample the exploration node's memory during a run.",
    )
    parser.add_argument("--container", default=DEFAULT_CONTAINER)
    parser.add_argument("--process", default=DEFAULT_PROCESS)
    parser.add_argument("--interval", type=float, default=2.0, help="seconds")
    parser.add_argument("--duration", type=float, default=1800.0, help="seconds")
    parser.add_argument(
        "--wait", type=float, default=180.0, help="seconds to wait for the container"
    )
    parser.add_argument("--settle", type=float, default=DEFAULT_SETTLE_S, help="seconds")
    parser.add_argument("--out", type=Path, default=None, help="write the CSV here")
    parser.add_argument(
        "--run",
        default=None,
        help="a runs/*.yaml name, to compare against the predicted grid cost",
    )
    parser.add_argument(
        "--summarise", type=Path, default=None, help="re-read a CSV and stop"
    )
    return parser


def expected_grid_bytes(run_name: str) -> Optional[int]:
    """What ``mapsize`` says this run's voxel grid will cost.

    Args:
        run_name: A ``runs/*.yaml`` name, with or without the extension.

    Returns:
        Bytes, or None if the run file cannot be read or expanded.
    """
    from sparx_agency.tasks.planning.falcon_pegasus.mapsize import expand_run, load_run

    path = RUNS_DIR / (run_name if run_name.endswith(".yaml") else run_name + ".yaml")
    try:
        return int(expand_run(load_run(path)).cost.total_bytes)
    except (OSError, ValueError) as exc:
        print("[memwatch] could not cost {}: {}".format(path, exc), file=sys.stderr)
        return None


def wait_for_container(container: str, timeout_s: float, interval_s: float) -> bool:
    """Block until the container is up.

    Args:
        container: Container name.
        timeout_s: Give up after this long.
        interval_s: How often to check.

    Returns:
        True if it came up.
    """
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if container_is_running(container):
            return True
        time.sleep(min(interval_s, 2.0))
    return False


def collect(args: argparse.Namespace) -> List[Sample]:
    """Sample until the run ends, the budget expires, or the user interrupts.

    Args:
        args: Parsed command line.

    Returns:
        Everything sampled.
    """
    samples: List[Sample] = []
    started = time.monotonic()
    deadline = started + args.duration
    missed = 0

    try:
        while time.monotonic() < deadline:
            elapsed = time.monotonic() - started
            rss, total = sample_once(args.container, args.process)
            samples.append(Sample(elapsed, rss, total))

            if rss is None and total is None:
                missed += 1
                # Three misses in a row means the container has gone, which is
                # how a normal run ends. One is a slow `docker exec`.
                if missed >= 3 and samples and any(s.rss_bytes for s in samples):
                    print("[memwatch] container gone — stopping")
                    break
            else:
                missed = 0

            time.sleep(args.interval)
    except KeyboardInterrupt:
        print("\n[memwatch] interrupted")

    return samples


def main(argv: list = None) -> int:
    """Entry point.

    Args:
        argv: Arguments, or None to read ``sys.argv``.

    Returns:
        Process exit status.
    """
    args = build_parser().parse_args(argv)
    predicted = expected_grid_bytes(args.run) if args.run else None

    if args.summarise is not None:
        try:
            samples = read_csv(args.summarise.read_text(encoding="utf-8"))
        except OSError as exc:
            print("[memwatch] {}".format(exc), file=sys.stderr)
            return 1
        print("[memwatch] {}".format(args.summarise))
        print(format_summary(summarise(samples, args.settle), predicted))
        return 0

    print("[memwatch] waiting for container '{}'".format(args.container))
    if not wait_for_container(args.container, args.wait, args.interval):
        print(
            "[memwatch] container '{}' never came up".format(args.container),
            file=sys.stderr,
        )
        return 1

    print("[memwatch] sampling '{}' every {:.0f} s".format(args.process, args.interval))
    samples = collect(args)

    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(
            CSV_HEADER + "\n" + "\n".join(s.csv_row() for s in samples) + "\n",
            encoding="utf-8",
        )
        print("[memwatch] wrote {}".format(args.out))

    print(format_summary(summarise(samples, args.settle), predicted))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
