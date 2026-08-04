"""Expand a run file's area block, and say what it will cost.

    # just look
    python -m sparx_agency.tasks.planning.falcon_pegasus.mapsize runs/6_whole_office.yaml

    # what the run script does: expand for rosparam, and report
    python -m sparx_agency.tasks.planning.falcon_pegasus.mapsize runs/6_whole_office.yaml \
        --out /tmp/run/expanded.yaml

    # try a different resolution without touching the file
    python -m sparx_agency.tasks.planning.falcon_pegasus.mapsize runs/6_whole_office.yaml \
        --resolution 0.2

Exit status is 1 on an invalid area, which is what stops the container starting.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from sparx_agency.tasks.planning.falcon_pegasus.mapsize.expand import (
    expand_any,
    load_and_expand,
    load_any,
    write_expanded,
)
from sparx_agency.tasks.planning.falcon_pegasus.mapsize.report import format_report


def build_parser() -> argparse.ArgumentParser:
    """The command line."""
    parser = argparse.ArgumentParser(
        prog="python -m sparx_agency.tasks.planning.falcon_pegasus.mapsize",
        description="Expand a run file's exploration area and report its memory cost.",
    )
    parser.add_argument("run_file", type=Path, help="a runs/*.yaml")
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="write the expanded config here, for rosparam to load",
    )
    parser.add_argument(
        "--resolution",
        type=float,
        default=None,
        help="override the file's resolution, to compare without editing it",
    )
    parser.add_argument(
        "--detailed",
        action="store_true",
        help="break the memory down by array",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="write the output file but print nothing unless it fails",
    )
    return parser


def main(argv: list = None) -> int:
    """Entry point.

    Args:
        argv: Arguments, or None to read ``sys.argv``.

    Returns:
        Process exit status.
    """
    args = build_parser().parse_args(argv)

    try:
        if args.resolution is None:
            expanded = load_and_expand(args.run_file)
        else:
            config = load_any(args.run_file)
            config["map_config"]["area"] = dict(
                config["map_config"].get("area", {}), resolution=args.resolution
            )
            expanded = expand_any(config)
    except (OSError, ValueError) as exc:
        print("[mapsize] {}: {}".format(args.run_file, exc), file=sys.stderr)
        return 1

    if args.out is not None:
        write_expanded(expanded, args.out)

    if not args.quiet:
        print("[mapsize] {}".format(args.run_file))
        print(format_report(expanded, detailed=args.detailed))
        if args.out is not None:
            print("  ->    {}".format(args.out))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
