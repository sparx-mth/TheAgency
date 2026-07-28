#!/usr/bin/env python3
"""Fly every route twice — anticipation off, then on — and print the difference.

    .venv/bin/python -m sparx_agency.tasks.planning.turn_anticipation_rig.compare
    ... --survey --plot /tmp/turns.png
    ... --no-yaw-bite          # idealised airframe, for the honest second number

Read the table as a trade, not a win: the ``secs`` column is expected to go UP
at every corner (a crab is capped by the weak lateral axis) while ``spin`` and
``stop`` go to zero. Which of those matters more is a question about the drone,
not about the code — see the README.
"""
from __future__ import annotations

import argparse
import sys
from math import degrees
from typing import List, Optional, Sequence, Tuple

from .airframe import AirframeParams
from .flight import FlightResult, fly
from .routes import CORRIDORS, corner_angles, survey_routes
from .tuning import anticipation, controller_params, deployed_dials

_HEADER = ("%-30s %-4s %6s %6s %6s %6s %6s %8s %7s %6s" % (
    "route", "yla", "secs", "TURN", "esc", "spin", "stop", "xtrack", "arrive",
    "lead"))


def _row(name, on, result):
    # type: (str, bool, FlightResult) -> str
    return "%-30s %-4s %6.1f %6.1f %6.1f %6.1f %6.1f %8.3f %7.3f %6.0f%s" % (
        name, "ON" if on else "off", result.seconds, result.turn_ticks * 0.1,
        result.escape_s, result.spin_s, result.stopped_s,
        result.worst_xtrack_m, result.arrive_err_m, result.peak_lead_deg,
        "" if result.reached else "   NOT REACHED")


def run(routes, yaw_bite=True, overrides=None):
    # type: (Sequence[Tuple[str, List]], bool, Optional[dict]) -> List[Tuple[str, FlightResult, FlightResult]]
    """Fly each route with the anticipation off and on, and return both runs."""
    dials = deployed_dials()
    airframe = AirframeParams(yaw_bite=yaw_bite)
    classic = controller_params(dials=dials)
    anticipating = controller_params(
        yaw_lookahead=anticipation(dials=dials, **(overrides or {})),
        dials=dials)
    out = []
    for name, waypoints in routes:
        out.append((name,
                    fly(waypoints, classic, airframe),
                    fly(waypoints, anticipating, airframe)))
    return out


def summarise(results):
    # type: (Sequence[Tuple[str, FlightResult, FlightResult]]) -> List[str]
    """The table, plus the totals line that is the actual answer."""
    lines = [_HEADER, "-" * len(_HEADER)]
    totals = [0.0] * 8
    for name, off, on in results:
        lines.append(_row(name, False, off))
        lines.append(_row(name, True, on))
        for i, value in enumerate((off.seconds, on.seconds,
                                   off.turn_ticks * 0.1, on.turn_ticks * 0.1,
                                   off.escape_s, on.escape_s,
                                   off.stopped_s, on.stopped_s)):
            totals[i] += value
    lines.append("-" * len(_HEADER))
    lines.append(
        "TOTAL   flight %.1f s -> %.1f s (%+.1f%%)   |   in TURN %.1f s -> "
        "%.1f s   |   escaping %.1f s -> %.1f s   |   stopped %.1f s -> %.1f s"
        % (totals[0], totals[1],
           100.0 * (totals[1] - totals[0]) / max(totals[0], 1e-6),
           totals[2], totals[3], totals[4], totals[5], totals[6], totals[7]))
    return lines


def plot(results, path):
    # type: (Sequence[Tuple[str, FlightResult, FlightResult]], str) -> None
    """Draw both tracks per route, with the nose drawn every second."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from math import cos, sin

    count = len(results)
    columns = min(3, count)
    rows = (count + columns - 1) // columns
    fig, axes = plt.subplots(rows, columns, figsize=(5.5 * columns, 5.0 * rows),
                             squeeze=False)
    for index, (name, off, on) in enumerate(results):
        ax = axes[index // columns][index % columns]
        for result, colour, label in ((off, "tab:red", "classic"),
                                      (on, "tab:blue", "anticipating")):
            xs = [p[0] for p in result.track]
            ys = [p[1] for p in result.track]
            ax.plot(xs, ys, color=colour, lw=1.4, label=label, zorder=2)
            # The nose, once a second: this is the picture the table cannot show.
            for i in range(0, len(result.track), 10):
                x, y, yaw = result.track[i]
                ax.plot([x, x + 0.35 * cos(yaw)], [y, y + 0.35 * sin(yaw)],
                        color=colour, lw=0.8, alpha=0.55, zorder=1)
        ax.set_title("%s\n%.1f s / %.1f s spinning  ->  %.1f s / %.1f s"
                     % (name, off.seconds, off.spin_s, on.seconds, on.spin_s),
                     fontsize=9)
        ax.set_aspect("equal", adjustable="datalim")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8, loc="best")
    for index in range(count, rows * columns):
        axes[index // columns][index % columns].axis("off")
    fig.suptitle("drift_pid turn anticipation: track and nose direction",
                 fontsize=12)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main(argv=None):
    # type: (Optional[Sequence[str]]) -> int
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--survey", action="store_true",
                        help="also fly real A* routes across the office survey")
    parser.add_argument("--survey-only", action="store_true",
                        help="fly ONLY the survey routes")
    parser.add_argument("--survey-count", type=int, default=4)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--no-yaw-bite", action="store_true",
                        help="idealised airframe: every commanded yaw delivered "
                             "in full, whatever the drone is doing")
    parser.add_argument("--start-m", type=float, default=None,
                        help="override the anticipation distance (m)")
    parser.add_argument("--plot", default=None, help="write a PNG of the tracks")
    args = parser.parse_args(argv)

    routes = [] if args.survey_only else list(CORRIDORS)
    if args.survey or args.survey_only:
        routes.extend(survey_routes(count=args.survey_count, seed=args.seed))
    if not routes:
        print("no routes to fly", file=sys.stderr)
        return 1

    overrides = {} if args.start_m is None else {"start_m": args.start_m}
    results = run(routes, yaw_bite=not args.no_yaw_bite, overrides=overrides)

    print("airframe: %s" % ("measured yaw/translation coupling"
                            if not args.no_yaw_bite
                            else "IDEALISED (every yaw delivered in full)"))
    print("tuning:   %s" % ("mission.yaml" if deployed_dials()
                            else "core defaults (mission.yaml not readable)"))
    for name, waypoints in routes:
        turns = [degrees(t) for t in corner_angles(waypoints)]
        real = [t for t in turns if abs(t) >= 25.0]
        print("  %-30s %2d waypoints, %d real corner(s): %s"
              % (name, len(waypoints), len(real),
                 ", ".join("%+.0f" % t for t in real) or "-"))
    print()
    for line in summarise(results):
        print(line)
    if args.plot:
        plot(results, args.plot)
        print("\nwrote %s" % args.plot)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
