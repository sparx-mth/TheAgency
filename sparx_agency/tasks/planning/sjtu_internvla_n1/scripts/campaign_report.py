#!/usr/bin/env python3
"""Compare the runs of a campaign: what each one flew, and where it got to.

Five runs of the same instruction are only worth recording if they can be read
side by side, and the per-run verdict line in ``record_campaign.sh`` answers
"did this produce a result" rather than "what did the policy do". This answers
the second: how much route, of what kind, how many deliberate rotations, how far
the aircraft actually travelled, and how close it came to the thing it was told
to stop at.

**Everything here comes out of each run's ``nodes.log``**, which the policy node
already writes one line per decision into -- no rosbag decoding, no ROS, no
model. That keeps it runnable in the plain ``.venv`` and, more usefully, keeps it
runnable on a campaign copied off the machine that flew it.

The positions are the aircraft's pose **at each decision**, not a continuous
track: the distance columns are therefore a lower bound on the ground covered,
which is the right way round for judging whether a run went anywhere.

Usage::

    .venv/bin/python -m sparx_agency.tasks.planning.sjtu_internvla_n1.scripts.campaign_report \\
        ~/sjtu_n1_recordings/room_right_20260826_183931 --target 1.22 -5.61
"""
from __future__ import annotations

import argparse
import math
import os
import re
import sys

_COMMIT = re.compile(
    r"committed #(\d+): (\d+) pts, ([\d.]+) m, from \(([-\d.]+), ([-\d.]+)\) "
    r"after (.+?) \[(curve|action)\]")
_TURN = re.compile(r"turn #(\d+): ([-+][\d.]+) deg")
_ESCAPE = re.compile(r"BLOCKED ESCAPE (\d+)")
_STOP = re.compile(r"N1 STOP")
_FPS = re.compile(r"System1=([\d.]+) Hz\s+System2=([\d.]+) Hz")


class Run(object):
    """One recording, read back out of its log."""

    def __init__(self, path):
        self.path = path
        self.name = os.path.basename(os.path.dirname(path))
        self.commits = []       # (points, metres, x, y, reason, kind)
        self.turns = []         # degrees
        self.escapes = 0
        self.stops = 0
        self.blocks = 0
        self.s1_fps = None
        self.s2_fps = None
        self.capsized = False
        self._read()

    def _read(self):
        with open(self.path, "r", errors="replace") as handle:
            for line in handle:
                match = _COMMIT.search(line)
                if match:
                    self.commits.append((int(match.group(2)), float(match.group(3)),
                                         float(match.group(4)), float(match.group(5)),
                                         match.group(6), match.group(7)))
                    continue
                match = _TURN.search(line)
                if match:
                    self.turns.append(float(match.group(2)))
                    continue
                if _ESCAPE.search(line):
                    self.escapes += 1
                if _STOP.search(line):
                    self.stops += 1
                if "HARD BLOCKED" in line:
                    self.blocks += 1
                if "CAPSIZED" in line:
                    self.capsized = True
                match = _FPS.search(line)
                if match:
                    self.s1_fps, self.s2_fps = float(match.group(1)), float(match.group(2))

    # ── what it flew ────────────────────────────────────────────────
    @property
    def curves(self):
        return [c for c in self.commits if c[5] == "curve"]

    @property
    def actions(self):
        return [c for c in self.commits if c[5] == "action"]

    @property
    def route_m(self):
        """Total length of every route committed, metres."""
        return sum(c[1] for c in self.commits)

    @property
    def curve_share(self):
        return (100.0 * len(self.curves) / len(self.commits)) if self.commits else None

    # ── where it went ───────────────────────────────────────────────
    @property
    def positions(self):
        return [(c[2], c[3]) for c in self.commits]

    @property
    def travelled_m(self):
        """Ground covered between decisions -- a LOWER BOUND on the real track."""
        pts = self.positions
        return sum(math.hypot(b[0] - a[0], b[1] - a[1])
                   for a, b in zip(pts, pts[1:]))

    @property
    def displacement_m(self):
        pts = self.positions
        if len(pts) < 2:
            return 0.0
        return math.hypot(pts[-1][0] - pts[0][0], pts[-1][1] - pts[0][1])

    def closest_to(self, target):
        """Nearest the aircraft got to ``target``, over the decisions."""
        if target is None or not self.positions:
            return None
        return min(math.hypot(x - target[0], y - target[1])
                   for (x, y) in self.positions)

    @property
    def verdict(self):
        """One word for what happened, in the order that matters."""
        if self.capsized:
            return "CAPSIZED"
        if not self.commits:
            return "NO ROUTE"
        if self.travelled_m < 0.5:
            return "WEDGED"
        if self.stops:
            return "STOPPED"
        return "FLEW"


def find_runs(root):
    """Every ``<run>/nodes.log`` under a campaign directory, in run order."""
    out = []
    for name in sorted(os.listdir(root)):
        log = os.path.join(root, name, "nodes.log")
        if os.path.isfile(log):
            out.append(Run(log))
    return out


def report(runs, target=None):
    """Print the comparison table and a one-line summary of the campaign."""
    head = ("run", "verdict", "routes", "curve%", "route m", "moved m", "net m",
            "turns", "esc", "blk", "S2 Hz")
    if target is not None:
        head = head + ("to target",)
    widths = [12, 9, 6, 6, 7, 7, 6, 5, 4, 4, 6] + ([9] if target is not None else [])
    print("  ".join(h.rjust(w) for h, w in zip(head, widths)))
    print("  ".join("-" * w for w in widths))
    for run in runs:
        row = [run.name, run.verdict, str(len(run.commits)),
               "--" if run.curve_share is None else "%.0f" % run.curve_share,
               "%.1f" % run.route_m, "%.1f" % run.travelled_m,
               "%.1f" % run.displacement_m, str(len(run.turns)),
               str(run.escapes), str(run.blocks),
               "--" if run.s2_fps is None else "%.2f" % run.s2_fps]
        if target is not None:
            near = run.closest_to(target)
            row.append("--" if near is None else "%.2f" % near)
        print("  ".join(c.rjust(w) for c, w in zip(row, widths)))

    if not runs:
        print("\nno runs found")
        return
    flown = [r for r in runs if r.commits]
    print("\n%d runs, %d with a committed route." % (len(runs), len(flown)))
    if flown:
        curve = sum(len(r.curves) for r in flown)
        action = sum(len(r.actions) for r in flown)
        print("decisions: %d curves, %d action steps (%.0f%% continuous)"
              % (curve, action, 100.0 * curve / max(1, curve + action)))
        print("rotations: %d, blocked-forward escapes: %d, hard blocks: %d"
              % (sum(len(r.turns) for r in flown), sum(r.escapes for r in flown),
                 sum(r.blocks for r in flown)))
        moved = [r.travelled_m for r in flown]
        print("ground covered per run: min %.1f m, median %.1f m, max %.1f m"
              % (min(moved), sorted(moved)[len(moved) // 2], max(moved)))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("campaign", help="the campaign directory")
    parser.add_argument("--target", nargs=2, type=float, metavar=("X", "Y"),
                        help="world point the instruction names, to measure "
                             "closest approach against")
    args = parser.parse_args(argv)
    runs = find_runs(args.campaign)
    report(runs, tuple(args.target) if args.target else None)
    return 0


if __name__ == "__main__":
    sys.exit(main())
