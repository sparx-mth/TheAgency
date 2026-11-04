"""Read a recording's ``steps.jsonl`` and summarise every stair traversal in it.

One line per TRAVERSE / RETREAT / CONFIRM_DESTINATION run: its action span,
the height the agent started and ended at, the net XY distance walked, how
many forward steps advanced less than five centimetres along the heading
(the wall-grind signature), and how it ended. A campaign whose traversals
all read "z 2.64 -> 2.64, grind 285/294" has the stair-well bug of the first
recorded campaign (2026-09-29); a healthy climb reads "z 0.05 -> 2.64" in a
few dozen actions.

Usage: ``python -m ...tests.traversal_report <recordings/<key>> [...]``
"""
from __future__ import annotations

import json
import math
from pathlib import Path
import sys

STAIR_PHASES = ("APPROACH_STAIRS", "TRAVERSE", "RETREAT", "CONFIRM_DESTINATION")


def runs(rows):
    """Consecutive steps sharing a stair phase, as (phase, rows) pairs, in order."""
    out, current, phase = [], [], None
    for row in rows:
        this = row["command"]["info"].get("phase")
        this = this if this in STAIR_PHASES else None
        if this != phase:
            if current:
                out.append((phase, current))
            current, phase = [], this
        current.append(row)
    if current:
        out.append((phase, current))
    return [(phase, block) for phase, block in out if phase is not None]


def advance_m(a, b):
    """How far the agent advanced from row ``a`` to row ``b`` along ``a``'s heading."""
    pa, pb = a["pose"], b["pose"]
    return (pb["x"] - pa["x"]) * math.cos(pa["yaw"]) + (pb["y"] - pa["y"]) * math.sin(pa["yaw"])


def describe(path):
    rows = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    print("%s  steps=%d" % (path, len(rows)))
    by_step = {row["step"]: row for row in rows}
    for phase, block in runs(rows):
        first, last = block[0], block[-1]
        forwards = [r for r in block if r["decision"]["action"] == "MOVE_FORWARD" and r["step"] + 1 in by_step]
        grind = sum(1 for r in forwards if advance_m(r, by_step[r["step"] + 1]) < 0.05)
        xy = math.dist((first["pose"]["x"], first["pose"]["y"]), (last["pose"]["x"], last["pose"]["y"]))
        info = last["command"]["info"]
        print("  %-19s steps %4d-%-4d (%3d)  z %5.2f -> %5.2f  net xy %.2f m  grind %d/%d  dir=%s tight=%s" % (
            phase, first["step"], last["step"], len(block), first["pose"]["z"], last["pose"]["z"], xy,
            grind, len(forwards), info.get("direction", "-"), info.get("tight", "-")))
    last = rows[-1]["decision"]["info"].get("policy", {})
    print("  final phase=%s kind=%s floor=%s" % (last.get("phase"), last.get("kind"), last.get("floor_id")))


if __name__ == "__main__":
    for argument in sys.argv[1:]:
        target = Path(argument)
        describe(target / "steps.jsonl" if target.is_dir() else target)

