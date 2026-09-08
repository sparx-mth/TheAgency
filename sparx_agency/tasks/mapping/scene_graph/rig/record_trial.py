#!/usr/bin/env python3
"""Extract one trial's outcome from its logs and append it to trials.jsonl.

Every number here is read out of a flight's own logs rather than measured by
the harness, and that is deliberate: the harness's wall clock says how long
the machine took, and this world runs well below real time at a ratio that
varies with whatever else is on the GPU. Sim seconds are the only comparable
unit, so time-to-find is taken from the stamps the nodes themselves wrote.

Fields, one JSON object per line:
    tag, target, start, repeat, backend      -- the cell
    outcome        found | timeout | aborted | bringup_failed
    t_find_s       sim seconds from takeoff to the latch; null unless found
    t_censored_s   the cap, for a timeout -- a LOWER BOUND on time-to-find,
                   recorded rather than dropped so a summary cannot quietly
                   average only its successes
    rooms_done, selections, arrivals, mapped, budget_spent, stalls
    found_xy, found_class, confirmations
    wall_s, oracle_llm_ticks, oracle_fallback_ticks
"""
from __future__ import annotations

import argparse
import json
import os
import re
from typing import Any, Dict, Optional

TAKEOFF_RE = re.compile(r"aircraft is airborne")
STAMP_RE = re.compile(r"\[INFO\] \[(\d+\.\d+)\]")
FOUND_RE = re.compile(r"TARGET FOUND\s+target='([^']*)'\s+matched class='([^']*)'")
XY_RE = re.compile(r"world XY = \(([-\d.]+), ([-\d.]+)\)\s+confirmations=(\d+)")
HB_RE = re.compile(
    r"sel=(\d+) arr=(\d+) mapped=(\d+) spent=(\d+) stall=(\d+) .*?done=(\d+)")


def read(path: str) -> str:
    try:
        with open(path, "r", errors="replace") as handle:
            return handle.read()
    except OSError:
        return ""


def sim_stamp_of(line: str) -> Optional[float]:
    """The node's own sim stamp on a log line, or None."""
    match = STAMP_RE.search(line)
    return float(match.group(1)) if match else None


def first_line_matching(text: str, pattern) -> Optional[str]:
    for line in text.splitlines():
        if pattern.search(line):
            return line
    return None


def collect(trial_dir: str) -> Dict[str, Any]:
    """Everything the logs will tell us about this trial."""
    out = {}  # type: Dict[str, Any]
    search_log = read(os.path.join(trial_dir, "object_search_node.log"))
    watcher_log = read(os.path.join(trial_dir, "target_watcher_node.log"))
    oracle_log = read(os.path.join(trial_dir, "llm_oracle_node.log"))

    takeoff_line = first_line_matching(search_log, TAKEOFF_RE)
    found_line = first_line_matching(watcher_log, FOUND_RE)
    t0 = sim_stamp_of(takeoff_line) if takeoff_line else None
    t1 = sim_stamp_of(found_line) if found_line else None
    # Both stamps come from nodes sharing use_sim_time, so the difference is
    # sim seconds of actual flying.
    out["t_find_s"] = (round(t1 - t0, 2) if (t0 is not None and t1 is not None)
                       else None)
    out["t_takeoff_stamp"] = t0
    out["t_found_stamp"] = t1

    if found_line:
        match = FOUND_RE.search(found_line)
        out["found_class"] = match.group(2) if match else None
    xy_line = first_line_matching(watcher_log, XY_RE)
    if xy_line:
        match = XY_RE.search(xy_line)
        if match:
            out["found_xy"] = [float(match.group(1)), float(match.group(2))]
            out["confirmations"] = int(match.group(3))

    # The LAST heartbeat is the run's final tally.
    last_hb = None
    for line in search_log.splitlines():
        if HB_RE.search(line):
            last_hb = line
    if last_hb:
        match = HB_RE.search(last_hb)
        if match:
            (out["selections"], out["arrivals"], out["mapped"],
             out["budget_spent"], out["stalls"], out["rooms_done"]) = (
                int(match.group(i)) for i in range(1, 7))

    # How often the oracle actually answered. A run whose ranking was uniform
    # throughout tested the geometry, not the method.
    out["oracle_llm_ticks"] = oracle_log.count("source=llm")
    out["oracle_fallback_ticks"] = oracle_log.count("uniform_fallback")
    out["confined_rooms"] = search_log.count("fencing FALCON into")
    out["fences_lifted"] = search_log.count("lifted the fence")
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--target", required=True)
    parser.add_argument("--start", required=True)
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--backend", default="falcon")
    parser.add_argument("--outcome", required=True)
    parser.add_argument("--cap-s", type=float, required=True)
    parser.add_argument("--trial-dir", required=True)
    parser.add_argument("--wall-s", type=float, default=0.0)
    args = parser.parse_args()

    record = {
        "tag": args.tag,
        "target": args.target,
        "start": args.start,
        "repeat": args.repeat,
        "backend": args.backend,
        "outcome": args.outcome,
        "wall_s": args.wall_s,
        "t_find_s": None,
        # A timeout is a real outcome with a lower bound, not a missing value.
        "t_censored_s": args.cap_s if args.outcome != "found" else None,
    }
    record.update(collect(args.trial_dir))
    if args.outcome != "found":
        record["t_find_s"] = None

    with open(args.out, "a") as handle:
        handle.write(json.dumps(record) + "\n")
    print("[record] %s %s t_find_s=%s rooms_done=%s"
          % (args.tag, args.outcome, record.get("t_find_s"),
             record.get("rooms_done")))


if __name__ == "__main__":
    main()
