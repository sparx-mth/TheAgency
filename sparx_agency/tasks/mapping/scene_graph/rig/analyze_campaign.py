#!/usr/bin/env python3
"""Summarise a search campaign's trials.jsonl, censoring honestly.

The one statistical point this file exists to enforce: **a timeout is data.**
A trial that ran to the cap without finding the target has a time-to-find of
"at least the cap", and dropping it inflates every summary of the trials that
did succeed. A method that finds the object in 60 s half the time and never
otherwise is not a 60 s method.

So the headline numbers are the FIND RATE and the median over all trials with
timeouts held at the cap, and the mean over successes only is printed beside
them clearly labelled as what it is.
"""
from __future__ import annotations

import json
import sys
from typing import Any, Dict, List


def load(path: str) -> List[Dict[str, Any]]:
    rows = []
    with open(path) as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def median(values: List[float]) -> float:
    if not values:
        return float("nan")
    s = sorted(values)
    n = len(s)
    mid = n // 2
    return s[mid] if n % 2 else 0.5 * (s[mid - 1] + s[mid])


def censored_times(rows: List[Dict[str, Any]]) -> List[float]:
    """Time-to-find with every non-find held at its cap (a lower bound)."""
    out = []
    for r in rows:
        if r.get("outcome") == "found" and r.get("t_find_s") is not None:
            out.append(float(r["t_find_s"]))
        elif r.get("t_censored_s") is not None:
            out.append(float(r["t_censored_s"]))
    return out


def summarise(rows: List[Dict[str, Any]], label: str) -> None:
    if not rows:
        return
    found = [r for r in rows if r.get("outcome") == "found"
             and r.get("t_find_s") is not None]
    times = censored_times(rows)
    rate = 100.0 * len(found) / len(rows)
    line = ("%-28s n=%-3d found=%-3d (%5.1f%%)  median_censored=%7.1fs"
            % (label, len(rows), len(found), rate, median(times)))
    if found:
        succ = [float(r["t_find_s"]) for r in found]
        line += "  mean_of_finds=%7.1fs  best=%6.1fs" % (
            sum(succ) / len(succ), min(succ))
    else:
        line += "  (no finds)"
    print(line)


def main() -> None:
    if len(sys.argv) < 2:
        print("usage: analyze_campaign.py trials.jsonl", file=sys.stderr)
        raise SystemExit(2)
    rows = load(sys.argv[1])
    if not rows:
        print("no trials recorded")
        return

    print("=" * 100)
    print("SEARCH CAMPAIGN -- time to find, sim seconds from takeoff")
    print("A timeout is recorded at the cap, NOT dropped: median_censored is "
          "over ALL trials.")
    print("=" * 100)
    summarise(rows, "ALL")
    print("-" * 100)

    for key, title in (("target", "BY TARGET"), ("start", "BY START"),
                       ("backend", "BY BACKEND")):
        values = sorted({str(r.get(key)) for r in rows})
        if len(values) <= 1 and key != "target":
            continue
        print(title)
        for value in values:
            summarise([r for r in rows if str(r.get(key)) == value],
                      "  " + value[:26])
        print("-" * 100)

    outcomes = {}  # type: Dict[str, int]
    for r in rows:
        outcomes[str(r.get("outcome"))] = outcomes.get(str(r.get("outcome")), 0) + 1
    print("OUTCOMES     " + "  ".join("%s=%d" % kv for kv in sorted(outcomes.items())))

    rooms = [int(r.get("rooms_done") or 0) for r in rows]
    fences = [int(r.get("confined_rooms") or 0) for r in rows]
    llm = sum(int(r.get("oracle_llm_ticks") or 0) for r in rows)
    fb = sum(int(r.get("oracle_fallback_ticks") or 0) for r in rows)
    print("ROOMS SEARCHED  total=%d  median/trial=%.1f" % (sum(rooms), median([float(x) for x in rooms])))
    print("FENCES APPLIED  total=%d  median/trial=%.1f" % (sum(fences), median([float(x) for x in fences])))
    # A campaign whose oracle was in fallback throughout measured the geometry,
    # not the method.
    total_ticks = llm + fb
    if total_ticks:
        print("ORACLE          llm=%d  uniform_fallback=%d  (%.0f%% answered)"
              % (llm, fb, 100.0 * llm / total_ticks))


if __name__ == "__main__":
    main()
