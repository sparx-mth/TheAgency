"""Survey every recorded flight for episodes of the airframe being HELD.

Walks the campaign's `runs/` tree, replays each flight's telemetry through
:class:`~sparx_agency.core.planning.recovery.contact_hold_detector.ContactHoldDetector`
and writes one JSON record per episode -- **including where it happened**, so a
later visualization can show the map locations that trap the aircraft rather
than just a count.

This is the evidence base for LOOP_BUGS.md B33 (37 % of flights, 3.6 % of all
flight time) and the input a blockage/avoidance policy would need.

Usage, from the repo root::

    PYTHONPATH=. .venv/bin/python -m sparx_agency.tools.falcon_campaign.contact_hold_survey \\
        --runs runs --out runs/_analysis/contact_holds.jsonl
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from sparx_agency.core.planning.recovery.contact_hold_detector import (
    ContactHoldDetector, ContactHoldParams)
from sparx_agency.tools.falcon_campaign import analyze


def episodes_in(run_dir):
    """Find every contact-hold episode in one run.

    Args:
        run_dir: A campaign run directory containing ``truth.jsonl``.

    Returns:
        A list of episode dicts, empty when the run has no usable telemetry.
    """
    path = run_dir / "truth.jsonl"
    if not path.exists():
        return []
    records, _ = analyze._load_jsonl(path)
    samples = analyze._normalize(records)
    if len(samples) < 500:
        return []

    detector = ContactHoldDetector(ContactHoldParams())
    out, current, held_before = [], None, False
    for s in samples:
        t, roll, pitch = s["t"], s["roll"], s["pitch"]
        if None in (t, roll, pitch) or s["vx"] is None or s["vz"] is None:
            continue
        tilt = math.degrees(max(abs(roll), abs(pitch)))
        ratio = (s["ranger"] / s["z"]
                 if s["ranger"] and s["z"] and s["z"] > 0.2 else None)
        verdict = detector.update(
            t, tilt, math.hypot(s["vx"], s["vy"]), s["vz"], ratio)
        if verdict.held:
            if not held_before:
                current = dict(run=run_dir.name, t_start=round(t - verdict.since_s, 2),
                               x=s["x"], y=s["y"], z=s["z"], tilt_deg=round(tilt, 1),
                               corroborated=verdict.corroborated, tilt_max_deg=round(tilt, 1))
            elif current is not None:
                current["tilt_max_deg"] = max(current["tilt_max_deg"], round(tilt, 1))
                current["corroborated"] = current["corroborated"] or verdict.corroborated
        elif held_before and current is not None:
            current["duration_s"] = round(t - current["t_start"], 1)
            out.append(current)
            current = None
        held_before = verdict.held
    if current is not None and samples:
        current["duration_s"] = round(samples[-1]["t"] - current["t_start"], 1)
        out.append(current)
    return out


def main(argv=None):
    """Scan the runs tree and write the episode dataset."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--runs", default="runs", type=Path)
    ap.add_argument("--out", default=None, type=Path)
    args = ap.parse_args(argv)

    run_dirs = sorted(d for d in args.runs.iterdir()
                      if d.is_dir() and (d / "truth.jsonl").exists())
    episodes, scanned = [], 0
    for d in run_dirs:
        try:
            found = episodes_in(d)
        except (OSError, ValueError, KeyError):
            continue
        scanned += 1
        episodes.extend(found)

    out = args.out or (args.runs / "_analysis" / "contact_holds.jsonl")
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as fh:
        for e in episodes:
            fh.write(json.dumps(e) + "\n")

    affected = len(set(e["run"] for e in episodes))
    total = sum(e["duration_s"] for e in episodes)
    print("scanned %d runs; %d episodes across %d flights (%.0f%%), %.0f s held total"
          % (scanned, len(episodes), affected,
             100.0 * affected / scanned if scanned else 0.0, total))
    print("written: %s" % out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
