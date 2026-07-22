"""apriltag_quality_report.py -- turn an apriltag quality CSV into a per-tag verdict.

Reads a CSV written by :mod:`apriltag_quality_log` and answers the operator's
question directly: for each tag, is it GOOD, or does it need work on the wall, and
why. The four failure modes it separates, each with a different fix:

  * **MIS-MAPPED** -- the tag is read fine but reprojects badly under the shared
    pose. Its recorded position / orientation / size in the tag map is wrong.
    Fix: re-measure that tag's map entry.
  * **RARELY SEEN** -- the tag is almost never in view. Fix: reposition it, or add
    one, to cover the stretch of the route where localization goes blind.
  * **HARD TO READ** -- low detector margin or a small image size whenever it does
    appear. Fix: bigger tag, better light, less glancing mounting angle.
  * **OFTEN DROPPED** -- detected but repeatedly not trusted (outlier-rejected).
    Usually a milder mis-map; check its map entry.

Run it:  ``python3 -m sparx_agency.tasks.localization.apriltag_quality_report \\
          /tmp/falcon/apriltag_YYYYmmdd_HHMMSS.csv``

Pure stdlib, Python 3.8 compatible.
"""
from __future__ import annotations

import csv
import sys
from typing import Dict, List, Optional

# Verdict thresholds. Deliberately lenient -- they only sort the tags into "look
# here first" buckets; the printed numbers are the real evidence.
_MISMAP_RMS_PX = 4.0      # worst per-tag reprojection above this = map entry suspect
_RARELY_SEEN_FRAC = 0.05  # detected in fewer than this fraction of frames
_LOW_MARGIN = 20.0        # mean decision_margin below this = hard to read
_SMALL_PX = 22.0          # median apparent size below this = too far / too grazing
_OFTEN_DROPPED_USED = 0.5  # trusted in fewer than this fraction of its detections


def _f(value) -> Optional[float]:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _median(values: List[float]) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    mid = len(s) // 2
    return s[mid] if len(s) % 2 else 0.5 * (s[mid - 1] + s[mid])


def summarize(rows: List[dict]) -> List[dict]:
    """Aggregate per-tag quality from the CSV rows.

    Args:
        rows: Parsed rows (dicts) from an apriltag quality CSV.

    Returns:
        One summary dict per tag id, worst-first (most in need of attention), each
        carrying the evidence and a ``verdict`` / ``reason``.
    """
    frames = set()
    per_tag: Dict[int, dict] = {}
    for r in rows:
        frames.add(r.get("stamp"))
        tid = r.get("tag_id")
        if tid in (None, ""):
            continue                          # a blank-tag (blind) row
        tid = int(tid)
        t = per_tag.setdefault(tid, {
            "tag_id": tid, "seen": 0, "used": 0, "in_map": 0,
            "margins": [], "sizes": [], "reproj": [], "dists": []})
        t["seen"] += 1
        t["used"] += 1 if r.get("used") == "1" else 0
        t["in_map"] += 1 if r.get("in_map") == "1" else 0
        for key, col in (("margins", "decision_margin"), ("sizes", "apparent_px"),
                         ("reproj", "tag_reproj_rms_px"), ("dists", "dist_m")):
            v = _f(r.get(col))
            if v is not None:
                t[key].append(v)

    total = max(1, len(frames))
    out = []
    for t in per_tag.values():
        seen_frac = t["seen"] / total
        used_frac = t["used"] / max(1, t["seen"])
        mean_margin = sum(t["margins"]) / len(t["margins"]) if t["margins"] else 0.0
        med_px = _median(t["sizes"])
        worst_reproj = max(t["reproj"]) if t["reproj"] else None
        mean_reproj = sum(t["reproj"]) / len(t["reproj"]) if t["reproj"] else None
        mean_dist = sum(t["dists"]) / len(t["dists"]) if t["dists"] else None
        in_map = t["in_map"] > 0

        if not in_map:
            verdict, reason, sev = "OFF-MAP", "detected but not in the tag map", 5
        elif worst_reproj is not None and worst_reproj > _MISMAP_RMS_PX:
            verdict, reason, sev = ("MIS-MAPPED",
                "worst reproj %.1f px -- re-measure its map pose/size" % worst_reproj, 4)
        elif seen_frac < _RARELY_SEEN_FRAC:
            verdict, reason, sev = ("RARELY SEEN",
                "in view only %.0f%% of frames -- reposition / add coverage"
                % (100 * seen_frac), 3)
        elif mean_margin < _LOW_MARGIN or med_px < _SMALL_PX:
            verdict, reason, sev = ("HARD TO READ",
                "margin %.0f, median %.0f px -- bigger tag / light / angle"
                % (mean_margin, med_px), 2)
        elif used_frac < _OFTEN_DROPPED_USED:
            verdict, reason, sev = ("OFTEN DROPPED",
                "trusted in only %.0f%% of sightings -- check map entry"
                % (100 * used_frac), 1)
        else:
            verdict, reason, sev = "GOOD", "", 0

        out.append({
            "tag_id": t["tag_id"], "verdict": verdict, "reason": reason,
            "_sev": sev, "seen": t["seen"], "seen_frac": seen_frac,
            "used_frac": used_frac, "mean_margin": mean_margin, "median_px": med_px,
            "mean_reproj_px": mean_reproj, "worst_reproj_px": worst_reproj,
            "mean_dist_m": mean_dist})
    out.sort(key=lambda d: (-d["_sev"], -d["seen"]))
    return out


def _fmt(value, spec: str) -> str:
    return "-" if value is None else spec % value


def main(argv: Optional[List[str]] = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv:
        print(__doc__)
        return 2
    rows = list(csv.DictReader(open(argv[0])))
    summary = summarize(rows)
    frames = len({r.get("stamp") for r in rows})
    print("AprilTag quality over %d frames  (%s)\n" % (frames, argv[0]))
    print("%-6s %-13s %6s %6s %6s %7s %7s %7s  %s" % (
        "tag", "verdict", "seen%", "used%", "margin", "med_px",
        "reproj", "dist_m", "note"))
    print("-" * 92)
    for s in summary:
        print("%-6d %-13s %5.0f%% %5.0f%% %6.0f %7.0f %7s %7s  %s" % (
            s["tag_id"], s["verdict"], 100 * s["seen_frac"],
            100 * s["used_frac"], s["mean_margin"], s["median_px"],
            _fmt(s["worst_reproj_px"], "%.1f"), _fmt(s["mean_dist_m"], "%.1f"),
            s["reason"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
