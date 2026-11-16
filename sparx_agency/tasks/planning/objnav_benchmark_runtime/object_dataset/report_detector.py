"""Markdown report of one detector on the object views: recall and wrong boxes by category and distance, random views, K-view voting."""
import argparse
import json
from pathlib import Path

from sparx_agency.tasks.planning.objnav_benchmark_runtime.object_dataset.geometry import is_held_out
from sparx_agency.tasks.planning.objnav_benchmark_runtime.object_dataset.scoring import (
    KIND_OF_CATEGORY, score_view, summarise, vote_summary)

MIN_TARGET_PIXELS = 1000
THRESHOLDS = (0.3, 0.5, 0.7)
BUCKETS = (0.8, 1.2, 1.8, 2.5, 3.5, 4.5)


def load(views, name, held_out_fraction, include_held_out):
    rows = []
    for folder in sorted(d for d in views.iterdir() if (d / ("det_%s.json" % name)).exists()):
        if is_held_out(folder.name, held_out_fraction) and not include_held_out:
            continue
        found = json.loads((folder / ("det_%s.json" % name)).read_text())
        rows += [(json.loads(line), found) for line in (folder / "views.jsonl").read_text().splitlines()]
    return rows


def scored_views(rows, threshold):
    objects = []
    for row, found in rows:
        if row["role"] != "object":
            continue
        bucket = min(BUCKETS, key=lambda d: abs(d - row["distance_m"]))
        score = score_view(row, found[row["frame"]], threshold)
        usable = row["target_visible"] and score["target_pixels"] >= MIN_TARGET_PIXELS and not score["target_truncated"]
        objects.append(dict(score, target_visible=usable, bucket=bucket,
                            category=row["category"], scene=row["scene"], target_instance=row["target_instance"],
                            sequence=row["sequence"], step=row["step"]))
    return objects


def random_view_alarms(rows, threshold):
    views = [(row, found) for row, found in rows if row["role"] == "random"]
    wrong = sum(score_view(row, found[row["frame"]], threshold, category=c)["wrong"] for row, found in views for c in KIND_OF_CATEGORY)
    return len(views), wrong


def table(summary, label):
    lines = ["| %s | views | usable target | recall (target) | recall (any goal object) | wrong boxes per view |" % label, "|---|---|---|---|---|---|"]
    for name, cell in summary.items():
        percent = lambda v: "-" if v is None else "%.0f%%" % (100 * v)
        lines.append("| %s | %d | %d | %s | %s | %.2f |" % (name, cell["views"], cell["visible"], percent(cell["recall"]),
                                                           percent(cell["recall_goal"]), cell["wrong_per_view"]))
    return "\n".join(lines)


def report(rows, name):
    out = ["# Detector `%s` on the object views (%d views)\n" % (name, len(rows)),
           "A target is usable when at least %d px of it are visible and the frame does not cut it. A box is a hit on the target when it covers that instance (IoU 0.3), "
           "a hit on the goal set when it covers any goal object of the kind, and wrong when it lies on an object of the kind outside the goal set or on nothing.\n" % MIN_TARGET_PIXELS]
    for threshold in THRESHOLDS:
        scored = scored_views(rows, threshold)
        visible = [r for r in scored if r["target_visible"]]
        views, alarms = random_view_alarms(rows, threshold)
        out += ["## Confidence >= %.1f\n" % threshold,
                "Overall: recall %.0f%% on the target and %.0f%% on the goal set over %d usable views; %d views have the target cut by the frame or tiny; "
                "wrong boxes per object view %.2f; on %d random views %.2f wrong boxes per view over the 6 kinds.\n"
                % (100 * sum(r["hit_target"] for r in visible) / max(1, len(visible)),
                   100 * sum(r["hit_goal"] for r in visible) / max(1, len(visible)), len(visible), len(scored) - len(visible),
                   sum(r["wrong"] for r in scored) / max(1, len(scored)), views, alarms / max(1, views)),
                table(summarise(scored, "category"), "category"), "", table(summarise(scored, "bucket"), "distance m"), ""]
        vote = vote_summary(scored, need=2)
        out += ["K-view vote (2 of the 3 headings at one approach step), %d groups: a box on the target in %.1f%% of groups by vote vs %.1f%% of single views; "
                "a wrong box in %.1f%% of groups by vote vs %.1f%% of single views.\n"
                % (vote["groups"], 100 * vote["hit"] / max(1, vote["groups"]), 100 * vote["single_hit"] / max(1, vote["views"]),
                   100 * vote["wrong"] / max(1, vote["groups"]), 100 * vote["single_wrong"] / max(1, vote["views"]))]
    return "\n".join(out)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--views", type=Path, required=True)
    p.add_argument("--name", required=True)
    p.add_argument("--held-out-fraction", type=float, default=0.25)
    p.add_argument("--include-held-out", action="store_true")
    args = p.parse_args(argv)
    text = report(load(args.views, args.name, args.held_out_fraction, args.include_held_out), args.name)
    (args.views / ("report_%s.md" % args.name)).write_text(text)
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
