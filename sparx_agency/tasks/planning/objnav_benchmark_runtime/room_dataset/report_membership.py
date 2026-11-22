"""How many frames since the previous pick see each frontier cluster: the share of clusters seen by none, 1-2, 3-5 and 6 or more frames, for all frames of the window and for the last 10 or 20 steps of it."""
import argparse
import json
from pathlib import Path

import numpy as np

BUCKETS = (("0", 0, 0), ("1-2", 1, 2), ("3-5", 3, 5), ("6+", 6, 10 ** 9))


def bucket_shares(counts):
    return [float(np.mean([low <= c <= high for c in counts])) for _, low, high in BUCKETS]


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--rows", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args(argv)
    rows = [json.loads(line) for line in args.rows.read_text().splitlines()]
    picks = {(r["recording"], r["pick_step"]) for r in rows}
    lines = ["Clusters: %d, at %d picks in %d recordings. Window: median %d frames (the frames since the previous pick)." % (
        len(rows), len(picks), len({r["recording"] for r in rows}), int(np.median([r["window"] for r in rows]))), "",
        "| frames counted | median per cluster | " + " | ".join("seen by %s" % b[0] for b in BUCKETS) + " |", "|---|---|" + "---|" * len(BUCKETS)]
    for label, key in (("whole window", "seen"), ("last 10 steps", "seen_last10"), ("last 20 steps", "seen_last20"), ("every frame since the start", "seen_history"), ("whole window, no line-of-sight test", "facing"), ("every frame since the start, no line-of-sight test", "facing_history")):
        counts = [r[key] for r in rows]
        lines.append("| %s | %d | %s |" % (label, int(np.median(counts)), " | ".join("%.0f%%" % (100 * s) for s in bucket_shares(counts))))
    ages = [r["age_of_latest"] for r in rows if r["age_of_latest"] is not None]
    lines += ["", "Steps since the latest frame that saw the cluster (clusters seen at all): median %d, 90th percentile %d." % (np.median(ages), np.percentile(ages, 90)), "",
              "Different 30-degree headings among the frames that see a cluster (clusters seen by at least one frame): median %d." % int(np.median([r["yaw_bins"] for r in rows if r["seen"]])),
              "Poses on known free cells (a sanity check of the frames): %.0f%%." % (100 * np.mean([r["pose_on_free"] for r in rows])), "",
              "| cluster size (cells) | clusters | median frames seeing it |", "|---|---|---|"]
    for label, low, high in (("4-7", 4, 7), ("8-15", 8, 15), ("16+", 16, 10 ** 9)):
        part = [r["seen"] for r in rows if low <= r["size_cells"] <= high]
        lines.append("| %s | %d | %s |" % (label, len(part), int(np.median(part)) if part else "-"))
    args.out.write_text("\n".join(lines))
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
