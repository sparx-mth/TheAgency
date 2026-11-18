"""How well rays on the occupancy grid stand in for the rendered depth when choosing room views: agreement of what each view sees, of which views are usable, and the floor coverage of the picks."""
import argparse
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

from sparx_agency.tasks.planning.objnav_benchmark_runtime.room_dataset.compare_grid_depth import SCORES
from sparx_agency.tasks.planning.objnav_benchmark_runtime.room_dataset.view_selection import select

PICKS = 3
RANDOM_DRAWS = 20


def coverage(cells, picks):
    return float(np.any(cells[picks], axis=0).mean()) if len(picks) else 0.0


def mean_overlap(a, b):
    union = (a | b).sum(axis=1)
    seeing = union > 0
    return float(((a & b).sum(axis=1)[seeing] / union[seeing]).mean()) if seeing.any() else float("nan")


def as_candidates(arrays, prefix):
    return [{"cells": c, "near": n, "inside": i, "objects": []} for c, n, i in zip(arrays[prefix + "cells"], arrays[prefix + "near"], arrays[prefix + "inside"])]


def compare_room(arrays, count, rng):
    depth, grid = as_candidates(arrays, "depth_"), as_candidates(arrays, "grid_")
    truth_cells = arrays["depth_cells"]
    usable_depth = np.array([c["near"] <= 0.4 and c["inside"] >= 0.7 for c in depth])
    usable_grid = np.array([c["near"] <= 0.4 and c["inside"] >= 0.7 for c in grid])
    seen_depth, seen_grid = truth_cells.sum(axis=1), arrays["grid_cells"].sum(axis=1)
    pool = np.flatnonzero(usable_depth)
    random = [coverage(truth_cells, list(rng.permutation(pool)[:count])) for _ in range(RANDOM_DRAWS)] if len(pool) else [0.0]
    return {"overlap": mean_overlap(truth_cells, arrays["grid_cells"]),
            "rank": float(spearmanr(seen_depth, seen_grid)[0]) if seen_depth.std() and seen_grid.std() else float("nan"),
            "usable_precision": float((usable_depth & usable_grid).sum() / max(1, usable_grid.sum())), "usable_recall": float((usable_depth & usable_grid).sum() / max(1, usable_depth.sum())),
            "depth_picks": coverage(truth_cells, select(depth, count, with_objects=False)), "grid_picks": coverage(truth_cells, select(grid, count, with_objects=False)),
            "random_picks": float(np.mean(random))}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args(argv)
    rng, rooms = np.random.default_rng(0), []
    for path in sorted(args.out.glob("*/" + SCORES)):
        loaded = np.load(path)
        for room in sorted({key.split("|")[0] for key in loaded.files}):
            rooms.append(compare_room({key.split("|")[1]: loaded[key] for key in loaded.files if key.split("|")[0] == room}, PICKS, rng))
    mean = lambda key: float(np.nanmean([r[key] for r in rooms]))
    ratio = np.array([r["grid_picks"] / r["depth_picks"] for r in rooms if r["depth_picks"] > 0])
    lines = ["%d regions, %d spots x 12 headings each" % (len(rooms), 24),
             "what a view sees, grid against depth: overlap (intersection over union) %.2f, rank correlation of cells seen %.2f" % (mean("overlap"), mean("rank")),
             "usable views (not a close-up, mostly inside): grid precision %.2f, recall %.2f" % (mean("usable_precision"), mean("usable_recall")),
             "floor coverage of the first %d picks, measured by depth: depth picks %.3f, grid picks %.3f, random usable views %.3f" % (PICKS, mean("depth_picks"), mean("grid_picks"), mean("random_picks")),
             "grid picks cover %.0f%% of what depth picks cover on average; at least 90%% in %.0f%% of regions, at least 75%% in %.0f%%" % (100 * ratio.mean(), 100 * (ratio >= 0.9).mean(), 100 * (ratio >= 0.75).mean())]
    text = "\n".join(lines)
    (args.out / "report_grid_vs_depth.txt").write_text(text)
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
