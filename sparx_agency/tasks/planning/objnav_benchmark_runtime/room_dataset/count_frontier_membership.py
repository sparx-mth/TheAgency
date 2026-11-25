"""For every frontier cluster the planner listed when it picked a goal, count the frames since its previous pick that saw the cluster; one jsonl row per cluster, from recorded planner runs."""
import argparse
import json
import math
from functools import partial
from multiprocessing import Pool
from pathlib import Path

from sparx_agency.tasks.planning.objnav_benchmark_runtime.review_recording import MapAtStep, load_steps
from sparx_agency.tasks.planning.objnav_benchmark_runtime.room_dataset.frontier_membership import MAX_RANGE_M, MapFrame, FREE, in_view, plan_steps, sees

RECENT = (10, 20)
YAW_BIN_DEG = 30


def recording_rows(recording, max_range_m=MAX_RANGE_M):
    steps = load_steps(recording)
    maps = MapAtStep(recording)
    frame = MapFrame(maps.origin, maps.resolution)
    states = [s["method"]["state"] for s in steps]
    after_warmup = [next(i for i, state in enumerate(states) if state != "warmup")] if "warmup" in states and set(states) != {"warmup"} else []
    picks = sorted(set(plan_steps(states)) | set(after_warmup))
    rows, previous = [], -1
    for pick in picks:
        goals = steps[pick]["method"]["search"]["frontier_goals"]
        window = steps[previous + 1:pick + 1]
        grid = maps.at(steps[pick]["step"], steps[pick]["method"].get("floor_id") or 0)[0]
        poses = [(s["pose"]["x"], s["pose"]["y"], s["pose"]["yaw"]) for s in window]
        history = [(s["pose"]["x"], s["pose"]["y"], s["pose"]["yaw"]) for s in steps[:pick + 1]]
        for goal in goals:
            ever = [i for i, pose in enumerate(history) if sees(pose, tuple(goal["xy"]), grid, frame, max_range_m=max_range_m)]
            seen = [i - (previous + 1) for i in ever if i > previous]
            facing = [i for i, pose in enumerate(history) if in_view(pose, tuple(goal["xy"]), max_range_m=max_range_m)]
            yaws = {int(math.degrees(poses[i][2]) // YAW_BIN_DEG) % (360 // YAW_BIN_DEG) for i in seen}
            rows.append({"recording": str(recording), "pick_step": pick, "window": len(window), "seen": len(seen), "seen_history": len(ever), "facing": sum(i > previous for i in facing), "facing_history": len(facing), "age_of_latest": pick - ever[-1] if ever else None, "yaw_bins": len(yaws),
                         **{"seen_last%d" % k: sum(i >= len(poses) - k for i in seen) for k in RECENT},
                         "room": goal["room"], "size_cells": goal["size_cells"], "geodesic_m": goal["geodesic_m"],
                         "pose_on_free": float(sum(grid[frame.cell_of(p[0], p[1])[::-1]] == FREE for p in poses)) / max(len(poses), 1)})
        previous = pick
    return rows


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--batches", type=Path, nargs="+", required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--max-range", type=float, default=MAX_RANGE_M, help="metres a frame can see a cluster from")
    p.add_argument("--limit", type=int, help="only the first N recordings")
    args = p.parse_args(argv)
    every = sorted(r for b in args.batches for r in b.glob("*/recordings/*") if (r / "steps.jsonl").exists())
    recordings = [r for r in every if (r / "map_journal.npz").exists()][:args.limit]
    print("recordings %d, with the map of every action %d" % (len(every), len(recordings)), flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with Pool(args.workers) as pool, open(args.out, "w") as handle:
        for done, rows in enumerate(pool.imap(partial(recording_rows, max_range_m=args.max_range), recordings), 1):
            handle.writelines(json.dumps(r) + "\n" for r in rows)
            print(done, "of", len(recordings), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
