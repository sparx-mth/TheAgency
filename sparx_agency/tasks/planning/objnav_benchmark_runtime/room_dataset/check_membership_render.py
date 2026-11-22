"""On a sample of planner picks, render the recorded frames and compare three ways of saying a frame sees a frontier cluster: it faces the cluster, the line to it is free on the map, and the rendered depth shows nothing in front of it."""
import argparse
import os
from pathlib import Path

import numpy as np

from sparx_agency.tasks.planning.objnav_benchmark_runtime.hm3d.protocol import protocol_for
from sparx_agency.tasks.planning.objnav_benchmark_runtime.object_dataset.geometry import habitat_position
from sparx_agency.tasks.planning.objnav_benchmark_runtime.review_recording import MapAtStep, load_steps
from sparx_agency.tasks.planning.objnav_benchmark_runtime.room_dataset.frontier_membership import MapFrame, in_view, sees
from sparx_agency.tasks.planning.objnav_benchmark_runtime.room_dataset.geometry import visible_cells
from sparx_agency.tasks.planning.objnav_benchmark_runtime.room_dataset.report_target_score import read_rows
from sparx_agency.tasks.planning.objnav_benchmark_runtime.room_dataset.select_hm3d_views import Hm3dScene
from sparx_agency.tasks.planning.objnav_benchmark_runtime.room_dataset.select_planner_peeks import MAX_PITCH_RAD, habitat_pose, render_pose, trajectory

DEPTH_TOLERANCE_M = 0.3


def level_frames(trajectory_rows, last=None):
    """Indices of the steps with a level camera, only the latest ``last`` of them when given."""
    level = [i for i, row in enumerate(trajectory_rows) if abs(row["camera_pitch"]) <= MAX_PITCH_RAD]
    return level[-last:] if last else level


def agreement(clusters):
    """Shares of frames, summed over clusters: seen in depth among facing, seen in depth among those the map line allows, and the map line among those seen in depth."""
    total = {k: sum(c[k] for c in clusters) for k in ("facing", "los", "depth", "both")}
    ratio = lambda a, b: None if total[b] == 0 else total[a] / total[b]
    return {"depth_among_facing": ratio("depth", "facing"), "depth_among_los": ratio("both", "los"), "los_among_depth": ratio("both", "depth")}


def check_pick(scene, recording, pick, window, protocol, max_frames):
    steps, rows = load_steps(recording), trajectory(recording)
    maps = MapAtStep(recording)
    frame = MapFrame(maps.origin, maps.resolution)
    grid = maps.at(steps[pick]["step"], steps[pick]["method"].get("floor_id") or 0)[0]
    goals = [tuple(g["xy"]) for g in steps[pick]["method"]["search"]["frontier_goals"]]
    frames = [i for i in level_frames(rows[:pick + 1]) if any(in_view((steps[i]["pose"]["x"], steps[i]["pose"]["y"], steps[i]["pose"]["yaw"]), g) for g in goals)][-max_frames:]
    clusters = [{"facing": 0, "los": 0, "depth": 0, "both": 0, "facing_window": 0, "depth_window": 0} for _ in goals]
    for i in frames:
        pose = (steps[i]["pose"]["x"], steps[i]["pose"]["y"], steps[i]["pose"]["yaw"])
        rgb, depth, rendered = render_pose(scene, *habitat_pose(rows[i]))
        camera = (*habitat_position(dict(zip("xyz", rendered[:3]))), rendered[3])
        for goal, c in zip(goals, clusters):
            if not in_view(pose, goal):
                continue
            cell = habitat_position({"x": goal[0], "y": goal[1], "z": rendered[2]})[None]
            seen_depth = bool(visible_cells(depth, camera, cell, protocol.width, protocol.height, protocol.hfov_deg, DEPTH_TOLERANCE_M)[0])
            seen_map = sees(pose, goal, grid, frame)
            in_window = i > pick - window
            c["facing"] += 1
            c["facing_window"] += in_window
            c["depth"] += seen_depth
            c["depth_window"] += seen_depth and in_window
            c["los"] += seen_map
            c["both"] += seen_depth and seen_map
    return clusters


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--membership", type=Path, required=True, help="the rows of count_frontier_membership")
    p.add_argument("--batches", type=Path, nargs="+", required=True, help="train-scene planner run folders only")
    p.add_argument("--rooms", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--picks", type=int, default=30)
    p.add_argument("--max-frames", type=int, default=60, help="the latest facing level frames rendered per pick")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--scenes-dir", default=os.environ.get("HM3D_SCENES_DIR"))
    args = p.parse_args(argv)
    roots = [str(b) for b in args.batches]
    picks = {}
    for row in read_rows(args.membership):
        if any(row["recording"].startswith(r) for r in roots) and (args.rooms / Path(row["recording"]).parts[-3] / "cells.npz").exists():
            picks[(row["recording"], row["pick_step"])] = row["window"]
    keys = sorted(picks)
    chosen = [keys[i] for i in np.random.default_rng(args.seed).permutation(len(keys))[:args.picks]]
    protocol = protocol_for("v1").with_split("train")
    by_scene = {}
    for recording, pick in chosen:
        by_scene.setdefault(Path(recording).parts[-3], []).append((recording, pick))
    clusters = []
    for scene_name, items in sorted(by_scene.items()):
        scene = Hm3dScene(scene_name, args.scenes_dir, protocol, 17, args.rooms / scene_name)
        for recording, pick in items:
            clusters += check_pick(scene, Path(recording), pick, picks[(recording, pick)], protocol, args.max_frames)
        scene.close()
        print(scene_name, len(clusters), "clusters so far", flush=True)
    lines = ["Picks: %d in %d scenes, clusters: %d (level-camera frames only, the latest %d facing frames per pick)." % (len(chosen), len(by_scene), len(clusters), args.max_frames), "",
             "| count over clusters | facing the cluster | line free on the map | rendered depth shows it | depth and map agree |", "|---|---|---|---|---|"]
    for label, keys_ in (("every frame so far", ("facing", "los", "depth", "both")),):
        lines.append("| %s | %s |" % (label, " | ".join(str(sum(c[k] for c in clusters)) for k in keys_)))
    agree = agreement(clusters)
    lines += ["", "Rendered depth shows the cluster in %.0f%% of the frames that face it and in %.0f%% of the frames the map line allows; the map line allows %.0f%% of the frames the depth shows." % tuple(100 * (agree[k] or 0) for k in ("depth_among_facing", "depth_among_los", "los_among_depth")), "",
              "| frames per cluster | median | share of clusters seen by 0 | 1-2 | 3-5 | 6+ |", "|---|---|---|---|---|---|"]
    for label, key in (("facing, window", "facing_window"), ("depth, window", "depth_window"), ("depth, rendered history", "depth")):
        counts = [c[key] for c in clusters]
        shares = [np.mean([lo <= n <= hi for n in counts]) for lo, hi in ((0, 0), (1, 2), (3, 5), (6, 10 ** 9))]
        lines.append("| %s | %d | %s |" % (label, int(np.median(counts)), " | ".join("%.0f%%" % (100 * s) for s in shares)))
    args.out.write_text("\n".join(lines))
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
