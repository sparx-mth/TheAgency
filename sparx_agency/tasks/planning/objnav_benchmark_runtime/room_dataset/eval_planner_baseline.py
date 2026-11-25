"""The planner's own room belief (its LLM probability and its exit count) against the VLM door and walk scores, on the rooms each recorded planner episode walked into, for that episode's own target: AUROC and the RPT* ordering test."""
import argparse
import json
from pathlib import Path

import numpy as np

from sparx_agency.tasks.planning.objnav_benchmark_runtime.review_recording import load_steps
from sparx_agency.tasks.planning.objnav_benchmark_runtime.room_dataset.eval_rpt_order import UNREACHABLE, area_percent, cluster_bootstrap, episode_matrix, episode_results
from sparx_agency.tasks.planning.objnav_benchmark_runtime.room_dataset.hm3d_room_distances import FILE
from sparx_agency.tasks.planning.objnav_benchmark_runtime.room_dataset.mp3d_target_score import kept_frames, vlm_file
from sparx_agency.tasks.planning.objnav_benchmark_runtime.room_dataset.planner_peeks import pose_visits, region_at
from sparx_agency.tasks.planning.objnav_benchmark_runtime.room_dataset.planner_scores import exits_percent, planner_view_of_room
from sparx_agency.tasks.planning.objnav_benchmark_runtime.room_dataset.report_target_score import read_rows, room_infos, visible_anywhere, vlm_scores
from sparx_agency.tasks.planning.objnav_benchmark_runtime.room_dataset.select_planner_peeks import MIN_STEPS_INSIDE, habitat_pose, trajectory
from sparx_agency.tasks.planning.objnav_benchmark_runtime.room_dataset.target_score import TARGETS, auroc, room_holds

MIN_ROOMS = 3
SOURCES = ("vlm_walk", "vlm_door", "area", "planner_prob", "exits_many", "exits_few")


def planner_episodes(batch, rooms_dir):
    """``(scene, target, {room name: planner view})`` for each recording: the rooms its poses stood in for a stretch, with what the planner believed on arrival."""
    for scene_dir in sorted(d for d in batch.iterdir() if (d / "recordings").is_dir()):
        scene = scene_dir.name
        if not (rooms_dir / scene / "rooms.json").exists():
            continue
        by_region = {r["region"]: r["room"] for r in json.loads((rooms_dir / scene / "rooms.json").read_text())}
        cells = np.load(rooms_dir / scene / "cells.npz")
        for recording in sorted((scene_dir / "recordings").iterdir()):
            target = json.loads((recording / "episode.json").read_text())["target"]
            steps, poses = load_steps(recording), trajectory(recording)
            regions = [region_at(habitat_pose(row)[0], cells["cells"], cells["regions"]) for row in poses]
            seen = {}
            for region, (arrival, end) in pose_visits(regions, MIN_STEPS_INSIDE).items():
                view = planner_view_of_room(steps, arrival, min(end, len(steps) - 1)) if region in by_region and arrival < len(steps) else None
                if view and view["probability"] is not None and view["exits"] is not None:
                    seen[by_region[region]] = view
            yield scene, target, seen


def dataset_episodes(batch, rooms_dir, peeks, style):
    """The usable ordering episodes of one planner run: rooms with a planner view and VLM scores, and what is needed to score them."""
    walk_f = {r: f for r, (_, f) in kept_frames(peeks / "walk", 1).items()}
    door_f = {r: f for r, (_, f) in kept_frames(peeks / "door", 1, ("door0", "door4")).items()}
    walk_rows = read_rows(peeks / "scores" / vlm_file("qwen3-vl:4b-instruct", "walk", style))
    door_rows = read_rows(peeks / "scores" / vlm_file("qwen3-vl:4b-instruct", "door", style))
    vlm = {"vlm_walk": vlm_scores(walk_rows, walk_f), "vlm_door": vlm_scores(door_rows, door_f)}
    info = {r["room"]: r for r in room_infos(rooms_dir)}
    distances = json.loads((rooms_dir / FILE).read_text())
    seen = visible_anywhere(peeks / "door")
    for scene, target, views in planner_episodes(batch, rooms_dir):
        if target not in TARGETS or scene not in distances:
            continue
        names = [n for n in views if n in distances[scene]["rooms"] and n in info and all(vlm[m].get((n, target)) is not None for m in vlm)]
        if len(names) >= MIN_ROOMS:
            yield scene, target, names, views, vlm, info, distances[scene], seen


def episode_percents(names, views, vlm, info, target):
    exits = [views[n]["exits"] for n in names]
    return {"vlm_walk": [vlm["vlm_walk"][(n, target)] for n in names], "vlm_door": [vlm["vlm_door"][(n, target)] for n in names],
            "area": area_percent([info[n]["area_m2"] for n in names]), "planner_prob": [100.0 * views[n]["probability"] for n in names],
            "exits_many": exits_percent(exits), "exits_few": exits_percent(exits, few_first=True)}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--set", nargs=3, action="append", metavar=("BATCH", "ROOMS", "PEEKS"), required=True)
    p.add_argument("--style", default="percent")
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args(argv)
    rng = np.random.default_rng(0)
    order_results, room_scores, counts = {}, {}, {"episodes": 0, "rooms": 0}
    for batch, rooms_dir, peeks in args.set:
        for scene, target, names, views, vlm, info, entry, seen in dataset_episodes(Path(batch), Path(rooms_dir), Path(peeks), args.style):
            counts["episodes"] += 1
            counts["rooms"] += len(names)
            percents = episode_percents(names, views, vlm, info, target)
            holds = [room_holds(info[n], target) for n in names]
            hidden = not any(seen.get((n, target), True) for n, h in zip(names, holds) if h)
            for kind in ("all",) + (("hidden",) if hidden else ()):
                for source, values in percents.items():
                    room_scores.setdefault(kind, {}).setdefault(source, {}).setdefault(scene, []).append((values, holds))
            index = {n: i for i, n in enumerate(entry["rooms"])}
            for start in names:
                if room_holds(info[start], target):
                    continue
                others = [n for n in names if n != start]
                matrix = episode_matrix(entry["matrix"], index[start], [index[n] for n in others])
                reachable = [i for i, n in enumerate(others) if matrix[0][i + 1] < UNREACHABLE]
                held = {j for j, i in enumerate(reachable) if room_holds(info[others[i]], target)}
                if not held:
                    continue
                kept = [others[i] for i in reachable]
                matrix = episode_matrix(entry["matrix"], index[start], [index[n] for n in kept])
                sub = {s: [percents[s][names.index(n)] for n in kept] for s in percents}
                triples = episode_results(sub, held, matrix, rng)
                for kind in ("all",) + (("hidden",) if hidden else ()):
                    for name, triple in triples.items():
                        order_results.setdefault(kind, {}).setdefault(name, {}).setdefault(scene, []).append(triple)
    lines = ["Planner episodes with at least %d rooms having a planner view and VLM scores: %d episodes, %d rooms." % (MIN_ROOMS, counts["episodes"], counts["rooms"]), ""]
    for kind in ("all", "hidden"):
        if kind not in order_results:
            continue
        lines += ["### Ordering, %s" % ("all episodes" if kind == "all" else "target in view of no frame of any room that holds it"), "",
                  "| source | episodes | efficiency [95% over scenes] | minus nearest-first |", "|---|---|---|---|"]
        base = order_results[kind]["nearest_first"]
        for name, by_scene in order_results[kind].items():
            scenes = sorted(by_scene)
            eff = cluster_bootstrap([[t[2] for t in by_scene[s]] for s in scenes], rng)
            diff = cluster_bootstrap([[a[2] - b[2] for a, b in zip(by_scene[s], base[s])] for s in scenes], rng)
            lines.append("| %s | %d | %.3f [%.3f, %.3f] | %+.3f [%+.3f, %+.3f] |" % (name, sum(len(v) for v in by_scene.values()), *eff, *diff))
        lines += ["", "### Which room holds the target (AUROC within an episode's rooms), %s" % kind, "", "| source | episodes with both kinds | mean AUROC [95% over scenes] |", "|---|---|---|"]
        for source in SOURCES:
            per_scene = {s: [a for a in (auroc(v, h) for v, h in items) if not np.isnan(a)] for s, items in room_scores[kind][source].items()}
            per_scene = [v for v in per_scene.values() if v]
            if per_scene:
                lines.append("| %s | %d | %.3f [%.3f, %.3f] |" % (source, sum(map(len, per_scene)), *cluster_bootstrap(per_scene, rng)))
        lines.append("")
    args.out.write_text("\n".join(lines))
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
