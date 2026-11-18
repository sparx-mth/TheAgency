"""Room ordering by RPT* given each source's score per room, on HM3D houses: the walk until the first room that holds the target, from every non-holding room as the start."""
import argparse
import json
from pathlib import Path

import numpy as np

from sparx_agency.core.planning.routing.rpt_star import metric_closure
from sparx_agency.tasks.planning.objnav_benchmark_runtime.room_dataset.hm3d_room_distances import FILE
from sparx_agency.tasks.planning.objnav_benchmark_runtime.room_dataset.mp3d_target_score import VLM_FILE, kept_frames
from sparx_agency.tasks.planning.objnav_benchmark_runtime.room_dataset.report_target_score import read_rows, room_infos, visible_anywhere, vlm_scores
from sparx_agency.tasks.planning.objnav_benchmark_runtime.room_dataset.rpt_order_eval import adjusted_probability, efficiency, nearest_first, rpt_order, walk_cost
from sparx_agency.tasks.planning.objnav_benchmark_runtime.room_dataset.target_score import TARGETS, room_holds

UNREACHABLE = 1e4
RANDOM_DRAWS = 20
BOOTSTRAP = 1000


def episode_matrix(distances, start, rooms):
    """The solver's matrix: the start (middle of room ``start``) at 0, then ``rooms``, as shortest routes (RPT* needs the triangle inequality); a pair with no route at all costs ``UNREACHABLE``."""
    points = [start] + list(rooms)
    direct = [[0.0 if a == b else UNREACHABLE if distances[a][b] is None else distances[a][b] for b in points] for a in points]
    closed = metric_closure(direct)
    return [[0.0 if i == j else min(UNREACHABLE, closed[i][j]) for j in range(len(points))] for i in range(len(points))]


def area_percent(areas):
    """Room area as a percent within a house: the largest 100, the smallest 0."""
    low, high = min(areas), max(areas)
    return [100.0 * (a - low) / (high - low) if high > low else 50.0 for a in areas]


def episode_results(percents, holders, matrix, rng):
    """``{source: (distance, rooms, efficiency)}`` for one episode; ``percents`` maps a score source to its percent per candidate room."""
    out = {}
    orders = {name: rpt_order([adjusted_probability(p) for p in values], matrix) for name, values in percents.items()}
    orders["nearest_first"] = nearest_first(matrix)
    orders["oracle"] = rpt_order([0.99 if i in holders else 0.01 for i in range(len(matrix) - 1)], matrix)
    for name, order in orders.items():
        d, v = walk_cost(order, matrix, holders)
        out[name] = (d, v, efficiency(d, matrix, holders))
    draws = [walk_cost(list(rng.permutation(len(matrix) - 1)), matrix, holders) for _ in range(RANDOM_DRAWS)]
    out["random"] = (float(np.mean([d for d, _ in draws])), float(np.mean([v for _, v in draws])), float(np.mean([efficiency(d, matrix, holders) for d, _ in draws])))
    return out


def cluster_bootstrap(per_scene, rng):
    """``(mean, low, high)`` of the mean over episodes, resampling scenes; ``per_scene`` is a list of lists of values."""
    flat = [v for scene in per_scene for v in scene]
    means = []
    for _ in range(BOOTSTRAP):
        pick = rng.integers(0, len(per_scene), len(per_scene))
        values = [v for i in pick for v in per_scene[i]]
        means.append(float(np.mean(values)))
    return float(np.mean(flat)), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--rooms", type=Path, required=True)
    p.add_argument("--walk", type=Path, required=True)
    p.add_argument("--door", type=Path, required=True)
    p.add_argument("--scores", type=Path, required=True)
    p.add_argument("--vlm", default="qwen3-vl:4b-instruct")
    args = p.parse_args(argv)
    model = args.vlm.replace(":", "_")
    walk_f = {r: f for r, (_, f) in kept_frames(args.walk, 1).items()}
    door0_f = {r: f for r, (_, f) in kept_frames(args.door, 1, ("door0",)).items()}
    door_f = {r: f for r, (_, f) in kept_frames(args.door, 1, ("door0", "door4")).items()}
    walk_rows, door_rows = read_rows(args.scores / (VLM_FILE % (model, "walk"))), read_rows(args.scores / (VLM_FILE % (model, "door")))
    scores = {"vlm_walk": vlm_scores(walk_rows, walk_f), "vlm_door": vlm_scores(door_rows, door_f), "vlm_door0": vlm_scores(door_rows, door0_f)}
    info = {r["room"]: r for r in room_infos(args.rooms)}
    distances = json.loads((args.rooms / FILE).read_text())
    seen = visible_anywhere(args.door)
    rng, results = np.random.default_rng(0), {"all": {}, "hidden": {}}
    for scene, entry in distances.items():
        names = entry["rooms"]
        index = {n: i for i, n in enumerate(names)}
        usable = [n for n in names if n in walk_f and n in door_f and all(scores[m].get((n, t)) is not None for m in scores for t in TARGETS)]
        for target in TARGETS:
            for start in usable:
                if room_holds(info[start], target):
                    continue
                matrix = episode_matrix(entry["matrix"], index[start], [index[n] for n in usable if n != start])
                rooms = [n for n, row in zip([n for n in usable if n != start], matrix[0][1:]) if row < UNREACHABLE]
                holders = {n for n in rooms if room_holds(info[n], target)}
                if not holders:
                    continue
                matrix = episode_matrix(entry["matrix"], index[start], [index[n] for n in rooms])
                held = {i for i, n in enumerate(rooms) if n in holders}
                percents = {m: [scores[m][(n, target)] for n in rooms] for m in scores}
                percents["area"] = area_percent([info[n]["area_m2"] for n in rooms])
                triples = episode_results(percents, held, matrix, rng)
                kinds = ["all"] + (["hidden"] if not any(seen.get((n, target), True) for n in holders) else [])
                for kind in kinds:
                    for name, triple in triples.items():
                        results[kind].setdefault(name, {}).setdefault(scene, []).append(triple)
    lines = []
    for kind, title in (("all", "All episodes"), ("hidden", "Episodes where the target is in view of no door or kept walk frame of any room that holds it")):
        lines += ["### " + title, "", "| source | episodes | mean walk (m) | mean rooms entered | efficiency [95% over scenes] | efficiency minus nearest-first |", "|---|---|---|---|---|---|"]
        base = results[kind]["nearest_first"]
        for name, by_scene in results[kind].items():
            scenes = sorted(by_scene)
            eff = cluster_bootstrap([[t[2] for t in by_scene[s]] for s in scenes], rng)
            diff = cluster_bootstrap([[a[2] - b[2] for a, b in zip(by_scene[s], base[s])] for s in scenes], rng)
            lines.append("| %s | %d | %.1f | %.2f | %.3f [%.3f, %.3f] | %+.3f [%+.3f, %+.3f] |" % (
                name, sum(len(v) for v in by_scene.values()), np.mean([t[0] for s in scenes for t in by_scene[s]]), np.mean([t[1] for s in scenes for t in by_scene[s]]), *eff, *diff))
        lines.append("")
    text = "\n".join(lines)
    (args.scores / "report_rpt_order.md").write_text(text)
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
