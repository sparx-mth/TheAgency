"""The spin scores judged inside each house, as the search meets them: AUROC among a house's own rooms and the share of houses whose top-scored room holds the target, against the chance of a random room."""
import argparse
import json
from pathlib import Path

import numpy as np

from sparx_agency.tasks.planning.objnav_benchmark_runtime.room_dataset.mp3d_target_score import vlm_file
from sparx_agency.tasks.planning.objnav_benchmark_runtime.room_dataset.report_spins import METHODS, VLM, room_scores
from sparx_agency.tasks.planning.objnav_benchmark_runtime.room_dataset.report_target_score import frame_scores, read_rows, room_infos
from sparx_agency.tasks.planning.objnav_benchmark_runtime.room_dataset.target_score import TARGETS, auroc, room_holds

BOOTSTRAP = 500


def within_scene_metrics(houses):
    """Means over the houses that have rooms both holding and lacking the target; ``houses`` maps a house to ``[(score, holds)]``."""
    mixed = [rooms for rooms in houses.values() if any(h for _, h in rooms) and not all(h for _, h in rooms)]
    tops = []
    for rooms in mixed:
        best = max(s for s, _ in rooms)
        tied = [h for s, h in rooms if s == best]
        tops.append(sum(tied) / len(tied))
    mean = lambda values: float(np.mean(values)) if values else float("nan")
    return {"auroc": mean([auroc([s for s, _ in rooms], [h for _, h in rooms]) for rooms in mixed]), "top1": mean(tops),
            "chance": mean([sum(h for _, h in rooms) / len(rooms) for rooms in mixed]), "houses": len(mixed)}


def target_houses(scores, holds, hidden, rooms_by_scene, only_hidden):
    """``{scene: [one list of (score, holds) per target]}`` over the rooms with a score, and the hidden ones only on request."""
    out = {}
    for scene, rooms in rooms_by_scene.items():
        for target in TARGETS:
            kept = [(scores[(r, target)], holds[r][target]) for r in rooms if scores.get((r, target)) is not None and (not only_hidden or hidden[(r, target)])]
            out.setdefault(scene, []).append(kept)
    return out


def bootstrap(by_scene, rng):
    """The metrics of all scenes, and the 95% interval of AUROC and of top-1 minus chance when resampling scenes."""
    scenes = sorted(by_scene)

    def measure(pick):
        return within_scene_metrics({(i, j): unit for i, s in enumerate(pick) for j, unit in enumerate(by_scene[s])})
    point = measure(scenes)
    draws = [measure([scenes[i] for i in rng.integers(0, len(scenes), len(scenes))]) for _ in range(BOOTSTRAP)]
    interval = lambda values: (float(np.nanpercentile(values, 2.5)), float(np.nanpercentile(values, 97.5)))
    return point, interval([d["auroc"] for d in draws]), interval([d["top1"] - d["chance"] for d in draws])


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--rooms", type=Path, nargs="+", required=True)
    p.add_argument("--spins", type=Path, required=True, help="hm3d_spins: views and scores")
    p.add_argument("--vlm", default=VLM)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args(argv)
    rng = np.random.default_rng(0)
    spins = {}
    for folder in sorted(d for d in (args.spins / "views").iterdir() if (d / "selection.json").exists()):
        for room, entry in json.loads((folder / "selection.json").read_text()).items():
            spins[room] = dict(entry, scene=folder.name)
    answers = frame_scores(read_rows(args.spins / "scores" / vlm_file(args.vlm, "walk", "percent")))
    info = {r["room"]: r for rooms in args.rooms for r in room_infos(rooms)}
    spins = {r: e for r, e in spins.items() if r in info and all((r, t) in answers for t in TARGETS)}
    holds = {r: {t: room_holds(info[r], t) for t in TARGETS} for r in spins}
    hidden = {(r, t): not any(v[t] for v in e["visible"].values()) for r, e in spins.items() for t in TARGETS}
    rooms_by_scene = {}
    for room, entry in spins.items():
        rooms_by_scene.setdefault(entry["scene"], []).append(room)
    lines = ["Rooms: %d in %d houses, each judged against the other rooms of its own house (%s, original percent question)." % (len(spins), len(rooms_by_scene), args.vlm), ""]
    for only_hidden, title in ((False, "All rooms"), (True, "Rooms where the target is in none of the spin's views")):
        lines += ["### " + title, "", "| spin | house-target pairs | AUROC in house [95%] | top-1 hits | chance | top-1 minus chance [95%] |", "|---|---|---|---|---|---|"]
        for m in METHODS:
            scores = room_scores(spins, answers, m)
            (point, auroc_range, gain_range) = bootstrap(target_houses(scores, holds, hidden, rooms_by_scene, only_hidden), rng)
            lines.append("| %s | %d | %.3f [%.3f, %.3f] | %.3f | %.3f | %+.3f [%+.3f, %+.3f] |" % (
                m, point["houses"], point["auroc"], *auroc_range, point["top1"], point["chance"], point["top1"] - point["chance"], *gain_range))
        lines.append("")
    args.out.write_text("\n".join(lines))
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
