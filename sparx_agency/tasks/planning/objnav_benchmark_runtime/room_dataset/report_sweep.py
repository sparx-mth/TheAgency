"""What the choice of views does to the target-in-room score, from a sweep of every position and heading of each room: the rules for choosing views, how many to take, where to stand, and the best any choice could do."""
import argparse
import json
from pathlib import Path

import numpy as np

from sparx_agency.tasks.planning.objnav_benchmark_runtime.room_dataset.mp3d_target_score import kept_frames, vlm_file
from sparx_agency.tasks.planning.objnav_benchmark_runtime.room_dataset.report_target_score import frame_scores, read_rows, room_infos, vlm_scores
from sparx_agency.tasks.planning.objnav_benchmark_runtime.room_dataset.sweep_rules import frame_picks, usable_frames
from sparx_agency.tasks.planning.objnav_benchmark_runtime.room_dataset.target_score import TARGETS, auroc, room_holds

DRAWS = 10
BOOTSTRAP = 500
MIN_EACH = 3
COUNTS = (1, 3, 6)


def mean_of(values):
    values = [v for v in values if v is not None and not np.isnan(v)]
    return float(np.mean(values)) if values else None


def load_sweep(views):
    """``{room: {"scene", "frames", "visible"}}`` of the sweep folder."""
    rooms = {}
    for folder in sorted(d for d in views.iterdir() if (d / "selection.json").exists()):
        for room, entry in json.loads((folder / "selection.json").read_text()).items():
            if "frames" in entry:
                rooms[room] = {"scene": folder.name, "frames": entry["frames"], "visible": entry["visible"]}
    return rooms


def rule_scores(sweep, answers, rule, count, rng):
    """``{(room, target): score}`` choosing ``count`` views by ``rule`` (the mean over the views) or, for ``oracle``, the views that agree best with the truth."""
    out = {}
    for room, entry in sweep.items():
        picks = frame_picks(entry["frames"], {"all": "floor", "oracle": "random"}.get(rule, rule), 10 ** 6 if rule == "all" else count, rng)
        for target in TARGETS:
            by_frame = {f: answers.get((room, target), {}).get(f) for f in usable_frames(entry["frames"])}
            if rule == "oracle":
                ranked = sorted((v, f) for f, v in by_frame.items() if v is not None)
                holds = entry["holds"][target]
                chosen = [f for _, f in (ranked[-count:] if holds else ranked[:count])]
                out[(room, target)] = mean_of([by_frame[f] for f in chosen])
            else:
                out[(room, target)] = mean_of([by_frame.get(f) for f in picks])
    return out


def mean_auroc(per_room, rooms, holds, hidden, only_hidden):
    """Mean over targets of the AUROC over ``rooms``; ``only_hidden`` keeps the rooms where the target is in view of no frame."""
    values = []
    for target in TARGETS:
        keep = [r for r in rooms if per_room.get((r, target)) is not None and (not only_hidden or hidden[(r, target)])]
        labels = [holds[r][target] for r in keep]
        if sum(labels) >= MIN_EACH and len(labels) - sum(labels) >= MIN_EACH:
            values.append(auroc([per_room[(r, target)] for r in keep], labels))
    return float(np.mean(values)) if values else float("nan")


def bootstrap(by_draw, rooms_by_scene, holds, hidden, only_hidden, rng, baseline=None):
    """``(mean, low, high)`` of the mean AUROC (minus ``baseline``'s when given), resampling scenes; ``by_draw`` is a list of score dicts averaged over."""
    scenes = sorted(rooms_by_scene)

    def value(pick):
        rooms = [r for s in pick for r in rooms_by_scene[s]]
        got = float(np.nanmean([mean_auroc(d, rooms, holds, hidden, only_hidden) for d in by_draw]))
        return got - float(np.nanmean([mean_auroc(d, rooms, holds, hidden, only_hidden) for d in baseline])) if baseline else got
    point = value(scenes)
    draws = [value([scenes[i] for i in rng.integers(0, len(scenes), len(scenes))]) for _ in range(BOOTSTRAP)]
    return point, float(np.nanpercentile(draws, 2.5)), float(np.nanpercentile(draws, 97.5))


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--rooms", type=Path, required=True)
    p.add_argument("--views", type=Path, required=True)
    p.add_argument("--scores", type=Path, required=True)
    p.add_argument("--reference", type=Path, help="the walk and door views and scores of the same rooms: a folder with walk, door and scores")
    p.add_argument("--vlm", default="qwen3-vl:4b-instruct")
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args(argv)
    rng = np.random.default_rng(0)
    sweep = load_sweep(args.views)
    answers = frame_scores(read_rows(args.scores / vlm_file(args.vlm, "walk", "percent")))
    info = {r["room"]: r for r in room_infos(args.rooms)}
    sweep = {r: e for r, e in sweep.items() if r in info and all((r, t) in answers for t in TARGETS)}
    for room, entry in sweep.items():
        entry["holds"] = {t: room_holds(info[room], t) for t in TARGETS}
    holds = {r: e["holds"] for r, e in sweep.items()}
    hidden = {(r, t): not any(v[t] for v in e["visible"].values()) for r, e in sweep.items() for t in TARGETS}
    rooms_by_scene = {}
    for room, entry in sweep.items():
        rooms_by_scene.setdefault(entry["scene"], []).append(room)
    rules = [("all views", "all", 1)] + [("%s, K=%d" % (label, k), rule, k) for k in COUNTS for label, rule in
                                         (("random", "random"), ("most floor", "floor"), ("nearest the door", "door"), ("farthest from the door", "far"), ("best any choice could do", "oracle"))]
    lines = ["Rooms swept: %d in %d scenes; views per room: median %d." % (len(sweep), len(rooms_by_scene), int(np.median([len(e["frames"]) for e in sweep.values()]))), ""]
    for only_hidden, title in ((False, "All rooms"), (True, "Rooms where the target is in view of no sweep view (the prior)")):
        scored = {}
        for label, rule, k in rules:
            draws = DRAWS if rule == "random" else 1
            scored[label] = [rule_scores(sweep, answers, rule, k, rng) for _ in range(draws)]
        lines += ["### " + title, "", "| choice of views | mean AUROC [95% over scenes] | minus random, same K |", "|---|---|---|"]
        for label, rule, k in rules:
            base = scored.get("random, K=%d" % k) if rule != "all" else None
            point = bootstrap(scored[label], rooms_by_scene, holds, hidden, only_hidden, rng)
            diff = bootstrap(scored[label], rooms_by_scene, holds, hidden, only_hidden, rng, base) if base and rule != "random" else None
            lines.append("| %s | %.3f [%.3f, %.3f] | %s |" % (label, *point, "%+.3f [%+.3f, %+.3f]" % diff if diff else ""))
        if args.reference:
            walk_f = {r: f for r, (_, f) in kept_frames(args.reference / "walk", 1).items()}
            door_f = {r: f for r, (_, f) in kept_frames(args.reference / "door", 1, ("door0", "door4")).items()}
            for label, frames_of, name in (("the free walk keeper (3 views)", walk_f, "walk"), ("the door peek (2 views)", door_f, "door")):
                ref = vlm_scores(read_rows(args.reference / "scores" / vlm_file(args.vlm, name, "percent")), frames_of)
                if any(r in frames_of for r in sweep):
                    point = bootstrap([ref], {s: [r for r in rs if r in frames_of] for s, rs in rooms_by_scene.items()}, holds, hidden, only_hidden, rng)
                    lines.append("| %s | %.3f [%.3f, %.3f] | |" % (label, *point))
        lines.append("")
    lines += ["### One view at a time: AUROC of a single view's score by where it stands (rooms where the target is in view of no sweep view)", "", "| view | views | AUROC |", "|---|---|---|"]
    entrance = sorted(f["entrance_m"] for e in sweep.values() for f in usable_frames(e["frames"]).values())
    floor = sorted(f["floor_share"] for e in sweep.values() for f in usable_frames(e["frames"]).values())
    cuts = lambda values: (values[len(values) // 3], values[2 * len(values) // 3])
    for name, key, values in (("distance to the entrance", "entrance_m", entrance), ("room floor in view", "floor_share", floor)):
        low, high = cuts(values)
        for label, test in (("lowest third", lambda v: v < low), ("middle third", lambda v: low <= v < high), ("highest third", lambda v: v >= high)):
            scores, labels = [], []
            for room, entry in sweep.items():
                for frame, f in usable_frames(entry["frames"]).items():
                    if test(f[key]):
                        for target in TARGETS:
                            value = answers[(room, target)].get(frame)
                            if value is not None and hidden[(room, target)]:
                                scores.append(value)
                                labels.append(entry["holds"][target])
            lines.append("| %s, %s | %d | %.3f |" % (name, label, len(scores), auroc(scores, labels)))
    args.out.write_text("\n".join(lines))
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
