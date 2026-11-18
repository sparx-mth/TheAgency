"""How well each score ranks rooms by whether they hold a target, over all rooms and over those where the target is not in view of the frames used: AUROC per target and the share of houses whose top-scored room holds it."""
import argparse
import json
from pathlib import Path

import numpy as np

from sparx_agency.tasks.planning.objnav_benchmark_runtime.room_dataset.mp3d_target_score import TEXT_FILE, VLM_FILE, kept_frames
from sparx_agency.tasks.planning.objnav_benchmark_runtime.room_dataset.target_score import TARGETS, auroc, auroc_difference, parse_percent, room_holds, room_score, scene_top1


def read_rows(path):
    return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []


def frame_scores(rows):
    scores = {}
    for r in rows:
        scores.setdefault((r["room"], r["target"]), {})[r["frame"]] = parse_percent(r["reply"])
    return scores


def vlm_scores(rows, frames_of):
    """``{(room, target): mean percent over the frames in frames_of[room]}`` for the rooms with every frame asked."""
    by_room = frame_scores(rows)
    return {key: room_score([answers.get(f) for f in frames_of[key[0]]]) for key, answers in by_room.items() if key[0] in frames_of}


def text_scores(rows):
    return {(r["room"], r["target"]): parse_percent(r["reply"]) for r in rows}


def visible_anywhere(door):
    """``{(room, target): True}`` where the target is in view in a door frame or a kept walk frame of the room."""
    seen = {}
    for folder in sorted(d for d in door.iterdir() if (d / "selection.json").exists()):
        for room, entry in json.loads((folder / "selection.json").read_text()).items():
            for per_target in entry.get("visible", {}).values():
                for target, flag in per_target.items():
                    seen[(room, target)] = seen.get((room, target), False) or flag
    return seen


def room_infos(rooms):
    """The room records of a folder of scenes: ``regions.json`` (Matterport3D) or ``rooms.json`` (HM3D) each."""
    for folder in sorted(d for d in rooms.iterdir() if d.is_dir()):
        if (folder / "regions.json").exists():
            yield from json.loads((folder / "regions.json").read_text())["rooms"]
        elif (folder / "rooms.json").exists():
            yield from json.loads((folder / "rooms.json").read_text())


def table(title, rooms, scores, info, hidden, rng):
    versus = "text_yolo" in scores
    lines = ["### " + title, "", "| target | rooms | holding | " + " | ".join(scores) + (" | vlm_walk minus text(yolo) |" if versus else " |"), "|---|---|---|" + "---|" * (len(scores) + versus)]
    tops = {m: [] for m in scores}
    for target in TARGETS:
        keys = [(room, target) for room in rooms if (room, target) in hidden and all(scores[m].get((room, target)) is not None for m in scores)]
        labels = [room_holds(info[room], target) for room, _ in keys]
        values = {m: [scores[m][k] for k in keys] for m in scores}
        row = [auroc(values[m], labels) for m in scores]
        diff = auroc_difference(values["vlm_walk"], values["text_yolo"], labels, rng) if versus and 0 < sum(labels) < len(labels) else (float("nan"),) * 3
        lines.append("| %s | %d | %d | %s%s |" % (target, len(keys), sum(labels), " | ".join("%.2f" % a for a in row), (" | %+.2f [%+.2f, %+.2f]" % diff) if versus else ""))
        for m in scores:
            houses = {}
            for (room, _), s, lab in zip(keys, values[m], labels):
                houses.setdefault(info[room]["scene"], []).append((s, lab))
            tops[m].append(scene_top1(houses))
    lines += ["", "top-scored room of a house holds the target (mean over targets): " + ", ".join("%s %.0f%%" % (m, 100 * np.nanmean(t)) for m, t in tops.items()), ""]
    return lines


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--rooms", type=Path, required=True)
    p.add_argument("--walk", type=Path, required=True)
    p.add_argument("--door", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--every", type=int, default=4)
    p.add_argument("--vlm", default="qwen3-vl:4b-instruct")
    p.add_argument("--text", default="qwen2.5:14b-instruct")
    args = p.parse_args(argv)
    model = args.vlm.replace(":", "_")
    walk_frames = {r: f for r, (_, f) in kept_frames(args.walk, args.every).items()}
    door0 = {r: f for r, (_, f) in kept_frames(args.door, 1, ("door0",)).items()}
    door = {r: f for r, (_, f) in kept_frames(args.door, 1, ("door0", "door4")).items()}
    rooms = sorted(set(walk_frames) & set(door))
    info = {r["room"]: r for r in room_infos(args.rooms)}
    walk_rows, door_rows = read_rows(args.out / (VLM_FILE % (model, "walk"))), read_rows(args.out / (VLM_FILE % (model, "door")))
    scores = {"vlm_walk": vlm_scores(walk_rows, walk_frames), "vlm_door0": vlm_scores(door_rows, door0), "vlm_door": vlm_scores(door_rows, door)}
    text_rows = read_rows(args.out / (TEXT_FILE % (args.text.replace(":", "_"), "yolo_world")))
    if text_rows:
        scores["text_yolo"] = text_scores(text_rows)
    scores["area"] = {(room, t): info[room]["area_m2"] for room in rooms for t in TARGETS}
    seen = visible_anywhere(args.door)
    everything = {(r, t): True for r in rooms for t in TARGETS}
    unseen = {(r, t): True for r in rooms for t in TARGETS if not seen.get((r, t), True)}
    rng = np.random.default_rng(0)
    lines = ["# Target-in-room score, %d rooms with walk and door views\n" % len(rooms)]
    lines += table("All rooms", rooms, scores, info, everything, rng)
    lines += table("Rooms where the target is in view of no door or kept walk frame (the prior)", rooms, scores, info, unseen, rng)
    text = "\n".join(lines)
    (args.out / "report_target_score.md").write_text(text)
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
