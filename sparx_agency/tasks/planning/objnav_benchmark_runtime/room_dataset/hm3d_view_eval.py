"""Room-type accuracy of the 4B vision model on Daphna's HM3D labels, for the old views and the floor-coverage views, first K of each."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np

from sparx_agency.tasks.planning.objnav_benchmark_runtime.room_dataset.label_app import Rooms
from sparx_agency.tasks.planning.objnav_benchmark_runtime.room_dataset.label_store import OPEN_PLAN
from sparx_agency.tasks.planning.objnav_benchmark_runtime.room_dataset.mp3d_eval import list_prompt, paired_difference, predict_group
from sparx_agency.tasks.planning.objnav_benchmark_runtime.room_dataset.mp3d_vlm_rooms import ask

KS = (1, 3, 6)
ACCEPTED = {"bedroom": {"bedroom"}, "bathroom": {"bathroom"}, "toilet": {"bathroom"}, "kitchen": {"kitchen"},
            "living room": {"living room"}, "family room or lounge": {"living room"}, "tv room": {"living room"},
            "dining room": {"dining room"}, OPEN_PLAN: {"kitchen", "living room", "dining room"},
            "office": {"office"}, "study": {"office"}, "library": {"library"},
            "playroom or kids room": {"recreation room", "bedroom"}, "game room": {"recreation room"}, "gym": {"recreation room"},
            "hallway": {"hallway"}, "stairs": {"stairs"}, "entryway": {"entryway"}, "closet": {"closet"},
            "laundry room": {"laundry or utility room"}, "utility room": {"laundry or utility room"}, "garage": {"garage"},
            "balcony or porch": {"outdoor"}, "outdoor": {"outdoor"}, "other room": {"other"}}
STRICT = {OPEN_PLAN: {"living room"}, "playroom or kids room": {"recreation room"}}


def accepted_groups(room_type, strict=False):
    """The model groups that count as right for a labelled room type; None for ``unsure`` and ``not a room``."""
    groups = (STRICT if strict else {}).get(room_type) or ACCEPTED.get(room_type)
    return None if groups is None else set(groups)


def correct_rooms(predicted, labels, strict=False):
    """``{room: right?}`` for the labelled rooms that are not left out; a room the model could not name is wrong."""
    result = {}
    for room, room_type in labels.items():
        groups = accepted_groups(room_type, strict)
        if groups is not None and room in predicted:
            result[room] = predicted[room] in groups
    return result


def view_paths(views_dir, selected):
    rooms = Rooms(Path(views_dir), None, selected=selected)
    return {room: [folder / "frames" / (v["frame"] + ".jpg") for v in shown] for room, (folder, _, shown) in rooms.chosen.items()}


def answers_for(paths_by_room, cache_file, url, model):
    cache = {}
    if cache_file.exists():
        cache = {json.loads(line)["path"]: json.loads(line)["answer"] for line in cache_file.read_text().splitlines()}
    with open(cache_file, "a") as handle:
        for paths in paths_by_room.values():
            for path in paths:
                if str(path) not in cache:
                    cache[str(path)] = ask(url, model, list_prompt(), path.read_bytes())
                    handle.write(json.dumps({"path": str(path), "answer": cache[str(path)]}) + "\n")
                    handle.flush()
    return {room: [cache[str(p)] for p in paths] for room, paths in paths_by_room.items()}


def report(old_answers, new_answers, labels, strict):
    rng = np.random.default_rng(0)
    rows = {}
    for name, answers in (("old views", old_answers), ("floor-coverage views", new_answers)):
        rows[name] = {k: correct_rooms({r: predict_group(a[:k]) for r, a in answers.items()}, labels, strict) for k in KS}
    out = ["| views | " + " | ".join("K=%d" % k for k in KS) + " |", "|---|" + "---|" * len(KS)]
    for name, by_k in rows.items():
        out.append("| %s | %s |" % (name, " | ".join("%.1f%%" % (100 * np.mean(list(by_k[k].values()))) for k in KS)))
    gains = ["%+.1f [%+.1f, %+.1f]" % tuple(100 * v for v in paired_difference(rows["floor-coverage views"][k], rows["old views"][k], rng)) for k in KS]
    out += ["", "Paired gain of the floor-coverage views over the old views: " + " | ".join("K=%d %s" % (k, g) for k, g in zip(KS, gains)),
            "Rooms scored: %d of %d labelled." % (len(rows["old views"][1]), len(labels))]
    return "\n".join(out)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--old", type=Path, required=True, help="the first HM3D room dataset (views chosen by annotation share)")
    p.add_argument("--new", type=Path, required=True, help="the floor-coverage set the labels were made on")
    p.add_argument("--labels", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--model", default="qwen3-vl:4b-instruct")
    p.add_argument("--url", default="http://127.0.0.1:11434/api/chat")
    args = p.parse_args(argv)
    with open(args.labels, newline="") as handle:
        labels = {row["room"]: row["room_type"] for row in csv.DictReader(handle)}
    args.out.mkdir(parents=True, exist_ok=True)
    cache = args.out / "answers.jsonl"
    old = answers_for({r: v for r, v in view_paths(args.old, False).items() if r in labels}, cache, args.url, args.model)
    new = answers_for({r: v for r, v in view_paths(args.new, True).items() if r in labels}, cache, args.url, args.model)
    text = "## Lenient (open plan accepts kitchen, living or dining; kids room accepts bedroom or recreation)\n\n%s\n\n## Strict\n\n%s\n" % (
        report(old, new, labels, False), report(old, new, labels, True))
    (args.out / "report.md").write_text(text)
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
