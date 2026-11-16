"""Room-type accuracy of every method on the Matterport3D regions, against the dataset's own labels: overall, per type, by number of frames."""
import argparse
import collections
import json
from pathlib import Path

import numpy as np

from sparx_agency.tasks.planning.objnav_benchmark_runtime.room_dataset.mp3d_eval import accuracy_by_frames, predict_group
from sparx_agency.tasks.planning.objnav_benchmark_runtime.room_dataset.mp3d_room_types import ROOM_GROUPS, group_of_answer

FRAME_COUNTS = (1, 3, 6)
REPEATS = 30


def load_truth(rooms):
    return {r["room"]: r["group"] for f in sorted(rooms.iterdir()) if (f / "regions.json").exists()
            for r in json.loads((f / "regions.json").read_text())["rooms"]}


def load_answers(path):
    """``{room: [answers]}`` from a per-frame (vlm_*) or per-room (text_*) jsonl."""
    answers = collections.defaultdict(list)
    for line in Path(path).read_text().splitlines():
        row = json.loads(line)
        answers[row["room"]].append(row["answer"])
    return answers


def method_row(name, answers, truth, rng):
    rooms = [r for r in answers if r in truth]
    predicted = {r: predict_group(answers[r]) for r in rooms}
    accuracy = np.mean([predicted[r] == truth[r] for r in rooms])
    known = [r for r in rooms if truth[r] != "other"]
    answered = sum(g is not None for answers_r in answers.values() for g in map(group_of_answer, answers_r))
    total = sum(len(a) for a in answers.values())
    by_frames = accuracy_by_frames({r: answers[r] for r in rooms}, truth, FRAME_COUNTS, REPEATS, rng) if len(answers[rooms[0]]) > 1 else {}
    return {"name": name, "rooms": len(rooms), "accuracy": float(accuracy), "without_other": float(np.mean([predicted[r] == truth[r] for r in known])),
            "unreadable": 1 - answered / max(1, total), "by_frames": by_frames, "predicted": predicted}


def report(rooms, files):
    truth, rng = load_truth(rooms), np.random.default_rng(0)
    rows = [method_row(Path(f).stem, load_answers(f), truth, rng) for f in files]
    counts = collections.Counter(truth.values())
    out = ["# Room types on %d Matterport3D regions (train scenes), against MP3D's labels\n" % len(truth),
           "Regions per type: " + ", ".join("%s %d" % (g, counts[g]) for g in ROOM_GROUPS if counts[g]) + ".\n",
           "Chance by always answering the most common type: %.0f%%.\n" % (100 * max(counts.values()) / len(truth)),
           "| method | regions | accuracy | accuracy without `other` | answers not readable as a type | accuracy with 1 / 3 / 6 frames |", "|---|---|---|---|---|---|"]
    for r in sorted(rows, key=lambda r: -r["accuracy"]):
        frames = " / ".join("%.0f%%" % (100 * r["by_frames"][k]) for k in FRAME_COUNTS) if r["by_frames"] else "-"
        out.append("| %s | %d | %.1f%% | %.1f%% | %.0f%% | %s |" % (r["name"], r["rooms"], 100 * r["accuracy"], 100 * r["without_other"], 100 * r["unreadable"], frames))
    best = max(rows, key=lambda r: r["accuracy"])
    out += ["", "## Per type, share of regions of that type the method names correctly\n", "| type | regions | " + " | ".join(r["name"] for r in rows) + " |",
            "|---|---|" + "---|" * len(rows)]
    for g in ROOM_GROUPS:
        if counts[g]:
            cells = ["%.0f%%" % (100 * np.mean([r["predicted"][x] == g for x in r["predicted"] if truth[x] == g] or [0])) for r in rows]
            out.append("| %s | %d | %s |" % (g, counts[g], " | ".join(cells)))
    confusion = collections.Counter((truth[x], best["predicted"][x] or "unreadable") for x in best["predicted"])
    out += ["", "## What the best method (%s) says when it is wrong\n" % best["name"], "| true type | said | regions |", "|---|---|---|"]
    out += ["| %s | %s | %d |" % (t, s, n) for (t, s), n in confusion.most_common() if t != s][:15]
    return "\n".join(out)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--rooms", type=Path, required=True)
    p.add_argument("files", nargs="*", type=Path)
    args = p.parse_args(argv)
    files = args.files or sorted(args.rooms.glob("vlm_*.jsonl")) + sorted(args.rooms.glob("text_*.jsonl"))
    text = report(args.rooms, files)
    (args.rooms / "report_rooms.md").write_text(text)
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
