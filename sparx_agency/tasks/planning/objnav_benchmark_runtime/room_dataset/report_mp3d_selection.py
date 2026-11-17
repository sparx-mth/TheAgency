"""Room-type accuracy of the 4B vision vote over the first K views each capture method keeps, against the old evenly spread views, with paired intervals."""
import argparse
import json
from pathlib import Path

import numpy as np

from sparx_agency.tasks.planning.objnav_benchmark_runtime.room_dataset.mp3d_eval import first_k_accuracy, paired_difference
from sparx_agency.tasks.planning.objnav_benchmark_runtime.room_dataset.mp3d_vlm_selected import selections
from sparx_agency.tasks.planning.objnav_benchmark_runtime.room_dataset.report_mp3d_rooms import load_answers, load_truth

KS = (1, 3, 6)
BASELINE = "spread"
VLM_FILE = "vlm_qwen3-vl_4b-instruct_list.jsonl"


def answers_by_method(out, vlm_file=VLM_FILE, only=None):
    """``{method: {room: [answers]}}`` for the rooms whose frames (those of the ``only`` methods, or all) all have an answer."""
    by_frame = {json.loads(line)["frame"]: json.loads(line)["answer"] for line in (out / vlm_file).read_text().splitlines()}
    methods = {}
    for _, room, entry in selections(out):
        chosen = {m: f for m, f in entry["methods"].items() if only is None or m in only}
        needed = set(entry["frames"]) if only is None else {f for frames in chosen.values() for f in frames}
        if needed and all(f in by_frame for f in needed):
            for method, frames in chosen.items():
                methods.setdefault(method, {})[room] = [by_frame[f] for f in frames]
    return methods


def report(rooms, out):
    truth = load_truth(rooms)
    methods = answers_by_method(out)
    scored = set.intersection(*(set(a) for a in methods.values()))
    methods[BASELINE] = {r: a for r, a in load_answers(rooms / VLM_FILE).items() if r in scored}
    correct = {m: {k: first_k_accuracy({r: a for r, a in methods[m].items() if r in scored}, truth, k) for k in KS} for m in methods}
    rng = np.random.default_rng(0)
    lines = ["# Which views to show the vision model: first K views per method, %d Matterport3D regions, 4B with the room list\n" % len(scored),
             "| method | " + " | ".join("K=%d" % k for k in KS) + " |", "|---|" + "---|" * len(KS)]
    for m in methods:
        lines.append("| %s | %s |" % (m, " | ".join("%.1f%%" % (100 * np.mean(list(correct[m][k].values()))) for k in KS)))
    lines += ["", "## Paired gain over `%s` (mean and 95%% interval over regions)\n" % BASELINE, "| method | " + " | ".join("K=%d" % k for k in KS) + " |", "|---|" + "---|" * len(KS)]
    for m in methods:
        if m != BASELINE:
            cells = ["%+.1f [%+.1f, %+.1f]" % tuple(100 * v for v in paired_difference(correct[m][k], correct[BASELINE][k], rng)) for k in KS]
            lines.append("| %s | %s |" % (m, " | ".join(cells)))
    return "\n".join(lines)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--rooms", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args(argv)
    text = report(args.rooms, args.out)
    (args.out / "report_selection.md").write_text(text)
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
