"""The target-in-room score of a spin in place at the arrival pose of each room, by how many 30-degree turns it takes: AUROC over all rooms and over rooms where the target is in none of the spin's views."""
import argparse
import json
from pathlib import Path

import numpy as np

from sparx_agency.tasks.planning.objnav_benchmark_runtime.room_dataset.mp3d_target_score import vlm_file
from sparx_agency.tasks.planning.objnav_benchmark_runtime.room_dataset.report_sweep import bootstrap, mean_of
from sparx_agency.tasks.planning.objnav_benchmark_runtime.room_dataset.report_target_score import frame_scores, read_rows, room_infos
from sparx_agency.tasks.planning.objnav_benchmark_runtime.room_dataset.target_score import TARGETS, room_holds

METHODS = ("spin0", "spin3", "spin6", "spin10")
VLM = "qwen3-vl:4b-instruct"


def room_scores(spins, answers, method, targets=TARGETS):
    """``{(room, target): mean percent over the views of ``method``}``; views without an answer are left out."""
    return {(room, t): mean_of([answers.get((room, t), {}).get(f) for f in entry["methods"][method]]) for room, entry in spins.items() for t in targets}


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
    scored = {m: [room_scores(spins, answers, m)] for m in METHODS}
    turns = {m: next(iter(spins.values()))["actions"][m] for m in METHODS}
    lines = ["Rooms: %d in %d scenes, one spin at the arrival pose of each room's first visit (4B model, the original percent question on every view)." % (len(spins), len(rooms_by_scene)), ""]
    for only_hidden, title in ((False, "All rooms"), (True, "Rooms where the target is in none of the spin's views")):
        lines += ["### " + title, "", "| spin | turns | views | mean AUROC [95% over scenes] | minus the arrival view alone |", "|---|---|---|---|---|"]
        for m in METHODS:
            point = bootstrap(scored[m], rooms_by_scene, holds, hidden, only_hidden, rng)
            gain = bootstrap(scored[m], rooms_by_scene, holds, hidden, only_hidden, rng, scored["spin0"]) if m != "spin0" else None
            lines.append("| %s | %d | %d | %.3f [%.3f, %.3f] | %s |" % (m, turns[m], turns[m] + 1, *point, "%+.3f [%+.3f, %+.3f]" % gain if gain else ""))
        lines.append("")
    args.out.write_text("\n".join(lines))
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
