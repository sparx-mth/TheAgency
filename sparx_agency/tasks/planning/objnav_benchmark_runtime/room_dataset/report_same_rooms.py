"""The same rooms seen three ways: frames kept along the planner's own path, a short scan commanded at the arrival pose, and an even sweep of the room; the target-in-room score of each, with the turns it costs."""
import argparse
import json
from pathlib import Path

import numpy as np

from sparx_agency.tasks.planning.objnav_benchmark_runtime.room_dataset.buffer_rules import buffer_picks, usable
from sparx_agency.tasks.planning.objnav_benchmark_runtime.room_dataset.mp3d_target_score import vlm_file
from sparx_agency.tasks.planning.objnav_benchmark_runtime.room_dataset.report_sweep import bootstrap, load_sweep, mean_of
from sparx_agency.tasks.planning.objnav_benchmark_runtime.room_dataset.report_target_score import frame_scores, read_rows, room_infos
from sparx_agency.tasks.planning.objnav_benchmark_runtime.room_dataset.sweep_rules import frame_picks
from sparx_agency.tasks.planning.objnav_benchmark_runtime.room_dataset.target_score import TARGETS, room_holds

DRAWS = 10
VLM = "qwen3-vl:4b-instruct"



def load_scan_methods(views):
    methods = {}
    for folder in sorted(d for d in views.iterdir() if (d / "selection.json").exists()):
        for room, entry in json.loads((folder / "selection.json").read_text()).items():
            methods[room] = entry["methods"]
    return methods


def load_answers(*score_dirs):
    answers = {}
    for folder in score_dirs:
        for key, by_frame in frame_scores(read_rows(folder / vlm_file(VLM, "walk", "percent"))).items():
            answers.setdefault(key, {}).update(by_frame)
    return answers


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--rooms", type=Path, nargs="+", required=True)
    p.add_argument("--buffers", type=Path, required=True, help="hm3d_buffers: views and scores")
    p.add_argument("--scans", type=Path, required=True, help="hm3d_scans: views and scores")
    p.add_argument("--sweeps", type=Path, nargs="+", required=True, help="sweep folders, each with views and scores")
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args(argv)
    rng = np.random.default_rng(0)
    buffers, scans = load_sweep(args.buffers / "views"), load_scan_methods(args.scans / "views")
    sweeps = {}
    for folder in args.sweeps:
        sweeps.update(load_sweep(folder / "views"))
    answers = load_answers(args.buffers / "scores", args.scans / "scores", *(f / "scores" for f in args.sweeps))
    info = {r["room"]: r for folder in args.rooms for r in room_infos(folder)}
    rooms = sorted(r for r in buffers if r in scans and r in sweeps and r in info and len(usable(buffers[r]["frames"])) >= 3 and all((r, t) in answers for t in TARGETS))
    holds = {r: {t: room_holds(info[r], t) for t in TARGETS} for r in rooms}
    seen = {r: {t: any(v[t] for source in (buffers, sweeps) for v in source[r]["visible"].values()) for t in TARGETS} for r in rooms}
    hidden = {(r, t): not seen[r][t] for r in rooms for t in TARGETS}
    by_scene = {}
    for room in rooms:
        by_scene.setdefault(buffers[room]["scene"], []).append(room)

    def scores(pick):
        return {(r, t): mean_of([answers[(r, t)].get(f) for f in pick(r)]) for r in rooms for t in TARGETS}

    sweep_usable = lambda r: {n: dict(f, entrance_m=f["entrance_m"] if f["entrance_m"] is not None else 0.0) for n, f in sweeps[r]["frames"].items()}
    choices = [
        ("planner path, 3 random frames", 0, lambda: [scores(lambda r: buffer_picks(buffers[r]["frames"], "random", 3, rng)) for _ in range(DRAWS)]),
        ("planner path, 3 frames with most floor", 0, lambda: [scores(lambda r: buffer_picks(buffers[r]["frames"], "floor", 3, rng))]),
        ("planner path, all usable frames", 0, lambda: [scores(lambda r: list(usable(buffers[r]["frames"])))]),
        ("scan at arrival, 3 views 30 degrees apart", 3, lambda: [scores(lambda r: scans[r]["scan3_30"])]),
        ("scan at arrival, 3 views 60 degrees apart", 6, lambda: [scores(lambda r: scans[r]["scan3_60"])]),
        ("scan at arrival, 5 views", 6, lambda: [scores(lambda r: scans[r]["scan5"])]),
        ("planner path (3 most floor) plus the 30-degree scan", 3, lambda: [scores(lambda r: buffer_picks(buffers[r]["frames"], "floor", 3, rng) + scans[r]["scan3_30"])]),
        ("even sweep of the room, 3 random views (not free)", "many", lambda: [scores(lambda r: frame_picks(sweep_usable(r), "random", 3, rng)) for _ in range(DRAWS)]),
        ("even sweep of the room, all views (not free)", "many", lambda: [scores(lambda r: list(frame_picks(sweep_usable(r), "random", 10 ** 6, rng)))]),
    ]
    computed = [(label, cost, make()) for label, cost, make in choices]
    lines = ["Rooms seen all three ways: %d in %d scenes.  Extra turns are counted from facing the way the agent arrived; the planner path costs none." % (len(rooms), len(by_scene)), ""]
    for only_hidden, title in ((False, "All rooms"), (True, "Rooms where the target is in view of no frame of any kind")):
        lines += ["### " + title, "", "| frames | extra turns | mean AUROC [95% over scenes] | minus planner path, 3 random |", "|---|---|---|---|"]
        base = computed[0][2]
        for label, cost, draws in computed:
            point = bootstrap(draws, by_scene, holds, hidden, only_hidden, rng)
            diff = bootstrap(draws, by_scene, holds, hidden, only_hidden, rng, base) if label != computed[0][0] else None
            lines.append("| %s | %s | %.3f [%.3f, %.3f] | %s |" % (label, cost, *point, "%+.3f [%+.3f, %+.3f]" % diff if diff else ""))
        lines.append("")
    args.out.write_text("\n".join(lines))
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
