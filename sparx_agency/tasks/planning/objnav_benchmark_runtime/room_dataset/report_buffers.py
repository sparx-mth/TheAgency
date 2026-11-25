"""From about twenty frames kept along the planner's own path into a room, which way of choosing a few of them gives the best target-in-room score."""
import argparse
from pathlib import Path

import numpy as np

from sparx_agency.tasks.planning.objnav_benchmark_runtime.room_dataset.buffer_rules import buffer_picks, usable
from sparx_agency.tasks.planning.objnav_benchmark_runtime.room_dataset.mp3d_target_score import vlm_file
from sparx_agency.tasks.planning.objnav_benchmark_runtime.room_dataset.report_sweep import bootstrap, load_sweep, mean_of
from sparx_agency.tasks.planning.objnav_benchmark_runtime.room_dataset.report_target_score import frame_scores, read_rows, room_infos
from sparx_agency.tasks.planning.objnav_benchmark_runtime.room_dataset.target_score import TARGETS, room_holds

DRAWS = 10
COUNTS = (1, 3, 5)
RULES = (("earliest", "first"), ("evenly spaced in time", "even_time"), ("most floor", "floor"), ("most different headings", "heading_spread"), ("random", "random"))


def rule_scores(rooms, answers, rule, count, rng):
    out = {}
    for room, entry in rooms.items():
        picks = buffer_picks(entry["frames"], rule, count, rng)
        for target in TARGETS:
            out[(room, target)] = mean_of([answers[(room, target)].get(f) for f in picks])
    return out


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--rooms", type=Path, nargs="+", required=True)
    p.add_argument("--views", type=Path, required=True)
    p.add_argument("--scores", type=Path, required=True)
    p.add_argument("--vlm", default="qwen3-vl:4b-instruct")
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args(argv)
    rng = np.random.default_rng(0)
    answers = frame_scores(read_rows(args.scores / vlm_file(args.vlm, "walk", "percent")))
    info = {r["room"]: r for folder in args.rooms for r in room_infos(folder)}
    rooms = {r: e for r, e in load_sweep(args.views).items() if r in info and all((r, t) in answers for t in TARGETS) and len(usable(e["frames"])) >= 3}
    holds = {r: {t: room_holds(info[r], t) for t in TARGETS} for r in rooms}
    hidden = {(r, t): not any(v[t] for v in e["visible"].values()) for r, e in rooms.items() for t in TARGETS}
    by_scene = {}
    for room, entry in rooms.items():
        by_scene.setdefault(entry["scene"], []).append(room)
    lines = ["Rooms: %d in %d scenes; usable frames per room: median %d." % (len(rooms), len(by_scene), int(np.median([len(usable(e["frames"])) for e in rooms.values()]))), ""]
    for only_hidden, title in ((False, "All rooms"), (True, "Rooms where the target is in view of no buffered frame")):
        scored = {("all frames", 0): [rule_scores(rooms, answers, "first", 10 ** 6, rng)]}
        for label, rule in RULES:
            for k in COUNTS:
                scored[(label, k)] = [rule_scores(rooms, answers, rule, k, rng) for _ in range(DRAWS if rule == "random" else 1)]
        lines += ["### " + title, "", "| choice | mean AUROC [95% over scenes] | minus random, same N |", "|---|---|---|"]
        for (label, k), draws in scored.items():
            base = scored.get(("random", k)) if label not in ("all frames", "random") else None
            point = bootstrap(draws, by_scene, holds, hidden, only_hidden, rng)
            diff = bootstrap(draws, by_scene, holds, hidden, only_hidden, rng, base) if base else None
            lines.append("| %s%s | %.3f [%.3f, %.3f] | %s |" % (label, ", N=%d" % k if k else "", *point, "%+.3f [%+.3f, %+.3f]" % diff if diff else ""))
        lines.append("")
    args.out.write_text("\n".join(lines))
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
