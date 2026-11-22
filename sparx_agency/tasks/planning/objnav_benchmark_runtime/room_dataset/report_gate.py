"""Does asking the model whether a view is informative pick better views than random choice or the most-floor rule: room scores from the chosen views, and single views by their gate logit."""
import argparse
from pathlib import Path

import numpy as np

from sparx_agency.tasks.planning.objnav_benchmark_runtime.room_dataset.frame_gate import GATE_FILE, gate_logits, gate_verdict
from sparx_agency.tasks.planning.objnav_benchmark_runtime.room_dataset.mp3d_target_score import vlm_file
from sparx_agency.tasks.planning.objnav_benchmark_runtime.room_dataset.report_sweep import DRAWS, bootstrap, load_sweep
from sparx_agency.tasks.planning.objnav_benchmark_runtime.room_dataset.report_sweep import rule_scores
from sparx_agency.tasks.planning.objnav_benchmark_runtime.room_dataset.report_target_score import frame_scores, read_rows, room_infos
from sparx_agency.tasks.planning.objnav_benchmark_runtime.room_dataset.sweep_rules import usable_frames
from sparx_agency.tasks.planning.objnav_benchmark_runtime.room_dataset.target_score import TARGETS, auroc, room_holds

COUNTS = (1, 3, 6)
DECIDING_COUNT = 3


def ranks(values):
    return np.argsort(np.argsort(values)).astype(float)


def single_view_rows(sweep, answers, gate, hidden):
    """``(gate logit, floor share, score, holds)`` for every usable gated view and target of the rooms where the target is hidden."""
    rows = []
    for room, entry in sweep.items():
        for frame, f in usable_frames(entry["frames"]).items():
            for target in TARGETS:
                score = answers[(room, target)].get(frame)
                if frame in gate and score is not None and hidden[(room, target)]:
                    rows.append((gate[frame], f["floor_share"], score, entry["holds"][target]))
    return rows


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--rooms", type=Path, required=True)
    p.add_argument("--sweeps", type=Path, nargs="+", required=True, help="sweep folders with views and scores")
    p.add_argument("--gate", type=Path, required=True)
    p.add_argument("--vlm", default="qwen3-vl:4b-instruct")
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args(argv)
    rng = np.random.default_rng(0)
    sweep, answers = {}, {}
    for folder in args.sweeps:
        sweep.update(load_sweep(folder / "views"))
        answers.update(frame_scores(read_rows(folder / "scores" / vlm_file(args.vlm, "walk", "percent"))))
    gate = gate_logits(read_rows(args.gate / (GATE_FILE % args.vlm.replace(":", "_"))))
    info = {r["room"]: r for r in room_infos(args.rooms)}
    sweep = {r: e for r, e in sweep.items() if r in info and all((r, t) in answers for t in TARGETS)}
    for room, entry in sweep.items():
        entry["holds"] = {t: room_holds(info[room], t) for t in TARGETS}
    holds = {r: e["holds"] for r, e in sweep.items()}
    hidden = {(r, t): not any(v[t] for v in e["visible"].values()) for r, e in sweep.items() for t in TARGETS}
    rooms_by_scene = {}
    for room, entry in sweep.items():
        rooms_by_scene.setdefault(entry["scene"], []).append(room)
    usable = [n for e in sweep.values() for n in usable_frames(e["frames"])]
    lines = ["Rooms: %d in %d scenes; usable views %d, gated %d (%.0f%% answered yes).  Bar set in advance: the lower end of the gain of gate choice over random choice at K=%d, in rooms where the target is in no view, is above 0." % (
        len(sweep), len(rooms_by_scene), len(usable), sum(n in gate for n in usable), 100 * np.mean([gate[n] > 0 for n in usable if n in gate]), DECIDING_COUNT), ""]
    decision = None
    for only_hidden, title in ((False, "All rooms"), (True, "Rooms where the target is in view of no sweep view")):
        scored = {}
        for k in COUNTS:
            for rule in ("random", "floor", "gate"):
                scored[(rule, k)] = [rule_scores(sweep, answers, rule, k, rng, gate) for _ in range(DRAWS if rule == "random" else 1)]
        lines += ["### " + title, "", "| choice of views | mean AUROC [95% over scenes] | minus random, same K |", "|---|---|---|"]
        for k in COUNTS:
            for rule, label in (("random", "random"), ("floor", "most floor"), ("gate", "highest gate logit")):
                point = bootstrap(scored[(rule, k)], rooms_by_scene, holds, hidden, only_hidden, rng)
                gain = bootstrap(scored[(rule, k)], rooms_by_scene, holds, hidden, only_hidden, rng, scored[("random", k)]) if rule != "random" else None
                if only_hidden and rule == "gate" and k == DECIDING_COUNT:
                    decision = gain
                lines.append("| %s, K=%d | %.3f [%.3f, %.3f] | %s |" % (label, k, *point, "%+.3f [%+.3f, %+.3f]" % gain if gain else ""))
        lines.append("")
    lines += ["Verdict on the gate (K=%d, hidden rooms): **%s** (gain %+.3f [%+.3f, %+.3f])." % ((DECIDING_COUNT, gate_verdict(decision), *decision)), ""]
    rows = single_view_rows(sweep, answers, gate, hidden)
    logits = np.array([r[0] for r in rows])
    lines += ["### One view at a time, rooms where the target is in view of no sweep view", "",
              "Rank correlation of the gate logit with the floor share of the view: %.2f." % np.corrcoef(ranks(logits), ranks(np.array([r[1] for r in rows])))[0, 1], "",
              "| gate logit | view-target pairs | AUROC of the view's score |", "|---|---|---|"]
    low, high = np.quantile(logits, [1 / 3, 2 / 3])
    for label, test in (("lowest third", lambda v: v < low), ("middle third", lambda v: low <= v < high), ("highest third", lambda v: v >= high)):
        kept = [r for r in rows if test(r[0])]
        lines.append("| %s | %d | %.3f |" % (label, len(kept), auroc([r[2] for r in kept], [r[3] for r in kept])))
    args.out.write_text("\n".join(lines))
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
