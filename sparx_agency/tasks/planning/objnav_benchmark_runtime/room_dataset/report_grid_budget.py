"""Room-type accuracy of the 4B vision vote for views planned on the occupancy grid under action budgets, with the actions each method spends, against the depth-chosen floor views."""
import argparse
import json
from pathlib import Path

import numpy as np

from sparx_agency.tasks.planning.objnav_benchmark_runtime.room_dataset.mp3d_eval import first_k_accuracy, paired_difference
from sparx_agency.tasks.planning.objnav_benchmark_runtime.room_dataset.mp3d_vlm_selected import selections
from sparx_agency.tasks.planning.objnav_benchmark_runtime.room_dataset.report_mp3d_rooms import load_truth
from sparx_agency.tasks.planning.objnav_benchmark_runtime.room_dataset.report_mp3d_selection import VLM_FILE, answers_by_method

K = 3
REFERENCE = ("mp3d_selected", "floor")


def summary_row(answers, actions, truth, k):
    viewed = {r: a for r, a in answers.items() if a}
    correct = first_k_accuracy(viewed, truth, k)
    return {"regions": len(answers), "with_views": len(viewed), "mean_views": float(np.mean([len(a) for a in answers.values()])),
            "mean_actions": float(np.mean([actions[r] for r in answers])), "accuracy": float(np.mean(list(correct.values()))) if correct else float("nan")}


def action_totals(out):
    totals = {}
    for _, room, entry in selections(out):
        for method, spent in entry.get("actions", {}).items():
            totals.setdefault(method, {})[room] = spent
    return totals


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--rooms", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--reference", type=Path, default=Path.home() / REFERENCE[0], help="folder of depth-chosen views, method '%s'" % REFERENCE[1])
    args = p.parse_args(argv)
    truth, rng = load_truth(args.rooms), np.random.default_rng(0)
    answers, actions = answers_by_method(args.out), action_totals(args.out)
    reference = answers_by_method(args.reference, VLM_FILE, {REFERENCE[1]})[REFERENCE[1]]
    lines = ["| method | regions | with a view | mean views | mean actions | accuracy K=3 (regions with a view) | gain over depth floor (same regions) |", "|---|---|---|---|---|---|---|"]
    for method in answers:
        row = summary_row(answers[method], actions[method], truth, K)
        shared = {r for r, a in answers[method].items() if a and r in reference}
        mine, theirs = (first_k_accuracy({r: d[r] for r in shared}, truth, K) for d in (answers[method], reference))
        gain = "%+.1f [%+.1f, %+.1f]" % tuple(100 * v for v in paired_difference(mine, theirs, rng)) if shared else "n/a"
        lines.append("| %s | %d | %d | %.2f | %.1f | %.1f%% | %s |" % (method, row["regions"], row["with_views"], row["mean_views"], row["mean_actions"], 100 * row["accuracy"], gain))
    text = "\n".join(lines)
    (args.out / "report_budget.md").write_text(text)
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
