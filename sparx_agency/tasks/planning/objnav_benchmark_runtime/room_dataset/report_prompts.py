"""Which question wording gives the best target-in-room score from the same three frames per room, by model: AUROC over all rooms and over rooms where the target is in none of the three frames."""
import argparse
import json
from pathlib import Path

import numpy as np

from sparx_agency.tasks.planning.objnav_benchmark_runtime.room_dataset.mp3d_target_score import vlm_file
from sparx_agency.tasks.planning.objnav_benchmark_runtime.room_dataset.report_sweep import bootstrap, load_sweep, mean_of
from sparx_agency.tasks.planning.objnav_benchmark_runtime.room_dataset.report_target_score import read_rows, room_infos
from sparx_agency.tasks.planning.objnav_benchmark_runtime.room_dataset.target_score import TARGETS, parse_percent, room_holds

MIN_AUROC = 0.70
MIN_LOWER_BOUND = 0.60
MODELS = ("qwen3-vl:4b-instruct", "qwen3-vl:8b-instruct")
STYLES = ("percent", "unseen", "nearby", "keep")
YES_NO_STYLES = ("keep",)


def reply_value(reply, style):
    """The number a reply stands for: its percent, or for a yes/no wording the logit read from the log-probabilities."""
    if style in YES_NO_STYLES:
        try:
            return float(json.loads(reply)["logit"])
        except (ValueError, KeyError, TypeError):
            return None
    return parse_percent(reply)


def verdict(interval):
    """``keep`` when the mean AUROC is at least ``MIN_AUROC`` and the lower end of its interval is above ``MIN_LOWER_BOUND``; set before the run."""
    mean, low, _ = interval
    return "keep" if mean >= MIN_AUROC and low > MIN_LOWER_BOUND else "not shown"


def room_scores(rooms, rows, style):
    by_key = {}
    for r in rows:
        by_key.setdefault((r["room"], r["target"]), []).append(reply_value(r["reply"], style))
    return {(room, t): mean_of(by_key.get((room, t), [])) for room in rooms for t in TARGETS}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--rooms", type=Path, nargs="+", required=True)
    p.add_argument("--views", type=Path, required=True)
    p.add_argument("--scores", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args(argv)
    rng = np.random.default_rng(0)
    rooms = load_sweep(args.views)
    info = {r["room"]: r for folder in args.rooms for r in room_infos(folder)}
    rooms = {r: e for r, e in rooms.items() if r in info}
    holds = {r: {t: room_holds(info[r], t) for t in TARGETS} for r in rooms}
    hidden = {(r, t): not any(v[t] for v in e["visible"].values()) for r, e in rooms.items() for t in TARGETS}
    by_scene = {}
    for room, entry in rooms.items():
        by_scene.setdefault(entry["scene"], []).append(room)
    lines = ["Rooms: %d in %d scenes, three frames each (the same frames for every wording).  Bar set in advance: AUROC >= %.2f with the lower end of the 95%% interval > %.2f where the target is in none of the three frames." % (len(rooms), len(by_scene), MIN_AUROC, MIN_LOWER_BOUND), ""]
    scored = {}
    for model in MODELS:
        for style in STYLES:
            rows = read_rows(args.scores / vlm_file(model, "walk", style))
            if rows:
                scored[(model, style)] = room_scores(rooms, rows, style)
    for only_hidden, title in ((False, "All rooms"), (True, "Rooms where the target is in none of the three frames")):
        lines += ["### " + title, "", "| model | wording | mean AUROC [95% over scenes] | minus the percent wording | " + ("verdict |" if only_hidden else "|"), "|---|---|---|---|" + ("---|" if only_hidden else "")]
        for (model, style), scores in scored.items():
            point = bootstrap([scores], by_scene, holds, hidden, only_hidden, rng)
            base = scored.get((model, "percent"))
            diff = bootstrap([scores], by_scene, holds, hidden, only_hidden, rng, [base]) if base and style != "percent" else None
            lines.append("| %s | %s | %.3f [%.3f, %.3f] | %s |%s" % (model, style, *point, "%+.3f [%+.3f, %+.3f]" % diff if diff else "", " %s |" % verdict(point) if only_hidden else ""))
        lines.append("")
    args.out.write_text("\n".join(lines))
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
