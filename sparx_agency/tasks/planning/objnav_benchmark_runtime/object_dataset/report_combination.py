"""One detector proposes boxes, another must find them too: right and wrong STOPs of the pair, alone and with the 3D check across approach steps."""
import argparse
from pathlib import Path

from sparx_agency.tasks.planning.objnav_benchmark_runtime.object_dataset.geometry import is_held_out
from sparx_agency.tasks.planning.objnav_benchmark_runtime.object_dataset.report_consistency import sequences_of_boxes
from sparx_agency.tasks.planning.objnav_benchmark_runtime.object_dataset.scoring import consistent_stop_summary

PAIRS = (("rfdetr", "llmdet_official"), ("yolo_world", "llmdet_official"), ("rfdetr", "yolo_world"), ("yolo_world", "rfdetr"))
PROPOSER_CONF = 0.3
CONFIRMER_CONFS = (0.3, 0.5)
STOP_RANGE_M = 2.5
RADIUS_M = 0.5
NEEDED_STEPS = (1, 2, 3)


def rows_for(folders, proposer, confirmer):
    sequences = [s for f in folders for s in sequences_of_boxes(f, proposer, PROPOSER_CONF, STOP_RANGE_M, confirmer)[0]]
    return [(need, consistent_stop_summary(sequences, need, RADIUS_M)) for need in NEEDED_STEPS]


def report(views, held_out_fraction, include_held_out):
    names = {n for pair in PAIRS for n in pair}
    folders = [d for d in sorted(views.iterdir()) if d.is_dir() and (d / "depth_check.json").exists()
               and all((d / ("det_%s.json" % n)).exists() for n in names) and (include_held_out or not is_held_out(d.name, held_out_fraction))]
    out = ["# A second detector must find the box too (steps within %.1f m, proposer confidence >= %.1f, agreement radius %.1f m, %d scenes)\n"
           % (STOP_RANGE_M, PROPOSER_CONF, RADIUS_M, len(folders)),
           "| proposer | confirmer (min conf) | steps that agree | right stops | wrong stops | wrong share | never stops |", "|---|---|---|---|---|---|---|"]
    for proposer, confirmer in PAIRS:
        for label, option in [("alone", None)] + [("%s (%.1f)" % (confirmer, c), (confirmer, c)) for c in CONFIRMER_CONFS]:
            if option is None and (proposer, confirmer) != PAIRS[0] and proposer in [p for p, _ in PAIRS[:PAIRS.index((proposer, confirmer))]]:
                continue
            for need, v in rows_for(folders, proposer, option):
                stops = v["stopped_right"] + v["stopped_wrong"]
                out.append("| %s | %s | %d | %d | %d | %.0f%% | %d |" % (proposer, label, need, v["stopped_right"], v["stopped_wrong"],
                                                                         100 * v["stopped_wrong"] / max(1, stops), v["never"]))
    return "\n".join(out)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--views", type=Path, required=True)
    p.add_argument("--held-out-fraction", type=float, default=0.25)
    p.add_argument("--include-held-out", action="store_true")
    args = p.parse_args(argv)
    text = report(args.views, args.held_out_fraction, args.include_held_out)
    (args.views / "report_combination.md").write_text(text)
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
