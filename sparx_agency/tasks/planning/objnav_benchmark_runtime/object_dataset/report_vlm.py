"""Markdown table of what vision-model answers on detector boxes kept of the right boxes and rejected of the wrong ones."""
import argparse
import json
from pathlib import Path

from sparx_agency.tasks.planning.objnav_benchmark_runtime.object_dataset.scoring import answers_within, vlm_summary

SETTINGS = (("all boxes, confidence >= 0.3", 0.0, 99.0), ("confidence >= 0.5", 0.5, 99.0), ("confidence >= 0.5 and <= 2.5 m", 0.5, 2.5))


def percent(part, whole):
    return "%d/%d (%.0f%%)" % (part, whole, 100 * part / max(1, whole))


def report(files):
    out = []
    for path in files:
        answers = [json.loads(line) for line in Path(path).read_text().splitlines()]
        out += ["## %s\n" % Path(path).stem, "| boxes | right boxes kept | wrong boxes rejected | unsure |", "|---|---|---|---|"]
        for label, min_conf, max_distance in SETTINGS:
            cell = vlm_summary(answers_within(answers, min_conf, max_distance))["all"]
            out.append("| %s | %s | %s | %d |" % (label, percent(cell["kept"], cell["right"]), percent(cell["rejected"], cell["wrong"]),
                                                   cell["unsure_right"] + cell["unsure_wrong"]))
        out += ["", "| category (all boxes) | right boxes kept | wrong boxes rejected |", "|---|---|---|"]
        for name, cell in vlm_summary(answers).items():
            if name != "all":
                out.append("| %s | %s | %s |" % (name, percent(cell["kept"], cell["right"]), percent(cell["rejected"], cell["wrong"])))
        out.append("")
    return "\n".join(out)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("files", nargs="+", type=Path)
    print(report(p.parse_args(argv).files))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
