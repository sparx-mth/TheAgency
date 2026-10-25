"""Read the exploration probes under a root: observed area, coverage rate and the prescan record per config."""
import argparse
import json
from pathlib import Path


def rows_of(config_dir):
    return [json.loads(line) for path in sorted(Path(config_dir).glob("**/episodes.jsonl"))
            for line in path.read_text().splitlines() if line.strip()]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path, help="<root>/<config>/**/episodes.jsonl")
    args = parser.parse_args()
    print("%-12s %3s %9s %9s %8s | %s" % ("config", "n", "area_m2", "m2/action", "steps", "prescan (actions, ended, falcon plans)"))
    for config_dir in sorted(d for d in args.root.iterdir() if d.is_dir()):
        rows = rows_of(config_dir)
        if not rows:
            continue
        metrics = [r["agent_info"]["policy"]["exploration_metrics"] for r in rows]
        area = [sum(m["observed_area_by_floor_m2"].values()) for m in metrics]
        prescans = [r["agent_info"].get("prescan") for r in rows]
        prescans = [p for p in prescans if p]
        note = ""
        if prescans:
            note = "mean %.0f actions; ended %s; mean %.0f falcon plans" % (
                sum(p["actions"] for p in prescans) / len(prescans),
                sorted({p["ended"] for p in prescans}, key=str),
                sum(p["stats"].get("falcon_plans", 0) for p in prescans) / len(prescans))
        print("%-12s %3d %9.1f %9.3f %8.0f | %s" % (
            config_dir.name, len(rows), sum(area) / len(area), sum(m["coverage_gain_m2_per_action"] for m in metrics) / len(metrics),
            sum(r["steps"] for r in rows) / len(rows), note))


if __name__ == "__main__":
    main()
