"""Summarise the trace folders written by ``trace_run.py`` (one folder per scene) into ``report.md`` and ``summaries.json``.

    PYTHONPATH=. python -m ...trace_tools.analyze_trace <trace dir> [--split train] [--version v1]
"""
import argparse
import collections
import gzip
import json
import re
from pathlib import Path

from sparx_agency.tasks.planning.objnav_benchmark_runtime.trace_tools.trace_summary import render_report, summarise_episode

DATASETS = Path.home() / "datasets" / "objectnav" / "hm3d"


def episode_name(episode_id):
    return re.sub(r"[^A-Za-z0-9_.-]", "_", episode_id)


def goal_centres(episodes_dir, episode_id, cache):
    scene, index = episode_id.split("/")
    code = scene.split("-")[1]
    if code not in cache:
        cache[code] = json.load(gzip.open(episodes_dir / ("%s.json.gz" % code)))
    data = cache[code]
    category = data["episodes"][int(index)]["object_category"]
    return [(-g["position"][2], -g["position"][0]) for g in data["goals_by_category"]["%s.basis.glb_%s" % (code, category)]]


def read_trace(folder):
    closing, poses = collections.defaultdict(list), collections.defaultdict(list)
    for line in open(folder / "trace.jsonl"):
        row = json.loads(line)
        if row["kind"] == "closing":
            closing[row["episode"]].append(row)
        elif row["kind"] == "detect" and row.get("pose"):
            poses[row["episode"]].append((row["pose"]["x"], row["pose"]["y"]))
    return closing, poses


def read_records(folder):
    return [json.loads(line) for path in sorted(folder.glob("run*/episodes.jsonl")) for line in open(path)]


def summarise_folder(folder, episodes_dir, cache):
    closing, poses = read_trace(folder)
    summaries = []
    for record in read_records(folder):
        name = episode_name(record["episode_id"])
        goals = goal_centres(episodes_dir, record["episode_id"], cache)
        summaries.append(summarise_episode(record["episode_id"], record, closing.get(name, []), poses.get(name, []), goals))
    return summaries


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace_dir", type=Path)
    parser.add_argument("--split", default="train")
    parser.add_argument("--version", default="v1")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args(argv)
    episodes_dir, cache = DATASETS / args.version / args.split / "content", {}
    summaries = [s for folder in sorted(args.trace_dir.iterdir()) if (folder / "trace.jsonl").exists()
                 for s in summarise_folder(folder, episodes_dir, cache)]
    out = args.out or args.trace_dir / "report.md"
    out.write_text(render_report(summaries))
    out.with_name("summaries.json").write_text(json.dumps(summaries, indent=1))
    print("%d episodes -> %s" % (len(summaries), out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
