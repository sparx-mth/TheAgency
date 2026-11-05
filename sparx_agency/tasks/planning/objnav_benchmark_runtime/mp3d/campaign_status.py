"""The live status of a sharded MP3D campaign, in one screen: ``python -m ...mp3d.campaign_status <root>``.

``progress.py`` describes one run directory; a sharded campaign has some two
hundred of them. This folds them into what an operator asks: how many of the
split's episodes are done, the running SR and SPL over them, which shards run
now, the throughput over the last hour (from the pool's heartbeat file) and the
ETA it implies, and one line per scene.

Read-only: it reads ``shard_plan.json``, the heartbeat, each scene's merged
``summary.json`` or its shards' ``progress.json``, and ``/proc`` for the shard
processes alive now.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import time

from sparx_agency.tasks.planning.objnav_benchmark_runtime.mp3d.shard_plan import read_plan
from sparx_agency.tasks.planning.objnav_benchmark_runtime.mp3d.sharded_campaign import HEARTBEAT_FILE

RUN_MODULE_MARK = "objnav_benchmark_runtime.mp3d.run"


def live_outputs():
    """The resolved ``--output`` directory of every ``mp3d.run`` process alive now."""
    outputs = set()
    for entry in os.listdir("/proc"):
        if not entry.isdigit():
            continue
        try:
            with open("/proc/%s/cmdline" % entry, "rb") as stream:
                argv = stream.read().decode(errors="replace").split("\0")
            cwd = os.readlink("/proc/%s/cwd" % entry)
        except OSError:
            continue
        if not any(RUN_MODULE_MARK in arg for arg in argv) or "--output" not in argv[:-1]:
            continue
        outputs.add(os.path.realpath(os.path.join(cwd, argv[argv.index("--output") + 1])))
    return outputs


def _read(path):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def throughput(root, window_s=3600.0, now=None):
    """``(episodes per hour, seconds spanned)`` over the heartbeat's last ``window_s``, or ``None``."""
    path = Path(root) / HEARTBEAT_FILE
    if not path.is_file():
        return None
    beats = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    now = time.time() if now is None else now
    recent = [beat for beat in beats if beat["unix"] >= now - window_s] or beats[-1:]
    first = next((beat for beat in beats if beat["unix"] >= recent[0]["unix"] - 1e-6), None)
    earlier = [beat for beat in beats if beat["unix"] < recent[0]["unix"]]
    start = earlier[-1] if earlier else first
    end = beats[-1]
    span = end["unix"] - start["unix"]
    if span <= 0:
        return None
    return 3600.0 * (end["done"] - start["done"]) / span, span


def scene_rows(root, plan, alive):
    """One dict per scene: episodes done and planned, SR and SPL so far, shard counts, merged or not."""
    root, rows = Path(root), []
    for scene in plan["scenes"]:
        shards = [shard for shard in plan["shards"] if shard["scene"] == scene]
        total = sum(shard["episodes"] for shard in shards)
        summary = _read(root / scene / "summary.json")
        if summary is not None:
            overall = summary["overall"]
            rows.append({"scene": scene, "done": overall["n_episodes"], "total": total, "merged": True,
                         "SR": overall["success_rate"], "SPL": overall["spl"],
                         "finished": len(shards), "running": 0, "pending": 0})
            continue
        done = finished = running = 0
        sr = spl = 0.0
        for shard in shards:
            directory = root / shard["key"]
            progress = _read(directory / "progress.json")
            if (directory / "summary.json").is_file():
                finished += 1
            elif os.path.realpath(str(directory)) in alive:
                running += 1
            if progress:
                n = int(progress["done"])
                done += n
                sr += n * float(progress["overall"]["SR"])
                spl += n * float(progress["overall"]["SPL"])
        rows.append({"scene": scene, "done": done, "total": total, "merged": False,
                     "SR": sr / done if done else None, "SPL": spl / done if done else None,
                     "finished": finished, "running": running, "pending": len(shards) - finished - running})
    return rows


def describe_campaign(root, now=None):
    """The campaign at ``root`` as a few lines, or ``None`` when it is not a sharded campaign."""
    plan = read_plan(root)
    if plan is None:
        return None
    rows = scene_rows(root, plan, live_outputs())
    done = sum(row["done"] for row in rows)
    total = sum(row["total"] for row in rows)
    scored = [row for row in rows if row["done"]]
    sr = sum(row["SR"] * row["done"] for row in scored) / done if done else 0.0
    spl = sum(row["SPL"] * row["done"] for row in scored) / done if done else 0.0
    lines = ["MP3D campaign %s: %d/%d episodes (%.1f%%) | SR %.3f SPL %.3f over the %d done | shards: %d finished, "
             "%d running, %d pending | %d/%d scenes merged"
             % (root, done, total, 100.0 * done / total if total else 0.0, sr, spl, done,
                sum(row["finished"] for row in rows), sum(row["running"] for row in rows),
                sum(row["pending"] for row in rows), sum(row["merged"] for row in rows), len(rows))]
    rate = throughput(root, now=now)
    if rate is not None and rate[0] > 0:
        eta_h = (total - done) / rate[0]
        finish = time.strftime("%Y-%m-%d %H:%M", time.localtime((time.time() if now is None else now) + eta_h * 3600))
        lines.append("  throughput %.1f episodes/h over the last %.0f min -> ETA %.1f h (about %s local)"
                     % (rate[0], rate[1] / 60.0, eta_h, finish))
    else:
        lines.append("  throughput: not measured yet (the pool writes a heartbeat every 5 min)")
    lines.append("  %-13s %9s  %5s  %5s  %s" % ("scene", "done", "SR", "SPL", "shards fin/run/pend"))
    for row in rows:
        lines.append("  %-13s %4d/%4d  %5s  %5s  %d/%d/%d%s" % (
            row["scene"], row["done"], row["total"],
            "-" if row["SR"] is None else "%.3f" % row["SR"], "-" if row["SPL"] is None else "%.3f" % row["SPL"],
            row["finished"], row["running"], row["pending"], "  merged" if row["merged"] else ""))
    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("root", type=Path)
    args = parser.parse_args(argv)
    text = describe_campaign(args.root)
    print(text if text is not None else "%s holds no shard_plan.json (not a sharded campaign)" % args.root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
