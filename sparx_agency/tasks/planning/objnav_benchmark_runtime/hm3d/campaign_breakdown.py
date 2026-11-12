"""The analysis a finished HM3D campaign is read with: metrics by category and storey, failure mechanisms, runtime, utilization.

``campaign.py aggregate`` writes the headline (``campaign_summary.json``: SR / SPL / SoftSPL over the rows, per
scene). This reads the same rows (``<root>/<scene>/episodes.jsonl``) and answers the questions the headline cannot:

* **SR, SPL and DTG** overall, per goal category, and split into **same-storey** and **cross-storey** episodes --
  whether the published goal's nearest view point stands on the storey the agent starts on, from the level survey
  (``hm3d/episode_levels.py``; ``sets/hm3d_v1_val_levels.csv`` ships the v1 val survey) -- each group through the
  harness's own :func:`~sparx_agency.tasks.planning.objnav_benchmark.aggregate.group_summary`, so the arithmetic
  and the intervals are the benchmark's.
* **Failure mechanisms** (:data:`MECHANISMS`), one tag per failed episode, read from the full per-episode
  diagnostics (``--full-records``: ``target_closing``, ``target_pursuit``, ``exploration_fallback``,
  ``building``) and the episode's storey -- a first reading to point the replay at, not a verdict. A lean record
  carries less and its tags say so (``limit_unknown`` / ``stop_unknown``).
* **The LLM's health** -- episodes with no oracle answer, room-LLM failures -- because a dead oracle is designed
  not to cost episodes and so also not to show (``LESSONS.md`` 2026-10-30).
* **Runtime** (the launcher's ``started_utc.txt`` / ``finished_utc.txt``, each scene's duration from the campaign
  log) and **utilization** (``utilization_log.py``'s CSV, when given).

::

    python -m sparx_agency.tasks.planning.objnav_benchmark_runtime.hm3d.campaign_breakdown runs/X \\
        [--levels sets/hm3d_v1_val_levels.csv] [--utilization runs/X_support/utilization.csv] \\
        [--launcher-log runs/X_support/launcher.out] [--json out.json] [--markdown out.md]

Standard library plus the harness's aggregate; no simulator, no model.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence

from sparx_agency.tasks.planning.objnav_benchmark.aggregate import group_summary
from sparx_agency.tasks.planning.objnav_benchmark.results_io import read_episode_rows

#: The shipped v1 val level survey (``episode_levels.py`` over the agent navmesh, 2026-11-05).
DEFAULT_LEVELS = Path(__file__).resolve().parent / "sets" / "hm3d_v1_val_levels.csv"

#: One tag per failed episode, the first that fits, in this order.
#:
#: Step-limit endings (no STOP):
#:   ``limit_cross_storey_goal``  the goal's view points lie on another storey and the search is confined to the
#:                                spawn storey (``allow_stair_traversal`` off): unreachable by construction;
#:   ``limit_never_seen``         same storey, the target was never in a frame the closing accepted;
#:   ``limit_wedged``             the closing released a lock for going nowhere, or forward was refused 40+ times;
#:   ``limit_seen_not_converted`` the target was seen (a lock, a pursuit or a bearing) and no STOP followed;
#:   ``limit_other``              none of the above held.
#: STOPs scored as failures:
#:   ``stop_cross_storey_goal``   a STOP with the goal on another storey (a same-category object of the spawn
#:                                storey, or a goal seen across levels);
#:   ``stop_near_miss``           within a metre of a view point -- the standing spot, not the object;
#:   ``stop_far``                 farther than that: a wrong instance or a geodesic the planar range did not show.
#: ``agent_error``: the harness recorded an exception; ``*_unknown``: a lean record, the counters are not there.
MECHANISMS = ("agent_error",
              "limit_cross_storey_goal", "limit_never_seen", "limit_wedged", "limit_seen_not_converted", "limit_other",
              "limit_unknown", "stop_cross_storey_goal", "stop_near_miss", "stop_far", "stop_unknown")
NEAR_MISS_M = 1.0
WEDGED_BLOCKED_FORWARD = 40


def load_levels(path: Path) -> Dict[str, dict]:
    """``episode id -> survey row`` of the level survey; ``goal_same_level`` as an int or None where unmeasured."""
    table = {}
    with path.open() as stream:
        for row in csv.DictReader(stream):
            same = row.get("goal_same_level", "")
            row["goal_same_level"] = int(same) if same not in ("", None) else None
            table[row["episode"]] = row
    return table


def load_rows(root: Path):
    """``(records, raw)`` of every scene's ``episodes.jsonl``: the harness records and the raw JSON rows, by id."""
    records, raw = [], {}
    for source in sorted(root.glob("*/episodes.jsonl")):
        for _, record in read_episode_rows(source):
            records.append(record)
        with source.open() as stream:
            for line in stream:
                if line.strip():
                    row = json.loads(line)
                    raw[row["episode_id"]] = row
    return records, raw


def storey_of(episode_id: str, levels: Dict[str, dict]) -> Optional[str]:
    """``"same"``, ``"cross"`` or None (no survey row, or the goal unmeasured)."""
    row = levels.get(episode_id)
    if row is None or row.get("goal_same_level") is None:
        return None
    return "same" if row["goal_same_level"] == 1 else "cross"


def mechanism(row: dict, storey: Optional[str]) -> str:
    """The failed episode's tag (:data:`MECHANISMS`), from its row alone."""
    if row.get("termination") == "agent_error" or row.get("agent_error"):
        return "agent_error"
    info = row.get("agent_info") or {}
    policy = info.get("policy") or {}
    lean = bool(info.get("lean"))
    closing = policy.get("target_closing") or {}
    pursuit = (policy.get("target_pursuit") or {}).get("stats") or {}
    dtg = row.get("distance_to_goal_m")
    if row.get("stop_called"):
        if storey == "cross":
            return "stop_cross_storey_goal"
        if dtg is None or (isinstance(dtg, float) and not math.isfinite(dtg)):
            return "stop_far"
        if dtg <= NEAR_MISS_M:
            return "stop_near_miss"
        return "stop_far" if closing or lean else "stop_unknown"
    if storey == "cross":
        return "limit_cross_storey_goal"
    if not closing:
        return "limit_unknown"
    seen = int(closing.get("last_seen_step", -1)) >= 0 or int(closing.get("releases", 0) or 0) > 0 \
        or int(pursuit.get("queries", 0) or 0) > 0 or int(closing.get("bearing_starts", 0) or 0) > 0
    blocked = int(info.get("blocked_forward", 0) or 0)
    if int(closing.get("stall_releases", 0) or 0) >= 1 or blocked >= WEDGED_BLOCKED_FORWARD:
        return "limit_wedged"
    if not seen:
        return "limit_never_seen"
    return "limit_seen_not_converted"


def release_reasons(rows: Sequence[dict]) -> Counter:
    """How the failed episodes' last closing lock ended (``target_closing.last_release``), where recorded."""
    reasons = Counter()
    for row in rows:
        closing = ((row.get("agent_info") or {}).get("policy") or {}).get("target_closing") or {}
        last = closing.get("last_release")
        if not last:
            continue
        text = last if isinstance(last, str) else (last.get("reason") or last.get("why") or json.dumps(last, sort_keys=True))
        reasons[str(text).split(":", 1)[-1].strip().split(";")[0][:70]] += 1
    return reasons


def llm_health(rows: Sequence[dict]) -> dict:
    """What the LLM did across the run: oracle answers used, episodes without one, room-LLM failures."""
    queries, none, label_queries, failures = [], 0, [], 0
    for row in rows:
        policy = (row.get("agent_info") or {}).get("policy") or {}
        q = policy.get("llm_queries")
        if isinstance(q, (int, float)):
            queries.append(int(q))
            none += int(q) == 0
        r = policy.get("room_label_queries")
        if isinstance(r, (int, float)):
            label_queries.append(int(r))
        failures += int(((policy.get("exploration_fallback") or {}).get("stats") or {}).get("room_llm_failures", 0) or 0)
    return {"episodes": len(rows), "oracle_calls_total": sum(queries), "oracle_calls_mean": round(sum(queries) / len(queries), 1) if queries else None,
            "episodes_without_oracle_answer": none, "label_calls_total": sum(label_queries),
            "label_calls_mean": round(sum(label_queries) / len(label_queries), 1) if label_queries else None,
            "room_llm_failures": failures}


def storey_events(rows: Sequence[dict]) -> Counter:
    """The building layer's events over every episode (storey confirmations, ignored departures, transitions)."""
    events = Counter()
    for row in rows:
        building = ((row.get("agent_info") or {}).get("policy") or {}).get("building") or {}
        for event in building.get("events") or []:
            events[event.get("event", "?")] += 1
        if building.get("completed_transitions"):
            events["completed_transitions"] += len(building["completed_transitions"])
    return events


def _group(key: str, records) -> Optional[dict]:
    if not records:
        return None
    summary = group_summary(key, records)
    return {"n": summary.n_episodes, "SR": round(summary.success_rate, 4), "SR_ci": [round(v, 4) for v in summary.success_rate_ci],
            "SPL": round(summary.spl, 4), "SPL_ci": [round(v, 4) for v in summary.spl_ci], "SoftSPL": round(summary.soft_spl, 4),
            "DTG_m": None if summary.distance_to_goal_m is None else round(summary.distance_to_goal_m, 3),
            "unreachable_end": summary.n_unreachable_end, "mean_steps": round(summary.mean_steps, 1)}


def scene_durations(launcher_log: Optional[Path]) -> Dict[str, dict]:
    """``scene -> {hours, exit}`` from the campaign's ``[campaign] end <scene> exit <code> after <h> h`` lines."""
    if launcher_log is None or not launcher_log.is_file():
        return {}
    out = {}
    pattern = re.compile(r"\[campaign\] end\s+(\S+) exit (-?\d+) after ([0-9.]+) h")
    for line in launcher_log.read_text(errors="replace").splitlines():
        match = pattern.search(line)
        if match:
            scene, code, hours = match.groups()
            entry = out.setdefault(scene, {"hours": 0.0, "exits": []})
            entry["hours"] += float(hours)
            entry["exits"].append(int(code))
    return out


def utilization_summary(path: Optional[Path]) -> Optional[dict]:
    if path is None or not path.is_file():
        return None
    from sparx_agency.tasks.planning.objnav_benchmark_runtime.utilization_log import summarise
    return summarise(path)


def breakdown(root: Path, levels: Dict[str, dict], launcher_log: Optional[Path] = None,
              utilization: Optional[Path] = None) -> dict:
    """The whole report as JSON-ready data."""
    records, raw = load_rows(root)
    if not records:
        raise ValueError("no episode rows under %s" % root)
    by_id = {record.episode_id: record for record in records}
    storeys = {episode_id: storey_of(episode_id, levels) for episode_id in by_id}
    categories = sorted({record.target_category for record in records})

    def pick(predicate):
        return [record for record in records if predicate(record)]

    metrics = {"overall": _group("all", records),
               "by_category": {c: _group(c, pick(lambda r, c=c: r.target_category == c)) for c in categories},
               "by_storey": {"same": _group("same", pick(lambda r: storeys[r.episode_id] == "same")),
                             "cross": _group("cross", pick(lambda r: storeys[r.episode_id] == "cross")),
                             "unsurveyed": _group("unsurveyed", pick(lambda r: storeys[r.episode_id] is None))},
               "by_category_and_storey": {c: {s: _group("%s/%s" % (c, s), pick(lambda r, c=c, s=s: r.target_category == c and storeys[r.episode_id] == s))
                                              for s in ("same", "cross")} for c in categories}}
    failures = [raw[i] for i, record in by_id.items() if not record.success]
    tags = {row["episode_id"]: mechanism(row, storeys[row["episode_id"]]) for row in failures}
    by_mechanism = Counter(tags.values())
    by_mechanism_and_category = defaultdict(Counter)
    for row in failures:
        by_mechanism_and_category[tags[row["episode_id"]]][row["target_category"]] += 1
    terminations = Counter(record.termination for record in records)
    counts_by_storey = Counter(v for v in storeys.values())
    return {"root": str(root), "episodes": len(records), "scenes": len({r.scene_id for r in records}),
            "storey_counts": {"same": counts_by_storey.get("same", 0), "cross": counts_by_storey.get("cross", 0),
                              "unsurveyed": counts_by_storey.get(None, 0)},
            "metrics": metrics, "terminations": dict(terminations),
            "failures": {"n": len(failures), "by_mechanism": {m: by_mechanism.get(m, 0) for m in MECHANISMS if by_mechanism.get(m)},
                         "by_mechanism_and_category": {m: dict(c) for m, c in by_mechanism_and_category.items()},
                         "last_release_reasons": dict(release_reasons(failures).most_common(15)),
                         "tags": tags},
            "llm": llm_health(list(raw.values())), "storey_events": dict(storey_events(list(raw.values()))),
            "runtime": {"started_utc": _read(root / "started_utc.txt"), "finished_utc": _read(root / "finished_utc.txt"),
                        "scene_hours": scene_durations(launcher_log),
                        "mean_wall_s_per_episode": round(sum(r.wall_s for r in records) / len(records), 1)},
            "utilization": utilization_summary(utilization)}


def _read(path: Path) -> Optional[str]:
    return path.read_text().strip() if path.is_file() else None


def _row(name: str, g: Optional[dict]) -> str:
    if g is None:
        return "| %s | 0 | – | – | – | – |" % name
    dtg = "–" if g["DTG_m"] is None else "%.2f" % g["DTG_m"]
    return "| %s | %d | %.3f [%.3f, %.3f] | %.3f [%.3f, %.3f] | %s | %.0f |" % (
        name, g["n"], g["SR"], g["SR_ci"][0], g["SR_ci"][1], g["SPL"], g["SPL_ci"][0], g["SPL_ci"][1], dtg, g["mean_steps"])


def markdown(report: dict) -> str:
    lines = ["# HM3D campaign breakdown: %s" % report["root"], "",
             "%d episodes in %d scenes; storeys: %d same, %d cross, %d unsurveyed." % (
                 report["episodes"], report["scenes"], report["storey_counts"]["same"], report["storey_counts"]["cross"],
                 report["storey_counts"]["unsurveyed"]), "",
             "| group | n | SR [95% CI] | SPL [95% CI] | DTG m | steps |", "|---|---|---|---|---|---|",
             _row("overall", report["metrics"]["overall"])]
    for name, g in report["metrics"]["by_category"].items():
        lines.append(_row(name, g))
    for name in ("same", "cross"):
        lines.append(_row("%s-storey" % name, report["metrics"]["by_storey"][name]))
    lines += ["", "## Per category, same- vs cross-storey", "",
              "| category | same n | same SR | same SPL | cross n | cross SR | cross SPL |", "|---|---|---|---|---|---|---|"]
    for name, pair in report["metrics"]["by_category_and_storey"].items():
        same, cross = pair["same"], pair["cross"]
        lines.append("| %s | %s | %s | %s | %s | %s | %s |" % (
            name, same["n"] if same else 0, "%.3f" % same["SR"] if same else "–", "%.3f" % same["SPL"] if same else "–",
            cross["n"] if cross else 0, "%.3f" % cross["SR"] if cross else "–", "%.3f" % cross["SPL"] if cross else "–"))
    f = report["failures"]
    lines += ["", "## Failures: %d of %d (terminations: %s)" % (f["n"], report["episodes"], json.dumps(report["terminations"])), "",
              "| mechanism | episodes | share of failures | by category |", "|---|---|---|---|"]
    for m, n in f["by_mechanism"].items():
        cats = ", ".join("%s %d" % (c, k) for c, k in sorted(f["by_mechanism_and_category"].get(m, {}).items(), key=lambda kv: -kv[1]))
        lines.append("| %s | %d | %.1f%% | %s |" % (m, n, 100.0 * n / max(1, f["n"]), cats))
    if f["last_release_reasons"]:
        lines += ["", "Last closing release among the failures: " + "; ".join("%s (%d)" % kv for kv in f["last_release_reasons"].items())]
    llm = report["llm"]
    lines += ["", "## LLM", "", "oracle calls %s (mean %s per episode), episodes without an oracle answer %s, room-label calls %s (mean %s), room-LLM failures %s" % (
        llm["oracle_calls_total"], llm["oracle_calls_mean"], llm["episodes_without_oracle_answer"], llm["label_calls_total"], llm["label_calls_mean"], llm["room_llm_failures"])]
    if report["storey_events"]:
        lines += ["", "## Storey events (building layer, all episodes)", "", ", ".join("%s %d" % kv for kv in sorted(report["storey_events"].items()))]
    rt = report["runtime"]
    lines += ["", "## Runtime", "", "started %s, finished %s; mean wall %.0f s per episode" % (rt["started_utc"], rt["finished_utc"], rt["mean_wall_s_per_episode"])]
    if rt["scene_hours"]:
        lines.append("scene durations (h): " + ", ".join("%s %.1f%s" % (s, v["hours"], "" if v["exits"] == [0] else " exit%s" % v["exits"])
                                                        for s, v in sorted(rt["scene_hours"].items(), key=lambda kv: -kv[1]["hours"])))
    u = report["utilization"]
    if u:
        lines += ["", "## Utilization (%d samples over %.1f h)" % (u["samples"], u["span_h"]), "",
                  "GPU util mean %.0f%% (median %.0f, p10 %.0f, p90 %.0f; %.0f%% of samples at 80%%+); VRAM mean %.1f GB, max %.1f GB; CPU mean %.0f%% (p90 %.0f); RAM mean %.1f GB, max %.1f GB" % (
                      u["gpu_util_pct"]["mean"], u["gpu_util_pct"]["median"], u["gpu_util_pct"]["p10"], u["gpu_util_pct"]["p90"], 100.0 * (u["gpu_util_share_above_80"] or 0),
                      u["gpu_mem_used_mib"]["mean"] / 1024.0, u["gpu_mem_used_mib"]["max"] / 1024.0, u["cpu_pct"]["mean"], u["cpu_pct"]["p90"],
                      u["ram_used_gb"]["mean"], u["ram_used_gb"]["max"])]
    return "\n".join(lines) + "\n"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("root", type=Path, help="The campaign root holding <scene>/episodes.jsonl")
    parser.add_argument("--levels", type=Path, default=DEFAULT_LEVELS, help="episode_levels.py survey CSV (default: the shipped v1 val survey)")
    parser.add_argument("--launcher-log", type=Path, help="The launcher's console log, for per-scene durations")
    parser.add_argument("--utilization", type=Path, help="utilization_log.py CSV")
    parser.add_argument("--json", type=Path, help="Write the full report here")
    parser.add_argument("--markdown", type=Path, help="Write the Markdown tables here")
    args = parser.parse_args(argv)
    levels = load_levels(args.levels) if args.levels.is_file() else {}
    report = breakdown(args.root, levels, args.launcher_log, args.utilization)
    text = markdown(report)
    if args.json:
        args.json.write_text(json.dumps(report, indent=2, sort_keys=True))
    if args.markdown:
        args.markdown.write_text(text)
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
