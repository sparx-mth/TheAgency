"""The single-storey study's table: a baseline campaign against each later step, on the fixed episode set.

Reads the ``episodes.jsonl`` rows of every campaign root named (``paired_ab.load_rows``: every scene under the
root) and prints, as Markdown:

* one row per campaign -- episodes, SR, SPL, SoftSPL, DTG, mean steps, STOPs / step limits, and against the
  first root (the baseline) the paired SR and SPL differences with their 95 % bootstrap intervals, the exact
  McNemar p of the success flips, and the flips themselves;
* per-category SR (with the count) and per-scene SR for every campaign side by side;
* for every campaign after the first, the episodes that changed outcome against the baseline.

Every number is over the episodes ALL the roots hold (an unfinished campaign shrinks the table and says so),
so a later step is never read against a baseline it did not run the same episodes of.

Usage::

    python -m sparx_agency.tasks.planning.objnav_benchmark_runtime.mp3d.single_storey_report \\
        baseline=runs/ss_baseline flat-p=runs/ss_flat_p step1=runs/ss_step1 [--json out.json]

Standard library only beyond ``paired_ab`` (``objnav_benchmark/stats.py`` for the intervals).
"""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from typing import Dict, List, Sequence

from sparx_agency.tasks.planning.objnav_benchmark.stats import bootstrap_mean_interval, mcnemar_exact_p
from sparx_agency.tasks.planning.objnav_benchmark_runtime.paired_ab import load_rows


def parse_arms(specs: Sequence[str]) -> List[tuple]:
    """``name=root[,root...]`` pairs, the first being the baseline; a bare root is named after its last path part."""
    arms = []
    for spec in specs:
        name, sep, roots = spec.partition("=")
        if not sep:
            name, roots = spec.rstrip("/").split("/")[-1], spec
        arms.append((name, roots))
    if len(arms) < 1:
        raise ValueError("name at least one campaign root")
    return arms


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else float("nan")


def _pct(value: float) -> str:
    return "%.1f %%" % (100.0 * value)


def arm_summary(rows: Dict[str, dict], ids: Sequence[str]) -> dict:
    """SR, SPL, SoftSPL, DTG, steps and the terminations over ``ids``."""
    picked = [rows[k] for k in ids]
    stops = sum(1 for r in picked if r.get("stop_called"))
    return {"episodes": len(picked), "sr": _mean([float(bool(r["success"])) for r in picked]),
            "spl": _mean([float(r["spl"]) for r in picked]),
            "soft_spl": _mean([float(r.get("soft_spl", 0.0)) for r in picked]),
            "dtg_m": _mean([float(r["distance_to_goal_m"]) for r in picked]),
            "steps": _mean([float(r["steps"]) for r in picked]),
            "stops": stops, "limits": len(picked) - stops,
            "wrong_stops": sum(1 for r in picked if r.get("stop_called") and not r["success"]),
            "near_misses_1m": sum(1 for r in picked if r.get("stop_called") and not r["success"]
                                  and float(r["distance_to_goal_m"]) <= 1.0)}


def paired(base: Dict[str, dict], other: Dict[str, dict], ids: Sequence[str], resamples: int = 10000) -> dict:
    sa = [bool(base[k]["success"]) for k in ids]
    sb = [bool(other[k]["success"]) for k in ids]
    dsr = [float(y) - float(x) for x, y in zip(sa, sb)]
    dspl = [float(other[k]["spl"]) - float(base[k]["spl"]) for k in ids]
    return {"sr_diff": _mean(dsr), "sr_ci": list(bootstrap_mean_interval(dsr, resamples=resamples, seed_key="ss/sr")),
            "spl_diff": _mean(dspl),
            "spl_ci": list(bootstrap_mean_interval(dspl, resamples=resamples, seed_key="ss/spl")),
            "mcnemar_p": mcnemar_exact_p(sa, sb),
            "to_success": [k for k, x, y in zip(ids, sa, sb) if y and not x],
            "to_failure": [k for k, x, y in zip(ids, sa, sb) if x and not y]}


def grouped_sr(rows: Dict[str, dict], ids: Sequence[str], key) -> Dict[str, tuple]:
    """``{group: (successes, n)}`` over ``ids`` for ``key(row)``."""
    out: Dict[str, List[int]] = defaultdict(lambda: [0, 0])
    for k in ids:
        group = key(rows[k])
        out[group][1] += 1
        out[group][0] += int(bool(rows[k]["success"]))
    return {g: (v[0], v[1]) for g, v in sorted(out.items())}


def report(arms: Sequence[tuple], resamples: int = 10000) -> dict:
    """Every table of the study as data; :func:`render` writes the Markdown."""
    loaded = [(name, load_rows(roots)) for name, roots in arms]
    common = set(loaded[0][1])
    for _, rows in loaded[1:]:
        common &= set(rows)
    ids = sorted(common)
    if not ids:
        raise ValueError("no episode was run by every campaign named")
    base_name, base = loaded[0]
    out = {"episodes": len(ids), "per_arm": {}, "sizes": {name: len(rows) for name, rows in loaded},
           "by_category": {}, "by_scene": {}, "paired": {}}
    for name, rows in loaded:
        out["per_arm"][name] = arm_summary(rows, ids)
        out["by_category"][name] = grouped_sr(rows, ids, lambda r: r["target_category"])
        out["by_scene"][name] = grouped_sr(rows, ids, lambda r: r["episode_id"].split("/")[0])
        if name != base_name:
            out["paired"][name] = paired(base, rows, ids, resamples)
    out["arms"] = [name for name, _ in loaded]
    return out


def render(data: dict) -> str:
    names = data["arms"]
    lines = ["Episodes in every campaign: %d%s" % (
        data["episodes"], "" if all(n == data["episodes"] for n in data["sizes"].values())
        else " (rows per campaign: %s)" % ", ".join("%s %d" % item for item in data["sizes"].items())), ""]
    lines.append("| campaign | n | SR | SPL | SoftSPL | DTG (m) | steps | STOP / limit | wrong STOPs (<= 1 m) "
                 "| dSR vs %s [95 %%] | McNemar p | dSPL [95 %%] | flips +/- |" % names[0])
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for name in names:
        s = data["per_arm"][name]
        p = data["paired"].get(name)
        pair = ("" if p is None else "%+.1f pts [%+.1f, %+.1f] | %.3f | %+.3f [%+.3f, %+.3f] | %d / %d" % (
            100 * p["sr_diff"], 100 * p["sr_ci"][0], 100 * p["sr_ci"][1], p["mcnemar_p"], p["spl_diff"],
            p["spl_ci"][0], p["spl_ci"][1], len(p["to_success"]), len(p["to_failure"])))
        lines.append("| %s | %d | %s | %s | %s | %.2f | %.0f | %d / %d | %d (%d) | %s |" % (
            name, s["episodes"], _pct(s["sr"]), _pct(s["spl"]), _pct(s["soft_spl"]), s["dtg_m"], s["steps"],
            s["stops"], s["limits"], s["wrong_stops"], s["near_misses_1m"], pair if pair else "-- | -- | -- | --"))
    lines += ["", "Per-category SR (successes / episodes):", "",
              "| category | n | " + " | ".join(names) + " |", "|---|---|" + "---|" * len(names)]
    categories = sorted(data["by_category"][names[0]])
    for category in categories:
        cells = []
        for name in names:
            won, n = data["by_category"][name][category]
            cells.append("%d (%s)" % (won, _pct(won / n)))
        lines.append("| %s | %d | %s |" % (category, data["by_category"][names[0]][category][1], " | ".join(cells)))
    lines += ["", "Per-scene SR:", "", "| scene | n | " + " | ".join(names) + " |", "|---|---|" + "---|" * len(names)]
    for scene in sorted(data["by_scene"][names[0]]):
        cells = ["%d (%s)" % (data["by_scene"][name][scene][0], _pct(data["by_scene"][name][scene][0] / data["by_scene"][name][scene][1]))
                 for name in names]
        lines.append("| %s | %d | %s |" % (scene, data["by_scene"][names[0]][scene][1], " | ".join(cells)))
    for name in names[1:]:
        p = data["paired"][name]
        lines += ["", "%s against %s: %d to success (%s); %d to failure (%s)" % (
            name, names[0], len(p["to_success"]), ", ".join(p["to_success"]) or "none",
            len(p["to_failure"]), ", ".join(p["to_failure"]) or "none")]
    return "\n".join(lines) + "\n"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("arms", nargs="+", help="name=root[,root...]; the first is the baseline")
    parser.add_argument("--resamples", type=int, default=10000)
    parser.add_argument("--json", help="also write the tables as JSON here")
    args = parser.parse_args(argv)
    data = report(parse_arms(args.arms), resamples=args.resamples)
    print(render(data), end="")
    if args.json:
        with open(args.json, "w") as fh:
            json.dump(data, fh, indent=2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
