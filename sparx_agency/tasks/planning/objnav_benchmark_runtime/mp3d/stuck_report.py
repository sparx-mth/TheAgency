"""Where the agent got stuck, before and after a change: two arms of a paired check, episode by episode.

``single_storey_report.py`` gives the outcome of the arms (SR, SPL, the
success flips). This module gives what the outcome does not show -- how often
a forward step was refused, how the closing's releases, relaxed standing
points and blind verification steps moved -- and the one number that says
whether the pairing can be trusted: in how many episodes the two arms walked
the identical trajectory. With a deterministic stand-in for the language
model (``stub_llm.py``, ``mp3d/cpu_stub_campaign.py``) an episode in which
the change never acted must be identical in both arms; every difference in
the table is then the change's.

Read from the lean ``episodes.jsonl`` rows alone (``paired_ab.load_rows``:
every scene under each root), so an arm kept from an earlier check can be
read against one run today::

    python -m sparx_agency.tasks.planning.objnav_benchmark_runtime.mp3d.stuck_report \\
        before=runs/cpu_ab/before after=runs/cpu_ab/after [--json out.json]

A counter an arm's code did not have reads 0 (the walk to the stop distance
before 2026-11-16), and one that was renamed is read under either name
(``retry_standoffs``, formerly ``squeezed_standoffs``).

Standard library only beyond ``paired_ab`` and ``objnav_benchmark/stats.py``.
"""
from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from typing import Callable, Dict, List, Sequence, Tuple

from sparx_agency.tasks.planning.objnav_benchmark.stats import bootstrap_mean_interval, mcnemar_exact_p
from sparx_agency.tasks.planning.objnav_benchmark_runtime.paired_ab import load_rows

#: An episode with at least this many verification steps taken toward a box without a usable depth holds a run of
#: them (the closing's verification budget is 12 actions; the paused clock allowed 24 more).
BLIND_RUN_STEPS = 20


def closing(row: dict, *names: str) -> float:
    """The first of the closing's counters ``names`` the row holds, as a number; 0 when it holds none."""
    counters = (((row.get("agent_info") or {}).get("policy") or {}).get("target_closing")) or {}
    for name in names:
        if counters.get(name) is not None:
            return float(counters[name])
    return 0.0


def refused(row: dict) -> float:
    """Forward steps the simulator refused in the episode."""
    return float((row.get("agent_info") or {}).get("blocked_forward", 0) or 0)


def forward(row: dict) -> float:
    """Forward steps asked for in the episode (at least 1, for a rate)."""
    return float(max(1, (row.get("action_counts") or {}).get("MOVE_FORWARD", 0) or 1))


def signature(row: dict) -> Tuple:
    """What two arms must share to have walked the same trajectory.

    Steps, every action's count, the path length and where it ended (the
    distance to the goal). The last is not redundant: the same actions in
    another order end somewhere else (one episode of the 2026-11-16 check
    had 21 steps, the same counts and the same 1.5 m in both arms, and ended
    5 cm apart).
    """
    return (row.get("steps"), json.dumps(row.get("action_counts"), sort_keys=True),
            round(float(row.get("path_length_m") or 0.0), 4), round(float(row.get("distance_to_goal_m") or 0.0), 4))


#: ``(name, row -> value)`` for every line of the table, in order.
METRICS: Tuple[Tuple[str, Callable[[dict], float]], ...] = (
    ("success", lambda r: float(bool(r["success"]))),
    ("SPL", lambda r: float(r["spl"])),
    ("steps", lambda r: float(r["steps"])),
    ("ended without a STOP", lambda r: float(not r.get("stop_called"))),
    ("distance to goal (m)", lambda r: float(r["distance_to_goal_m"])),
    ("refused forward steps", refused),
    ("refused per 100 forward steps", lambda r: 100.0 * refused(r) / forward(r)),
    ("closing: releases", lambda r: closing(r, "releases")),
    ("closing:   no route from here", lambda r: closing(r, "boxed_releases")),
    ("closing:   action bound", lambda r: closing(r, "bound_releases")),
    ("closing:   stalled", lambda r: closing(r, "stall_releases")),
    ("closing: nearest-floor walks", lambda r: closing(r, "nearest_routes")),
    ("closing: footing sweeps", lambda r: closing(r, "footing_sweeps")),
    ("closing: standoffs at the relaxed clearance", lambda r: closing(r, "retry_standoffs", "squeezed_standoffs")),
    ("closing: standoffs revoked", lambda r: closing(r, "standoffs_revoked")),
    ("closing: blind verification steps", lambda r: closing(r, "glimpse_steps")),
    ("closing: walks to the stop distance", lambda r: closing(r, "stop_approaches")),
    ("closing:   forward steps of those walks", lambda r: closing(r, "stop_approach_steps")),
    ("closing:   walks that ended short of it", lambda r: closing(r, "stop_approach_shortfalls")),
)


def sign_test_p(differences: Sequence[float]) -> float:
    """The exact two-sided sign test of ``differences`` against zero (ties dropped)."""
    moved = [d for d in differences if d != 0]
    return mcnemar_exact_p([d < 0 for d in moved], [d > 0 for d in moved]) if moved else 1.0


def paired_line(name: str, value: Callable[[dict], float], before: Dict[str, dict], after: Dict[str, dict],
                ids: Sequence[str], resamples: int = 10000) -> dict:
    """One metric over the paired episodes: both means, the mean and median difference, its interval, the sign test."""
    a, b = [value(before[k]) for k in ids], [value(after[k]) for k in ids]
    diff = [y - x for x, y in zip(a, b)]
    low, high = bootstrap_mean_interval(diff, resamples=resamples, seed_key="stuck/" + name)
    return {"name": name, "before": statistics.mean(a), "after": statistics.mean(b), "diff": statistics.mean(diff),
            "ci": [low, high], "median_diff": statistics.median(diff), "up": sum(d > 0 for d in diff),
            "down": sum(d < 0 for d in diff), "p": sign_test_p(diff)}


def report(before: Dict[str, dict], after: Dict[str, dict], resamples: int = 10000) -> dict:
    """Everything the text shows, over the episodes both arms hold.

    Returns:
        ``{"episodes", "only_before", "only_after", "identical", "lines",
        "lost", "won", "mcnemar_p", "blind_runs", "refused_total", "flips",
        "scenes"}`` -- see :func:`render`.

    Raises:
        ValueError: The two arms share no episode.
    """
    ids = sorted(set(before) & set(after))
    if not ids:
        raise ValueError("the two arms share no episode")

    def blind(rows, k):
        return closing(rows[k], "glimpse_steps") >= BLIND_RUN_STEPS

    def mean(rows, keys, value):
        return statistics.mean(value(rows[k]) for k in keys)

    lost = [k for k in ids if before[k]["success"] and not after[k]["success"]]
    won = [k for k in ids if not before[k]["success"] and after[k]["success"]]
    scenes: Dict[str, List[str]] = defaultdict(list)
    for k in ids:
        scenes[k.split("/")[0]].append(k)
    return {
        "episodes": len(ids), "only_before": len(set(before) - set(after)), "only_after": len(set(after) - set(before)),
        "identical": sum(signature(before[k]) == signature(after[k]) for k in ids),
        "lines": [paired_line(name, value, before, after, ids, resamples) for name, value in METRICS],
        "lost": lost, "won": won,
        "mcnemar_p": mcnemar_exact_p([bool(before[k]["success"]) for k in ids], [bool(after[k]["success"]) for k in ids]),
        "blind_runs": {"before": [k for k in ids if blind(before, k)], "after": [k for k in ids if blind(after, k)]},
        "refused_total": {"before": sum(refused(before[k]) for k in ids), "after": sum(refused(after[k]) for k in ids)},
        "flips": [{"id": k, "to": "failure" if k in lost else "success", "target": after[k].get("target_category"),
                   "before": [before[k]["steps"], round(float(before[k]["distance_to_goal_m"]), 2)],
                   "after": [after[k]["steps"], round(float(after[k]["distance_to_goal_m"]), 2)],
                   "stop_after": bool(after[k].get("stop_called"))} for k in lost + won],
        "scenes": {scene: {"episodes": len(keys),
                           "sr": [mean(before, keys, lambda r: float(bool(r["success"]))),
                                  mean(after, keys, lambda r: float(bool(r["success"])))],
                           "spl": [mean(before, keys, lambda r: float(r["spl"])), mean(after, keys, lambda r: float(r["spl"]))],
                           "steps": [mean(before, keys, lambda r: float(r["steps"])), mean(after, keys, lambda r: float(r["steps"]))],
                           "refused": [mean(before, keys, refused), mean(after, keys, refused)]}
                   for scene, keys in sorted(scenes.items())}}


def render(data: dict, names: Sequence[str] = ("before", "after")) -> str:
    """The report as text: the pairing, the table, the flips, the blind-step runs, the scenes."""
    out = ["%s against %s: %d paired episodes; identical trajectories in %d"
           % (names[1], names[0], data["episodes"], data["identical"])]
    if data["only_before"] or data["only_after"]:
        out.append("NOT PAIRED: %d episodes only in %s, %d only in %s"
                   % (data["only_before"], names[0], data["only_after"], names[1]))
    out.append("%-44s %8s %8s %9s  %-20s %7s  %s" % ("mean per episode", names[0][:8], names[1][:8], "diff",
                                                      "95% bootstrap CI", "median", "up/down (sign test p)"))
    for line in data["lines"]:
        out.append("%-44s %8.3f %8.3f %+9.3f  %-20s %+7.2f  %3d/%3d (p=%.3f)" % (
            line["name"], line["before"], line["after"], line["diff"], "[%+.3f, %+.3f]" % tuple(line["ci"]),
            line["median_diff"], line["up"], line["down"], line["p"]))
    out.append("success flips: %d lost, %d won (exact McNemar p = %.3f)" % (len(data["lost"]), len(data["won"]), data["mcnemar_p"]))
    for flip in data["flips"]:
        out.append("   to %-7s %-22s %-14s %s %3d steps, DTG %.2f m | %s %3d steps, DTG %.2f m, STOP %s" % (
            flip["to"], flip["id"], flip["target"], names[0], flip["before"][0], flip["before"][1], names[1],
            flip["after"][0], flip["after"][1], "yes" if flip["stop_after"] else "no"))
    runs = data["blind_runs"]
    out.append("blind verification-step runs (>= %d such steps in an episode): %d episodes %s, %d %s"
               % (BLIND_RUN_STEPS, len(runs["before"]), names[0], len(runs["after"]), names[1]))
    for arm, name in (("before", names[0]), ("after", names[1])):
        if runs[arm]:
            out.append("   %s: %s" % (name, ", ".join(runs[arm])))
    out.append("refused forward steps in all: %d %s, %d %s"
               % (data["refused_total"]["before"], names[0], data["refused_total"]["after"], names[1]))
    out.append("%-13s %3s  %-15s %-15s %-13s %-13s" % ("scene", "n", "SR", "SPL", "steps", "refused"))
    for scene, s in data["scenes"].items():
        out.append("%-13s %3d  %.3f -> %.3f  %.3f -> %.3f  %4.0f -> %4.0f  %5.2f -> %5.2f" % (
            scene, s["episodes"], s["sr"][0], s["sr"][1], s["spl"][0], s["spl"][1], s["steps"][0], s["steps"][1],
            s["refused"][0], s["refused"][1]))
    return "\n".join(out)


def parse_arm(spec: str) -> Tuple[str, str]:
    """``name=root[,root...]``; a bare root is named after its last path part."""
    name, sep, roots = spec.partition("=")
    return (name, roots) if sep else (spec.rstrip("/").split("/")[-1], spec)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("before", help="[name=]root[,root...] of the arm before the change")
    parser.add_argument("after", help="[name=]root[,root...] of the arm after it")
    parser.add_argument("--resamples", type=int, default=10000, help="bootstrap resamples per interval")
    parser.add_argument("--json", help="also write the numbers here")
    args = parser.parse_args(argv)
    (before_name, before_roots), (after_name, after_roots) = parse_arm(args.before), parse_arm(args.after)
    data = report(load_rows(before_roots), load_rows(after_roots), resamples=args.resamples)
    print(render(data, (before_name, after_name)))
    if args.json:
        with open(args.json, "w") as stream:
            json.dump(dict(data, arms=[before_name, after_name]), stream, indent=1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
