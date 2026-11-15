"""Two HM3D campaigns on the same episodes, compared episode by episode: arm A against arm B.

Written for Run 1 against Run 2 of 2026-11-12 (stair traversal off against on, everything else equal, B replaying
A's LLM answers), and general to any two roots that ran the same episode ids. For every group -- overall, each goal
category, same- and cross-storey goals (the level survey, ``campaign_breakdown.load_levels``), and each category by
storey -- it reports SR, SPL and DTG per arm and the **paired** B - A difference with a percentile-bootstrap interval
over the per-episode differences (``objnav_benchmark/stats.py``), the exact McNemar test on the success flips, the
sign-flip test on SPL, and the flips themselves. DTG is compared over the episodes finite in both arms and the
unreachable ends are counted beside it, as ``aggregate`` does.

Storey transitions per episode (``building.completed_transitions``) are reported per arm by group and by outcome,
since the stairs-on arm is read by them: a cross-storey goal needs one, a same-storey goal needs none.

The integrity check of a paired reading: an episode of B in which the change never acted -- no traversal started,
the stairs never offered to the room-search loop, every LLM answer replayed -- should end as in A. The count of such
"quiet" episodes and of those that still differ is reported; a large second number means the arms are not paired
the way the reading assumes (``hm3d/README.md``, "Reproducible runs and paired comparisons").

::

    python -m sparx_agency.tasks.planning.objnav_benchmark_runtime.hm3d.compare_campaigns runs/A runs/B \\
        [--label-a "stairs off" --label-b "stairs on"] [--levels CSV] [--json out.json] [--markdown out.md]

Standard library plus the harness's statistics; no simulator, no model.
"""
from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Sequence

from sparx_agency.tasks.planning.objnav_benchmark.stats import (bootstrap_mean_interval, mcnemar_exact_p,
                                                                 paired_permutation_p)
from sparx_agency.tasks.planning.objnav_benchmark_runtime.hm3d.campaign_breakdown import (
    DEFAULT_LEVELS, building_of, load_levels, load_rows, mechanism, storey_of, transitions_of, traversal_allowed)

RESAMPLES = 5000


def _finite(value) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(float(value))


def group_comparison(key: str, ids: Sequence[str], a: Dict[str, dict], b: Dict[str, dict]) -> Optional[dict]:
    """SR / SPL / DTG per arm and the paired differences over ``ids``; None for an empty group."""
    ids = list(ids)
    if not ids:
        return None
    sa = [bool(a[i]["success"]) for i in ids]
    sb = [bool(b[i]["success"]) for i in ids]
    pa = [float(a[i]["spl"]) for i in ids]
    pb = [float(b[i]["spl"]) for i in ids]
    n = len(ids)
    dsr = [float(y) - float(x) for x, y in zip(sa, sb)]
    dspl = [y - x for x, y in zip(pa, pb)]
    both_finite = [i for i in ids if _finite(a[i]["distance_to_goal_m"]) and _finite(b[i]["distance_to_goal_m"])]
    da = [float(a[i]["distance_to_goal_m"]) for i in ids if _finite(a[i]["distance_to_goal_m"])]
    db = [float(b[i]["distance_to_goal_m"]) for i in ids if _finite(b[i]["distance_to_goal_m"])]
    ddtg = [float(b[i]["distance_to_goal_m"]) - float(a[i]["distance_to_goal_m"]) for i in both_finite]
    out = {"n": n,
           "SR_a": round(sum(sa) / n, 4), "SR_b": round(sum(sb) / n, 4), "dSR": round(sum(dsr) / n, 4),
           "dSR_ci": [round(v, 4) for v in bootstrap_mean_interval(dsr, resamples=RESAMPLES, seed_key="compare/sr/" + key)] if n > 1 else None,
           "mcnemar_p": round(mcnemar_exact_p(sa, sb), 4),
           "flips_to_success": sum(1 for x, y in zip(sa, sb) if y and not x),
           "flips_to_failure": sum(1 for x, y in zip(sa, sb) if x and not y),
           "SPL_a": round(sum(pa) / n, 4), "SPL_b": round(sum(pb) / n, 4), "dSPL": round(sum(dspl) / n, 4),
           "dSPL_ci": [round(v, 4) for v in bootstrap_mean_interval(dspl, resamples=RESAMPLES, seed_key="compare/spl/" + key)] if n > 1 else None,
           "spl_sign_flip_p": round(paired_permutation_p(pa, pb, permutations=RESAMPLES, seed_key="compare/splp/" + key), 4) if n > 1 else None,
           "DTG_a": round(sum(da) / len(da), 3) if da else None, "DTG_b": round(sum(db) / len(db), 3) if db else None,
           "unreachable_end_a": n - len(da), "unreachable_end_b": n - len(db),
           "dDTG": round(sum(ddtg) / len(ddtg), 3) if ddtg else None,
           "dDTG_ci": [round(v, 3) for v in bootstrap_mean_interval(ddtg, resamples=RESAMPLES, seed_key="compare/dtg/" + key)] if len(ddtg) > 1 else None,
           "dDTG_n": len(ddtg),
           "steps_a": round(sum(a[i]["steps"] for i in ids) / n, 1), "steps_b": round(sum(b[i]["steps"] for i in ids) / n, 1),
           "transitions_a": round(sum(transitions_of(a[i]) for i in ids) / n, 3),
           "transitions_b": round(sum(transitions_of(b[i]) for i in ids) / n, 3)}
    return out


def transitions_by_outcome(rows: Dict[str, dict], ids: Sequence[str]) -> dict:
    counts = [transitions_of(rows[i]) for i in ids]
    succ = [transitions_of(rows[i]) for i in ids if rows[i]["success"]]
    fail = [transitions_of(rows[i]) for i in ids if not rows[i]["success"]]
    dist = Counter(counts)
    return {"mean": round(sum(counts) / len(counts), 3) if counts else None, "with_any": sum(1 for c in counts if c),
            "max": max(counts) if counts else 0, "distribution": {str(k): dist[k] for k in sorted(dist)},
            "mean_success": round(sum(succ) / len(succ), 3) if succ else None,
            "mean_failure": round(sum(fail) / len(fail), 3) if fail else None}


def guard_painted(row: dict) -> int:
    """Cells the spawn-floor guard wrote occupied in the episode (steps, drops, stair footprints); 0 when inert."""
    stats = ((((row.get("agent_info") or {}).get("policy") or {}).get("spawn_floor_guard") or {}).get("stats")) or {}
    return int(stats.get("step_cells", 0) or 0) + int(stats.get("drop_cells", 0) or 0) + int(stats.get("stair_cells", 0) or 0)


def quiet_in_b(row_b: dict, row_a: dict) -> bool:
    """True when the change under test never acted in the episode, so B should end exactly as A.

    For the stair flag that is: no traversal started in B, the stairs never offered to B's room-search loop, every
    LLM answer of B replayed -- and A's spawn-floor guard painted nothing, because the guard is inert once traversal
    is allowed and a step or drop it masked in A is a map difference B carries from its first action (00802/000001,
    2026-11-12: 952 step cells in A, none in B, the closing three steps shorter).
    """
    building = building_of(row_b)
    events = {e.get("event") for e in building.get("events") or []}
    policy = (row_b.get("agent_info") or {}).get("policy") or {}
    loop_stats = (policy.get("room_search_loop") or {}).get("stats") or {}
    answers = policy.get("llm_answers") or {}
    return ("traversal_started" not in events and not building.get("completed_transitions")
            and int(loop_stats.get("stairs_offered", 0) or 0) == 0 and int(answers.get("asked", 0) or 0) == 0
            and guard_painted(row_a) == 0)


def same_ending(x: dict, y: dict) -> bool:
    return (bool(x["success"]), round(float(x["spl"]), 9), int(x["steps"]), round(float(x["path_length_m"]), 6)) == \
           (bool(y["success"]), round(float(y["spl"]), 9), int(y["steps"]), round(float(y["path_length_m"]), 6))


def compare(root_a: Path, root_b: Path, levels: Dict[str, dict], label_a: str, label_b: str) -> dict:
    _, a = load_rows(root_a)
    _, b = load_rows(root_b)
    common = sorted(set(a) & set(b))
    if not common:
        raise ValueError("no episode is in both %s and %s" % (root_a, root_b))
    storeys = {i: storey_of(i, levels) for i in common}
    categories = sorted({a[i]["target_category"] for i in common})
    groups = {"overall": group_comparison("overall", common, a, b)}
    for c in categories:
        groups[c] = group_comparison(c, [i for i in common if a[i]["target_category"] == c], a, b)
    for s in ("same", "cross"):
        groups["%s-storey" % s] = group_comparison(s, [i for i in common if storeys[i] == s], a, b)
    by_cat_storey = {c: {s: group_comparison("%s/%s" % (c, s), [i for i in common if a[i]["target_category"] == c and storeys[i] == s], a, b)
                         for s in ("same", "cross")} for c in categories}
    flips = {"to_success": sorted(i for i in common if b[i]["success"] and not a[i]["success"]),
             "to_failure": sorted(i for i in common if a[i]["success"] and not b[i]["success"])}
    flips["to_success_by_storey"] = dict(Counter(storeys[i] for i in flips["to_success"]))
    flips["to_failure_by_storey"] = dict(Counter(storeys[i] for i in flips["to_failure"]))
    flips["to_failure_mechanisms_in_b"] = dict(Counter(mechanism(b[i], storeys[i]) for i in flips["to_failure"]))
    flips["to_success_mechanisms_in_a"] = dict(Counter(mechanism(a[i], storeys[i]) for i in flips["to_success"]))
    failures_b = [i for i in common if not b[i]["success"]]
    failures_a = [i for i in common if not a[i]["success"]]
    quiet = [i for i in common if quiet_in_b(b[i], a[i])]
    quiet_differs = [i for i in quiet if not same_ending(a[i], b[i])]
    answers = Counter()
    for i in common:
        for k, v in ((((b[i].get("agent_info") or {}).get("policy") or {}).get("llm_answers")) or {}).items():
            if isinstance(v, (int, float)):
                answers[k] += int(v)
    return {"root_a": str(root_a), "root_b": str(root_b), "label_a": label_a, "label_b": label_b,
            "pairs": len(common), "a_only": len(set(a) - set(b)), "b_only": len(set(b) - set(a)),
            "allowed_a": sorted({str(traversal_allowed(a[i])) for i in common}), "allowed_b": sorted({str(traversal_allowed(b[i])) for i in common}),
            "storey_counts": dict(Counter(storeys[i] or "unsurveyed" for i in common)),
            "groups": groups, "by_category_and_storey": by_cat_storey,
            "transitions": {"a": {s: transitions_by_outcome(a, [i for i in common if storeys[i] == s or s == "all"]) for s in ("all", "same", "cross")},
                            "b": {s: transitions_by_outcome(b, [i for i in common if storeys[i] == s or s == "all"]) for s in ("all", "same", "cross")}},
            "mechanisms": {"a": dict(Counter(mechanism(a[i], storeys[i]) for i in failures_a).most_common()),
                           "b": dict(Counter(mechanism(b[i], storeys[i]) for i in failures_b).most_common())},
            "flips": flips,
            "integrity": {"b_quiet": len(quiet), "b_quiet_but_differs": len(quiet_differs), "examples": quiet_differs[:10],
                          "b_answers": dict(answers)},
            "wall_h": {"a": round(sum(a[i]["wall_s"] for i in common) / 3600.0, 1), "b": round(sum(b[i]["wall_s"] for i in common) / 3600.0, 1)}}


def _fmt_ci(ci) -> str:
    return "" if not ci else " [%+.3f, %+.3f]" % (ci[0], ci[1])


def _row(name: str, g: Optional[dict]) -> str:
    if g is None:
        return "| %s | 0 | – | – | – | – | – | – | – | – |" % name
    dtg_a = "–" if g["DTG_a"] is None else "%.2f" % g["DTG_a"]
    dtg_b = "–" if g["DTG_b"] is None else "%.2f" % g["DTG_b"]
    ddtg = "–" if g["dDTG"] is None else "%+.2f%s" % (g["dDTG"], _fmt_ci(g["dDTG_ci"]).replace(".3f", ".2f") if False else (" [%+.2f, %+.2f]" % tuple(g["dDTG_ci"]) if g["dDTG_ci"] else ""))
    return "| %s | %d | %.3f → %.3f | %+.3f%s (p=%s) | %.3f → %.3f | %+.3f%s | %s → %s | %s | %.0f → %.0f | %.2f → %.2f |" % (
        name, g["n"], g["SR_a"], g["SR_b"], g["dSR"], _fmt_ci(g["dSR_ci"]), g["mcnemar_p"], g["SPL_a"], g["SPL_b"], g["dSPL"], _fmt_ci(g["dSPL_ci"]),
        dtg_a, dtg_b, ddtg, g["steps_a"], g["steps_b"], g["transitions_a"], g["transitions_b"])


def markdown(r: dict) -> str:
    A, B = r["label_a"], r["label_b"]
    head = "| group | n | SR %s → %s | ΔSR [95%% CI] (McNemar p) | SPL %s → %s | ΔSPL [95%% CI] | DTG m %s → %s | ΔDTG [95%% CI] | steps | transitions/episode %s → %s |" % (A, B, A, B, A, B, A, B)
    lines = ["# %s (A) against %s (B): %d paired episodes" % (A, B, r["pairs"]),
             "", "A: `%s` (allow_stair_traversal %s) · B: `%s` (allow_stair_traversal %s) · storeys: %s" % (
                 r["root_a"], ", ".join(r["allowed_a"]), r["root_b"], ", ".join(r["allowed_b"]), json.dumps(r["storey_counts"])),
             "", head, "|---|---|---|---|---|---|---|---|---|---|"]
    order = ["overall"] + [k for k in r["groups"] if k not in ("overall", "same-storey", "cross-storey")] + ["same-storey", "cross-storey"]
    for k in order:
        lines.append(_row(k, r["groups"][k]))
    lines += ["", "## Per category by storey", "", "| category | storey | n | SR A → B | ΔSR | SPL A → B | ΔSPL | DTG A → B | transitions A → B |", "|---|---|---|---|---|---|---|---|---|"]
    for c, pair in r["by_category_and_storey"].items():
        for s in ("same", "cross"):
            g = pair[s]
            if g is None:
                continue
            lines.append("| %s | %s | %d | %.3f → %.3f | %+.3f%s | %.3f → %.3f | %+.3f | %s → %s | %.2f → %.2f |" % (
                c, s, g["n"], g["SR_a"], g["SR_b"], g["dSR"], _fmt_ci(g["dSR_ci"]), g["SPL_a"], g["SPL_b"], g["dSPL"],
                "–" if g["DTG_a"] is None else "%.2f" % g["DTG_a"], "–" if g["DTG_b"] is None else "%.2f" % g["DTG_b"], g["transitions_a"], g["transitions_b"]))
    t = r["transitions"]
    lines += ["", "## Storey transitions per episode", "", "| arm | group | mean | episodes with ≥1 | max | distribution | mean in successes | mean in failures |", "|---|---|---|---|---|---|---|---|"]
    for arm, label in (("a", A), ("b", B)):
        for s in ("all", "same", "cross"):
            x = t[arm][s]
            lines.append("| %s | %s | %s | %d | %d | %s | %s | %s |" % (label, s, x["mean"], x["with_any"], x["max"], json.dumps(x["distribution"]), x["mean_success"], x["mean_failure"]))
    f = r["flips"]
    lines += ["", "## Flips", "", "B succeeds where A failed: **%d** (by storey %s; what A's failure was: %s)" % (len(f["to_success"]), json.dumps(f["to_success_by_storey"]), json.dumps(f["to_success_mechanisms_in_a"])),
              "", "B fails where A succeeded: **%d** (by storey %s; what B's failure is: %s)" % (len(f["to_failure"]), json.dumps(f["to_failure_by_storey"]), json.dumps(f["to_failure_mechanisms_in_b"]))]
    lines += ["", "## Failure mechanisms", "", "| mechanism | %s | %s |" % (A, B), "|---|---|---|"]
    for m in sorted(set(r["mechanisms"]["a"]) | set(r["mechanisms"]["b"]), key=lambda m: -(r["mechanisms"]["a"].get(m, 0) + r["mechanisms"]["b"].get(m, 0))):
        lines.append("| %s | %d | %d |" % (m, r["mechanisms"]["a"].get(m, 0), r["mechanisms"]["b"].get(m, 0)))
    i = r["integrity"]
    lines += ["", "## Pairing integrity", "", "B episodes in which traversal never started, the stairs were never offered, every LLM answer was replayed and A's spawn-floor guard painted nothing: %d; of those, ending differently from A: %d%s. B's answers: %s." % (
        i["b_quiet"], i["b_quiet_but_differs"], (" (e.g. %s)" % ", ".join(i["examples"][:5])) if i["examples"] else "", json.dumps(i["b_answers"])),
        "", "Worker time: %s %.1f h, %s %.1f h." % (A, r["wall_h"]["a"], B, r["wall_h"]["b"])]
    return "\n".join(lines) + "\n"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("root_a", type=Path)
    parser.add_argument("root_b", type=Path)
    parser.add_argument("--label-a", default="A")
    parser.add_argument("--label-b", default="B")
    parser.add_argument("--levels", type=Path, default=DEFAULT_LEVELS)
    parser.add_argument("--json", type=Path)
    parser.add_argument("--markdown", type=Path)
    args = parser.parse_args(argv)
    levels = load_levels(args.levels) if args.levels.is_file() else {}
    report = compare(args.root_a, args.root_b, levels, args.label_a, args.label_b)
    text = markdown(report)
    if args.json:
        args.json.write_text(json.dumps(report, indent=2, sort_keys=True))
    if args.markdown:
        args.markdown.write_text(text)
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
