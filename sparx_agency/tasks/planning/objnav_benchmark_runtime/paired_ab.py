"""A paired A/B of two campaigns on the same episodes, read from their ``episodes.jsonl`` rows.

Reports SR and SPL per arm, the B - A differences with paired-bootstrap intervals, the exact McNemar test on the
success flips and the sign-flip test on SPL (``objnav_benchmark/stats.py``), the flip rate, the decision rule, and
how many pairs a decision would need at the measured flip rate.

The integrity check is the reason the arms are paired through the LLM answer store
(``core/mapping/topology/llm_answer_cache.py``): arm B replays arm A's answers, so in an episode where B's change
never acted -- for the geometric-glance A/B, no geometric glance taken -- B asked A's questions in A's order and must
end exactly as A did. Any such episode that differs is a bug in the pairing, and the numbers are not to be read.

The decision rule (the user's, 2026-11-01): keep geometric glances ON only if SR improves significantly (the 95 %
interval of B - A above zero) and SPL does not drop significantly (its interval not entirely below zero).

Usage::

    python -m sparx_agency.tasks.planning.objnav_benchmark_runtime.paired_ab <arm A root>[,<root>...] \\
        <arm B root>[,<root>...] [--hours-per-pair H] [--json out.json]
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
from typing import Dict, List, Optional, Sequence

from sparx_agency.tasks.planning.objnav_benchmark.stats import (bootstrap_mean_interval, mcnemar_exact_p,
                                                                 paired_permutation_p)

Z_TWO_SIDED_95 = 1.959964
Z_POWER_80 = 0.841621


def load_rows(roots: str) -> Dict[str, dict]:
    """Every episode row under the comma-separated campaign roots, by episode id (a later root wins)."""
    out = {}
    for root in roots.split(","):
        for path in sorted(glob.glob(os.path.join(root, "**", "episodes.jsonl"), recursive=True)):
            with open(path) as fh:
                for line in fh:
                    if line.strip():
                        row = json.loads(line)
                        out[row["episode_id"]] = row
    return out


def geometric_glances(row: dict) -> int:
    """Geometric glances an episode took: the glance scheduler's starts less its cue glances."""
    stats = (((row.get("agent_info") or {}).get("policy") or {}).get("glances") or {}).get("stats") or {}
    return int(stats.get("started", 0)) - int(stats.get("cues", 0))


def mcnemar_pairs_needed(flip_rate: float, delta: float, power_z: float = Z_POWER_80,
                         alpha_z: float = Z_TWO_SIDED_95) -> Optional[int]:
    """Pairs for the McNemar test to detect a success-rate difference ``delta`` when ``flip_rate`` of pairs differ.

    Connor's normal approximation, ``n = (z_a sqrt(p_d) + z_b sqrt(p_d - delta^2))^2 / delta^2``; None when no
    such design exists (``delta`` zero or ``|delta|`` above what the flip rate allows).
    """
    if delta == 0 or flip_rate <= delta * delta or abs(delta) > flip_rate:
        return None
    return int(math.ceil((alpha_z * math.sqrt(flip_rate) + power_z * math.sqrt(flip_rate - delta * delta)) ** 2
                         / (delta * delta)))


def compare(a: Dict[str, dict], b: Dict[str, dict], resamples: int = 10000) -> dict:
    """The paired summary of the episodes both arms ran."""
    common = sorted(set(a) & set(b))
    if not common:
        raise ValueError("no episode is in both arms")
    sa = [bool(a[k]["success"]) for k in common]
    sb = [bool(b[k]["success"]) for k in common]
    pa = [float(a[k]["spl"]) for k in common]
    pb = [float(b[k]["spl"]) for k in common]
    n = len(common)
    dsr = [float(y) - float(x) for x, y in zip(sa, sb)]
    dspl = [y - x for x, y in zip(pa, pb)]
    sr_ci = bootstrap_mean_interval(dsr, resamples=resamples, seed_key="paired_ab/sr")
    spl_ci = bootstrap_mean_interval(dspl, resamples=resamples, seed_key="paired_ab/spl")
    b_only = [k for k, x, y in zip(common, sa, sb) if y and not x]
    a_only = [k for k, x, y in zip(common, sa, sb) if x and not y]
    quiet = [k for k in common if geometric_glances(b[k]) == 0]
    broken = [k for k in quiet if (a[k]["success"], round(float(a[k]["spl"]), 9), a[k]["steps"],
                                   round(float(a[k]["path_length_m"]), 6))
              != (b[k]["success"], round(float(b[k]["spl"]), 9), b[k]["steps"], round(float(b[k]["path_length_m"]), 6))]
    answers = {"replayed": 0, "repeated": 0, "asked": 0}
    for k in common:
        got = (((b[k].get("agent_info") or {}).get("policy") or {}).get("llm_answers")) or {}
        for key in answers:
            answers[key] += int(got.get(key, 0))
    summary = {
        "pairs": n, "a_only": sorted(set(a) - set(b)), "b_only": sorted(set(b) - set(a)),
        "sr_a": sum(sa) / n, "sr_b": sum(sb) / n, "sr_diff": sum(dsr) / n, "sr_ci": list(sr_ci),
        "mcnemar_p": mcnemar_exact_p(sa, sb), "flips_to_success": b_only, "flips_to_failure": a_only,
        "flip_rate": (len(b_only) + len(a_only)) / n,
        "spl_a": sum(pa) / n, "spl_b": sum(pb) / n, "spl_diff": sum(dspl) / n, "spl_ci": list(spl_ci),
        "spl_sign_flip_p": paired_permutation_p(pa, pb, permutations=10000),
        "spl_diff_sd": _sd(dspl),
        "b_never_glanced": len(quiet), "b_never_glanced_but_differs": broken,
        "b_answers": answers,
    }
    summary["decision"] = decision(summary)
    return summary


def decision(summary: dict) -> str:
    """The rule: ON only if SR's interval is above zero and SPL's is not entirely below zero."""
    if summary["b_never_glanced_but_differs"]:
        return "not readable: episodes in which B never glanced end differently from A"
    sr_up = summary["sr_ci"][0] > 0
    spl_down = summary["spl_ci"][1] < 0
    if sr_up and not spl_down:
        return "keep geometric glances ON"
    if spl_down:
        return "OFF: SPL drops significantly"
    return "OFF for now: the SR gain is not significant at this size"


def pairs_for_decision(summary: dict, deltas: Sequence[float] = (0.02, 0.03)) -> List[dict]:
    """Pairs needed (power 0.8) at the measured flip rate, for the observed SR difference and the given ones."""
    rate = summary["flip_rate"]
    wanted = []
    if summary["sr_diff"] > 0:
        wanted.append(("observed", summary["sr_diff"]))
    wanted += [("%.0f points" % (100 * d), d) for d in deltas]
    return [{"effect": name, "delta": d, "pairs": mcnemar_pairs_needed(rate, d)} for name, d in wanted]


def _sd(values: Sequence[float]) -> float:
    m = sum(values) / len(values)
    return math.sqrt(sum((v - m) ** 2 for v in values) / max(1, len(values) - 1))


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("a", help="arm A's campaign roots, comma-separated")
    p.add_argument("b", help="arm B's campaign roots, comma-separated")
    p.add_argument("--resamples", type=int, default=10000)
    p.add_argument("--hours-per-pair", type=float, help="measured wall hours per pair (both arms), for the estimate")
    p.add_argument("--json", help="also write the summary here")
    args = p.parse_args(argv)
    s = compare(load_rows(args.a), load_rows(args.b), args.resamples)
    print("%d pairs (only in A: %d, only in B: %d)" % (s["pairs"], len(s["a_only"]), len(s["b_only"])))
    print("SR   A %.3f  B %.3f  B-A %+.3f  95%% CI [%+.3f, %+.3f]  McNemar exact p = %.3g" % (
        s["sr_a"], s["sr_b"], s["sr_diff"], s["sr_ci"][0], s["sr_ci"][1], s["mcnemar_p"]))
    print("SPL  A %.3f  B %.3f  B-A %+.3f  95%% CI [%+.3f, %+.3f]  sign-flip p = %.3g  (sd of the differences %.3f)" % (
        s["spl_a"], s["spl_b"], s["spl_diff"], s["spl_ci"][0], s["spl_ci"][1], s["spl_sign_flip_p"], s["spl_diff_sd"]))
    print("flips: %d to success, %d to failure; flip rate %.3f" % (
        len(s["flips_to_success"]), len(s["flips_to_failure"]), s["flip_rate"]))
    print("integrity: B never glanced in %d episodes; %d of them end differently from A%s" % (
        s["b_never_glanced"], len(s["b_never_glanced_but_differs"]),
        (": " + ", ".join(s["b_never_glanced_but_differs"][:10])) if s["b_never_glanced_but_differs"] else ""))
    total = sum(s["b_answers"].values())
    if total:
        print("arm B's LLM answers: %d replayed from A (%.0f %%), %d repeated, %d asked" % (
            s["b_answers"]["replayed"], 100.0 * s["b_answers"]["replayed"] / total, s["b_answers"]["repeated"],
            s["b_answers"]["asked"]))
    print("decision: " + s["decision"])
    s["pairs_for_decision"] = pairs_for_decision(s)
    for row in s["pairs_for_decision"]:
        more = None if row["pairs"] is None else max(0, row["pairs"] - s["pairs"])
        hours = "" if more is None or not args.hours_per_pair else ", ~%.0f h more" % (more * args.hours_per_pair)
        print("  to detect %s (%+.3f): %s pairs%s" % (
            row["effect"], row["delta"], "n/a" if row["pairs"] is None else "%d (%d more)" % (row["pairs"], more), hours))
    if args.json:
        with open(args.json, "w") as fh:
            json.dump(s, fh, indent=1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
