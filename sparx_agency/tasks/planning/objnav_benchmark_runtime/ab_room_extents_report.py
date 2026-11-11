"""The report of the room-extents A/B: two campaigns on the same episodes, the code before the mechanism and with it.

Reads both arms' ``episodes.jsonl`` rows and, where ``--record`` kept them,
their recordings, and writes:

* SR, SPL, SoftSPL and DTG per arm on the common episodes, with the paired
  differences, bootstrap intervals and the exact McNemar test from
  :mod:`paired_ab` (``objnav_benchmark/stats.py`` underneath); the same split
  by same- and cross-storey goals when the levels survey is at hand;
* the paired comparison: how many episodes improved (failed before, found
  now), how many got worse, how many stayed; SPL and DTG up / down counts;
* for every episode whose outcome changed, WHERE the arms parted: the first
  oracle round whose visit order differs (the rows' ``solver_records`` and
  ``estimate_events``, zipped), both arms' terminations, steps and DTG, and
  -- from the recordings -- the room lines each arm's oracle was shown at that
  round, the numbers it gave and its words, with the rooms whose coverage or
  size the arm-with's words name flagged as COVERAGE-DRIVEN decisions;
* the examples: the coverage-driven partings with the biggest outcome change.

Usage::

    python -m sparx_agency.tasks.planning.objnav_benchmark_runtime.ab_room_extents_report <without root> <with root> \\
        [--markdown report.md] [--json report.json] [--examples 8]

Python 3.8 syntax; standard library plus the runtime's own modules.
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import math
import os
import re
import sys
from typing import Any, Dict, List, Optional, Sequence, Tuple

from sparx_agency.core.mapping.topology.search_node_oracle import ROOM, STAIRS, OPENING, SearchNode, format_node
from sparx_agency.tasks.planning.objnav_benchmark_runtime.hm3d.representative_set import SETS_DIR, load_levels
from sparx_agency.tasks.planning.objnav_benchmark_runtime.paired_ab import compare, load_rows

#: Words in the oracle's ``why`` (or ``house`` / ``stage``) that name the coverage or size facts the mechanism added.
COVERAGE_WORDS = ("cover", "left", "unseen", "wall", "m2", "mapped", "at least", "most of", "unexplored part", "remaining",
                  "lower bound", "size")
NODE_FIELDS = {f for f in SearchNode.__dataclass_fields__}  # type: ignore[attr-defined]


def recording_key(episode_id: str) -> str:
    return hashlib.sha256(episode_id.encode()).hexdigest()[:12]


def find_recording(root: str, episode_id: str) -> Optional[str]:
    """The ``recordings/<key>`` directory of ``episode_id`` under ``root``, or None."""
    key = recording_key(episode_id)
    hits = glob.glob(os.path.join(root, "**", "recordings", key), recursive=True)
    return hits[0] if hits else None


def load_steps(directory: str) -> List[dict]:
    path = os.path.join(directory, "steps.jsonl")
    if not os.path.isfile(path):
        return []
    rows = []
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def rounds(row: dict) -> List[dict]:
    """The episode's oracle rounds in order: ``{step, order, top, reading, pass, house}`` from the row's two ledgers."""
    policy = ((row.get("agent_info") or {}).get("policy") or {})
    events = ((policy.get("room_search_loop") or {}).get("estimate_events") or [])
    solves = policy.get("solver_records") or []
    out = []
    for index, event in enumerate(events):
        solve = solves[index] if index < len(solves) else {}
        order = [int(v) for v in (solve.get("order") or [])]
        out.append({"index": index, "step": int(event.get("step", -1)), "order": order,
                    "top": order[0] if order else None, "reading": event.get("reading") or {},
                    "pass": event.get("pass_verdict"), "house": event.get("house", ""),
                    "p_present": event.get("p_present")})
    return out


def first_parting(a_rounds: Sequence[dict], b_rounds: Sequence[dict]) -> Optional[Tuple[dict, dict]]:
    """The first oracle round whose NEXT room differs between the arms (the same index in both), or None."""
    for ra, rb in zip(a_rounds, b_rounds):
        if ra["top"] != rb["top"] or ra["order"] != rb["order"]:
            return ra, rb
    if len(a_rounds) != len(b_rounds):
        shorter = min(len(a_rounds), len(b_rounds))
        if shorter < len(a_rounds):
            return a_rounds[shorter], {"index": shorter, "step": None, "order": [], "top": None, "reading": {}, "pass": None,
                                       "house": "", "p_present": None}
        return ({"index": shorter, "step": None, "order": [], "top": None, "reading": {}, "pass": None, "house": "",
                 "p_present": None}, b_rounds[shorter])
    return None


def oracle_at(steps: Sequence[dict], step: Optional[int]) -> Optional[dict]:
    """The recorded reasoning at ``step``: the nodes, the probabilities, the reasons, the reading."""
    if step is None:
        return None
    for row in steps:
        if int(row.get("step", -1)) == int(step):
            reasoning = (row.get("method") or {}).get("reasoning") or {}
            if reasoning.get("nodes"):
                return reasoning
    # The round's reasoning may be recorded on the action after the oracle call: take the nearest later row.
    later = [r for r in steps if int(r.get("step", -1)) > int(step) and ((r.get("method") or {}).get("reasoning") or {}).get("nodes")]
    return ((later[0].get("method") or {}).get("reasoning") or {}) if later else None


def render_nodes(reasoning: dict) -> List[dict]:
    """Each node of a recorded reasoning as ``{id, line, p, why}``, sorted by probability."""
    probs = {str(k): float(v) for k, v in ((reasoning.get("oracle") or {}).get("probs") or {}).items()}
    reasons = {str(k): str(v) for k, v in ((reasoning.get("oracle") or {}).get("reasons") or {}).items()}
    out = []
    for node in reasoning.get("nodes") or []:
        fields = {k: v for k, v in node.items() if k in NODE_FIELDS}
        for key in ("objects", "glimpsed"):
            if key in fields and fields[key] is not None:
                fields[key] = tuple(fields[key])
        if fields.get("kind") not in (ROOM, STAIRS, OPENING):
            fields["kind"] = ROOM
        try:
            line = format_node(SearchNode(**fields))
        except Exception as exc:  # a recording older than a field: show what there is
            line = "id=%s (unrenderable: %s)" % (node.get("id"), exc)
        nid = str(node.get("id"))
        out.append({"id": node.get("id"), "line": line, "p": probs.get(nid), "why": reasons.get(nid, "")})
    out.sort(key=lambda item: -(item["p"] if item["p"] is not None else -1.0))
    return out


def coverage_driven(b_nodes: Sequence[dict], a_top: Any, b_top: Any) -> Tuple[bool, str]:
    """Whether arm-with's choice reads as made on coverage or size: its words for the rooms that swapped name them."""
    if a_top == b_top:
        return False, ""
    named = []
    for item in b_nodes:
        if item["id"] in (a_top, b_top):
            text = (item["why"] or "").lower()
            if any(word in text for word in COVERAGE_WORDS):
                named.append("room %s: %s" % (item["id"], item["why"]))
    return bool(named), "; ".join(named)


def episode_detail(episode_id: str, a_row: dict, b_row: dict, a_root: str, b_root: str) -> dict:
    """Everything the report says about one episode whose outcome changed."""
    a_rounds, b_rounds = rounds(a_row), rounds(b_row)
    parting = first_parting(a_rounds, b_rounds)
    detail = {
        "episode_id": episode_id, "target": a_row.get("target_category"),
        "without": {"success": bool(a_row.get("success")), "spl": float(a_row.get("spl", 0.0)),
                    "dtg_m": a_row.get("distance_to_goal_m"), "steps": a_row.get("steps"),
                    "termination": a_row.get("termination"), "rounds": len(a_rounds)},
        "with": {"success": bool(b_row.get("success")), "spl": float(b_row.get("spl", 0.0)),
                 "dtg_m": b_row.get("distance_to_goal_m"), "steps": b_row.get("steps"),
                 "termination": b_row.get("termination"), "rounds": len(b_rounds)},
        "parted": None, "coverage_driven": False, "coverage_words": "",
    }
    if parting is None:
        detail["parted"] = {"where": "never at an oracle round: the orders agreed round for round" if a_rounds else
                            "no oracle round in either arm"}
        return detail
    ra, rb = parting
    parted = {"round": ra["index"], "without_step": ra["step"], "with_step": rb["step"],
              "without_order": ra["order"], "with_order": rb["order"],
              "without_reading": ra["reading"], "with_reading": rb["reading"],
              "without_house": ra["house"], "with_house": rb["house"]}
    a_dir, b_dir = find_recording(a_root, episode_id), find_recording(b_root, episode_id)
    if a_dir and ra["step"] is not None:
        reasoning = oracle_at(load_steps(a_dir), ra["step"])
        parted["without_nodes"] = render_nodes(reasoning) if reasoning else []
    if b_dir and rb["step"] is not None:
        reasoning = oracle_at(load_steps(b_dir), rb["step"])
        parted["with_nodes"] = render_nodes(reasoning) if reasoning else []
    detail["parted"] = parted
    flagged, words = coverage_driven(parted.get("with_nodes", []), ra["top"], rb["top"])
    detail["coverage_driven"], detail["coverage_words"] = flagged, words
    return detail


def mean(values: Sequence[float]) -> Optional[float]:
    clean = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    return sum(clean) / len(clean) if clean else None


def arm_summary(rows: Dict[str, dict], ids: Sequence[str]) -> dict:
    return {"episodes": len(ids),
            "sr": mean([1.0 if rows[i].get("success") else 0.0 for i in ids]),
            "spl": mean([rows[i].get("spl", 0.0) for i in ids]),
            "soft_spl": mean([rows[i].get("soft_spl") for i in ids]),
            "dtg_m": mean([rows[i].get("distance_to_goal_m") for i in ids]),
            "steps": mean([rows[i].get("steps") for i in ids])}


def build_report(a_root: str, b_root: str, examples: int = 8, levels: Optional[Dict[str, Dict[str, str]]] = None) -> dict:
    a, b = load_rows(a_root), load_rows(b_root)
    common = sorted(set(a) & set(b))
    if not common:
        raise ValueError("no episode is in both arms")
    report = {"without_root": a_root, "with_root": b_root, "common_episodes": len(common),
              "only_without": sorted(set(a) - set(b)), "only_with": sorted(set(b) - set(a)),
              "without": arm_summary(a, common), "with": arm_summary(b, common),
              "paired": compare(a, b)}
    improved = [i for i in common if b[i].get("success") and not a[i].get("success")]
    worse = [i for i in common if a[i].get("success") and not b[i].get("success")]
    spl_up = [i for i in common if float(b[i].get("spl", 0)) > float(a[i].get("spl", 0)) + 0.01]
    spl_down = [i for i in common if float(b[i].get("spl", 0)) < float(a[i].get("spl", 0)) - 0.01]
    dtg_pairs = [(i, a[i].get("distance_to_goal_m"), b[i].get("distance_to_goal_m")) for i in common]
    dtg_better = [i for i, x, y in dtg_pairs if x is not None and y is not None and float(y) < float(x) - 0.1]
    dtg_worse = [i for i, x, y in dtg_pairs if x is not None and y is not None and float(y) > float(x) + 0.1]
    report["changes"] = {"improved": improved, "worse": worse, "same_outcome": len(common) - len(improved) - len(worse),
                         "spl_up": len(spl_up), "spl_down": len(spl_down), "dtg_better": len(dtg_better), "dtg_worse": len(dtg_worse)}
    if levels:
        same = [i for i in common if str(levels.get(i, {}).get("goal_same_level", "")).strip() == "1"]
        cross = [i for i in common if i in levels and i not in same]
        report["same_storey"] = {"episodes": len(same), "without": arm_summary(a, same) if same else None,
                                 "with": arm_summary(b, same) if same else None}
        report["cross_storey"] = {"episodes": len(cross), "without": arm_summary(a, cross) if cross else None,
                                  "with": arm_summary(b, cross) if cross else None}
    details = [episode_detail(i, a[i], b[i], a_root, b_root) for i in improved + worse]
    report["changed_episodes"] = details
    # Where did the arms part, over every common episode? The mechanism acts at an oracle round; an episode whose
    # orders agreed throughout and still ended differently parted elsewhere (perception, kinematics, a fresh answer).
    parted_at_round = parted_never = 0
    for i in common:
        if first_parting(rounds(a[i]), rounds(b[i])) is None:
            parted_never += 1
        else:
            parted_at_round += 1
    report["parting"] = {"at_an_oracle_round": parted_at_round, "orders_agreed_throughout": parted_never}
    flagged = [d for d in details if d["coverage_driven"]]
    flagged.sort(key=lambda d: -abs(d["with"]["spl"] - d["without"]["spl"]))
    report["coverage_examples"] = flagged[:max(0, int(examples))]
    return report


def _fmt(value: Optional[float], digits: int = 3) -> str:
    return "-" if value is None else ("%%.%df" % digits) % value


def markdown(report: dict) -> str:
    w, v, p, c = report["without"], report["with"], report["paired"], report["changes"]
    lines = ["# Room size and coverage: A/B on %d common episodes" % report["common_episodes"], "",
             "| arm | SR | SPL | SoftSPL | DTG (m) | steps |", "|---|---|---|---|---|---|",
             "| without (`%s`) | %s | %s | %s | %s | %s |" % (report["without_root"], _fmt(w["sr"]), _fmt(w["spl"]), _fmt(w["soft_spl"]), _fmt(w["dtg_m"], 2), _fmt(w["steps"], 0)),
             "| with (`%s`) | %s | %s | %s | %s | %s |" % (report["with_root"], _fmt(v["sr"]), _fmt(v["spl"]), _fmt(v["soft_spl"]), _fmt(v["dtg_m"], 2), _fmt(v["steps"], 0)),
             ""]
    lines.append("Paired (with - without): SR %s, SPL %s; McNemar exact p = %s, SPL sign-flip p = %s; flip rate %s" % (
        _ci(p, "sr"), _ci(p, "spl"), _fmt(p.get("mcnemar_p"), 3), _fmt(p.get("spl_sign_flip_p"), 3), _fmt(p.get("flip_rate"), 3)))
    lines += ["", "## Paired comparison", "",
              "- improved (failed without, found with): **%d** -- %s" % (len(c["improved"]), ", ".join(c["improved"]) or "none"),
              "- worse (found without, failed with): **%d** -- %s" % (len(c["worse"]), ", ".join(c["worse"]) or "none"),
              "- same outcome: %d; SPL up on %d / down on %d (over 0.01); DTG better on %d / worse on %d (over 0.1 m)"
              % (c["same_outcome"], c["spl_up"], c["spl_down"], c["dtg_better"], c["dtg_worse"]),
              "- the arms parted at an oracle round in %d episodes; the orders agreed throughout in %d"
              % (report["parting"]["at_an_oracle_round"], report["parting"]["orders_agreed_throughout"])]
    for key, title in (("same_storey", "Same-storey goals"), ("cross_storey", "Cross-storey goals")):
        block = report.get(key)
        if block and block.get("without"):
            lines += ["", "### %s (%d)" % (title, block["episodes"]),
                      "without: SR %s SPL %s DTG %s | with: SR %s SPL %s DTG %s" % (
                          _fmt(block["without"]["sr"]), _fmt(block["without"]["spl"]), _fmt(block["without"]["dtg_m"], 2),
                          _fmt(block["with"]["sr"]), _fmt(block["with"]["spl"]), _fmt(block["with"]["dtg_m"], 2))]
    lines += ["", "## Episodes whose outcome changed, and where the arms parted", ""]
    for d in report["changed_episodes"]:
        lines += _episode_md(d)
    lines += ["", "## Examples where coverage changed the oracle's decision", ""]
    if not report["coverage_examples"]:
        lines.append("(none flagged: no parting round where arm-with's words for the swapped rooms named coverage or size)")
    for d in report["coverage_examples"]:
        lines += _episode_md(d, full=True)
    return "\n".join(lines) + "\n"


def _ci(paired: dict, key: str) -> str:
    """``+0.020 [-0.010, 0.050]`` from ``paired_ab.compare``'s ``<key>_diff`` and ``<key>_ci``."""
    diff, ci = paired.get("%s_diff" % key), paired.get("%s_ci" % key) or []
    if diff is None:
        return "-"
    text = "%+.3f" % float(diff)
    if len(ci) == 2:
        text += " [%s, %s]" % (_fmt(ci[0]), _fmt(ci[1]))
    return text


def _episode_md(d: dict, full: bool = False) -> List[str]:
    w, v = d["without"], d["with"]
    verdict = "IMPROVED" if v["success"] and not w["success"] else "WORSE" if w["success"] and not v["success"] else "changed"
    lines = ["### %s (%s) -- %s%s" % (d["episode_id"], d["target"], verdict, "  [coverage-driven]" if d["coverage_driven"] else ""),
             "without: %s, SPL %.3f, DTG %s m, %s steps, %s | with: %s, SPL %.3f, DTG %s m, %s steps, %s" % (
                 "success" if w["success"] else "failure", w["spl"], _fmt(w["dtg_m"], 2), w["steps"], w["termination"],
                 "success" if v["success"] else "failure", v["spl"], _fmt(v["dtg_m"], 2), v["steps"], v["termination"])]
    parted = d.get("parted") or {}
    if "where" in parted:
        lines.append("parted: %s" % parted["where"])
        return lines + [""]
    lines.append("parted at oracle round %s (without: action %s, order %s; with: action %s, order %s)" % (
        parted.get("round"), parted.get("without_step"), parted.get("without_order"), parted.get("with_step"), parted.get("with_order")))
    if d["coverage_words"]:
        lines.append("arm-with's words: %s" % d["coverage_words"])
    if full or d["coverage_driven"]:
        for arm, key in (("without", "without_nodes"), ("with", "with_nodes")):
            nodes = parted.get(key) or []
            if nodes:
                lines.append("")
                lines.append("%s -- the oracle's lines and numbers at that round:" % arm)
                lines.append("")
                lines.append("```")
                for item in nodes[:6]:
                    lines.append("%s  -> p=%s  %s" % (item["line"], _fmt(item["p"], 2), item["why"]))
                lines.append("```")
                reading = parted.get("%s_reading" % arm) or {}
                if reading:
                    lines.append("reading: %s" % json.dumps(reading))
    return lines + [""]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("without_root")
    parser.add_argument("with_root")
    parser.add_argument("--markdown", default=None)
    parser.add_argument("--json", default=None)
    parser.add_argument("--examples", type=int, default=8)
    parser.add_argument("--levels", default=str(SETS_DIR / "hm3d_v1_val_levels.csv"))
    args = parser.parse_args(argv)
    levels = load_levels(args.levels) if args.levels else None
    report = build_report(args.without_root, args.with_root, examples=args.examples, levels=levels or None)
    text = markdown(report)
    if args.markdown:
        with open(args.markdown, "w", encoding="utf-8") as handle:
            handle.write(text)
    if args.json:
        with open(args.json, "w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=1, default=str)
    sys.stdout.write(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
