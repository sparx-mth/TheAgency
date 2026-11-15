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

The oracle rounds come from the rows' ledgers where the run kept them. The
launchers write ``--lean`` rows by default, which drop those ledgers; then the
rounds are read from the arms' ANSWER STORES (``LLM_ANSWERS_RECORD``, the
``<root>_answers`` directory beside each root unless ``--answers-without`` /
``--answers-with`` say otherwise): every oracle prompt with its room lines and
every answer with its numbers and words, in order, per episode. A round from
the store carries the state the oracle was shown -- the rooms, their types and
their evidence, without the size and coverage fields -- so a parting is also
said to be on THE SAME STATE (the mechanism's lines were the only difference,
and they changed the decision) or after an earlier divergence (the arms had
already parted elsewhere: a fresh room label, a different scan).

Usage::

    python -m sparx_agency.tasks.planning.objnav_benchmark_runtime.ab_room_extents_report <without root> <with root> \\
        [--markdown report.md] [--json report.json] [--examples 8] [--answers-without DIR] [--answers-with DIR]

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
from collections import defaultdict
from typing import Any, Dict, List, Optional, Sequence, Tuple

from sparx_agency.core.mapping.topology.search_node_oracle import ROOM, STAIRS, OPENING, SearchNode, format_node
from sparx_agency.tasks.planning.objnav_benchmark_runtime.hm3d.representative_set import SETS_DIR, load_levels
from sparx_agency.tasks.planning.objnav_benchmark_runtime.paired_ab import compare, load_rows

#: Words in the oracle's ``why`` (or ``house`` / ``stage``) that name the coverage or size facts the mechanism added.
COVERAGE_WORDS = ("cover", "left", "unseen", "wall", "m2", "mapped", "at least", "most of", "unexplored part", "remaining",
                  "lower bound", "size")
NODE_FIELDS = {f for f in SearchNode.__dataclass_fields__}  # type: ignore[attr-defined]
#: A room line of the oracle prompt, both before and with the mechanism: ``id=3  ROOM  type=... size=... ...``.
NODE_LINE = re.compile(r"^id=(\d+)\s+(ROOM|STAIRS|OPENING)\s+(.*)$")
#: The answer-store route the node oracle records under (``llm_client`` names the label route ``default``).
ORACLE_ROUTE = "reasoning"


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


def default_answer_store(root: str) -> Optional[str]:
    """The ``<root>_answers`` directory the A/B launcher records beside an arm's root, when it exists."""
    base = os.path.normpath(root)
    candidate = os.path.join(os.path.dirname(base), os.path.basename(base) + "_answers")
    return candidate if os.path.isdir(candidate) else None


def load_answer_store(path: Optional[str]) -> Dict[str, List[dict]]:
    """The node oracle's records of an answer store, by episode, in the order they were recorded.

    Every ``*.jsonl`` file under ``path`` holds one record per line (``llm_answer_cache``); a line that is not JSON
    (a write cut by a crash) is skipped. Only the oracle's route is kept: the room-label route asks nothing the
    report reads.
    """
    by_episode = defaultdict(list)  # type: Dict[str, List[dict]]
    if not path or not os.path.isdir(path):
        return {}
    serial = 0
    for file in sorted(glob.glob(os.path.join(path, "*.jsonl"))):
        with open(file, encoding="utf-8", errors="replace") as handle:
            for line in handle:
                line = line.strip()
                if not line.startswith("{"):
                    continue
                try:
                    record = json.loads(line)
                except ValueError:
                    continue
                if record.get("route") != ORACLE_ROUTE:
                    continue
                serial += 1
                record["_serial"] = serial
                by_episode[str(record.get("scope"))].append(record)
    for records in by_episode.values():
        records.sort(key=lambda r: (str(r.get("recorded_utc") or ""), r["_serial"]))
    return dict(by_episode)


def parse_node_line(line: str) -> Optional[dict]:
    """One room line of the oracle prompt as ``{id, kind, fields, line}``; None for any other line.

    Fields are ``key=value`` groups two or more spaces apart; ``seen: a, b`` is the room's evidence. The format
    before the mechanism (``size=14m2  frontier=3  exits=2 (~35m2 unseen behind them)  ...  here=yes  seen: ...``)
    and with it (``size=at least 14m2 (...)  covered=7m2 (...)  seen: ...  frontier=3 (~33m2 unknown behind them)``)
    both parse; the signature below keeps what the mechanism did not touch.
    """
    match = NODE_LINE.match(line.strip())
    if not match:
        return None
    fields = {}  # type: Dict[str, str]
    for part in re.split(r"\s{2,}", match.group(3).strip()):
        if part.startswith("seen:"):
            fields["seen"] = part[len("seen:"):].strip()
        elif "=" in part:
            key, value = part.split("=", 1)
            fields[key.strip()] = value.strip()
    return {"id": int(match.group(1)), "kind": match.group(2), "fields": fields, "line": line.strip()}


def _leading_int(text: Optional[str]) -> Optional[int]:
    match = re.match(r"\s*(\d+)", text or "")
    return int(match.group(1)) if match else None


def state_signature(nodes: Sequence[dict]) -> Tuple[Tuple[Any, ...], ...]:
    """What the oracle was shown that the mechanism does not change: each room's id, kind, type, evidence, frontier
    and exit counts and its status word. Equal signatures in both arms mean the same situation, differently worded."""
    out = []
    for node in nodes:
        fields = node["fields"]
        status = re.match(r"[a-z_]+", fields.get("status", "") or "")
        out.append((node["id"], node["kind"], fields.get("type"), fields.get("seen"),
                    _leading_int(fields.get("frontier")), _leading_int(fields.get("exits")),
                    status.group(0) if status else None))
    return tuple(out)


def store_rounds(records: Sequence[dict]) -> List[dict]:
    """The episode's oracle rounds from its answer-store records, in the shape :func:`rounds` gives, plus
    ``nodes`` (the room lines with the numbers and words the oracle gave them) and ``state`` (:func:`state_signature`).

    The order is the oracle's own numbers, highest first (ties by id): the solver's order, which the ledgers hold,
    is not in the store. ``step`` is unknown and None.
    """
    out = []
    for index, record in enumerate(records):
        prompt = str(record.get("user") or "")
        answer = record.get("answer")
        if isinstance(answer, str):
            try:
                answer = json.loads(answer)
            except ValueError:
                answer = {}
        answer = answer if isinstance(answer, dict) else {}
        nodes = [n for n in (parse_node_line(line) for line in prompt.splitlines()) if n]
        scores, reasons = {}, {}  # type: Dict[int, float], Dict[int, str]
        for item in answer.get("nodes") or []:
            try:
                nid = int(item.get("id"))
                scores[nid] = float(item.get("p"))
            except (TypeError, ValueError):
                continue
            reasons[nid] = str(item.get("why") or "")
        order = sorted(scores, key=lambda nid: (-scores[nid], nid))
        house = next((line[len("HOUSE:"):].strip() for line in prompt.splitlines() if line.startswith("HOUSE:")), "")
        rendered = [{"id": n["id"], "line": n["line"], "p": scores.get(n["id"]), "why": reasons.get(n["id"], "")} for n in nodes]
        rendered.sort(key=lambda item: -(item["p"] if item["p"] is not None else -1.0))
        out.append({"index": index, "step": None, "order": order, "top": order[0] if order else None,
                    "reading": {str(k): v for k, v in scores.items()}, "pass": answer.get("pass"), "house": house,
                    "p_present": None, "nodes": rendered, "state": state_signature(nodes)})
    return out


def pick_rounds(a_row: dict, b_row: dict, a_store: Dict[str, List[dict]], b_store: Dict[str, List[dict]],
                episode_id: str) -> Tuple[List[dict], List[dict], str]:
    """Both arms' rounds from one source: the ledgers when both rows carry them, else the answer stores."""
    a_rounds, b_rounds = rounds(a_row), rounds(b_row)
    if a_rounds or b_rounds:
        return a_rounds, b_rounds, "ledgers"
    a_records, b_records = a_store.get(episode_id) or [], b_store.get(episode_id) or []
    if a_records or b_records:
        return store_rounds(a_records), store_rounds(b_records), "answer store"
    return [], [], "none"


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


def episode_detail(episode_id: str, a_row: dict, b_row: dict, a_root: str, b_root: str,
                   a_store: Optional[Dict[str, List[dict]]] = None, b_store: Optional[Dict[str, List[dict]]] = None) -> dict:
    """Everything the report says about one episode whose outcome changed."""
    a_rounds, b_rounds, source = pick_rounds(a_row, b_row, a_store or {}, b_store or {}, episode_id)
    parting = first_parting(a_rounds, b_rounds)
    detail = {
        "episode_id": episode_id, "target": a_row.get("target_category"),
        "without": {"success": bool(a_row.get("success")), "spl": float(a_row.get("spl", 0.0)),
                    "dtg_m": a_row.get("distance_to_goal_m"), "steps": a_row.get("steps"),
                    "termination": a_row.get("termination"), "rounds": len(a_rounds)},
        "with": {"success": bool(b_row.get("success")), "spl": float(b_row.get("spl", 0.0)),
                 "dtg_m": b_row.get("distance_to_goal_m"), "steps": b_row.get("steps"),
                 "termination": b_row.get("termination"), "rounds": len(b_rounds)},
        "rounds_source": source, "parted": None, "same_state_at_parting": None,
        "coverage_driven": False, "coverage_words": "",
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
    if source == "answer store":
        parted["without_nodes"], parted["with_nodes"] = ra.get("nodes") or [], rb.get("nodes") or []
        if ra.get("state") is not None and rb.get("state") is not None:
            detail["same_state_at_parting"] = ra["state"] == rb["state"]
    else:
        a_dir, b_dir = find_recording(a_root, episode_id), find_recording(b_root, episode_id)
        if a_dir and ra["step"] is not None:
            reasoning = oracle_at(load_steps(a_dir), ra["step"])
            parted["without_nodes"] = render_nodes(reasoning) if reasoning else []
        if b_dir and rb["step"] is not None:
            reasoning = oracle_at(load_steps(b_dir), rb["step"])
            parted["with_nodes"] = render_nodes(reasoning) if reasoning else []
    detail["parted"] = parted
    flagged, words = coverage_driven(parted.get("with_nodes", []), ra["top"], rb["top"])
    # On the same state the mechanism's lines (and its rules) were the only difference the oracle saw: a changed
    # top room there is its doing whether or not the words say so.
    detail["coverage_driven"] = flagged or (bool(detail["same_state_at_parting"]) and ra["top"] != rb["top"])
    detail["coverage_words"] = words
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


def build_report(a_root: str, b_root: str, examples: int = 8, levels: Optional[Dict[str, Dict[str, str]]] = None,
                 a_answers: Optional[str] = None, b_answers: Optional[str] = None) -> dict:
    a, b = load_rows(a_root), load_rows(b_root)
    common = sorted(set(a) & set(b))
    if not common:
        raise ValueError("no episode is in both arms")
    a_answers = a_answers if a_answers is not None else default_answer_store(a_root)
    b_answers = b_answers if b_answers is not None else default_answer_store(b_root)
    a_store, b_store = load_answer_store(a_answers), load_answer_store(b_answers)
    report = {"without_root": a_root, "with_root": b_root, "common_episodes": len(common),
              "only_without": sorted(set(a) - set(b)), "only_with": sorted(set(b) - set(a)),
              "answer_stores": {"without": a_answers, "with": b_answers},
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
    details = [episode_detail(i, a[i], b[i], a_root, b_root, a_store, b_store) for i in improved + worse]
    report["changed_episodes"] = details
    # Where did the arms part, over every common episode? The mechanism acts at an oracle round; an episode whose
    # orders agreed throughout and still ended differently parted elsewhere (perception, kinematics, a fresh answer).
    # From the answer stores a round also says whether both arms faced the same state there.
    parted_at_round = parted_never = no_round = same_state = state_differed = 0
    sources = set()
    same_state_episodes = []  # type: List[str]
    for i in common:
        a_rounds, b_rounds, source = pick_rounds(a[i], b[i], a_store, b_store, i)
        sources.add(source)
        if not a_rounds and not b_rounds:
            no_round += 1
            continue
        parting = first_parting(a_rounds, b_rounds)
        if parting is None:
            parted_never += 1
            continue
        parted_at_round += 1
        ra, rb = parting
        if ra.get("state") is not None and rb.get("state") is not None:
            if ra["state"] == rb["state"]:
                same_state += 1
                same_state_episodes.append(i)
            else:
                state_differed += 1
    report["parting"] = {"source": ", ".join(sorted(s for s in sources if s != "none")) or "none",
                         "at_an_oracle_round": parted_at_round, "orders_agreed_throughout": parted_never,
                         "no_oracle_round": no_round, "same_state_at_parting": same_state,
                         "state_already_differed_at_parting": state_differed,
                         "same_state_episodes": same_state_episodes}
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
              "- the arms parted at an oracle round in %d episodes; the orders agreed throughout in %d; no oracle round in %d "
              "(rounds from the %s)"
              % (report["parting"]["at_an_oracle_round"], report["parting"]["orders_agreed_throughout"],
                 report["parting"].get("no_oracle_round", 0), report["parting"].get("source", "ledgers"))]
    if report["parting"].get("same_state_at_parting") or report["parting"].get("state_already_differed_at_parting"):
        lines.append("- of the partings, %d were on the SAME state (the same rooms and evidence in both arms: the size and "
                     "coverage lines alone changed the oracle's choice) and %d after the arms had already diverged"
                     % (report["parting"]["same_state_at_parting"], report["parting"]["state_already_differed_at_parting"]))
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
    lines.append("parted at oracle round %s (without: action %s, order %s; with: action %s, order %s)%s" % (
        parted.get("round"), parted.get("without_step"), parted.get("without_order"), parted.get("with_step"), parted.get("with_order"),
        "" if d.get("same_state_at_parting") is None else
        " -- the same rooms and evidence in both arms; the size and coverage lines alone differed" if d["same_state_at_parting"] else
        " -- the rooms or evidence already differed: the arms had parted before this round"))
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
    parser.add_argument("--answers-without", default=None,
                        help="arm-without's answer store (LLM_ANSWERS_RECORD); default: <without root>_answers if it exists")
    parser.add_argument("--answers-with", default=None,
                        help="arm-with's answer store; default: <with root>_answers if it exists")
    args = parser.parse_args(argv)
    levels = load_levels(args.levels) if args.levels else None
    report = build_report(args.without_root, args.with_root, examples=args.examples, levels=levels or None,
                          a_answers=args.answers_without, b_answers=args.answers_with)
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
