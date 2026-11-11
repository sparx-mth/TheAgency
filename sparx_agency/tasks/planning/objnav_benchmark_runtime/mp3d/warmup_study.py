"""The warm-up study (2026-11-11): its episode subset and its comparison table.

The study measures the vantage warm-up (``RPTSettings.warmup.mode == "vantage"``,
``core/planning/object_search/exploration/vantage_warmup.py``) against the ten-turn
spin, under ground-truth perception, on a small fixed subset of the single-storey
set -- episodes the running MP3D ground-truth campaign has already finished, so the
campaign's own rows are the baseline arm and cost the study nothing, and its LLM
answer store is replayed wherever an arm asks a question the campaign asked.

Two commands::

    # the subset: finished campaign episodes in the single-storey set, N of them, stratified by scene
    python -m ...mp3d.warmup_study select --campaign runs/gt_oracle_mp3d_... --n 40 --seed 17 \\
        --out sparx_agency/tasks/planning/objnav_benchmark_runtime/mp3d/sets/warmup_study_v1.json

    # the table: the baseline (campaign root) against each arm, over the subset
    python -m ...mp3d.warmup_study report --set .../warmup_study_v1.json \\
        spin=runs/gt_oracle_mp3d_... p1_b15=runs/wu_p1_b15 p3_b25=runs/wu_p3_b25 [--markdown out.md] [--json out.json]

Beside SR, SPL, SoftSPL, DTG and steps (``single_storey_report`` for the paired
statistics) the table carries the warm-up's own measures, read from every
episode's record and recording:

* ``warm-up actions``: actions the warm-up took (the spin's ten; the vantage
  warm-up's turns and steps together); ``moves``: how many walks it took;
  ``walked (m)``: the path length of those actions.
* ``area @ warm-up end``: the observed map's area (``exploration_metrics``
  ``coverage_curve``) on the last warm-up action; ``area @10/@20/@30``: the
  same at fixed actions, so budgets of different length are read on equal
  terms.
* ``first sight``: the first action a box of the target's class at or above the
  detection confidence was in the frame -- the policy's own
  ``perception.target_first_seen_step`` where the record has it (code after
  2026-11-11), else read off the recording's ``steps.jsonl`` detections -- as
  the median over the episodes that saw it, the share that saw it within 30
  actions, and the share that never did. ``first lock``: the first action the
  target closing took over.
* the local exploration (experiment 2): the actions spent in explore visits
  (the ``frontier`` and ``viewpoint`` phases), the observed area they gained
  per action, the visits, and the viewpoint visit's own counts (frontier
  retired as not worth an action, steps inside).
* with ``--explain``, why every changed episode changed: for each episode
  whose success flipped or whose SPL moved by 0.05, the two arms' outcomes,
  steps, DTG and terminations, their warm-ups, first sights and first locks,
  how many LLM answers each asked anew, and the first action at which their
  poses parted (read off both recordings) with the phase each was in there.

Standard library only beyond the runtime's own modules.
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import random
import statistics
from collections import defaultdict
from typing import Dict, Iterable, List, Optional, Sequence

from sparx_agency.tasks.planning.objnav_benchmark_runtime.mp3d.single_storey_report import (
    _mean, _pct, arm_summary, grouped_sr, paired, parse_arms)
from sparx_agency.tasks.planning.objnav_benchmark_runtime.paired_ab import load_rows

AREA_HORIZONS = (10, 20, 30, 50)
FIRST_SIGHT_WITHIN = 30
EXPLORE_PHASES = ("frontier", "viewpoint")


# -- select ---------------------------------------------------------------------------------------------------
def finished_rows(root: str) -> Dict[str, dict]:
    """Episode rows of the campaign's FINISHED shards / scenes (a ``run.json`` with ``finished_utc``), by id."""
    out = {}
    for path in sorted(glob.glob(os.path.join(root, "**", "episodes.jsonl"), recursive=True)):
        run = os.path.join(os.path.dirname(path), "run.json")
        try:
            finished = json.load(open(run)).get("finished_utc") is not None
        except (OSError, ValueError):
            finished = False
        if not finished:
            continue
        with open(path) as fh:
            for line in fh:
                if line.strip():
                    row = json.loads(line)
                    out[row["episode_id"]] = row
    return out


def select_subset(candidates: Iterable[str], n: int, seed: int) -> List[str]:
    """``n`` episode ids drawn round-robin over their scenes, each scene's order shuffled by ``seed``."""
    by_scene: Dict[str, List[str]] = defaultdict(list)
    for episode_id in sorted(set(candidates)):
        by_scene[episode_id.split("/")[0]].append(episode_id)
    rng = random.Random(seed)
    queues = []
    for scene in sorted(by_scene):
        ids = list(by_scene[scene])
        rng.shuffle(ids)
        queues.append(ids)
    rng.shuffle(queues)
    picked: List[str] = []
    while len(picked) < n and any(queues):
        for queue in queues:
            if queue and len(picked) < n:
                picked.append(queue.pop())
    return sorted(picked)


def select(args) -> int:
    base = json.loads(open(args.set).read()) if args.set else None
    rows = finished_rows(args.campaign)
    candidates = [k for k in rows if base is None or k in set(base)]
    picked = select_subset(candidates, args.n, args.seed)
    meta = {"campaign": os.path.abspath(args.campaign), "set": None if args.set is None else os.path.abspath(args.set),
            "finished_in_campaign": len(rows), "candidates": len(candidates), "n": len(picked), "seed": args.seed,
            "by_scene": dict(sorted(defaultdict(int, {s: sum(1 for k in picked if k.startswith(s + "/"))
                                                     for s in {k.split("/")[0] for k in picked}}).items())),
            "baseline_sr": _mean([float(bool(rows[k]["success"])) for k in picked]) if picked else None}
    with open(args.out, "w") as fh:
        json.dump(picked, fh, indent=0)
        fh.write("\n")
    with open(os.path.splitext(args.out)[0] + ".meta.json", "w") as fh:
        json.dump(meta, fh, indent=2, sort_keys=True)
        fh.write("\n")
    print(json.dumps(meta, indent=2, sort_keys=True))
    return 0


# -- the warm-up's measures ----------------------------------------------------------------------------------
def policy_of(row: dict) -> dict:
    return ((row.get("agent_info") or {}).get("policy") or {})


def coverage_at(curve: Sequence[dict], action: int) -> Optional[float]:
    """The observed area on ``action`` (the last sample at or before it)."""
    value = None
    for sample in curve:
        if int(sample["action"]) <= action:
            value = float(sample["observed_m2"])
        else:
            break
    return value


def warmup_measures(row: dict) -> dict:
    """The warm-up's actions, moves, the area when it ended and at fixed horizons, from the episode's record."""
    policy = policy_of(row)
    warmup = policy.get("warmup") or {}
    vantage = warmup.get("vantage") or {}
    curve = ((policy.get("exploration_metrics") or {}).get("coverage_curve")) or []
    warm = [s for s in curve if s.get("phase") == "warmup"]
    end_area = float(warm[-1]["observed_m2"]) if warm else (float(curve[0]["observed_m2"]) if curve else None)
    # ``first_lock`` is the recording's (the closing's ``started_step`` is its LAST takeover's start, not its first).
    out = {"warmup_actions": int(warmup.get("actions", 0)), "warmup_mode": warmup.get("mode", "spin"),
           "moves": int((vantage.get("stats") or {}).get("moves", 0)),
           "area_end_m2": end_area, "first_seen": (policy.get("perception") or {}).get("target_first_seen_step"),
           "first_lock": None}
    for horizon in AREA_HORIZONS:
        out["area_%d_m2" % horizon] = coverage_at(curve, horizon)
    explore = [s for s in curve if s.get("phase") in EXPLORE_PHASES]
    loop_stats = (policy.get("room_search_loop") or {}).get("stats") or {}
    out["explore_actions"] = len(explore)
    out["explore_gain_m2"] = sum(float(s.get("new_m2", 0.0)) for s in explore)
    out["explore_visits"] = int(loop_stats.get("explore_visits", 0))
    out["viewpoint_retired"] = int(loop_stats.get("viewpoint_retired_frontiers", 0))
    out["viewpoint_steps_inside"] = int(loop_stats.get("viewpoint_steps_inside", 0))
    return out


def recording_dir(roots: str, episode_id: str) -> Optional[str]:
    """The recording directory of ``episode_id`` under the roots, or None."""
    for root in roots.split(","):
        for path in glob.glob(os.path.join(root, "**", "recordings", "*", "episode.json"), recursive=True):
            try:
                if json.load(open(path)).get("episode_id") == episode_id:
                    return os.path.dirname(path)
            except (OSError, ValueError):
                continue
    return None


def _target_names(category: str):
    """A predicate on a detector label: the MP3D target's own acceptance rule when the mapper knows the category."""
    try:
        from sparx_agency.core.planning.objnav.labels.datasets.mp3d import mp3d_label_mapper
        target = mp3d_label_mapper().target_labels(category)
        return target.accepts
    except Exception:
        plain = {category, category.replace("_", " ")}
        return lambda label: str(label) in plain


def recording_measures(directory: str, category: str, confidence: float = 0.3) -> dict:
    """First sight of the target, first lock and the warm-up's walked path, read off ``steps.jsonl``.

    Reads the log once, in order, and stops when every measure is known and the
    warm-up is over (the log of a 500-action episode is ~15 MB).
    """
    accepts = _target_names(category)
    first_seen = first_lock = None
    walked = 0.0
    last_xy = None
    warm_over = False
    path = os.path.join(directory, "steps.jsonl")
    if not os.path.exists(path):
        return {}
    with open(path) as fh:
        for line in fh:
            if not line.strip():
                continue
            step = json.loads(line)
            method = step.get("method") or {}
            pose = step.get("pose") or {}
            phase = (((step.get("command") or {}).get("info") or {}).get("phase")) or method.get("state")
            xy = (float(pose.get("x", 0.0)), float(pose.get("y", 0.0)))
            if phase == "warmup":
                # The walk of the warm-up: the distance between consecutive warm-up poses (a turn moves nothing).
                if last_xy is not None:
                    walked += math.dist(last_xy, xy)
                last_xy = xy
            else:
                last_xy = None
            if first_seen is None:
                for det in method.get("detections") or []:
                    if accepts(str(det.get("label"))) and float(det.get("confidence", 0.0)) >= confidence:
                        first_seen = int(step["step"])
                        break
            if first_lock is None and str(phase or "").startswith("target_"):
                first_lock = int(step["step"])
            search = method.get("search") or {}
            warm = search.get("warmup") or {}
            if warm and int(warm.get("actions", 0)) >= int(warm.get("budget", 0)):
                warm_over = True
            if first_seen is not None and first_lock is not None and int(step["step"]) > 60 and warm_over:
                break
    return {"first_seen_rec": first_seen, "first_lock_rec": first_lock, "walked_m": walked}


def episode_measures(rows: Dict[str, dict], roots: str, ids: Sequence[str], recordings: bool = True) -> Dict[str, dict]:
    out = {}
    for episode_id in ids:
        row = rows[episode_id]
        m = warmup_measures(row)
        if recordings:
            directory = recording_dir(roots, episode_id)
            if directory is not None:
                rec = recording_measures(directory, row.get("target_category", ""))
                if m.get("first_seen") is None:
                    m["first_seen"] = rec.get("first_seen_rec")
                if m.get("first_lock") is None:
                    m["first_lock"] = rec.get("first_lock_rec")
                m["walked_m"] = rec.get("walked_m")
        out[episode_id] = m
    return out


def summarise(measures: Dict[str, dict], ids: Sequence[str]) -> dict:
    picked = [measures[k] for k in ids]
    seen = [m["first_seen"] for m in picked if m.get("first_seen") is not None]
    locks = [m["first_lock"] for m in picked if m.get("first_lock") is not None]
    walked = [m["walked_m"] for m in picked if m.get("walked_m") is not None]
    out = {"warmup_actions": _mean([m["warmup_actions"] for m in picked]),
           "moves": _mean([m["moves"] for m in picked]),
           "walked_m": _mean(walked) if walked else float("nan"),
           "area_end_m2": _mean([m["area_end_m2"] for m in picked if m.get("area_end_m2") is not None]),
           "first_seen_median": statistics.median(seen) if seen else float("nan"),
           "first_seen_within": sum(1 for s in seen if s <= FIRST_SIGHT_WITHIN) / len(picked) if picked else float("nan"),
           "never_seen": 1.0 - len(seen) / len(picked) if picked else float("nan"),
           "first_lock_median": statistics.median(locks) if locks else float("nan"),
           "never_locked": 1.0 - len(locks) / len(picked) if picked else float("nan")}
    for horizon in AREA_HORIZONS:
        values = [m["area_%d_m2" % horizon] for m in picked if m.get("area_%d_m2" % horizon) is not None]
        out["area_%d_m2" % horizon] = _mean(values) if values else float("nan")
    actions = sum(m.get("explore_actions", 0) for m in picked)
    out["explore_actions"] = _mean([m.get("explore_actions", 0) for m in picked])
    out["explore_gain_per_action"] = (sum(m.get("explore_gain_m2", 0.0) for m in picked) / actions) if actions else float("nan")
    out["explore_visits"] = _mean([m.get("explore_visits", 0) for m in picked])
    out["viewpoint_retired"] = _mean([m.get("viewpoint_retired", 0) for m in picked])
    out["viewpoint_steps_inside"] = _mean([m.get("viewpoint_steps_inside", 0) for m in picked])
    return out


# -- report ----------------------------------------------------------------------------------------------------
def report(arms: Sequence[tuple], ids: Optional[Sequence[str]], recordings: bool = True, resamples: int = 10000,
           explain_changes: bool = False) -> dict:
    loaded = [(name, roots, load_rows(roots)) for name, roots in arms]
    common = set(loaded[0][2]) if ids is None else set(ids)
    for _, _, rows in loaded:
        common &= set(rows)
    ids = sorted(common)
    if not ids:
        raise ValueError("no episode of the set was run by every arm named")
    base_name, base_roots, base = loaded[0]
    out = {"episodes": len(ids), "arms": [name for name, _, _ in loaded], "per_arm": {}, "warmup": {}, "paired": {},
           "by_scene": {}, "sizes": {name: len(set(rows) & set(ids)) for name, _, rows in loaded}, "per_episode": {},
           "explanations": {}}
    measures_by_arm = {}
    for name, roots, rows in loaded:
        out["per_arm"][name] = arm_summary(rows, ids)
        measures = episode_measures(rows, roots, ids, recordings=recordings)
        measures_by_arm[name] = measures
        out["warmup"][name] = summarise(measures, ids)
        out["by_scene"][name] = grouped_sr(rows, ids, lambda r: r["episode_id"].split("/")[0])
        out["per_episode"][name] = {k: dict(measures[k], success=bool(rows[k]["success"]), spl=float(rows[k]["spl"]),
                                            steps=int(rows[k]["steps"])) for k in ids}
        if name != base_name:
            p = paired(base, rows, ids, resamples)
            dspl = {k: float(rows[k]["spl"]) - float(base[k]["spl"]) for k in ids}
            p["spl_better"] = sorted(k for k, v in dspl.items() if v > 0.01)
            p["spl_worse"] = sorted(k for k, v in dspl.items() if v < -0.01)
            out["paired"][name] = p
            if explain_changes:
                changed = sorted({k for k in ids if bool(rows[k]["success"]) != bool(base[k]["success"])
                                  or abs(dspl[k]) >= SPL_CHANGE})
                out["explanations"][name] = [
                    explain(base[k], rows[k], measures_by_arm[base_name][k], measures[k],
                            recording_dir(base_roots, k) if recordings else None,
                            recording_dir(roots, k) if recordings else None) for k in changed]
    return out


SPL_CHANGE = 0.05


def divergence(base_dir: Optional[str], arm_dir: Optional[str]) -> Optional[dict]:
    """The first action at which two recordings' poses part, with each arm's phase there, or None."""
    if not base_dir or not arm_dir:
        return None
    a, b = os.path.join(base_dir, "steps.jsonl"), os.path.join(arm_dir, "steps.jsonl")
    if not (os.path.exists(a) and os.path.exists(b)):
        return None
    with open(a) as fa, open(b) as fb:
        for la, lb in zip(fa, fb):
            if not la.strip() or not lb.strip():
                continue
            sa, sb = json.loads(la), json.loads(lb)
            pa, pb = sa.get("pose") or {}, sb.get("pose") or {}
            if any(abs(float(pa.get(k, 0.0)) - float(pb.get(k, 0.0))) > 1e-3 for k in ("x", "y", "yaw", "camera_pitch")):
                phase = lambda s: (((s.get("command") or {}).get("info") or {}).get("phase")) or (s.get("method") or {}).get("state")
                return {"step": int(sa["step"]), "phase_base": phase(sa), "phase_arm": phase(sb)}
    return None


def explain(base_row: dict, arm_row: dict, base_m: dict, arm_m: dict, base_dir: Optional[str], arm_dir: Optional[str]) -> dict:
    """The facts behind one changed episode, side by side."""
    def llm(row):
        return (policy_of(row).get("llm_answers") or {})
    def closing(row):
        c = policy_of(row).get("target_closing") or {}
        return {"failure": c.get("failure"), "last_release": c.get("last_release"), "releases": c.get("releases")}
    return {"episode": base_row["episode_id"], "target": base_row.get("target_category"),
            "success": [bool(base_row["success"]), bool(arm_row["success"])],
            "spl": [round(float(base_row["spl"]), 3), round(float(arm_row["spl"]), 3)],
            "steps": [int(base_row["steps"]), int(arm_row["steps"])],
            "dtg_m": [round(float(base_row["distance_to_goal_m"]), 2), round(float(arm_row["distance_to_goal_m"]), 2)],
            "termination": [base_row.get("termination"), arm_row.get("termination")],
            "warmup_actions": [base_m.get("warmup_actions"), arm_m.get("warmup_actions")],
            "moves": [base_m.get("moves"), arm_m.get("moves")],
            "walked_m": [base_m.get("walked_m"), arm_m.get("walked_m")],
            "first_seen": [base_m.get("first_seen"), arm_m.get("first_seen")],
            "first_lock": [base_m.get("first_lock"), arm_m.get("first_lock")],
            "llm_asked": [llm(base_row).get("asked"), llm(arm_row).get("asked")],
            "llm_replayed": [llm(base_row).get("replayed"), llm(arm_row).get("replayed")],
            "closing": [closing(base_row), closing(arm_row)],
            "diverged": divergence(base_dir, arm_dir)}


def render_explanation(name: str, base_name: str, item: dict) -> str:
    def pair(key, fmt="%s"):
        a, b = item[key]
        return "%s -> %s" % (fmt % a if a is not None else "--", fmt % b if b is not None else "--")
    outcome = ("%s -> %s" % ("success" if item["success"][0] else "failure", "success" if item["success"][1] else "failure"))
    parts = ["%s (%s): %s; SPL %s; steps %s; DTG %s m; ended %s" % (
        item["episode"], item["target"], outcome, pair("spl", "%.3f"), pair("steps"), pair("dtg_m", "%.2f"), pair("termination"))]
    parts.append("warm-up actions %s, walks %s, walked %s m" % (pair("warmup_actions"), pair("moves"),
                                                               pair("walked_m", "%.2f") if None not in item["walked_m"] else pair("walked_m")))
    parts.append("target first in frame at action %s, first takeover at %s" % (pair("first_seen"), pair("first_lock")))
    parts.append("LLM answers asked %s, of which replayed %s" % (pair("llm_asked"), pair("llm_replayed")))
    d = item["diverged"]
    if d is not None:
        parts.append("the two runs part at action %d (%s was in %s, %s in %s)" % (
            d["step"], base_name, d["phase_base"], name, d["phase_arm"]))
    for label, c in zip((base_name, name), item["closing"]):
        if c.get("failure") or c.get("last_release"):
            parts.append("%s's closing: %s" % (label, c.get("failure") or ("last release %s" % (c.get("last_release"),))))
    return "- " + "; ".join(parts)


def _num(value, fmt="%.1f") -> str:
    return "--" if value is None or (isinstance(value, float) and math.isnan(value)) else fmt % value


def render(data: dict) -> str:
    names = data["arms"]
    lines = ["Episodes in every arm: %d" % data["episodes"], ""]
    lines.append("| arm | n | SR | SPL | SoftSPL | DTG (m) | steps | dSR vs %s [95 %%] | McNemar p | dSPL [95 %%] | flips +/- |"
                 % names[0])
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|")
    for name in names:
        s = data["per_arm"][name]
        p = data["paired"].get(name)
        pair = ("-- | -- | -- | --" if p is None else "%+.1f pts [%+.1f, %+.1f] | %.3f | %+.3f [%+.3f, %+.3f] | %d / %d" % (
            100 * p["sr_diff"], 100 * p["sr_ci"][0], 100 * p["sr_ci"][1], p["mcnemar_p"], p["spl_diff"],
            p["spl_ci"][0], p["spl_ci"][1], len(p["to_success"]), len(p["to_failure"])))
        lines.append("| %s | %d | %s | %s | %s | %.2f | %.0f | %s |" % (
            name, s["episodes"], _pct(s["sr"]), _pct(s["spl"]), _pct(s["soft_spl"]), s["dtg_m"], s["steps"], pair))
    lines += ["", "The warm-up itself:", "",
              "| arm | warm-up actions | moves | walked (m) | area @ warm-up end (m2) | "
              + " | ".join("area @%d" % h for h in AREA_HORIZONS)
              + " | first sight: median action | seen within %d | never seen | first lock: median | never locked |" % FIRST_SIGHT_WITHIN,
              "|---|---|---|---|---|" + "---|" * len(AREA_HORIZONS) + "---|---|---|---|---|"]
    for name in names:
        w = data["warmup"][name]
        lines.append("| %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s |" % (
            name, _num(w["warmup_actions"]), _num(w["moves"], "%.2f"), _num(w["walked_m"], "%.2f"), _num(w["area_end_m2"]),
            " | ".join(_num(w["area_%d_m2" % h]) for h in AREA_HORIZONS),
            _num(w["first_seen_median"], "%.0f"), _pct(w["first_seen_within"]), _pct(w["never_seen"]),
            _num(w["first_lock_median"], "%.0f"), _pct(w["never_locked"])))
    lines += ["", "The local exploration (the explore visits):", "",
              "| arm | explore actions / episode | area gained per explore action (m2) | explore visits / episode "
              "| frontier retired as not worth it / episode | steps inside / episode |",
              "|---|---|---|---|---|---|"]
    for name in names:
        w = data["warmup"][name]
        lines.append("| %s | %s | %s | %s | %s | %s |" % (
            name, _num(w["explore_actions"]), _num(w["explore_gain_per_action"], "%.3f"), _num(w["explore_visits"], "%.2f"),
            _num(w["viewpoint_retired"], "%.2f"), _num(w["viewpoint_steps_inside"], "%.2f")))
    lines += ["", "Per-scene SR:", "", "| scene | n | " + " | ".join(names) + " |", "|---|---|" + "---|" * len(names)]
    for scene in sorted(data["by_scene"][names[0]]):
        cells = ["%d (%s)" % (data["by_scene"][name][scene][0], _pct(data["by_scene"][name][scene][0] / data["by_scene"][name][scene][1]))
                 for name in names]
        lines.append("| %s | %d | %s |" % (scene, data["by_scene"][names[0]][scene][1], " | ".join(cells)))
    for name in names[1:]:
        p = data["paired"][name]
        lines += ["", "%s against %s: %d to success (%s); %d to failure (%s); SPL better in %d episodes, worse in %d "
                      "(by more than 0.01)" % (
                          name, names[0], len(p["to_success"]), ", ".join(p["to_success"]) or "none",
                          len(p["to_failure"]), ", ".join(p["to_failure"]) or "none",
                          len(p.get("spl_better", [])), len(p.get("spl_worse", [])))]
        items = data.get("explanations", {}).get(name)
        if items is not None:
            lines += ["", "Why the changed episodes changed (%s against %s; a flip of success, or SPL moved by %.2f):" % (
                name, names[0], SPL_CHANGE)]
            lines += [render_explanation(name, names[0], item) for item in items] or ["- none"]
    return "\n".join(lines) + "\n"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    sel = sub.add_parser("select", help="draw the study's episode subset from a campaign's finished episodes")
    sel.add_argument("--campaign", required=True, help="the campaign root whose finished episodes are the population")
    sel.add_argument("--set", default=None, help="restrict to the ids of this JSON list (the single-storey set)")
    sel.add_argument("--n", type=int, default=40)
    sel.add_argument("--seed", type=int, default=17)
    sel.add_argument("--out", required=True)
    rep = sub.add_parser("report", help="the comparison table over the subset: the first arm is the baseline")
    rep.add_argument("arms", nargs="+", help="name=root[,root] (the first is the baseline)")
    rep.add_argument("--set", default=None, help="the subset's JSON list; every arm is read on these episodes")
    rep.add_argument("--no-recordings", action="store_true", help="skip the steps.jsonl reads (first sight, walked)")
    rep.add_argument("--explain", action="store_true", help="say why every changed episode changed (reads both recordings)")
    rep.add_argument("--markdown", default=None)
    rep.add_argument("--json", dest="json_out", default=None)
    args = parser.parse_args(argv)
    if args.command == "select":
        return select(args)
    ids = json.loads(open(args.set).read()) if args.set else None
    data = report(parse_arms(args.arms), ids, recordings=not args.no_recordings, explain_changes=args.explain)
    text = render(data)
    print(text)
    if args.markdown:
        with open(args.markdown, "w") as fh:
            fh.write(text)
    if args.json_out:
        with open(args.json_out, "w") as fh:
            json.dump(data, fh, indent=2, sort_keys=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
