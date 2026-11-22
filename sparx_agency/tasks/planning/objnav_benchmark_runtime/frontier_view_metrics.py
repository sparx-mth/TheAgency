"""What a frontier-view replay says: how many nodes had a picture, how many questions that costs, and what the pictures showed.

``frontier_view_replay.replay`` walks a recording and scores its frontier
nodes; ``frontier_view_hindsight`` measures each scoring against the rest of
the recording. This module turns both into the numbers a decision is made
on, per variant and per episode, and pools episodes.

**Coverage.** Of the nodes an episode had: how many were scored at all, how
many on the action they appeared (a picture already existed), how many only
later, and how many never -- with the selector's own reason, and with how
long each lived (a boundary that flickered for two actions and had no
picture is noise; one that stood for forty is a hole). Beside it the share
of node-actions with a score in hand: at a random action, what part of the
nodes an order would be planned over have a number.

**Cost.** Every scoring is one model call under the score-once rule, so the
count of scorings IS the count of calls; how many distinct pictures they
were asked with says how much a batch could share. A call spent on a node
that was gone two actions later bought nothing an order could use.

**Fidelity.** On how many actions the frontier inventory the replay rebuilt
was the one the recording lists (``frontier_view_replay.same_as_recorded``):
the nodes are only as true as that.

Pooling adds counts and joins the per-scoring values BEFORE any median is
taken: the median of six episodes' medians is not the median of their
scorings.

Python 3.9 syntax, numpy.
"""
from __future__ import annotations

from collections import Counter
from typing import Dict, Iterable, Optional, Sequence

import numpy as np

from sparx_agency.core.planning.object_search.frontier_views.score_ledger import BLOCKED

#: A node that lived at least this many actions and was never scored is a hole, not a flicker.
LONG_LIVED_STEPS = 5
#: A scoring "looked past the boundary" when its frames see through at least this share of the node's cells.
THROUGH_SHARE = 0.1
#: A call was spent on a passing node when the node was gone within this many actions of its scoring.
GONE_SOON_STEPS = 2
#: In hindsight a node had "(nearly) nothing" to show when no frame saw this share of what was mapped behind it.
NOTHING_SHARE = 0.05


def _median(values: Iterable[float]) -> Optional[float]:
    values = [float(v) for v in values]
    return round(float(np.median(values)), 3) if values else None


def _mean(values: Iterable[float]) -> Optional[float]:
    values = [float(v) for v in values]
    return round(float(np.mean(values)), 3) if values else None


def _rounded(record: Dict) -> Dict:
    return {key: (round(value, 3) if isinstance(value, float) else value) for key, value in record.items()}


def _event_row(event) -> Dict:
    """One scoring, JSON-ready."""
    views = event.views
    row = {"step": event.step, "node": event.node.node_id, "cells": event.node.size_cells, "parts": event.node.parts,
           "first_step": event.first_step, "wait_steps": event.step - event.first_step,
           "score": round(views.score, 3), "seen_fraction": round(views.seen_fraction, 3),
           "through_fraction": round(views.through_fraction, 3),
           "views": [{"name": view.name, "step": view.step, "age_steps": event.step - view.step,
                      "gain": round(view.gain, 3), "seen_fraction": round(view.seen_fraction, 3),
                      "through_fraction": round(view.through_fraction, 3), "range_m": round(view.range_m, 2),
                      "window": [round(v, 1) for v in view.window]} for view in views.views]}
    if event.hindsight is not None:
        row["hindsight"] = _rounded({key: value for key, value in event.hindsight.items() if key != "cells"})
    if event.later is not None:
        row["on_its_last_action"] = _rounded(event.later)
    return row


def variant_raw(replayed, name: str) -> Dict:
    """The per-node and per-scoring values of one variant, as plain numbers and lists: what :func:`variant_summary` and :func:`pooled` reduce."""
    ledger, events = replayed.ledgers[name], replayed.events[name]
    last = replayed.last_views[name]
    lives = {record.node_id: record.last_step - record.first_step + 1 for record in ledger.nodes.values()}
    unscored = [record for record in ledger.nodes.values() if record.scored_step is None]
    long_unscored = [record for record in unscored if lives[record.node_id] >= LONG_LIVED_STEPS]
    scored_at = {record.node_id: record.scored_step for record in ledger.nodes.values() if record.scored_step is not None}
    presences = [(step["step"], node_id) for step in replayed.steps for node_id in step["nodes"]]
    hindsights = [event.hindsight for event in events if event.hindsight is not None and "shown" in event.hindsight]
    waited = [event for event in events if event.later is not None and event.later["usable"]]
    return {
        "nodes": len(ledger.nodes), "steps": len(replayed.steps), "calls": len(events),
        "immediate": sum(1 for event in events if event.step == event.first_step),
        "waits": [event.step - event.first_step for event in events if event.step > event.first_step],
        "unscored": len(unscored), "unscored_long": len(long_unscored),
        "unscored_reasons": dict(Counter(last[record.node_id].reason if record.node_id in last else "never offered a frame"
                                         for record in long_unscored)),
        "blocked": sum(1 for record in ledger.nodes.values() if record.state == BLOCKED),
        "presences": len(presences),
        "presences_scored": sum(1 for step, node_id in presences if scored_at.get(node_id, step + 1) <= step),
        "gone_soon": sum(1 for event in events
                         if ledger.nodes[event.node.node_id].last_step - event.step <= GONE_SOON_STEPS),
        "pictures": len({event.views.names[0] for event in events}),
        "frames_per_call": [len(event.views.views) for event in events],
        "score": [event.views.score for event in events],
        "seen": [event.views.seen_fraction for event in events],
        "through": [event.views.through_fraction for event in events],
        "range_m": [event.views.views[0].range_m for event in events],
        "window_px": [event.views.views[0].window[2] - event.views.views[0].window[0] for event in events],
        "age_steps": [event.step - event.views.views[0].step for event in events],
        "later": {"score_then": [event.later["score"] for event in waited],
                  "score_at_scoring": [event.views.score for event in waited],
                  "range_then": [event.later["range_m"] for event in waited],
                  "range_at_scoring": [event.views.views[0].range_m for event in waited],
                  "doubled": [1.0 if event.later["score"] >= 2.0 * event.views.score else 0.0 for event in waited]},
        "later_lost": sum(1 for event in events if event.later is not None and not event.later["usable"]),
        "hindsight": {key: [h[key] for h in hindsights] for key in ("shown", "available", "best_frame", "current_frame")},
        "hindsight_skipped": sum(1 for event in events if event.hindsight is not None and "shown" not in event.hindsight),
    }


def variant_summary(raw: Dict) -> Dict:
    """The numbers of one variant, from :func:`variant_raw` (of one episode, or pooled)."""
    calls, nodes, later, hind = raw["calls"], raw["nodes"], raw["later"], raw["hindsight"]
    shown, available = sum(hind["shown"]), sum(hind["available"])
    return {
        "nodes": nodes, "calls": calls, "calls_per_100_steps": round(100.0 * calls / max(1, raw["steps"]), 1),
        "scored_share": round(calls / nodes, 3) if nodes else None,
        "scored_on_appearance": raw["immediate"], "scored_later": len(raw["waits"]),
        "median_wait_steps": _median(raw["waits"]),
        "never_scored": raw["unscored"], "never_scored_long_lived": raw["unscored_long"],
        "never_scored_reasons": raw["unscored_reasons"], "blocked": raw["blocked"],
        "node_actions_with_a_score": round(raw["presences_scored"] / raw["presences"], 3) if raw["presences"] else None,
        "calls_on_nodes_gone_soon": raw["gone_soon"], "distinct_pictures": raw["pictures"],
        "frames_per_call": {str(n): raw["frames_per_call"].count(n) for n in sorted(set(raw["frames_per_call"]))},
        "median_score": _median(raw["score"]), "median_seen_fraction": _median(raw["seen"]),
        "looked_past_share": _mean(1.0 if value >= THROUGH_SHARE else 0.0 for value in raw["through"]),
        "median_range_m": _median(raw["range_m"]), "median_window_px": _median(raw["window_px"]),
        "median_age_steps": _median(raw["age_steps"]),
        "on_its_last_action": {"scorings": len(later["score_then"]), "no_view_by_then": raw["later_lost"],
                               "median_score": _median(later["score_then"]),
                               "median_score_at_scoring": _median(later["score_at_scoring"]),
                               "median_range_m": _median(later["range_then"]),
                               "median_range_m_at_scoring": _median(later["range_at_scoring"]),
                               "score_doubled_share": _mean(later["doubled"])},
        "hindsight": {"scorings": len(hind["shown"]), "too_little_mapped_later": raw["hindsight_skipped"],
                      "shown": _mean(hind["shown"]), "available": _mean(hind["available"]),
                      "best_frame": _mean(hind["best_frame"]), "current_frame": _mean(hind["current_frame"]),
                      "shown_of_available": round(shown / available, 3) if available > 0 else None,
                      "nothing_available_share": _mean(1.0 if value < NOTHING_SHARE else 0.0
                                                       for value in hind["available"])},
    }


def summarise(replayed) -> Dict:
    """One replayed episode as a JSON-ready record: the frames, the nodes, every variant's numbers, the first variant's scorings."""
    counts = [len(step["nodes"]) for step in replayed.steps]
    compared = [step["as_recorded"] for step in replayed.steps if step.get("as_recorded") is not None]
    raw = {variant.name: variant_raw(replayed, variant.name) for variant in replayed.variants}
    first = replayed.variants[0]
    return {
        "recording": str(replayed.directory), "scene": replayed.episode.get("scene"),
        "episode": replayed.episode.get("episode_id"), "target": replayed.episode.get("target"),
        "steps": len(replayed.steps),
        "frames": {"count": len(replayed.frames.records), "source": replayed.frames.source},
        "inventory": {"compared": len(compared), "as_recorded": sum(1 for same in compared if same)},
        "nodes": {"ids": len(replayed.ledgers[first.name].nodes), "per_action_mean": _mean(counts),
                  "per_action_max": max(counts) if counts else 0,
                  "clusters_per_action_mean": _mean(step["clusters"] for step in replayed.steps),
                  "min_cells": replayed.node_settings.min_cells, "merge_m": replayed.node_settings.merge_m},
        "variants": {variant.name: dict(variant_summary(raw[variant.name]), about=variant.about)
                     for variant in replayed.variants},
        "raw": raw,
        "scorings": [_event_row(event) for event in replayed.events[first.name]],
    }


def pooled(results: Sequence[Dict]) -> Dict:
    """Several episodes' results as one: counts added, per-scoring values pooled before any median is taken."""
    variants = {}
    for name in (list(results[0]["variants"]) if results else []):
        total = None
        for result in results:
            total = _added(total, result["raw"][name])
        summary = variant_summary(total)
        summary["about"] = results[0]["variants"][name].get("about", "")
        # A frame's name is its step: the same name in two episodes is two pictures.
        summary["distinct_pictures"] = sum(result["variants"][name]["distinct_pictures"] for result in results)
        variants[name] = summary
    steps = sum(result["steps"] for result in results)

    def weighted(key):
        return round(sum((result["nodes"][key] or 0.0) * result["steps"] for result in results) / max(1, steps), 3)

    return {"recording": "%d episodes pooled" % len(results), "scene": None, "episode": None, "target": None,
            "steps": steps, "frames": {"count": sum(result["frames"]["count"] for result in results),
                                       "source": ", ".join(sorted({result["frames"]["source"] for result in results}))},
            "inventory": {key: sum(result["inventory"][key] for result in results) for key in ("compared", "as_recorded")},
            "nodes": {"ids": sum(result["nodes"]["ids"] for result in results),
                      "per_action_mean": weighted("per_action_mean"),
                      "per_action_max": max((result["nodes"]["per_action_max"] for result in results), default=0),
                      "clusters_per_action_mean": weighted("clusters_per_action_mean"),
                      "min_cells": results[0]["nodes"]["min_cells"] if results else None,
                      "merge_m": results[0]["nodes"]["merge_m"] if results else None},
            "variants": variants}


def _added(total: Optional[Dict], raw: Dict) -> Dict:
    """``total`` with ``raw`` added: numbers summed, lists joined, dictionaries merged the same way."""
    if total is None:
        return _added({}, raw)
    out = dict(total)
    for key, value in raw.items():
        if isinstance(value, dict):
            out[key] = _added(total.get(key) or {}, value)
        elif isinstance(value, list):
            out[key] = list(total.get(key) or []) + list(value)
        else:
            out[key] = (total.get(key) or 0) + value
    return out


def _percent(value: Optional[float]) -> str:
    return "n/a" if value is None else "%.0f%%" % (100.0 * value)


def _number(value: Optional[float], form: str = "%.1f") -> str:
    return "n/a" if value is None else form % value


def _pair(record: Dict, key: str, form: str) -> str:
    return "%s / %s" % (_number(record[key], form), _number(record[key + "_at_scoring"], form))


#: The report's table: a label and what to print under each variant; a row with no function is a heading.
_ROWS = (
    ("nodes scored (= model calls)", lambda s: "%d of %d" % (s["calls"], s["nodes"])),
    ("  share of nodes", lambda s: _percent(s["scored_share"])),
    ("  on the action they appeared", lambda s: "%d" % s["scored_on_appearance"]),
    ("  later (median wait, actions)", lambda s: "%d (%s)" % (s["scored_later"], _number(s["median_wait_steps"], "%.0f"))),
    ("never scored", lambda s: "%d" % s["never_scored"]),
    ("  of them lived %d+ actions" % LONG_LIVED_STEPS, lambda s: "%d" % s["never_scored_long_lived"]),
    ("node-actions with a score in hand", lambda s: _percent(s["node_actions_with_a_score"])),
    ("calls per 100 actions", lambda s: _number(s["calls_per_100_steps"])),
    ("calls on nodes gone %d actions later" % GONE_SOON_STEPS, lambda s: "%d" % s["calls_on_nodes_gone_soon"]),
    ("distinct pictures asked with", lambda s: "%d" % s["distinct_pictures"]),
    ("frames per call 1 / 2 / 3", lambda s: " / ".join(str(s["frames_per_call"].get(n, 0)) for n in ("1", "2", "3"))),
    ("median share of the node seen", lambda s: _percent(s["median_seen_fraction"])),
    ("calls that look PAST the boundary", lambda s: _percent(s["looked_past_share"])),
    ("median range of the first frame", lambda s: _number(s["median_range_m"], "%.1f m")),
    ("median window width", lambda s: _number(s["median_window_px"], "%.0f px")),
    ("median age of the first frame", lambda s: _number(s["median_age_steps"], "%.0f act.")),
    ("still there after a visit (blocked)", lambda s: "%d" % s["blocked"]),
    ("had the node been scored on its LAST action instead (every frame until then on offer):", None),
    ("  median score then / at scoring", lambda s: _pair(s["on_its_last_action"], "median_score", "%.2f")),
    ("  median range then / at scoring, m", lambda s: _pair(s["on_its_last_action"], "median_range_m", "%.1f")),
    ("  nodes whose score at least doubles", lambda s: _percent(s["on_its_last_action"]["score_doubled_share"])),
    ("in hindsight -- of the space behind the node that was mapped later, the share in the pictures:", None),
    ("  scorings measured", lambda s: "%d" % s["hindsight"]["scorings"]),
    ("  shown by the chosen frames", lambda s: _percent(s["hindsight"]["shown"])),
    ("  in any frame logged so far", lambda s: _percent(s["hindsight"]["available"])),
    ("  chosen / any", lambda s: _percent(s["hindsight"]["shown_of_available"])),
    ("  nodes with (nearly) none in any", lambda s: _percent(s["hindsight"]["nothing_available_share"])),
)


def report(result: Dict) -> str:
    """One result (an episode's, or the pooled one) as text: the header, the table of variants, the reasons of the misses."""
    nodes, inventory, variants = result["nodes"], result["inventory"], result["variants"]
    title = result["recording"] if result["episode"] is None else "%s  (%s, target %s)" % (
        result["recording"], result["episode"], result["target"])
    lines = ["", "== %s ==" % title,
             "%d actions, %d frames (%s)" % (result["steps"], result["frames"]["count"], result["frames"]["source"]),
             "frontier inventory rebuilt = the recorded one on %d of the %d actions the recording built its own"
             % (inventory["as_recorded"], inventory["compared"]),
             "frontier nodes: %d ids, %.1f an action (max %d) from %.1f clusters an action; at least %d cells, merged within %.1f m"
             % (nodes["ids"], nodes["per_action_mean"] or 0.0, nodes["per_action_max"],
                nodes["clusters_per_action_mean"] or 0.0, nodes["min_cells"], nodes["merge_m"]),
             "", "%-36s" % "" + "".join("%15s" % name for name in variants)]
    for label, value in _ROWS:
        lines.append(label if value is None else "%-36s" % label + "".join("%15s" % value(s) for s in variants.values()))
    for name, summary in variants.items():
        if summary["never_scored_reasons"]:
            lines.append("never scored, long-lived, %s: %s" % (name, "; ".join(
                "%d %s" % (count, reason)
                for reason, count in sorted(summary["never_scored_reasons"].items(), key=lambda item: (-item[1], item[0])))))
    return "\n".join(lines)
