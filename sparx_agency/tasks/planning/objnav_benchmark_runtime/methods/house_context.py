"""The HOUSE line of the node oracle's prompt: the big picture the model reasons from.

The node oracle (``core/mapping/topology/search_node_oracle.py``) is asked to
judge every node against the whole home -- which room types have been found
(scanned ones included, so a living room already seen tells it a sofa is
unlikely in the rooms still unidentified), how many rooms are still
unidentified, how many openings are not yet looked into, and how much of the
action budget is spent. This module writes that one line from the scene graph
and the scan ledger. Nothing here judges: the words are facts in the prompt's
vocabulary, and the budget is shown in coarse steps so an unchanged map gives
an unchanged prompt (the oracle reuses a byte-identical prompt's reply).
"""
from __future__ import annotations

from sparx_agency.core.mapping.topology.search_node_oracle import budget_line
from sparx_agency.tasks.planning.objnav_benchmark_runtime.methods.room_scans import (
    FRAGMENT, SCAN_POINT_INSIDE, SEEN_FROM_SCAN, SEEN_THROUGH)

#: The scan ledger's verdicts in the prompt's words (``status=scanned(<how>, <ago>)``).
SCAN_WORDS = {SCAN_POINT_INSIDE: "full rotation", SEEN_FROM_SCAN: "seen from another room's scan",
              SEEN_THROUGH: "looked into, no frontier left", FRAGMENT: "fragment"}


def scan_words(how):
    """The prompt's words for a scan ledger verdict ``how`` (an unknown verdict reads as itself)."""
    return SCAN_WORDS.get(how, str(how))


def house_line(graph, scanned, openings_left, actions_used, budget, frontier_left=None):
    """One line on the home so far, in the prompt's vocabulary.

    ``rooms found: bedroom (never entered), kitchen (scanned), living_room
    (scanned); 2 rooms unidentified (1 scanned); 1 opening not yet looked
    into; unexplored frontier still reachable; actions used about 150 of 500``.

    Args:
        graph: The floor's :class:`~scene_graph.ObservedSceneGraph`.
        scanned: ``{pid: ...}`` -- the rooms the scan ledger has finished
            (only membership is read).
        openings_left: How many openings of the floor are not yet looked into.
        actions_used: The current action.
        budget: The episode's action budget.
        frontier_left: Whether a reachable unexplored boundary is left
            anywhere on the storey; None leaves it unsaid.
    """
    rooms = getattr(getattr(graph, "registry", None), "rooms", {}) or {}
    tracker = getattr(graph, "label_tracker", None)
    labels = getattr(tracker, "labels", {}) or {}
    metadata = getattr(tracker, "metadata", {}) or {}
    last_inside = getattr(graph, "last_inside", {}) or {}
    named = []
    unknown = unknown_scanned = 0
    for pid in rooms:
        label = labels[pid].label if pid in labels else "unknown"
        if not label or label == "unknown":
            unknown += 1
            if pid in scanned:
                unknown_scanned += 1
            continue
        weak = (metadata.get(pid) or {}).get("strength") == "weak"
        status = "scanned" if pid in scanned else ("entered" if last_inside.get(pid) is not None else "never entered")
        named.append("%s%s (%s)" % (label, "?" if weak else "", status))
    parts = ["rooms found: %s" % (", ".join(sorted(named)) if named else "none yet")]
    if unknown:
        parts.append("%d room%s unidentified%s" % (unknown, "" if unknown == 1 else "s",
                                                   " (%d scanned)" % unknown_scanned if unknown_scanned else ""))
    else:
        parts.append("every known room identified")
    parts.append("%d opening%s not yet looked into" % (int(openings_left), "" if int(openings_left) == 1 else "s"))
    if frontier_left is not None:
        parts.append("unexplored frontier still reachable" if frontier_left else "no unexplored frontier left")
    parts.append(budget_line(actions_used, budget))
    return "; ".join(parts)
