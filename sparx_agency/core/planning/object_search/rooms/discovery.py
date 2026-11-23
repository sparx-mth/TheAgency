"""Shared startup, semantic interrupts and local-task resumption for ObjectNav."""
from __future__ import annotations

import numpy as np

from sparx_agency.core.planning.object_search.exploration.doorway_candidates import threshold_cells
from sparx_agency.core.planning.object_search.rooms.room_search_loop import RoomReasoningUnavailable
from sparx_agency.core.planning.object_search.exploration.warmup import rotation_command

SUSPENDED_PHASES = ("warmup", "doorway_peek", "room_return", "discovery_fallback", "glance")


def discover(p, obs, world, cost):
    """One synchronous discovery decision; None hands control to the room task.

    The warm-up is one full rotation in place -- the look-around the whole
    method is built on, taken where the agent stands before any room is
    chosen: the map gets its first rooms, the detector its first objects,
    and when it ends the agent's position goes into the scan ledger, so the
    room it stands in is finished before the first room is ever chosen. It
    runs again on every storey the agent first sets foot on. Its length is
    the camera's: as many turns as the field of view needs to have seen
    every bearing once (``policy.warmup_turns``, ten at the benchmark's
    30 degrees and 79-degree camera), not the twelve that bring the body
    back to its start heading.
    """
    if p.hierarchy is not None:
        p.hierarchy.sync_floor(obs, world)
    command = p.peek.plan(obs, world)
    if command is not None:
        p._action_owner = "doorway_peek"
        return command
    command = warmup_rotation(p, obs, world)
    if command is not None:
        return command
    return reconsider_rooms(p, obs, world, cost)


def warmup_rotation(p, obs, world):
    """Complete the in-place scan of a newly entered floor before selecting a destination."""
    command = rotation_command(p, obs)
    if command is not None:
        return command
    if getattr(p, "_warmup_pending", False):
        p._warmup_pending = False
        ledger = getattr(p, "scans", None)
        if ledger is not None:
            pid = p.graph.room_at(world, (obs.pose.x, obs.pose.y))
            ledger.record(obs, p.graph.registry.rooms.get(pid) if pid is not None else None, source="warmup")
    return None


def reconsider_rooms(p, obs, world, cost):
    """Text-room policy only: reconsider the offer after a semantic change."""
    signature = semantic_signature(p)
    previous = p._semantic_signature
    loop = getattr(p, "loop", None)
    if loop is not None and getattr(loop, "_last_reason_step", -1) >= getattr(p, "_semantic_signature_step", -1):
        # The loop valued the nodes since this was last read: what changed (its own exclusions, the
        # rooms it finished) is its doing, not a clue to interrupt the walk for.
        previous = None
    p._semantic_signature_step = int(obs.step)
    event = p.peek.pending_reason or (previous is not None and signature != previous)
    # A room's type changing is a semantic clue, acted on at once; rooms appearing on or leaving the offer
    # is geometry -- the watershed renumbering at a boundary -- which the loop debounces.
    semantic = bool(p.peek.pending_reason) or (previous is not None and semantic_change(previous, signature))
    if event and p.graph.registry.rooms:
        try:
            if p.hierarchy is None:
                p.loop.reconsider(obs, world, semantic=semantic)
            else:
                p.hierarchy.reconsider(obs, world)
        except RoomReasoningUnavailable:
            # Keep the interrupted task's budget intact while a service is away.
            p._action_owner = "discovery_fallback"
            return p.fallback.plan(obs, world, cost, reason="semantic reassessment unavailable")
        p.peek.pending_reason = False
        # The loop's exclusions just moved rooms off the offer: read the signature after them, or the
        # next action compares against the offer as it stood before and reads its own doing as a clue.
        signature = semantic_signature(p)
    p._semantic_signature = signature
    return resume_after_peek(p, obs, world)


def semantic_signature(p):
    """What a change of warrants a fresh decision mid-walk: the storey, the rooms on offer, their types.

    Until 2026-11-09 the signature held every room's full object-class set,
    so a vase first credited to the kitchen, or a room renumbered by the
    watershed, interrupted the walk and bought an oracle call: QUCTc6BB5sX/
    000069 (MP3D, ground truth) asked 27 times in 230 actions in a 5 m patch,
    often on consecutive actions, over the same two or three nodes. A clue is
    a room appearing on or leaving the offer, or a room's type changing; the
    target's own class is the closing's business. (Until 2026-11-15 a "home
    object" of the target -- a sink for a toilet, from a hand-written table
    -- appearing on the storey was a third clue; it went with
    ``room_priors.py``. A new object changes the oracle's reading only
    through the room type the classifier gives it, or at the next loop
    point, where the node list carries it.)

    Returns ``(floor, offered, types)``: the offered room ids and the type of
    each offered room by id -- a dict, so :func:`semantic_change` can compare
    the rooms common to two readings and is not fooled by the ids the
    renumbering moves.
    """
    rooms = p.graph.registry.rooms
    loop = getattr(p, "loop", None)
    excluded = getattr(loop, "_excluded", None) or {}
    offered = tuple(pid for pid in sorted(rooms) if pid not in excluded)
    tracker = getattr(p.graph, "label_tracker", None)
    labels = getattr(tracker, "labels", {}) or {}
    types = {pid: (labels[pid].label if pid in labels else "unknown") for pid in offered}
    return (p.mapping.floor_id, offered, types)


def semantic_change(previous, signature):
    """Whether the difference between two signatures is one the oracle should be asked about at once.

    A room on offer in both readings whose type changed (or a change of
    storey). A room appearing on or leaving the offer -- the watershed
    renumbering, a merge -- is geometry, which the loop debounces
    (``LoopSettings.reconsider_min_actions``); a new object is read at the
    next loop point, where the node list carries it.
    """
    if previous[0] != signature[0]:
        return True
    before_types, types = previous[2], signature[2]
    return any(pid in before_types and before_types[pid] != label for pid, label in types.items())


def resume_after_peek(p, obs, world):
    """A retained SEARCH resumes its remaining budget only after return transit."""
    if p._resume_room is None:
        return None
    floor, pid, actions = p._resume_room
    current = p.supervisor.room_id if p.hierarchy is None else p.hierarchy.regions.room_id
    searching = p.supervisor.state == "search" if p.hierarchy is None else p.hierarchy.machine.phase == "local_exploration"
    room = p.graph.registry.rooms.get(pid)
    if floor != p.mapping.floor_id or not searching or current != pid or room is None:
        p._resume_room = None
        return None
    if p.graph.room_at(world, (obs.pose.x, obs.pose.y)) == pid:
        p._resume_room = None
        return None
    inventory = p.graph.frontier_inventory
    if inventory is not None and actions < p.settings.doorway_peek.return_actions:
        ys, xs = np.nonzero(threshold_cells(p, world, room) & np.isfinite(inventory.distance_m))
        if len(xs):
            index = int(np.argmin(inventory.distance_m[ys, xs]))
            goal = world.grid_to_world(int(xs[index]), int(ys[index]))
            command = p._navigate(obs, world, goal, "room_return")
            if command is not None:
                p._action_owner = "room_return"
                return command
    p._resume_room = None
    if p.hierarchy is None:
        p.supervisor.finish("unreachable", "peek return unavailable", p._floor_time)
        p.loop.room_id = None
        p.loop._needs_reason = True
    else:
        p.hierarchy.machine.end("peek_return_unavailable")
    p.route_memory.clear("peek_return_unavailable")
    p._route = p._goal = None
    return None
