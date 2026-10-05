"""Shared startup, semantic interrupts and local-task resumption for ObjectNav."""
from __future__ import annotations

from dataclasses import replace
import math
import numpy as np

from sparx_agency.core.common.types import normalize_angle
from sparx_agency.core.planning.objnav.types.command import NavigationCommand
from sparx_agency.tasks.planning.objnav_benchmark_runtime.methods.doorway_candidates import threshold_cells
from sparx_agency.tasks.planning.objnav_benchmark_runtime.methods.room_search_loop import RoomReasoningUnavailable

SUSPENDED_PHASES = ("warmup", "doorway_peek", "room_return", "discovery_fallback", "glance")


def discover(p, obs, world, cost):
    """One synchronous discovery decision; None hands control to the room task.

    The warm-up is one full rotation in place -- the look-around the whole
    method is built on, taken where the agent stands before any room is
    chosen: the map gets its first rooms, the detector its first objects,
    and when it ends the agent's position goes into the scan ledger, so the
    room it stands in is finished before the first room is ever chosen. It
    runs again on every storey the agent first sets foot on.
    """
    if p.hierarchy is not None:
        p.hierarchy.sync_floor(obs, world)
    command = p.peek.plan(obs, world)
    if command is not None:
        p._action_owner = "doorway_peek"
        return command
    if p.warmup_actions < p.settings.warmup_steps:
        p._action_owner = "warmup"
        here = (float(obs.pose.x), float(obs.pose.y))
        anchor = getattr(p, "_warmup_xy", None)
        if anchor is not None and p.warmup_actions > 0 and math.dist(here, anchor) > 0.5:
            # A takeover or a stair walk moved the agent mid-rotation: the turns so far
            # looked at another place. One rotation from one spot, or the ledger's
            # record lands where only part of the circle was turned.
            p.warmup_actions = 0
        if p.warmup_actions == 0:
            p._warmup_xy = here
        turn = p.episode.action_spec.turn_angle_rad
        command = NavigationCommand.hold(final_yaw=normalize_angle(obs.pose.yaw + turn), info={"kind": "warmup_scan"})
        return replace(command, info=dict(command.info, kind="warmup", warmup_step=p.warmup_actions + 1,
                                          warmup_budget=p.settings.warmup_steps))
    if getattr(p, "_warmup_pending", False):
        p._warmup_pending = False
        ledger = getattr(p, "scans", None)
        if ledger is not None:
            pid = p.graph.room_at(world, (obs.pose.x, obs.pose.y))
            ledger.record(obs, p.graph.registry.rooms.get(pid) if pid is not None else None, source="warmup")
    signature = (p.mapping.floor_id, tuple(sorted(p.graph.registry.rooms)),
                 tuple((pid, tuple(sorted(set(p.graph.objects_in(pid))))) for pid in sorted(p.graph.registry.rooms)))
    previous = p._semantic_signature
    event = p.peek.pending_reason or (previous is not None and signature != previous)
    if event and p.graph.registry.rooms:
        try:
            if p.hierarchy is None:
                p.loop.reconsider(obs, world)
            else:
                p.hierarchy.reconsider(obs, world)
        except RoomReasoningUnavailable:
            # Keep the interrupted task's budget intact while a service is away.
            p._action_owner = "discovery_fallback"
            return p.fallback.plan(obs, world, cost, reason="semantic reassessment unavailable")
        p.peek.pending_reason = False
    p._semantic_signature = signature
    return resume_after_peek(p, obs, world)


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
