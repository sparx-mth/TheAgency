"""Shared startup, semantic interrupts and local-task resumption for ObjectNav."""
from __future__ import annotations

from dataclasses import replace
import numpy as np

from sparx_agency.core.planning.objnav.types.command import NavigationCommand
from sparx_agency.tasks.planning.objnav_benchmark_runtime.methods.doorway_candidates import threshold_cells
from sparx_agency.tasks.planning.objnav_benchmark_runtime.methods.room_search_loop import RoomReasoningUnavailable

SUSPENDED_PHASES = ("warmup", "doorway_peek", "room_return", "discovery_fallback")


def discover(p, obs, world, cost):
    """One synchronous discovery decision; None hands control to the room task."""
    if p.hierarchy is not None:
        p.hierarchy.sync_floor(obs, world)
    command = p.peek.plan(obs, world)
    if command is not None:
        p._action_owner = "doorway_peek"
        return command
    if p.warmup_actions < p.settings.warmup_steps:
        p._action_owner = "warmup"
        command = p.sweep.explore(obs, world, cost, np.ones(world.grid.shape, dtype=bool))
        if command is None:
            command = NavigationCommand.hold(final_yaw=obs.pose.yaw + p.episode.action_spec.turn_angle_rad,
                                             info={"kind": "warmup_scan"})
        return replace(command, info=dict(command.info, kind="warmup", warmup_step=p.warmup_actions + 1,
                                          warmup_budget=p.settings.warmup_steps))
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
