"""One-shot doorway inspection with an isolated route and resumable search task."""
from __future__ import annotations

import math
import numpy as np
from sparx_agency.core.common.types import normalize_angle
from sparx_agency.core.planning.objnav.types.command import NavigationCommand
from sparx_agency.tasks.planning.objnav_benchmark_runtime.methods.doorway_candidates import doorway_candidates, same_region, threshold_cells
from sparx_agency.tasks.planning.objnav_benchmark_runtime.methods.route_memory import CommittedRoute
from sparx_agency.tasks.planning.objnav_benchmark_runtime.methods.peek_stairs import peek_planning_world, stair_peek_mask


class DoorwayPeek:
    """Approach a nearby new room, scan its inward semicircle, then reconsider.

    No nested/repeated peeks. The allowance is consumed on start, even on
    cancellation. Actual measured yaw still distinguishes a completed scan
    from a mere attempt; both attempts and initial labels exempt future peeks.
    """

    def __init__(self, policy):
        self.policy, self.settings = policy, policy.settings.doorway_peek
        self.active = None
        self.records, self.events = [], []
        self.initialized_floors = set()
        self.pending_reason = False
        self._step, self._command = None, None

    def _record(self, pid, mask):
        floor = self.policy.mapping.floor_id
        for record in self.records:
            if record["floor"] != floor or record["mask"].shape != mask.shape:
                continue
            overlap = int(np.count_nonzero(record["mask"] & mask))
            ids = record.get("room_ids", [record["room"]])
            consumed = record.get("room_peeked", False) or record["done"] or record.get("classified", False) or record["attempts"] > 0
            # Registry IDs never recycle. Keep their history through modest
            # drift; a retired-ID region or split child must overlap spatially.
            if ((pid in ids and overlap > 0) or same_region(record["mask"], mask)
                    or (consumed and overlap >= 0.6 * max(1, min(int(record["mask"].sum()), int(mask.sum()))))):
                record["room_ids"] = sorted(set(ids) | {pid})
                return record
        return None

    def _exemption(self, pid, room):
        record = self._record(pid, room.mask)
        if record is not None:
            if record["done"]:
                return "scanned"
            if record.get("room_peeked", False) or record["attempts"] > 0:
                return "already_peeked"
            if record.get("classified", False):
                return "classified"
        info = self.policy.graph.label_info(pid) or {}
        if info.get("label") not in (None, "", "unknown"):
            if record is None:
                record = dict(floor=self.policy.mapping.floor_id, room=pid, mask=room.mask.copy(),
                              done=False, attempts=0, retry=0, room_peeked=False)
                self.records.append(record)
            record.update(classified=True, initial_label=info["label"], reason="classified")
            return "classified"
        return None

    def pending_rooms(self):
        """Unknown regions still eligible for their single peek; never demand repeats."""
        pending = []
        world = getattr(self.policy, "last_world", None)
        excluded = stair_peek_mask(self.policy, world) if world is not None else None
        for pid, room in self.policy.graph.registry.rooms.items():
            if excluded is not None and (room.mask & excluded).any():
                interior = threshold_cells(self.policy, world, room)
                if interior.any() and not (interior & ~excluded).any():
                    continue  # a stair connector is not an unexplored room doorway
            if self._exemption(pid, room) is None:
                pending.append(pid)
        return sorted(pending)

    def floor_ready(self):
        """No vacuous success on a newly arrived, not-yet-segmented storey."""
        return bool(self.policy.graph.registry.rooms) and not self.pending_rooms() and self.active is None

    def _remember_start(self, obs, world):
        """Remember entry for diagnostics, never mistake it for a completed peek."""
        p = self.policy
        floor = p.mapping.floor_id
        if self.active is not None:
            return
        pid = p.graph.room_at(world, (obs.pose.x, obs.pose.y))
        if pid is not None and self._record(pid, p.graph.registry.rooms[pid].mask) is None:
            self.records.append(dict(floor=floor, room=pid, mask=p.graph.registry.rooms[pid].mask.copy(),
                                     done=False, entered=True, attempts=0, retry=0,
                                     reason="spawn_room" if floor not in self.initialized_floors else "entered_without_scan"))
            self.initialized_floors.add(floor)

    def plan(self, obs, world, *, force=False):
        """One unknown-room peek; force never bypasses exemptions or stair safety."""
        p = self.policy
        if ((p.building is not None and p.building.committed) or p.mapping.atlas.in_transition
                or getattr(getattr(p, "closing", None), "active", False)):
            self.cancel(obs, "exclusive_navigation", restore=False)
            return None
        if not self.settings.enabled and not force and self.active is None:
            return None
        if self._step == obs.step and (self._command is not None or not force):
            return self._command
        self._step, self._command = obs.step, None
        excluded = stair_peek_mask(p, world)
        gx, gy = world.world_to_grid(obs.pose.x, obs.pose.y)
        if world.in_bounds(gx, gy) and excluded[gy, gx]:
            self.cancel(obs, "stair_exclusion")
            return None
        self._remember_start(obs, world)
        if self.active is None:
            # A single confirmed-object clue may supply the initial label
            # before a peek steals control from the normal reasoning cadence.
            for pid in self.pending_rooms():
                p.loop._reclassify(obs, pid, "peek_candidate")
            candidates = doorway_candidates(self.policy, obs, world, self.settings, nearby=not force,
                                             room_ids=set(self.pending_rooms()))
            for distance, pid, xy, heading, mask in candidates:
                record = self._record(pid, mask)
                if record and (record["done"] or record.get("room_peeked", False) or record["attempts"] >= 1):
                    continue
                self._begin(obs, pid, xy, heading, mask, record, distance)
                break
        if self.active is not None:
            self._command = self._advance(obs, world)
        return self._command

    def _begin(self, obs, pid, xy, heading, mask, record, distance=0.0):
        p = self.policy
        if record is None:
            record = dict(floor=p.mapping.floor_id, room=pid, mask=mask.copy(), done=False, attempts=0, retry=0)
            self.records.append(record)
        record["attempts"] += 1
        record["room_peeked"] = True
        record["peek_started_step"] = obs.step
        saved = (p.route_memory, p._route, p._goal, p._last_plan_s, p._blocked_since)
        p.route_memory = CommittedRoute(p.episode.action_spec, p.converter_params, p.route_settings)
        p._route = p._goal = None
        self.active = dict(room=pid, xy=xy, heading=heading, mask=mask.copy(), floor=p.mapping.floor_id,
                           start=obs.step, phase="approach", actions=0, scan_actions=0,
                           swept=0.0, last_yaw=None, record=record, saved=saved)
        self.active["approach_limit"] = self.settings.approach_actions + int(math.ceil(
            max(0.0, distance - self.settings.trigger_distance_m) / p.episode.action_spec.forward_step_m))
        self.active["resume_room"] = p.supervisor.room_id if p.hierarchy is None and p.supervisor.state == "search" else None
        if p.hierarchy is not None and p.hierarchy.machine.phase == "local_exploration":
            self.active["resume_room"] = p.hierarchy.regions.room_id
        self.events.append(dict(step=obs.step, event="peek_started", floor=p.mapping.floor_id, room=pid, entry=list(xy)))

    def _advance(self, obs, world):
        p, active = self.policy, self.active
        if active["floor"] != p.mapping.floor_id:
            self.cancel(obs, "floor_changed", restore=False)
            return None
        # Track a room that was renumbered while approaching it, but never
        # silently jump the inspection to a geometrically unrelated room.
        room = p.graph.registry.rooms.get(active["room"])
        if room is None or not same_region(active["mask"], room.mask):
            room = next((r for r in p.graph.registry.rooms.values() if same_region(active["mask"], r.mask)), None)
        if room is None:
            self.cancel(obs, "room_disappeared")
            return None
        active["room"] = room.id
        p.loop._reclassify(obs, room.id, "peek_active")
        info = p.graph.label_info(room.id) or {}
        if info.get("label") not in (None, "", "unknown"):
            active["record"].update(classified=True, initial_label=info["label"])
            self.cancel(obs, "classified")
            self.pending_reason = True
            return None
        excluded = stair_peek_mask(p, world)
        ex, ey = world.world_to_grid(*active["xy"])
        if world.in_bounds(ex, ey) and excluded[ey, ex]:
            self.cancel(obs, "stair_exclusion")
            return None
        gx, gy = world.world_to_grid(obs.pose.x, obs.pose.y)
        inside = world.in_bounds(gx, gy) and room.mask[gy, gx]
        if active["phase"] == "approach":
            if inside and math.dist((obs.pose.x, obs.pose.y), active["xy"]) <= p.converter_params.goal_tolerance_m + world.resolution:
                active["phase"] = "align"
                active["record"]["entered"] = True
                p.route_memory.clear("peek_interior_reached")
                p._route = p._goal = None
            elif active["actions"] >= active["approach_limit"]:
                self.cancel(obs, "approach_budget")
                return None
            else:
                command = p._navigate(obs, peek_planning_world(p, world, excluded), active["xy"], "doorway_peek_approach")
                if command is None:
                    self.cancel(obs, "entry_unreachable")
                return command
        if not inside:
            self.cancel(obs, "threshold_invalidated")
            return None
        turn = p.episode.action_spec.turn_angle_rad
        scan_limit = max(self.settings.scan_actions, int(math.ceil(2 * math.pi / turn)) + 2)
        if active["scan_actions"] >= scan_limit:
            self.cancel(obs, "scan_budget")
            return None
        if active["phase"] == "align":
            start_yaw = normalize_angle(active["heading"] - math.pi / 2)
            if abs(normalize_angle(start_yaw - obs.pose.yaw)) > turn * 0.51:
                return NavigationCommand.hold(final_yaw=start_yaw, info={"kind": "doorway_peek_align"})
            active["phase"], active["last_yaw"] = "scan", obs.pose.yaw
        delta = normalize_angle(obs.pose.yaw - active["last_yaw"])
        active["swept"] = max(0.0, active["swept"] + delta)
        active["last_yaw"] = obs.pose.yaw
        if active["swept"] >= math.pi - 1e-6:
            active["record"].update(done=True, reason="scanned", room=room.id, mask=room.mask.copy(),
                                    scan_cell=[gy, gx], scan_xy=[obs.pose.x, obs.pose.y],
                                    completed_step=obs.step, swept_degrees=math.degrees(active["swept"]))
            self.events.append(dict(step=obs.step, event="peek_completed", floor=active["floor"], room=room.id,
                                    scan_xy=[obs.pose.x, obs.pose.y], swept_degrees=math.degrees(active["swept"])))
            self._restore(obs)
            self.pending_reason = True
            return None
        return NavigationCommand.hold(final_yaw=normalize_angle(obs.pose.yaw + turn),
                                      info={"kind": "doorway_peek_scan", "room": room.id,
                                            "swept_degrees": math.degrees(active["swept"])})

    def charge(self):
        if self.active is not None:
            self.active["actions"] += 1
            if self.active["phase"] != "approach":
                self.active["scan_actions"] += 1

    def _restore(self, obs):
        p, active = self.policy, self.active
        memory, route, goal, planned, blocked = active["saved"]
        paused = max(0, obs.step - active["start"])
        memory.resume_after_pause(paused)
        p.route_memory, p._route, p._goal = memory, route, goal
        seconds = paused * p.settings.action_time_s
        p._last_plan_s = None if planned is None else planned + seconds
        p._blocked_since = None if blocked is None else blocked + seconds
        p._last_pose = (obs.pose.x, obs.pose.y)
        if active["resume_room"] is not None:
            p._resume_room = (active["floor"], active["resume_room"], 0)
        self.active = None

    def cancel(self, obs, reason, restore=True):
        """Target pursuit and committed stair motion outrank every inspection."""
        if self.active is None:
            return
        active = self.active
        active["record"].update(retry=obs.step + self.settings.retry_actions, reason=reason)
        # No cancellation refunds the one-time allowance, including target or stair takeover.
        self.events.append(dict(step=obs.step, event="peek_cancelled", floor=active["floor"],
                                room=active["room"], reason=reason))
        if restore and active["floor"] == self.policy.mapping.floor_id:
            self._restore(obs)
        else:
            self.active = None
        self._step, self._command = None, None

    def diagnostics(self):
        active = self.active
        return {"enabled": self.settings.enabled,
                "active": None if active is None else {k: active[k] for k in ("room", "floor", "phase", "actions", "swept")},
                "coverage": {"floor": self.policy.mapping.floor_id, "known_rooms": len(self.policy.graph.registry.rooms),
                             "pending_rooms": self.pending_rooms(), "floor_ready": self.floor_ready()},
                "records": [{k: v for k, v in r.items() if k != "mask"} for r in self.records],
                "events": list(self.events)}
