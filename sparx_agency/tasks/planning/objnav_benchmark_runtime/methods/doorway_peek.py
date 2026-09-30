"""One-shot doorway inspection with an isolated route and resumable search task."""
from __future__ import annotations

import math
from sparx_agency.core.common.types import normalize_angle
from sparx_agency.core.planning.objnav.types.command import NavigationCommand
from sparx_agency.tasks.planning.objnav_benchmark_runtime.methods.doorway_candidates import doorway_candidates, same_region
from sparx_agency.tasks.planning.objnav_benchmark_runtime.methods.route_memory import CommittedRoute


class DoorwayPeek:
    """Approach a nearby new room, scan its inward semicircle, then reconsider.

    No nested peeks. Actual measured yaw, not plan-call count, completes the
    scan. Failed routes/scans are bounded and cooled; completed regions are
    remembered across floor revisits and modest room-ID changes.
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
        return next((r for r in self.records if r["floor"] == floor
                     and (r["room"] == pid or same_region(r["mask"], mask))), None)

    def _remember_start(self, obs, world):
        p = self.policy
        floor = p.mapping.floor_id
        if self.active is not None:
            return
        pid = p.graph.room_at(world, (obs.pose.x, obs.pose.y))
        if pid is not None and self._record(pid, p.graph.registry.rooms[pid].mask) is None:
            self.records.append(dict(floor=floor, room=pid, mask=p.graph.registry.rooms[pid].mask.copy(),
                                     done=True, attempts=0, retry=0,
                                     reason="spawn_room" if floor not in self.initialized_floors else "already_entered"))
            self.initialized_floors.add(floor)

    def plan(self, obs, world):
        if not self.settings.enabled:
            return None
        if self._step == obs.step:
            return self._command
        self._step, self._command = obs.step, None
        self._remember_start(obs, world)
        if self.active is None:
            for _, pid, xy, heading, mask in doorway_candidates(self.policy, obs, world, self.settings):
                record = self._record(pid, mask)
                if record and (record["done"] or record["attempts"] >= self.settings.max_attempts or obs.step < record["retry"]):
                    continue
                self._begin(obs, pid, xy, heading, mask, record)
                break
        if self.active is not None:
            self._command = self._advance(obs, world)
        return self._command

    def _begin(self, obs, pid, xy, heading, mask, record):
        p = self.policy
        if record is None:
            record = dict(floor=p.mapping.floor_id, room=pid, mask=mask.copy(), done=False, attempts=0, retry=0)
            self.records.append(record)
        record["attempts"] += 1
        saved = (p.route_memory, p._route, p._goal, p._last_plan_s, p._blocked_since)
        p.route_memory = CommittedRoute(p.episode.action_spec, p.converter_params, p.route_settings)
        p._route = p._goal = None
        self.active = dict(room=pid, xy=xy, heading=heading, mask=mask.copy(), floor=p.mapping.floor_id,
                           start=obs.step, phase="approach", actions=0, scan_actions=0,
                           swept=0.0, last_yaw=None, record=record, saved=saved)
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
        gx, gy = world.world_to_grid(obs.pose.x, obs.pose.y)
        inside = world.in_bounds(gx, gy) and room.mask[gy, gx]
        if active["phase"] == "approach":
            if inside and math.dist((obs.pose.x, obs.pose.y), active["xy"]) <= p.converter_params.goal_tolerance_m + world.resolution:
                active["phase"] = "align"
                p.route_memory.clear("peek_threshold_reached")
                p._route = p._goal = None
            elif active["actions"] >= self.settings.approach_actions:
                self.cancel(obs, "approach_budget")
                return None
            else:
                command = p._navigate(obs, world, active["xy"], "doorway_peek_approach")
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
            active["record"].update(done=True, reason="scanned", room=room.id, mask=room.mask.copy())
            self.events.append(dict(step=obs.step, event="peek_completed", floor=active["floor"], room=room.id,
                                    swept_degrees=math.degrees(active["swept"])))
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
        if reason == "target_priority":
            active["record"]["attempts"] = max(0, active["record"]["attempts"] - 1)
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
                "records": [{k: v for k, v in r.items() if k != "mask"} for r in self.records],
                "events": list(self.events)}
