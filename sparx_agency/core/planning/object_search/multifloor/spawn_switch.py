"""One switch to the other storey of the flight the agent spawned on, once the first storey is searched out.

With stair traversal forbidden (the benchmark default) a spawn on a staircase is walked to the NEARER storey the
flight joins (:class:`~multifloor_policy.MultiFloorSearch`, the provisional spawn storey), which becomes the spawn
storey. That storey holds the target in 112 of the 174 HM3D v1 val episodes that start on stairs; in most of the rest
it is at the other end of the same flight (``runs/stair_spawn/episode_levels.csv``, 2026-11-02). So when the first
storey is searched out -- the exploration fallback has no frontier left and asks the building for a last resort --
the agent walks back to where it came off the flight and on along it to the storey at its far end:

* **armed only by a stair spawn** -- a provisional spawn storey settled on a storey (an atlas arrival, or the
  guard's re-measure onto a floor a step below), never a spawn on a floor;
* **at most once onto the flight** -- the switch is spent once its traversal starts, whether it arrives, is
  retreated from or abandoned; an approach given up before the flight (a cramped corner on the way, 00814/000004)
  leaves the agent on the same storey and is tried once more, ``spawn_switch_min_floor_actions`` later;
* **final** -- the storey it reaches becomes the spawn storey, and nothing arms again;
* **somewhere new** -- it never ends on a floor known when it set off: not the first storey, not the spawn's own
  tread on the way past (``avoid_floors``; the floor guard neither settles nor adopts there while it traverses);
* **armed only by a storey** -- a plateau smaller than ``min_floor_area_m2`` reached from a stair spawn is a
  landing, never adopted as the spawn storey, so it arms nothing.

Two storeys at most, and never back to the first: the agent cannot go up and down between floors.
"""
from __future__ import annotations

import math
from typing import Optional

SELECTED_BY_SPAWN_SWITCH = "spawn_switch"
#: Approaches to the flight one episode may start; only one of them can reach it.
APPROACH_ATTEMPTS = 2


class SpawnStairsSwitch:
    """The switch's state for one episode, owned by the building coordinator.

    Args:
        coordinator: The :class:`MultiFloorSearch` it switches for.
    """

    def __init__(self, coordinator):
        self.coordinator = coordinator
        self.flight: Optional[dict] = None      # the storey settled on and the way back to the flight
        self.state: Optional[str] = None        # None, "approach", "traverse", "done", "abandoned"
        self.portal_id: Optional[int] = None
        self.failures = 0
        self.attempts = 0
        self.retry_at: Optional[int] = None     # the action an approach given up before the flight may start again

    @property
    def params(self):
        return self.coordinator.params

    def settled(self, obs, floor_id, old_height, new_height, how):
        """A provisional (stair) spawn storey settled on a storey: remember the way to the other end of the flight."""
        if self.flight is not None or self.state is not None or not self.params.spawn_switch:
            return
        towards = -1 if float(new_height) > float(old_height) else 1     # back along the flight, past the spawn
        self.flight = {"floor_id": int(floor_id), "elevation_m": float(new_height), "direction": towards,
                       "entry": (float(obs.pose.x), float(obs.pose.y), float(new_height)), "step": int(obs.step),
                       "guide": self._guide(int(floor_id), towards)}
        self.coordinator.events.append({"action": obs.step, "event": "spawn_switch_armed", "floor_id": int(floor_id),
                                        "towards": "up" if towards > 0 else "down", "how": how,
                                        "entry": [round(v, 2) for v in self.flight["entry"]],
                                        "guide_points": len(self.flight["guide"])})

    def _guide(self, floor_id, towards):
        """The way back onto the flight: the walk the atlas measured from the spawn to this storey, reversed.

        Without it the switch's traversal had only the point it settled at to aim from, and from the top of a
        flight the treads below are out of sight: 00814/000004 (HM3D v1 val, 2026-11-02) wandered the storey's
        floor for its 30-action budget beside the flight it had climbed. Past the trail's end -- the spawn -- the
        traversal continues on the support it sees. A storey reached by a step, re-measured with no transition,
        has no trail; the flight seen from the spawn in the switch's direction stands in for it.
        """
        atlas = self.coordinator.policy.mapping.atlas
        last = atlas.last_connection
        if last is not None and floor_id in (atlas.connections[last].source, atlas.connections[last].destination):
            return [tuple(float(v) for v in p) for p in atlas.connections[last].oriented(floor_id)[1]]
        for flight in self.coordinator._spawn_flights:
            if int(flight["direction"]) == towards and flight.get("path"):
                return [tuple(float(v) for v in p) for p in flight["path"]]
        return []

    def owns(self, portal) -> bool:
        return portal is not None and self.portal_id is not None and portal.get("id") == self.portal_id

    def due(self, obs, exhausted) -> bool:
        """Whether to switch now, with the budget left for the far storey and no target known on this one.

        The first storey searched out (``exhausted``, after ``spawn_switch_min_floor_actions``), or searched for
        ``spawn_switch_after_actions`` without finding the target: 00800/000038 settled on the storey below a
        midway spawn (1.47 m down, 1.53 m up), whose fourteen rooms kept frontier for all 450 actions left, and
        the plant was upstairs.
        """
        c, p = self.coordinator, self.params
        if self.flight is None or self.state is not None:
            return False
        if c.floor_id != self.flight["floor_id"] or c.traversing:
            return False
        if self.retry_at is not None:
            if obs.step < self.retry_at:
                return False
        else:
            searched = obs.step - self.flight["step"]
            if searched < (p.spawn_switch_min_floor_actions if exhausted else p.spawn_switch_after_actions):
                return False
        if self._target_known():
            return False
        return int(c.policy.episode.max_steps) - int(obs.step) >= p.spawn_switch_min_actions_left

    def _target_known(self) -> bool:
        """A target seen on this storey -- remembered by the pursuit or the closing -- keeps the agent here."""
        policy, floor = self.coordinator.policy, self.coordinator.floor_id
        pursuit = getattr(policy, "pursuit", None)
        if pursuit is not None and any(t.get("floor") == floor for t in getattr(pursuit, "targets", ())):
            return True
        closing = getattr(policy, "closing", None)
        return closing is not None and (getattr(closing, "active", False) or getattr(closing, "memory", None) is not None)

    def begin(self, obs):
        """Aim the building at the flight the agent came off, toward its other end."""
        c = self.coordinator
        portal = c._portal(obs, self.flight["direction"], [self.flight["entry"]] + list(self.flight["guide"]))
        portal["selected_by"] = SELECTED_BY_SPAWN_SWITCH
        portal["avoid_floors"] = sorted(c.policy.mapping.atlas.floors)   # every floor known now is one it left
        c.active, self.portal_id, self.state, self.failures = portal, portal["id"], "approach", 0
        self.attempts += 1
        c.policy.route_memory.clear("spawn switch")
        c.events.append({"action": obs.step, "event": "spawn_switch_started", "portal_id": portal["id"],
                         "direction": "up" if self.flight["direction"] > 0 else "down",
                         "searched_actions": int(obs.step - self.flight["step"]),
                         "reason": "the storey the stair spawn settled on is searched out; the flight's other end "
                                   "is the only other storey the episode may hold"})

    def plan(self, obs, world):
        """Walk to the flight's entry and start its traversal there; None hands the action back."""
        c = self.coordinator
        entry = tuple(c.active["entry"][:2])
        if math.dist((obs.pose.x, obs.pose.y), entry) <= 0.65:
            if c._start(obs, force=True):          # rooms owed a peek on a storey the switch is leaving: no veto
                self.state = "traverse"
                return c.transition.plan(obs)
            return None
        command = c.policy._navigate(obs, world, entry, "portal/%d" % c.active["id"])
        if command is not None:
            self.failures = 0
            return command
        self.failures += 1
        if self.failures >= self.params.approach_failures:
            c._abandon(obs, "spawn switch: the flight's entry is unreachable")
        return None

    def accept_arrival(self, obs):
        """The switch's traversal stands settled on the far storey and the atlas calls it a landing: adopt it.

        The atlas refuses a storey within ``min_floor_separation_m`` of every floor it knows, and the spawn's own
        "floor" -- a tread of this very flight -- is one of them.
        """
        c = self.coordinator
        atlas = c.policy.mapping.atlas
        t = c.transition
        if self.state != "traverse" or t is None or not atlas.in_transition or t.retreat_path is not None:
            return
        if t.phase == "CONFIRM_DESTINATION" and t.arrival_allowed:
            floor = atlas.adopt_level(obs.pose, float(obs.pose.z))
            c.events.append({"action": obs.step, "event": "spawn_switch_adopted", "floor_id": int(floor),
                             "height_m": round(float(obs.pose.z), 3)})

    def arrived(self, obs, floor_id, height):
        """The switch reached its storey: it becomes the spawn storey, and the switch is spent."""
        if self.state not in ("approach", "traverse") or not self.owns(self.coordinator.active):
            return
        self.state = "done"
        guard = getattr(self.coordinator.policy, "floor_guard", None)
        if guard is not None and guard.enabled:
            guard.rebase(obs, int(floor_id), float(height), "the spawn switch reached the other storey of the flight")
        self.coordinator.events.append({"action": obs.step, "event": "spawn_switch_arrived", "floor_id": int(floor_id),
                                        "height_m": round(float(height), 3)})

    def abandoned(self, obs, reason):
        if self.state == "approach" and self.attempts < APPROACH_ATTEMPTS:
            self.state, self.portal_id = None, None
            self.retry_at = int(obs.step) + int(self.params.spawn_switch_min_floor_actions)
            self.coordinator.events.append({"action": obs.step, "event": "spawn_switch_deferred", "reason": reason,
                                            "retry_at": self.retry_at})
            return
        if self.state in ("approach", "traverse"):
            self.state = "abandoned"
            self.coordinator.events.append({"action": obs.step, "event": "spawn_switch_abandoned", "reason": reason})

    def diagnostics(self):
        return {"state": self.state, "armed": self.flight is not None, "attempts": self.attempts,
                "floor_id": None if self.flight is None else self.flight["floor_id"],
                "guide_points": None if self.flight is None else len(self.flight["guide"])}
