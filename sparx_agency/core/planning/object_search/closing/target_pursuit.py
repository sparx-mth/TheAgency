"""Locked targets released for want of a path, walked back to once the map holds a route to one of them.

The closing gives a LOCKED target up when no standoff has a route from where
the agent stands (``TargetClosing._no_path``, "no safe path from here"): the
spot is remembered as boxed in and the search, which can walk, carries on.
The target itself was forgotten with the lock. The map then grows -- a door
opens up, the treads of a step down come into view -- and nothing brought the
agent back: the GT-oracle sweep of 2026-10-28 released 00880/000067's toilet
at action 337 and 00839/000044's television at action 141 this way and
reached neither, and the step-down re-run of 00835/000006 (2026-10-31) found
the steps to its television at action 90, forty-seven actions after the
release, and walked on to other rooms.

:class:`TargetPursuit` keeps those targets. Every ``retry_actions`` it asks
the closing's own standoff selection (:class:`~target_path.TargetApproachPath`,
the same ring, planners and navmesh projection) for a route from where the
agent stands now; when one exists the agent walks it, facing the target on
arrival, and the next fresh box re-locks it as any takeover would -- the
boxed-in spot it was released from no longer refuses it, since a route now
exists. A pursuit that loses its route, outlives ``max_walk_actions`` or
arrives without a sighting for ``arrival_actions`` ends, and a target is
tried ``max_attempts`` times at most within ``max_age_actions`` of its
release.

A **bearing approach** given up because a wall stopped it or the class left
the frame (``TargetClosing._end_bearing``) hands over too: a box without a
depth is beyond the depth range, so its target is placed on the bearing
``bearing_reach_m`` past the camera's ``max_depth_m`` and pursued the same
way -- around the wall, through the unknown if need be -- until a frame
measures it. 00853-5cdEh9F2hJL/000065 (2026-10-31) saw its television from
beyond 5 m three times, walked three steps into a wall each time, and locked
it with depth at action 487.

**A bearing is a guess, and the guesses are bounded as a whole** (since
2026-11-11). Z6MFQCViBuw's atrium house showed the other side of the rule:
chairs on the far side of the void are seen without depth again and again,
every abandoned bearing approach remembered a NEW point 6 m along its
bearing, and each was walked toward for up to ``max_walk_actions`` -- eight
of the single-storey set's episodes spent 275-453 of their 500 actions
"walking back to the released chair" (MP3D, ground truth, filter on). Since
then a bearing target within ``bearing_merge_m`` of a remembered one of the
same label is the same object (its guess is updated, its attempts kept), a
bearing pursuit walks at most ``bearing_max_walk_actions``, and every
pursuit of the episode together walks at most ``max_total_walk_actions``:
the search, not the chase, owns the rest.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import math
from types import SimpleNamespace
from typing import Dict, List, Optional

from sparx_agency.core.common.types import normalize_angle
from sparx_agency.core.planning.objnav.types.command import NavigationCommand
from sparx_agency.core.planning.object_search.closing.target_path import TargetApproachPath


@dataclass(frozen=True)
class PursuitSettings:
    """Bounds of the pursuit; counts in actions.

    Attributes:
        enabled: Whether released targets are pursued at all.
        retry_actions: Actions between two route queries for one target.
        max_age_actions: A target released longer ago than this is no
            longer pursued.
        max_attempts: Pursuits started per target.
        max_walk_actions: Actions one pursuit may walk before it ends.
        arrival_actions: Actions at the standoff, facing the target, before
            a pursuit without a sighting ends.
        bearing_reach_m: How far beyond the camera's depth range a target
            seen without depth is placed along its bearing.
        bearing_merge_m: A bearing target placed within this of a remembered
            bearing target of the same label is that target: its guess is
            updated, its attempts are kept (one chair glimpsed across a void
            six times is one chair, not six).
        bearing_max_walk_actions: Actions one pursuit of a BEARING target may
            walk -- a guess along a line, not a measured position.
        max_total_walk_actions: Actions every pursuit of the episode may walk
            together; spent, released targets are left to the search.
    """

    enabled: bool = True
    retry_actions: int = 8
    max_age_actions: int = 300
    max_attempts: int = 3
    max_walk_actions: int = 120
    arrival_actions: int = 4
    bearing_reach_m: float = 1.0
    bearing_merge_m: float = 1.5
    bearing_max_walk_actions: int = 48
    max_total_walk_actions: int = 150

    def __post_init__(self):
        if type(self.enabled) is not bool:
            raise ValueError("PursuitSettings.enabled must be a bool")
        if isinstance(self.bearing_reach_m, bool) or not 0.0 <= float(self.bearing_reach_m) < 10.0:
            raise ValueError("PursuitSettings.bearing_reach_m must lie in [0, 10) m")
        if isinstance(self.bearing_merge_m, bool) or not 0.0 <= float(self.bearing_merge_m) < 10.0:
            raise ValueError("PursuitSettings.bearing_merge_m must lie in [0, 10) m (0: never merge)")
        for name in ("retry_actions", "max_age_actions", "max_attempts", "max_walk_actions", "arrival_actions",
                     "bearing_max_walk_actions", "max_total_walk_actions"):
            value = getattr(self, name)
            if type(value) is not int or value < 1:
                raise ValueError("PursuitSettings.%s must be a positive int, got %r" % (name, value))


class TargetPursuit:
    """The targets the closing released for want of a path, and the walk back to one of them."""

    def __init__(self, policy, settings: Optional[PursuitSettings] = None):
        self.policy = policy
        self.settings = settings or PursuitSettings()
        self.targets: List[Dict] = []
        self.active: Optional[Dict] = None
        self.stats = {"remembered": 0, "merged": 0, "queries": 0, "started": 0, "arrived": 0, "reacquired": 0,
                      "ended": 0, "walked": 0, "budget_spent": 0}
        self.events: List[Dict] = []

    # -- what the closing hands over ----------------------------------------------------------------------------
    def remember(self, obs, closing):
        """Keep the target of a lock ``closing`` is about to release for want of a path."""
        if closing.xyz is not None:
            self.remember_at(obs, closing.xyz, closing.label, closing.floor_id, "no_path",
                             lowest_m=getattr(closing, "lowest_m", None),
                             face_normal_xy=getattr(closing, "face_normal_xy", None))

    def remember_bearing(self, obs, label, yaw, floor_id):
        """Keep a target seen only without depth: on its bearing, just beyond the depth range."""
        reach = float(obs.camera.max_depth_m) + float(self.settings.bearing_reach_m)
        xyz = (float(obs.pose.x) + reach * math.cos(yaw), float(obs.pose.y) + reach * math.sin(yaw), float(obs.pose.z))
        self.remember_at(obs, xyz, label, floor_id, "bearing")

    def remember_at(self, obs, xyz, label, floor_id, source, lowest_m=None, face_normal_xy=None):
        """Keep a target to walk back to; ``lowest_m`` and ``face_normal_xy`` are the lock's lowest measured point and
        facing normal, so its standoff is chosen as the closing would choose it (``TargetApproachPath``)."""
        if not self.settings.enabled:
            return
        if source == "bearing" and self.settings.bearing_merge_m > 0.0:
            for entry in self.targets:
                if (entry["source"] == "bearing" and entry["label"] == label and entry["floor"] == floor_id
                        and math.dist(entry["xyz"][:2], (float(xyz[0]), float(xyz[1]))) <= self.settings.bearing_merge_m):
                    # The same object glimpsed again: a fresher guess at where it stands, the same attempts.
                    entry["xyz"] = tuple(float(v) for v in xyz)
                    entry["released"] = int(obs.step)
                    entry["from_xy"] = (float(obs.pose.x), float(obs.pose.y))
                    self.stats["merged"] += 1
                    self._log(obs, "merged", label=label, attempts=entry["attempts"])
                    return
        self.targets.append({"xyz": tuple(float(v) for v in xyz), "floor": floor_id, "label": label,
                             "released": int(obs.step), "last_query": int(obs.step), "attempts": 0,
                             "from_xy": (float(obs.pose.x), float(obs.pose.y)), "source": source,
                             "lowest_m": None if lowest_m is None else float(lowest_m),
                             "face_normal_xy": None if face_normal_xy is None else tuple(float(v) for v in face_normal_xy)})
        self.stats["remembered"] += 1
        self._log(obs, "remembered", label=label, source=source)

    def notify_takeover(self, obs):
        """The closing locked a target again: the pursuit in force, if any, did its work."""
        if self.active is not None:
            self.stats["reacquired"] += 1
            self._end(obs, "re-acquired")

    # -- the walk ---------------------------------------------------------------------------------------------
    def command(self, obs, world) -> Optional[NavigationCommand]:
        """The pursuit's command for this action, or None to leave it to the search."""
        p, s = self.policy, self.settings
        if not s.enabled or p.closing.active:
            return None
        if self.active is not None:
            return self._continue(obs, world)
        if self.stats["walked"] >= s.max_total_walk_actions:
            if not self.stats["budget_spent"]:
                self.stats["budget_spent"] = 1
                self._log(obs, "budget_spent", walked=self.stats["walked"])
            return None                          # the episode's pursuits have walked their share
        step = int(obs.step)
        for entry in reversed(self.targets):
            if (entry["floor"] != p.mapping.floor_id or entry["attempts"] >= s.max_attempts
                    or step - entry["released"] > s.max_age_actions or step - entry["last_query"] < s.retry_actions):
                continue
            entry["last_query"] = step
            self.stats["queries"] += 1
            path = TargetApproachPath(p)
            target = self._shim(entry)
            # A query that finds nothing must leave the search's committed route as it was: a ring goal the
            # floor guard refuses clears the route memory inside ``_navigate``.
            saved = (dict(vars(p.route_memory)), p._route, p._goal)
            command = path.command(target, obs, world)
            if command is None:
                vars(p.route_memory).clear()
                vars(p.route_memory).update(saved[0])
                p._route, p._goal = saved[1], saved[2]
                continue
            entry["attempts"] += 1
            self.active = {"entry": entry, "path": path, "target": target, "started": step, "arrived_at": None}
            self.stats["started"] += 1
            self._unbox(entry)
            self._log(obs, "started", label=entry["label"], attempt=entry["attempts"])
            return self._dress(obs, command, entry, "a route to a standoff exists now")
        return None

    def _continue(self, obs, world):
        s, a = self.settings, self.active
        entry = a["entry"]
        bound = s.bearing_max_walk_actions if entry["source"] == "bearing" else s.max_walk_actions
        if obs.step - a["started"] > bound:
            return self._end(obs, "walked %d actions" % bound)
        if self.stats["walked"] >= s.max_total_walk_actions:
            return self._end(obs, "the episode's pursuits have walked %d actions" % self.stats["walked"])
        self.stats["walked"] += 1
        if a["arrived_at"] is not None:
            if obs.step - a["arrived_at"] >= s.arrival_actions:
                return self._end(obs, "arrived without a sighting")
            return NavigationCommand.hold(final_yaw=self._bearing(obs, entry), info={
                "kind": "target_pursuit", "reason": "at the standoff: facing the released %s" % entry["label"]})
        command = a["path"].command(a["target"], obs, world)
        if command is None:
            return self._end(obs, "the route was lost")
        if a["path"].arrived:
            a["arrived_at"] = int(obs.step)
            self.stats["arrived"] += 1
            return NavigationCommand.hold(final_yaw=self._bearing(obs, entry), info={
                "kind": "target_pursuit", "reason": "arrived at the standoff: facing the released %s" % entry["label"]})
        return self._dress(obs, command, entry, "walking back to the released %s" % entry["label"])

    # -- internals ----------------------------------------------------------------------------------------------
    def _shim(self, entry):
        """What :class:`TargetApproachPath` reads of a target: its 3-D memory, the closing's settings, its height
        above the base (from the lock's lowest point, when the memory kept one), four counters."""
        closing = self.policy.closing
        lowest, normal = entry.get("lowest_m"), entry.get("face_normal_xy")

        def frontal_normal(obs):
            if normal is None or lowest is None or not closing.settings.frontal_standoff:
                return None
            return normal if float(lowest) - float(obs.pose.z) >= closing.settings.frontal_min_elevation_m else None

        return SimpleNamespace(xyz=entry["xyz"], settings=closing.settings, face_normal_xy=normal, frontal_normal=frontal_normal,
                               elevation_above_base=lambda obs: 0.0 if lowest is None else closing.elevation_above_base(obs, lowest),
                               stale_standoffs_kept=0, widened_standoffs=0, through_unknown_routes=0, frontal_standoffs=0,
                               frontal_reselections=0)

    def _unbox(self, entry):
        """A route exists now: the spot the target was released from no longer refuses its takeover."""
        closing = self.policy.closing
        radius = closing.settings.boxed_in_radius_m
        closing.boxed_in = [b for b in closing.boxed_in
                            if not (b[1] == entry["floor"] and math.dist(b[0], entry["from_xy"]) <= radius)]

    def _bearing(self, obs, entry):
        return normalize_angle(math.atan2(entry["xyz"][1] - obs.pose.y, entry["xyz"][0] - obs.pose.x))

    def _dress(self, obs, command, entry, reason):
        self.policy._action_owner = "target_pursuit"
        return replace(command, info=dict(command.info, kind="target_pursuit", reason=reason,
                                          pursuit={"label": entry["label"], "attempt": entry["attempts"],
                                                   "released": entry["released"]}))

    def _end(self, obs, why):
        self.stats["ended"] += 1
        self._log(obs, "ended", why=why)
        self.active = None
        self.policy.route_memory.clear("pursuit_ended")
        return None

    def _log(self, obs, event, **fields):
        if len(self.events) < 40:
            self.events.append(dict(fields, action=int(obs.step), event=event))

    def diagnostics(self):
        return {"settings": asdict(self.settings), "stats": dict(self.stats), "targets": len(self.targets),
                "active": None if self.active is None else self.active["entry"]["label"], "events": list(self.events)}
