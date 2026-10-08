"""The safe backing manoeuvre: a target that appears at arm's length is re-verified from farther back.

The agent turns and the target is right in front of it -- 0.7 m, half the
frame, no history. Locking and STOPping on that single viewpoint is how a
counter front became a bed. Before the closing commits, this manoeuvre takes
the agent two or three steps back into space it knows to be clear, turns it
to face the remembered target coordinates, and hands the frame back to
verification from a wider perspective; only then may the lock, the approach
and the terminal inspection follow.

Clear space is checked before a step is taken, in two ledgers the agent
already keeps: the trail of poses it stood at on this storey
(``SightLedger.poses`` -- walked once, known free) and the observed grid with
the floor guard and the collision-qualified A* planner (``_plan_to``). A
candidate on the spawn floor's forbidden cells, off the local NavMesh when
one is bound, in unknown or occupied cells, or without a path from here is
not clear space. When nothing qualifies the manoeuvre is skipped and recorded
as such: verification proceeds in place, as before.
"""
from __future__ import annotations

from dataclasses import replace
import math
from typing import Dict, Optional, Sequence, Tuple

from sparx_agency.core.common.types import normalize_angle
from sparx_agency.core.planning.objnav.action_converter.action_choice import turn_action
from sparx_agency.core.planning.objnav.types.command import NavigationCommand

#: Bearings tried around "straight away from the target", degrees: behind first, then the sides.
RETREAT_OFFSETS_DEG = (0, 30, -30, 60, -60, 90, -90)


class BackoffManeuver:
    """Retreat to a verified clear spot, face the target, then yield to re-verification."""

    def __init__(self, policy, steps: int, max_actions: int, min_gain_m: float = 0.30):
        if type(steps) is not int or steps < 1 or type(max_actions) is not int or max_actions < 1:
            raise ValueError("Backoff needs positive integer steps and action bound")
        self.policy = policy
        self.steps, self.max_actions, self.min_gain_m = steps, max_actions, float(min_gain_m)
        self.state = "pending"              # pending -> retreat -> face -> done
        self.goal: Optional[Tuple[float, float]] = None
        self.source: Optional[str] = None   # "trail" or "ring": where the clear spot came from
        self.skipped: Optional[str] = None  # why no manoeuvre was made
        self.outcome: Optional[str] = None  # how a manoeuvre that was made ended
        self.actions = 0
        self.retreat_actions = 0
        self.started: Optional[int] = None
        self.finished: Optional[int] = None
        self._last_xy: Optional[Tuple[float, float]] = None
        self._stalled = 0                   # consecutive retreat actions without displacement

    @property
    def active(self) -> bool:
        return self.state in ("retreat", "face")

    def begin(self, obs, world, target_xyz: Sequence[float]) -> bool:
        """Choose the retreat spot; False (and ``skipped`` set) when no clear space is known."""
        self.started = int(obs.step)
        goal = self._retreat_goal(obs, world, target_xyz)
        if goal is None:
            self.skipped = "no navigable clear space behind or beside the agent"
            self.state = "done"
            self.finished = int(obs.step)
            return False
        self.goal, self.source = goal
        self.state = "retreat"
        return True

    def command(self, obs, world, target_xyz: Sequence[float], pitch: Optional[float]) -> Optional[NavigationCommand]:
        """This action's command, or None once the agent stands back and faces the target."""
        if not self.active:
            return None
        self.actions += 1
        if self.actions > self.max_actions:
            return self._finish(obs, "action bound reached; verifying from here")
        here = (float(obs.pose.x), float(obs.pose.y))
        spec = self.policy.episode.action_spec
        tilt = spec.has_camera_tilt
        info = {"kind": "target_closing", "phase": "BACK_OFF", "target_visible": False, "backoff_state": self.state,
                "backoff_goal": [round(v, 2) for v in self.goal], "backoff_source": self.source}
        if self.state == "retreat":
            # Arrival is the converter's: within its goal tolerance the route is complete and
            # its idle result is executed as a TURN (Markleeville 2026-10-08: eleven turns on
            # the spot, 0.25 m from the goal, until the action bound). Half a step of slack,
            # and standing still for three retreat actions is arrival too.
            tolerance = self.policy.converter_params.goal_tolerance_m + 0.5 * spec.forward_step_m
            toward = math.atan2(self.goal[1] - here[1], self.goal[0] - here[0])
            facing = turn_action(normalize_angle(toward - obs.pose.yaw), spec.turn_angle_rad) is None
            if facing and self._last_xy is not None and math.dist(self._last_xy, here) < 1e-3:
                self._stalled += 1                      # facing the goal and not moving: the converter is idling there
            elif not facing or self._last_xy is None or math.dist(self._last_xy, here) >= 1e-3:
                self._stalled = 0
            self._last_xy = here
            budget = 2 * int(math.ceil(math.pi / spec.turn_angle_rad)) + self.steps + 2   # turn out, step, turn back
            if math.dist(here, self.goal) <= tolerance or self._stalled >= 3 or self.retreat_actions >= budget:
                self.state = "face"
            else:
                command = self.policy._navigate(obs, world, self.goal, "target_backoff")
                if command is None:
                    self.state = "face"                      # the path is gone: face the target from here
                else:
                    self.retreat_actions += 1
                    # Level in transit: the retreat frames are ordinary mapping frames.
                    return replace(command, camera_pitch=0.0 if tilt else None,
                                   info=dict(command.info, **info, reason="backing off to re-verify from a wider perspective"))
        bearing = math.atan2(float(target_xyz[1]) - here[1], float(target_xyz[0]) - here[0])
        error = normalize_angle(bearing - obs.pose.yaw)
        if turn_action(error, self.policy.episode.action_spec.turn_angle_rad) is None:
            return self._finish(obs, "facing the remembered target; re-verifying")
        return NavigationCommand.hold(camera_pitch=pitch if tilt else None, final_yaw=bearing,
                                      info=dict(info, backoff_state="face", reason="turning back to face the remembered target"))

    def _finish(self, obs, why: str) -> None:
        self.state = "done"
        self.finished = int(obs.step)
        self.outcome = why
        return None

    def _retreat_goal(self, obs, world, target_xyz) -> Optional[Tuple[Tuple[float, float], str]]:
        """The nearest clear spot ``steps`` forward steps away that widens the range to the target."""
        p = self.policy
        spec = p.episode.action_spec
        reach = self.steps * spec.forward_step_m
        here = (float(obs.pose.x), float(obs.pose.y))
        target_xy = (float(target_xyz[0]), float(target_xyz[1]))
        range_now = math.dist(here, target_xy)
        candidates = []
        for entry in reversed(list(p.sight.poses(p.mapping.floor_id))[-60:]):
            xy = (float(entry[1]), float(entry[2]))
            if 0.8 * reach <= math.dist(here, xy) <= 1.6 * reach:
                candidates.append((xy, "trail"))
        away = math.atan2(here[1] - target_xy[1], here[0] - target_xy[0])
        for offset in RETREAT_OFFSETS_DEG:
            angle = away + math.radians(offset)
            candidates.append(((here[0] + reach * math.cos(angle), here[1] + reach * math.sin(angle)), "ring"))
        for xy, source in candidates:
            if math.dist(xy, target_xy) < range_now + self.min_gain_m:
                continue                                     # not a wider perspective
            spot = self._clear_spot(obs, world, xy)
            if spot is not None:
                return spot, source
        return None

    def _clear_spot(self, obs, world, xy) -> Optional[Tuple[float, float]]:
        """``xy`` snapped to the NavMesh when one is bound, if it is a free, allowed, reachable cell."""
        p = self.policy
        if p.target_projector is not None:
            snapped = p.target_projector((float(xy[0]), float(xy[1]), float(obs.pose.z)))
            if snapped is None or abs(snapped[2] - obs.pose.z) > 0.2 or math.dist(snapped[:2], xy) > 0.15:
                return None
            xy = (float(snapped[0]), float(snapped[1]))
        gx, gy = world.world_to_grid(*xy)
        if not world.in_bounds(gx, gy) or not world.is_free(gx, gy):
            return None
        if not p.floor_guard.goal_allowed(obs, world, xy, "target_backoff"):
            return None
        return xy if p._plan_to(obs, world, xy) is not None else None

    def diagnostics(self) -> Dict[str, object]:
        return {"state": self.state, "goal": None if self.goal is None else [round(v, 2) for v in self.goal],
                "source": self.source, "skipped": self.skipped, "outcome": self.outcome, "actions": self.actions,
                "retreat_actions": self.retreat_actions,
                "started_step": self.started, "finished_step": self.finished}
