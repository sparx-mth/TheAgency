"""The escape probe: a physical step out of a map the planner cannot leave.

The planner works on the observed grid inflated by the body radius, and
unknown is impassable to it. An agent that spawns in a strip behind a sofa,
or that the depth camera has ringed with the top of a counter, stands in a
pocket of free cells no inflated route leaves: every rung of the exploration
fallback finds nothing reachable, and the last one -- the idle turn -- does
not change the map. Collierville/000036 (``6bc7ace42dcf``, 2026-10-13) spent
488 of its 500 actions on it, twice, with the toilet 5.6 m away.

The simulator's collision check is the arbiter of what the body can pass,
not the map's resolution: the probe reads the grid WITHOUT inflation along
each of the turn headings, picks the heading whose occupied band is thinnest
with the most free or unknown floor beyond it, turns to face it and takes
one forward step. A step that moves the agent re-maps its surroundings and
the ordinary rungs take over; a step the simulator blocks is remembered for
that cell and heading (:meth:`EscapeProbe.note_blocked`) and the next probe
takes the next heading. The planner's own map is never rewritten: LESSONS
2026-10-08 is the reason -- opening "genuine" obstacles around the start is
the one way A* threads a real wall.

**Mapped floor beyond a band outranks unknown beyond it** (since 2026-10-26).
Nothing is ever mapped outside the house, so the thin exterior wall of a
room always reads as a one-cell band with the whole probe range "free or
unknown" behind it, and by the floor-beyond score it beat every real exit:
00835/000064 (``680ae26a733b``) probed its bedroom's west wall 71 times over
351 actions while the strip's one exit -- a one-cell bump bar with 1.8 m of
MAPPED floor behind it -- was ranked below it on every action. A heading
whose far side holds known free cells is tried before any heading whose far
side is unknown alone; the unknown-only headings keep their turn after it,
for the pocket the camera has never seen out of.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Dict, List, NamedTuple, Optional, Tuple

import numpy as np

from sparx_agency.core.common.types import normalize_angle
from sparx_agency.core.planning.objnav.types.command import NavigationCommand


class RayProfile(NamedTuple):
    """What one heading holds, in metres, read without inflation (:func:`ray_profile`).

    ``pocket_m`` and ``beyond_m`` count free OR unknown cells, as the planner's
    "impassable unknown" does not apply to a probing step; ``pocket_mapped_m``
    and ``beyond_mapped_m`` are the KNOWN free part of each -- the floor the
    camera has actually seen there.
    """

    pocket_m: float
    barrier_m: float
    beyond_m: float
    pocket_mapped_m: float
    beyond_mapped_m: float


@dataclass(frozen=True)
class EscapeSettings:
    """When a heading is worth a probing step. Metres and actions.

    Attributes:
        enabled: Probe before the fallback's idle turn (the default). Off,
            the hold is the last rung as before.
        range_m: How far along a heading the grid is read.
        max_barrier_m: The thickest occupied band a probe pushes against. A
            sofa's back or the rim of a counter seen from below is one to
            three cells; a wall the camera saw whole is wider, and beyond it
            is nothing the step can reach.
        min_beyond_m: The least free or unknown floor that must lie beyond
            the band (or straight ahead, when there is none) for the heading
            to lead anywhere.
        min_pocket_m: A heading with at least this much free or unknown
            floor BEFORE the first occupied cell is worth a step whatever
            lies beyond -- two forward steps by default (since 2026-10-25).
            The thinnest-barrier rule alone left a wedged agent spinning:
            00824/000066 stood for 450 actions in a pocket whose open side
            ran 0.6 m to a bed (a barrier too thick to probe) while every
            heading it did probe was a refused lateral touch. A step along
            the open side moves the body to a cell the planner may well
            leave; the simulator decides, as for every probe.
        memory_actions: A heading the simulator blocked at a cell is not
            probed again from that cell for this many actions.
    """

    enabled: bool = True
    range_m: float = 3.0
    max_barrier_m: float = 0.4
    min_beyond_m: float = 0.5
    min_pocket_m: float = 0.5
    memory_actions: int = 80

    def __post_init__(self):
        for name in ("range_m", "max_barrier_m", "min_beyond_m", "min_pocket_m"):
            if not float(getattr(self, name)) > 0.0:
                raise ValueError("%s must be positive, got %r" % (name, getattr(self, name)))
        if self.max_barrier_m >= self.range_m or self.min_beyond_m >= self.range_m or self.min_pocket_m >= self.range_m:
            raise ValueError("max_barrier_m, min_beyond_m and min_pocket_m must be under range_m")
        if int(self.memory_actions) < 1:
            raise ValueError("memory_actions must be at least 1")


def ray_profile(world, xy, heading, skip_m, range_m) -> RayProfile:
    """The :class:`RayProfile` along ``heading`` from ``xy``, read at half-cell steps without inflation.

    ``pocket_m`` is the free or unknown run from ``skip_m`` (the agent's own
    footprint, which the map clears by construction) to the first occupied
    cell; ``barrier_m`` the occupied band that follows, and ``beyond_m`` the
    free or unknown run after the band, up to the next occupied cell, the
    grid's edge or ``range_m``; the ``*_mapped_m`` fields are how much of
    each run is KNOWN free rather than unknown (the far side of a house's
    exterior wall has none). With no occupied cell in range the heading is
    open: the pocket is reported as the beyond too, and the barrier is zero.
    A band that reaches the range is returned as ``range_m`` with nothing
    beyond.
    """
    step = world.resolution / 2.0
    pocket = barrier = beyond = 0.0
    pocket_mapped = beyond_mapped = 0.0
    phase = "pocket"
    distance = skip_m
    while distance <= range_m:
        gx, gy = world.world_to_grid(xy[0] + distance * math.cos(heading), xy[1] + distance * math.sin(heading))
        if not world.in_bounds(gx, gy):
            break
        blocked = world.is_occupied(gx, gy)
        known = world.is_free(gx, gy)
        if phase == "pocket":
            if blocked:
                phase = "barrier"
            else:
                pocket += step
                pocket_mapped += step if known else 0.0
        if phase == "barrier":
            if blocked:
                barrier += step
            else:
                phase = "beyond"
        if phase == "beyond":
            if blocked:
                break
            beyond += step
            beyond_mapped += step if known else 0.0
        distance += step
    if phase == "pocket":
        beyond, beyond_mapped = pocket, pocket_mapped
    return RayProfile(pocket, min(barrier, range_m), beyond, pocket_mapped, beyond_mapped)


class EscapeProbe:
    """The fallback's rung before the idle turn: one probing step out of a pocket the planner cannot leave."""

    def __init__(self, policy, settings: Optional[EscapeSettings] = None):
        self.policy = policy
        self.settings = settings or EscapeSettings()
        self.stats = {"probes": 0, "turns": 0, "steps": 0, "blocked": 0}
        self._blocked: Dict[Tuple[int, int, int, int], int] = {}   # (floor, cell x, cell y, heading bin) -> action
        self.last: Optional[dict] = None                           # the last probe, for the record

    # -- memory ---------------------------------------------------------------
    def _key(self, obs, world, heading):
        p = self.policy
        gx, gy = world.world_to_grid(obs.pose.x, obs.pose.y)
        turn = p.episode.action_spec.turn_angle_rad
        return (int(p.mapping.floor_id), int(gx), int(gy), int(round(normalize_angle(heading) / turn)))

    def note_blocked(self, obs, world=None):
        """The simulator blocked a forward step: remember the cell and heading it was taken on."""
        world = world if world is not None else self.policy.mapping.worlds.get(self.policy.mapping.floor_id)
        if world is None:
            return
        self._blocked[self._key(obs, world, obs.pose.yaw)] = int(obs.step)
        self.stats["blocked"] += 1

    def _remembered(self, obs, world, heading):
        when = self._blocked.get(self._key(obs, world, heading))
        return when is not None and int(obs.step) - when < self.settings.memory_actions

    # -- the probe -------------------------------------------------------------
    def candidates(self, obs, world) -> List[dict]:
        """Every turn heading worth a step, best first: mapped floor on the far side before unknown alone; within that, a thin band with much floor beyond or a long free run, few turns to face it."""
        p, s = self.policy, self.settings
        spec = p.episode.action_spec
        turn = spec.turn_angle_rad
        turns = max(1, int(round(2.0 * math.pi / turn)))
        skip = p.settings.body_radius_m
        xy = (obs.pose.x, obs.pose.y)
        out = []
        for k in range(turns):
            offset = k if k <= turns // 2 else k - turns          # signed turns from the current yaw
            heading = normalize_angle(obs.pose.yaw + offset * turn)
            ray = ray_profile(world, xy, heading, skip, s.range_m)
            pocket, barrier, beyond = ray.pocket_m, ray.barrier_m, ray.beyond_m
            thin = barrier <= s.max_barrier_m and beyond >= s.min_beyond_m
            open_run = pocket >= s.min_pocket_m
            if (not thin and not open_run) or self._remembered(obs, world, heading):
                continue
            gain = max(beyond - 2.0 * barrier if thin else -math.inf, pocket if open_run else -math.inf)
            score = gain - 0.05 * abs(offset)
            # The floor the step leads onto: past the band for a probe, along the pocket for an open run.
            mapped = max(ray.beyond_mapped_m if thin else 0.0, ray.pocket_mapped_m if open_run else 0.0)
            out.append({"heading": float(heading), "turns": int(offset), "pocket_m": round(float(pocket), 2),
                        "barrier_m": round(float(barrier), 2), "beyond_m": round(float(beyond), 2),
                        "mapped_m": round(float(mapped), 2),
                        "rule": "thin_barrier" if thin else "open_run", "score": float(score)})
        # The far side of an exterior wall is never mapped; a heading onto known floor goes first (00835/000064).
        out.sort(key=lambda c: (c["mapped_m"] <= 0.0, -c["score"]))
        return out

    def plan(self, obs, world) -> Optional[NavigationCommand]:
        """Turn toward the best heading, or step along it; None when no heading is worth a step."""
        if not self.settings.enabled:
            return None
        found = self.candidates(obs, world)
        if not found:
            self.last = None
            return None
        best = found[0]
        self.stats["probes"] += 1
        spec = self.policy.episode.action_spec
        info = {"kind": "escape", "escape": dict(best, candidates=len(found)),
                "reason": "boxed in on the inflated map; probing the thinnest barrier by a forward step"}
        self.last = dict(best, step=int(obs.step))
        if abs(normalize_angle(best["heading"] - obs.pose.yaw)) > spec.turn_angle_rad / 2.0:
            self.stats["turns"] += 1
            return NavigationCommand.hold(final_yaw=best["heading"], info=info)
        self.stats["steps"] += 1
        reach = 2.0 * spec.forward_step_m
        ahead = (obs.pose.x + reach * math.cos(obs.pose.yaw), obs.pose.y + reach * math.sin(obs.pose.yaw))
        return NavigationCommand.follow([(float(obs.pose.x), float(obs.pose.y)), (float(ahead[0]), float(ahead[1]))],
                                        info=info)

    def diagnostics(self):
        return {"settings": {"enabled": self.settings.enabled, "range_m": self.settings.range_m,
                             "max_barrier_m": self.settings.max_barrier_m, "min_beyond_m": self.settings.min_beyond_m,
                             "min_pocket_m": self.settings.min_pocket_m, "memory_actions": self.settings.memory_actions},
                "stats": dict(self.stats), "last": self.last}
