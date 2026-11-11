"""The vantage warm-up: a greedy look-around that may walk to where the camera learns more.

The method begins every episode (and every storey first set foot on) with a
look-around before any room is chosen: the spawn room's walls and doors go
on the map, its furniture to the room labeller, and the search starts from
what it saw. The warm-up in force since the method began is ten turns in
place (``WarmupSettings.mode == "spin"``, :func:`~rooms.discovery.discover`)
-- the cheapest look there is, one action per 30 degrees, but taken where
the agent happens to stand: facing a wall at arm's length, in a corridor,
behind a wardrobe, the circle sees little, and the search starts blind.

This module is the alternative (``mode == "vantage"``): the greedy look
planner of :mod:`~sparx_agency.core.planning.object_search.exploration.greedy_look`
run unscoped within a budget of ``budget_steps`` actions -- at every action
the choice between the next sweep here and a walk to a better standing
point, by **expected useful area per action** (MapEx's viewpoint score on
the observed map with this camera's sensor model; see that module for the
rule). At most ``max_points`` walks: 0 is a greedy sweep in place, 1 "one
vantage point", more "several". The warm-up ends when no option clears
``min_rate_m2_per_action`` or the budget is spent, and the actions it did
not spend go to the search.

A spot the planner judges complete -- its circle covered by the frames, or
its uncovered arc promising no more than ``scan_skip_gain_m2`` -- goes into
the scan ledger as the spin's spot did
(:class:`~rooms.room_scans.RoomScanLedger`, ``source="warmup"``), so the
room it stands in is finished before the first room is chosen. The actions
are the episode's: every turn and step here counts against the 500 as the
spin's ten did, so a comparison between the two is on equal terms.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import math
from typing import Optional

from sparx_agency.core.planning.object_search.exploration.greedy_look import GreedyLookPlanner, LookSettings

SPIN, VANTAGE = "spin", "vantage"


@dataclass(frozen=True)
class WarmupSettings:
    """How the episode's first look is taken. Distances in metres, counts in actions.

    Attributes:
        mode: ``"spin"`` -- the look-around in place, as many turns as the
            camera needs to see the circle, bounded by
            ``RPTSettings.warmup_steps``; ``"vantage"`` -- the greedy
            planner of this module.
        budget_steps: ``vantage``: the most actions the warm-up may take,
            moves and turns together (the ``X`` of the study); the spin
            takes ten at the benchmark's camera.
        max_points: ``vantage``: how many times the warm-up may walk to a
            new standing point. 0 never moves, 1 allows one vantage point
            beside the spawn, more allow several.
        min_rate_m2_per_action: ``vantage``: the least expected useful area
            per action an option must promise to be taken; below it the
            warm-up ends and the remaining actions go to the search. 0
            spends the whole budget.
        max_travel_m: ``vantage``: how far along the floor a candidate may
            lie from the agent.
        candidate_spacing_m: ``vantage``: the lattice pitch the candidates
            are sampled on.
        min_move_m: ``vantage``: no candidate within this of a spot already
            looked around from (a shorter move shows the same room).
        detection_range_m: How far an object can be detected by the
            perception in force -- the range the gain is counted to, not
            the depth sensor's clip (``SensorModel.detection_range_m``).
            3.5 m by default, the method's own far limit for a clear view of
            a landmark (``RPTSettings.landmark_miss_max_range_m``) and the
            object-inclusion range 3D-Mem ablated to (ApexNav reads its
            frontiers as seen within 4.0 m of a 4.99 m depth clip); the
            ground-truth perception detects to the depth range, 5 m, and
            the study's arms set that.
        unknown_free_run_m: The survival prior on unknown space
            (``SensorModel.unknown_free_run_m``); None is MapEx's
            transparent unknown.
        near_horizon_m: Floor within this of a pose the agent stood at is
            as good as seen: a level camera 0.88 m up sees the floor from
            about 1.4 m out, so the unknown ring under its feet is floor
            nothing stands on (anything standing there was mapped), not
            area a walk would reveal.
        scan_skip_gain_m2: A spot whose uncovered arc promises no more than
            this is a completed look-around for the scan ledger even though
            the body did not turn through it. 1 m2: half of what one turn
            into open unknown promises under the default prior (1.8 m2), so
            an arc left unswept because every bearing in it meets a wall
            within a metre or two finishes the spot, and an arc with an
            unseen room behind it does not.
        plan_attempts: Candidates the planner is asked about per action
            before the turn option stands in.
    """

    mode: str = SPIN
    budget_steps: int = 15
    max_points: int = 1
    min_rate_m2_per_action: float = 0.25
    max_travel_m: float = 4.0
    candidate_spacing_m: float = 0.5
    min_move_m: float = 0.75
    detection_range_m: float = 3.5
    unknown_free_run_m: Optional[float] = 2.0
    near_horizon_m: float = 1.4
    scan_skip_gain_m2: float = 1.0
    plan_attempts: int = 3

    def __post_init__(self):
        if self.mode not in (SPIN, VANTAGE):
            raise ValueError("warmup.mode must be 'spin' or 'vantage'; no silent fallback")
        for name in ("budget_steps", "plan_attempts"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError("warmup.%s must be a positive integer" % name)
        if type(self.max_points) is not int or self.max_points < 0:
            raise ValueError("warmup.max_points must be a non-negative integer")
        for name in ("max_travel_m", "candidate_spacing_m", "min_move_m", "detection_range_m", "near_horizon_m"):
            value = getattr(self, name)
            if isinstance(value, bool) or not math.isfinite(value) or value <= 0:
                raise ValueError("warmup.%s must be positive and finite" % name)
        for name in ("min_rate_m2_per_action", "scan_skip_gain_m2"):
            value = getattr(self, name)
            if isinstance(value, bool) or not math.isfinite(value) or value < 0:
                raise ValueError("warmup.%s must be finite and non-negative" % name)
        if self.unknown_free_run_m is not None and (
                isinstance(self.unknown_free_run_m, bool) or not math.isfinite(self.unknown_free_run_m)
                or self.unknown_free_run_m <= 0):
            raise ValueError("warmup.unknown_free_run_m must be None or positive and finite")

    def look(self):
        """These knobs as the greedy look planner's :class:`~greedy_look.LookSettings`."""
        return LookSettings(detection_range_m=self.detection_range_m, unknown_free_run_m=self.unknown_free_run_m,
                            near_horizon_m=self.near_horizon_m, candidate_spacing_m=self.candidate_spacing_m,
                            min_move_m=self.min_move_m, max_travel_m=self.max_travel_m,
                            min_rate_m2_per_action=self.min_rate_m2_per_action, scan_skip_gain_m2=self.scan_skip_gain_m2,
                            plan_attempts=self.plan_attempts, max_points=self.max_points)


class VantageWarmup:
    """The vantage warm-up: the greedy look planner within the warm-up's budget, its complete spots in the scan ledger."""

    def __init__(self, policy, settings):
        self.policy = policy
        self.settings = settings
        self.planner = GreedyLookPlanner(policy, settings.look(), "warmup", on_spot_closed=self._spot_closed)
        self.reset()

    def reset(self):
        """A fresh warm-up: on reset and on every storey first set foot on."""
        self.planner.reset()
        self.scans_recorded = 0
        self.budget_ends = 0

    @property
    def moves(self):
        return self.planner.moves

    def plan(self, obs, world, cost):
        """One warm-up action, or None when the warm-up is over (nothing left worth an action, or the budget spent)."""
        p = self.policy
        remaining = int(p.warmup_turns) - int(p.warmup_actions)
        command = self.planner.plan(obs, world, cost, remaining)
        if command is None:
            self.finish(obs, world)
            return None
        return replace(command, info=dict(command.info, kind="warmup", mode=VANTAGE,
                                          warmup_step=int(p.warmup_actions) + 1, warmup_budget=int(p.warmup_turns)))

    def finish(self, obs, world):
        """The warm-up is over: close the spot in force and mark the policy's warm-up done (idempotent)."""
        p = self.policy
        if getattr(p, "_warmup_done", False):
            return
        self.planner.close(obs, world)
        if int(p.warmup_actions) >= int(p.warmup_turns):
            self.budget_ends += 1
        p._warmup_done = True

    def _spot_closed(self, obs, world, spot):
        """A spot whose look-around is complete is a completed scan: the room it stands in is finished."""
        if not spot.get("complete"):
            return
        p = self.policy
        ledger = getattr(p, "scans", None)
        if ledger is None or world is None:
            return
        pid = p.graph.room_at(world, tuple(spot["xy"]))
        room = p.graph.registry.rooms.get(pid) if pid is not None else None
        ledger.record(obs, room, source="warmup", xy=tuple(spot["xy"]))
        spot["recorded"] = True
        self.scans_recorded += 1
        self.planner._log(obs, "warmup_scan_recorded", xy=[round(v, 2) for v in spot["xy"]], room=pid,
                          uncovered_deg=spot.get("uncovered_deg"), remaining_gain_m2=spot.get("remaining_gain_m2"))

    def diagnostics(self):
        d = self.planner.diagnostics()
        d["stats"] = dict(d["stats"], scans_recorded=self.scans_recorded, budget_end=self.budget_ends)
        d["settings"] = asdict(self.settings)
        return d
