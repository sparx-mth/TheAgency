"""Informative glances along a route: where to stop and look, and the look itself.

The map is observed free space, and the camera's cone is narrow: an agent
walking a corridor sees the corridor. Doors on either side pass through
the edge of the frame or not at all, and the recordings of 2026-10-04 show
it -- at Hanson's action 35 an open door stood a step to the left of a
committed route and was never looked into. Turning in place costs no path
length (nothing under SPL), but under the 500-action budget a full circle
every few steps is unaffordable, so the question is *where* along the route
a look pays: a step too early and the door frame hides the room behind it,
a step too late and the frame hides it again; a left look is six actions, a
look to both sides twelve -- the price of a full circle, which sees more.

This module schedules at most one such **glance** on the route in force:

1. **Candidates** are the route's points ahead of the agent, one every
   ``stride_m`` up to ``horizon_m``, each with the route's heading there.
2. **Gain** per candidate is the unknown floor a glance from there would
   sweep -- left, right or the full circle
   (:func:`~sparx_agency.core.planning.exploration.view_gain.glance_gains`)
   -- minus what the walk reveals anyway: the forward cones along the whole
   route ahead. A doorway's room counts once, at the point it shows best.
3. **Choice**: the candidate and kind of greatest gain per action, if the
   gain clears ``min_gain_m2`` and the rate ``min_gain_per_action_m2``; a
   side glance costs the turns out and back, a circle as many turns as the
   camera's field of view needs to see the whole of it (ten at 30 degrees
   with a 79-degree camera -- not twelve). A candidate far along the route
   is discounted (``lookahead_discount``): a glance the agent can take now
   is worth more than one of the same rate planned six metres on, which
   the route's changes as often re-plan away as take.
4. **The look** begins when the agent reaches the chosen point: one in-place
   turn per action, toward the side (and back, which the route follower
   does by itself when the glance ends) or all the way round, read off the
   measured yaw as the room scan's rotation is. It is a *suspended phase*
   (``discovery.SUSPENDED_PHASES``): the room-search loop is not ticked and
   not charged, the supervisor's clocks pause, and the committed route's
   watchdogs are told how long the pause was. **A completed full glance is
   a completed look-around** and goes into the scan ledger
   (:class:`~room_scans.RoomScanLedger`, ``source="glance"``, since
   2026-10-13): the room the agent stood in is finished by it, exactly as a
   room scan finishes a room, and no second circle is turned a metre away.

A target sighting or a committed stair traversal ends a glance at once;
the warm-up, a peek's look, a scan's rotation and the target takeover never
start one; none is taken within ``scanned_m`` of a spot a full rotation
already stood on. The candidates are re-scored every ``revalue_actions`` and
whenever the route changes, and the map's growth as the agent walks moves
the best point with it -- the glance is taken where it is worth most *now*,
not where it was worth most three actions ago.

**Cue glances** (since 2026-10-05) are the second kind: the geometric
scorer values unknown floor, and a small room beside the route shows it
little, but the detector sees the room's furniture at the edge of the
frame. A confident box cut off by the frame's left or right edge is a
cue to turn that way -- enough turns to centre the object and one more,
not a right angle (:meth:`GlanceScheduler._cue`). The target's own class
and its home objects (a sink, for a toilet) are cues outright; any other
class only while that side still holds unknown floor. Hanson 2026-10-05,
action 101: a bathroom vanity read as ``cabinet 0.84`` on the left edge,
two right glances were taken toward a larger unknown, and the toilet
beside the vanity was reached 250 actions later. The unknown the scorer
counts excludes what the sight ledger has settled
(:class:`~sightlines.SightLedger`): a window is no longer worth a look
for the garden behind it.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from sparx_agency.core.common.types import normalize_angle
from sparx_agency.core.planning.exploration.view_gain import (
    UnknownView, camera_rotation_turns, cone_from_camera, glance_gains)
from sparx_agency.core.planning.objnav.types.command import NavigationCommand
from sparx_agency.tasks.planning.objnav_benchmark_runtime.methods.room_priors import home_object

LEFT, RIGHT, FULL = "left", "right", "full"


@dataclass(frozen=True)
class GlanceSettings:
    """What a glance is worth and when one is taken. Distances in metres, counts in actions.

    Attributes:
        enabled: Schedule glances at all (the ablation is ``False``).
        horizon_m: How far ahead along the route candidates are scored.
        stride_m: Spacing of candidate points along the route.
        arrival_m: Within this of the chosen point the glance begins.
        min_gain_m2: Least unknown floor a glance must promise beyond what
            the walk reveals.
        min_gain_per_action_m2: ... and per action it costs: a six-action
            side glance needs 3 m2 at the default, a twelve-action circle
            6 m2. The bar is set where walking stops being the better buy:
            six forward steps into unknown space sweep the cone over
            1.5 m of new floor, around 10 m2 in a room; a doorway beside
            the route shows nothing to the walk and 5-15 m2 to a glance
            (the Hanson routes of 2026-10-04 replayed on their final maps:
            14.8 m2 at action 33, 7.6 m2 at 164).
        side_rad: How far a side glance turns (a right angle).
        cooldown_actions: After a glance, no new one for this many actions.
            Four (eight until 2026-10-13): the Darden recording
            ``ec8d17aeaaaa`` took a two-turn cue glance at action 51 and the
            eight-action cooldown then swallowed a 15 m2 right glance whose
            point the agent walked past at action 56.
        revalue_actions: Candidates are re-scored at most this often while
            the route stands; a new route is scored at once.
        lookahead_discount: How much a candidate's rate is discounted for
            lying far ahead along the route: the score compared is
            ``rate * (1 - discount * along / horizon_m)``, so a glance the
            agent can take now beats one of equal rate planned six metres
            on -- a bird in the hand; the route changes, the map grows under
            it and a glance planned far ahead is as often re-planned away as
            taken (32 scheduled, 4 geometric glances taken in the Darden
            recording). 0 restores the plain rate order.
        max_range_m: The farthest a ray is followed; the camera's depth
            range when None.
        blind_m: The floor under the camera that no glance reveals.
        scanned_m: No glance within this of a completed full rotation (the
            warm-up, a room scan, a full glance -- the scan ledger): whatever
            is still unknown from there is beyond range or behind furniture,
            and a second look from nearly the same spot reveals none of it.
            1.5 m since 2026-10-13: the Darden recording glanced right for
            4.4 m2 of optimistic gain 1.0 m from a full glance it had taken
            ten actions earlier, and turned back for three more.
        repeat_m: No side glance whose sector a completed look already swept
            from within this distance (the scheduler's own memory of every
            side and cue glance: where it stood and the bearings its frames
            covered, the field of view included). The re-run of that
            recording looked left through the same doorway at the same
            dining room twice, 0.25 m and eight actions apart (actions 60
            and 68): the unknown beyond a doorway stays unknown after the
            look -- beyond range, or a hole in the mesh -- and the
            optimistic scorer values it again every time.
        cue_enabled: Take a **cue glance** toward a confident detection cut
            off by the frame's left or right edge (since 2026-10-05). The
            geometric scorer values unknown floor, and a small room beside
            the route shows little of it; the eye catches the object at the
            edge of vision and turns to it. Hanson action 101: a bathroom
            vanity read as ``cabinet 0.84`` at the left edge of the frame,
            the scheduler glanced right twice (toward a larger unknown),
            and the toilet beside the vanity was found 250 actions later.
        cue_confidence: Least detector confidence of a box that is a cue.
        cue_border_px: A box within this of the frame's edge is cut off.
        cue_min_box_frac: Least share of the frame's height the box must
            span: a close, substantial object (the vanity at action 101
            spanned 40 %), not a speck at the end of a corridor.
        cue_min_gain_m2: Unknown floor the cue's side must still hold for
            a cue of an ordinary class -- and never less than the geometric
            glance's own rate floor over the cue's actions out and back
            (``min_gain_per_action_m2 * 2 * turns``: 3 m2 for three turns),
            since 2026-10-13: the Darden re-run took nine cue glances of
            1.7-2.6 m2 at six actions each, a chair or a cabinet at the edge
            of a room already mapped; the target's own class and its home
            objects (:data:`room_priors.HOME_OBJECTS`) need none.
        cue_max_turns: Most turns a cue glance makes: enough to centre the
            object and see what stands beside it, not a right angle.
        cue_repeat_m: No second cue glance for the same class and side
            within this of one already taken.
    """

    enabled: bool = True
    horizon_m: float = 6.0
    stride_m: float = 0.5
    arrival_m: float = 0.35
    min_gain_m2: float = 3.0
    min_gain_per_action_m2: float = 0.5
    side_rad: float = math.pi / 2
    cooldown_actions: int = 4
    revalue_actions: int = 2
    lookahead_discount: float = 0.5
    max_range_m: Optional[float] = 5.0
    blind_m: float = 0.6
    scanned_m: float = 1.5
    repeat_m: float = 1.0
    cue_enabled: bool = True
    cue_confidence: float = 0.4
    cue_border_px: int = 8
    cue_min_box_frac: float = 0.3
    cue_min_gain_m2: float = 2.0
    cue_max_turns: int = 3
    cue_repeat_m: float = 2.0

    def __post_init__(self):
        if type(self.enabled) is not bool or type(self.cue_enabled) is not bool:
            raise ValueError("glances.enabled and glances.cue_enabled must be bools")
        for name in ("horizon_m", "stride_m", "arrival_m", "min_gain_m2", "min_gain_per_action_m2", "side_rad", "blind_m",
                     "scanned_m", "repeat_m", "cue_min_gain_m2", "cue_repeat_m"):
            value = getattr(self, name)
            if isinstance(value, bool) or not math.isfinite(value) or value <= 0:
                raise ValueError("glances.%s must be positive and finite" % name)
        if (isinstance(self.lookahead_discount, bool) or not math.isfinite(self.lookahead_discount)
                or not 0.0 <= self.lookahead_discount < 1.0):
            raise ValueError("glances.lookahead_discount must lie in [0, 1)")
        for name in ("cooldown_actions", "revalue_actions", "cue_max_turns"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError("glances.%s must be a positive integer" % name)
        if type(self.cue_border_px) is not int or self.cue_border_px < 0:
            raise ValueError("glances.cue_border_px must be a non-negative integer")
        for name in ("cue_confidence", "cue_min_box_frac"):
            value = getattr(self, name)
            if isinstance(value, bool) or not math.isfinite(value) or not 0 < value <= 1:
                raise ValueError("glances.%s must lie in (0, 1]" % name)
        if self.max_range_m is not None and (not math.isfinite(self.max_range_m) or self.max_range_m <= self.blind_m):
            raise ValueError("glances.max_range_m must exceed blind_m")
        if self.side_rad > math.pi:
            raise ValueError("glances.side_rad must be at most pi")


@dataclass(frozen=True)
class GlanceCandidate:
    """One scored point of the route.

    Attributes:
        index: The candidate's position in the resampled route (0 = here).
        xy: Where the glance would stand.
        heading: The route's heading there.
        along_m: Route distance from the agent.
        left_m2 / right_m2 / full_m2: The gains.
    """

    index: int
    xy: Tuple[float, float]
    heading: float
    along_m: float
    left_m2: float
    right_m2: float
    full_m2: float


@dataclass(frozen=True)
class GlancePlan:
    """The glance chosen on the route in force."""

    candidate: GlanceCandidate
    kind: str
    gain_m2: float
    actions: int

    @property
    def rate(self) -> float:
        return self.gain_m2 / max(1, self.actions)


class GlanceScheduler:
    """Scores the route in force for a glance, and performs the one it chose.

    Attributes:
        plan_in_force: The :class:`GlancePlan` waiting for the agent to reach
            its point, or None.
        active: The glance being performed, or None.
        events: One entry per scheduled, started, completed or aborted glance.
        stats: Counters for the record.
    """

    def __init__(self, policy, settings=None):
        self.policy = policy
        self.settings = settings or GlanceSettings()
        self.plan_in_force: Optional[GlancePlan] = None
        self.active: Optional[Dict] = None
        self.candidates: Tuple[GlanceCandidate, ...] = ()
        self.events: List[Dict] = []
        self.stats = {"evaluations": 0, "scheduled": 0, "started": 0, "completed": 0, "aborted": 0,
                      "turns": 0, "gain_m2": 0.0, "cues": 0, "cues_repeated": 0, "cues_without_gain": 0,
                      "cues_placed_elsewhere": 0, "rotations_recorded": 0}
        self._last_eval_step = -10 ** 9
        self._last_glance_step = -10 ** 9
        self._route_key = None
        self._cues_taken: List[Tuple[str, str, Tuple[float, float]]] = []
        #: Every completed side or cue glance on this episode: where it stood and the
        #: bearings its frames covered, ``(xy, centre_rad, half_rad, step)``.
        self._looked: List[Tuple[Tuple[float, float], float, float, int]] = []

    # -- the wrapper ----------------------------------------------------------
    def apply(self, obs, world, command):
        """The command for this action: a glance's turn, or ``command`` as the search produced it."""
        if not self.settings.enabled or command is None or command.stop:
            return command
        if self.active is not None:
            return self.continue_look(obs, world) or command
        if not command.waypoints:
            return command
        if not self._may_glance(obs):
            return command
        cue = self._cue(obs, world)
        if cue is not None:
            return self._start_cue(obs, cue)
        route = self._route_points()
        if route is None:
            return command
        key = (tuple(round(v, 2) for v in route[-1]), len(route))
        if key != self._route_key or obs.step - self._last_eval_step >= self.settings.revalue_actions:
            self._route_key = key
            self._evaluate(obs, world, route)
        plan = self.plan_in_force
        if plan is None:
            return command
        here = (float(obs.pose.x), float(obs.pose.y))
        if math.dist(here, plan.candidate.xy) <= self.settings.arrival_m or self._passed(route, here, plan):
            return self._start(obs, plan)
        return command

    def continue_look(self, obs, world=None):
        """The next turn of the glance in force, or None the action it is complete (the search resumes at once)."""
        if self.active is None:
            return None
        return self._continue(obs, world)

    def abort(self, obs, reason):
        """End the glance in force, if any, without finishing it (a target sighting, a stair traversal)."""
        if self.active is None:
            return
        self.stats["aborted"] += 1
        self._log(obs, "glance_aborted", kind=self.active["kind"], reason=reason, turns=self.active["turns"])
        self._finish_pause(self.active["turns"])
        self.active = None
        self.plan_in_force = None
        self._last_glance_step = int(obs.step)

    # -- gating ---------------------------------------------------------------
    def _may_glance(self, obs):
        p = self.policy
        if getattr(p, "_action_owner", "search") != "search":
            return False                                           # warm-up, a peek, a return leg
        if getattr(p, "warmup_actions", 0) < getattr(p, "warmup_turns", getattr(p.settings, "warmup_steps", 0)):
            return False
        closing = getattr(p, "closing", None)
        if closing is not None and getattr(closing, "active", False):
            return False
        if getattr(p, "_target_xy", None) is not None:
            return False
        building = getattr(p, "building", None)
        if building is not None and getattr(building, "traversing", False):
            return False
        if obs.step - self._last_glance_step < self.settings.cooldown_actions:
            return False
        return True

    def _route_points(self):
        path = getattr(self.policy.route_memory, "path", None)
        if path is None:
            return None
        points = [(float(q.x), float(q.y)) if hasattr(q, "x") else (float(q[0]), float(q[1]))
                  for q in getattr(path, "points", path)]
        return points if len(points) >= 2 else None

    # -- scoring --------------------------------------------------------------
    def _view(self, obs, world):
        camera = obs.camera
        k = camera.intrinsics
        max_range = self.settings.max_range_m
        if max_range is None or not math.isfinite(max_range):
            max_range = float(camera.max_depth_m)
        max_range = min(max_range, float(camera.max_depth_m)) if math.isfinite(camera.max_depth_m) else max_range
        sight = getattr(self.policy, "sight", None)
        resolved = sight.resolved(world) if sight is not None else None
        return UnknownView(world, cone_from_camera(k.width, k.fx, max_range, self.settings.blind_m), resolved=resolved)

    def _resample(self, obs, route):
        """Candidate points: the agent's position, then one every ``stride_m`` along the route ahead."""
        here = (float(obs.pose.x), float(obs.pose.y))
        nearest = min(range(len(route)), key=lambda i: math.dist(route[i], here))
        ahead = [here] + list(route[nearest + 1:])
        out = []
        along, since = 0.0, self.settings.stride_m
        for i in range(len(ahead)):
            if i > 0:
                seg = math.dist(ahead[i - 1], ahead[i])
                along += seg
                since += seg
            if along > self.settings.horizon_m:
                break
            if i == 0 or since >= self.settings.stride_m - 1e-9:
                nxt = ahead[i + 1] if i + 1 < len(ahead) else None
                prv = ahead[i - 1] if i > 0 else None
                if nxt is not None and math.dist(nxt, ahead[i]) > 1e-6:
                    heading = math.atan2(nxt[1] - ahead[i][1], nxt[0] - ahead[i][0])
                elif prv is not None and math.dist(prv, ahead[i]) > 1e-6:
                    heading = math.atan2(ahead[i][1] - prv[1], ahead[i][0] - prv[0])
                else:
                    heading = float(obs.pose.yaw)
                out.append((len(out), ahead[i], heading, along))
                since = 0.0
        return out

    def _evaluate(self, obs, world, route):
        """Score every candidate and keep the best glance, or none."""
        s = self.settings
        self.stats["evaluations"] += 1
        self._last_eval_step = int(obs.step)
        view = self._view(obs, world)
        points = self._resample(obs, route)
        walk = np.zeros(world.grid.shape, dtype=bool)
        for _, xy, heading, _ in points:
            walk |= view.unknown_ahead(xy, heading)
        scanned = self._scanned_points()
        turn = float(self.policy.episode.action_spec.turn_angle_rad)
        side_actions = 2 * int(math.ceil(s.side_rad / turn - 1e-9))
        full_actions = self._full_turns()
        half_fov = view.cone.half_fov_rad
        candidates, best, best_score = [], None, None
        for index, xy, heading, along in points:
            if any(math.dist(xy, point) <= s.scanned_m for point in scanned):
                continue                                           # a full rotation stood here: nothing more to see
            gain = glance_gains(view, xy, heading, s.side_rad, already=walk)
            candidate = GlanceCandidate(index, (float(xy[0]), float(xy[1])), float(heading), float(along),
                                        gain.left_m2, gain.right_m2, gain.full_m2)
            candidates.append(candidate)
            # A bird in the hand: the rate of a glance planned far along the route is
            # discounted, because the route and the map change before the agent gets there.
            discount = 1.0 - s.lookahead_discount * min(1.0, float(along) / s.horizon_m)
            for kind, value, actions in ((LEFT, gain.left_m2, side_actions), (RIGHT, gain.right_m2, side_actions),
                                         (FULL, gain.full_m2, full_actions)):
                if value < s.min_gain_m2 or value / actions < s.min_gain_per_action_m2:
                    continue
                if kind != FULL and self._already_looked(xy, heading + (0.5 if kind == LEFT else -0.5) * s.side_rad,
                                                         0.5 * s.side_rad + half_fov):
                    continue                                       # a look from beside here swept that sector already
                plan = GlancePlan(candidate, kind, float(value), int(actions))
                score = plan.rate * discount
                if best is None or score > best_score + 1e-9 or (abs(score - best_score) <= 1e-9
                                                                and plan.candidate.along_m < best.candidate.along_m):
                    best, best_score = plan, score
        self.candidates = tuple(candidates)
        previous = self.plan_in_force
        self.plan_in_force = best
        if best is not None and (previous is None or previous.kind != best.kind
                                 or math.dist(previous.candidate.xy, best.candidate.xy) > s.stride_m / 2):
            self.stats["scheduled"] += 1
            self._log(obs, "glance_scheduled", kind=best.kind, gain_m2=round(best.gain_m2, 2), actions=best.actions,
                      along_m=round(best.candidate.along_m, 2), xy=[round(v, 2) for v in best.candidate.xy],
                      candidates=len(candidates))

    def _scanned_points(self):
        """Where completed rotations stood on this storey (the scan ledger), as ``(x, y)`` pairs."""
        ledger = getattr(self.policy, "scans", None)
        if ledger is None or not hasattr(ledger, "on_floor"):
            return []
        return [tuple(record["xy"]) for record in ledger.on_floor()]

    def _already_looked(self, xy, centre_rad, half_rad, tolerance_rad=math.radians(15.0)):
        """Whether a completed side or cue glance within ``repeat_m`` of ``xy`` swept the sector ``centre +- half``.

        A sector is covered when it lies inside the remembered one, with
        ``tolerance_rad`` of slack at the edges: the frames of a look from a
        step away showed the same doorway, and the unknown beyond it that
        is still unknown is beyond range or behind the jamb -- a second look
        from beside the first reveals none of it.
        """
        for where, seen_centre, seen_half, _ in self._looked:
            if math.dist(where, xy) > self.settings.repeat_m:
                continue
            offset = abs(normalize_angle(float(centre_rad) - seen_centre))
            if offset + float(half_rad) <= seen_half + tolerance_rad:
                return True
        return False

    def _remember_look(self, obs, active):
        """Record the bearings a completed side or cue glance covered, for :meth:`_already_looked`."""
        k = obs.camera.intrinsics
        half_fov = math.atan(0.5 * float(k.width) / float(k.fx))
        start, end = float(active["start_yaw"]), float(obs.pose.yaw)
        swept = normalize_angle(end - start)
        centre = normalize_angle(start + 0.5 * swept)
        self._looked.append((tuple(active["xy_exact"]), float(centre), 0.5 * abs(swept) + half_fov, int(obs.step)))

    def _full_turns(self):
        """Turns of a full glance: what the camera's field of view needs to see the whole circle."""
        p = self.policy
        if callable(getattr(p, "rotation_turns", None)):
            return int(p.rotation_turns())
        turn = float(p.episode.action_spec.turn_angle_rad)
        k = getattr(getattr(p.episode, "camera", None), "intrinsics", None)
        if k is not None:
            return camera_rotation_turns(k.width, k.fx, turn)
        return int(math.ceil(2 * math.pi / turn - 1e-9))

    @staticmethod
    def _passed(route, here, plan):
        """Whether the agent has walked past the plan's point without coming within ``arrival_m`` of it."""
        nearest = min(range(len(route)), key=lambda i: math.dist(route[i], here))
        target = min(range(len(route)), key=lambda i: math.dist(route[i], plan.candidate.xy))
        return nearest > target

    # -- cues: an object cut off at the edge of the frame ----------------------------
    #: Projection verdicts that put a box somewhere a glance toward it cannot help.
    _CUE_EXCLUDED_STATUSES = frozenset(("wrong_floor_height", "unsupported_floor_association", "outside_observed_map"))

    def _cue(self, obs, world):
        """The best cue in this frame -- ``(side, class, conf, turns, gain_m2)`` -- or None.

        A detection at least ``cue_confidence`` confident, spanning at least
        ``cue_min_box_frac`` of the frame's height, whose box is cut by the
        frame's left or right edge; the target's own class and its
        home objects qualify outright, any other class only while the side
        still holds ``cue_min_gain_m2`` of unknown floor (a chair at the
        edge of a room already mapped is not worth two turns). The glance
        turns enough to centre the object and one turn more, at most
        ``cue_max_turns``; a class already glanced at on that side within
        ``cue_repeat_m`` is not a cue again.
        """
        s = self.settings
        if not s.cue_enabled:
            return None
        p = self.policy
        detections = getattr(getattr(p, "perception", None), "detections", None) or ()
        if not detections:
            return None
        # This frame's projection verdicts, by box: a box perception placed on another
        # storey, outside the map or on a candidate the takeover refuses is no cue
        # (Ranchester 2026-10-05, actions 53-59: a sofa on the storey below at the right
        # edge of a landing, three cue turns and three back, every two metres).
        placed_elsewhere = set()
        for row in getattr(getattr(p, "perception", None), "projections", None) or ():
            if (row.get("status") in self._CUE_EXCLUDED_STATUSES or row.get("target_evidence") == "refused_by_takeover"):
                placed_elsewhere.add(tuple(round(float(v), 3) for v in row.get("xyxy", ())))
        k = obs.camera.intrinsics
        turn = float(p.episode.action_spec.turn_angle_rad)
        here = (float(obs.pose.x), float(obs.pose.y))
        target = getattr(p, "target", None)
        view = None
        best = None
        for detection in detections:
            conf = float(getattr(detection, "conf", 0.0))
            if conf < s.cue_confidence:
                continue
            cls = str(getattr(detection, "cls", "") or "")
            x1, y1, x2, y2 = (float(v) for v in detection.xyxy)
            if (y2 - y1) < s.cue_min_box_frac * float(k.height):
                continue
            if tuple(round(float(v), 3) for v in detection.xyxy) in placed_elsewhere:
                self.stats["cues_placed_elsewhere"] += 1
                continue
            if x1 <= s.cue_border_px:
                side = LEFT
            elif x2 >= float(k.width) - s.cue_border_px:
                side = RIGHT
            else:
                continue
            if any(c == cls and sd == side and math.dist(xy, here) <= s.cue_repeat_m for c, sd, xy in self._cues_taken):
                self.stats["cues_repeated"] += 1
                continue
            u = 0.5 * (x1 + x2)
            bearing = abs(math.atan2(-(u - k.cx) / k.fx, 1.0))
            turns = max(1, min(s.cue_max_turns, int(math.ceil(bearing / turn - 1e-9)) + 1))
            direction = 1.0 if side == LEFT else -1.0
            own = bool(target is not None and (
                (callable(getattr(target, "accepts", None)) and target.accepts(cls)) or home_object(target, cls)))
            half_fov = math.atan(0.5 * float(k.width) / float(k.fx))
            if self._already_looked(here, float(obs.pose.yaw) + direction * 0.5 * turns * turn,
                                    0.5 * turns * turn + half_fov):
                self.stats["cues_repeated"] += 1                   # those frames were taken from beside here already
                continue
            gain = None
            if not own:
                if view is None:
                    view = self._view(obs, world)
                gains = glance_gains(view, here, float(obs.pose.yaw), turns * turn)
                gain = gains.left_m2 if side == LEFT else gains.right_m2
                # The cue costs its turns out and back like any side glance, and buys no
                # less unknown floor per action than the geometric scorer demands.
                floor = max(s.cue_min_gain_m2, s.min_gain_per_action_m2 * 2 * turns)
                if gain < floor:
                    self.stats["cues_without_gain"] += 1
                    continue
            rank = (1 if own else 0, conf)
            if best is None or rank > best[0]:
                best = (rank, (side, cls, conf, turns, gain))
        return None if best is None else best[1]

    def _start_cue(self, obs, cue):
        side, cls, conf, turns, gain = cue
        p = self.policy
        turn = float(p.episode.action_spec.turn_angle_rad)
        yaw = float(obs.pose.yaw)
        direction = 1.0 if side == LEFT else -1.0
        here = (float(obs.pose.x), float(obs.pose.y))
        self.active = dict(kind=side, direction=direction, target=float(normalize_angle(yaw + direction * turns * turn)),
                           started=int(obs.step), turns=0, swept=0.0, last_yaw=yaw, start_yaw=yaw, xy_exact=here,
                           gain_m2=0.0 if gain is None else float(gain), xy=[round(v, 2) for v in here],
                           cue=cls, cue_turns=int(turns))
        self._cues_taken.append((cls, side, here))
        self.stats["started"] += 1
        self.stats["cues"] += 1
        # The geometric plan in force survives a cue: a cue is two or three turns toward
        # an object at the frame's edge, and the glance planned along the route is still
        # owed where it was planned (the Darden recording lost a 15 m2 look that way).
        self._log(obs, "glance_started", kind=side, cue=cls, cue_conf=round(float(conf), 2), actions=int(turns),
                  gain_m2=None if gain is None else round(float(gain), 2), xy=self.active["xy"],
                  heading_deg=round(math.degrees(yaw), 1))
        return self._turn(obs)

    # -- the look -------------------------------------------------------------
    def _start(self, obs, plan):
        p = self.policy
        yaw = float(obs.pose.yaw)
        heading = plan.candidate.heading
        if plan.kind == FULL:
            direction = 1.0 if plan.candidate.left_m2 >= plan.candidate.right_m2 else -1.0
            target = None
        else:
            direction = 1.0 if plan.kind == LEFT else -1.0
            target = float(normalize_angle(heading + direction * self.settings.side_rad))
        self.active = dict(kind=plan.kind, direction=direction, target=target, started=int(obs.step),
                           turns=0, swept=0.0, last_yaw=yaw, start_yaw=yaw, gain_m2=plan.gain_m2,
                           xy=[round(v, 2) for v in plan.candidate.xy],
                           xy_exact=(float(obs.pose.x), float(obs.pose.y)))
        self.stats["started"] += 1
        self.plan_in_force = None
        self._log(obs, "glance_started", kind=plan.kind, gain_m2=round(plan.gain_m2, 2), actions=plan.actions,
                  xy=self.active["xy"], heading_deg=round(math.degrees(heading), 1))
        return self._turn(obs)

    def _continue(self, obs, world=None):
        active = self.active
        turn = float(self.policy.episode.action_spec.turn_angle_rad)
        yaw = float(obs.pose.yaw)
        active["swept"] += abs(normalize_angle(yaw - active["last_yaw"]))
        active["last_yaw"] = yaw
        done = False
        if active["kind"] == FULL:
            # The circle is seen when the field of view has covered it (ten turns at 30
            # degrees with a 79-degree camera), not when the body is back where it started.
            full_circle = self._full_turns()
            done = active["swept"] >= (full_circle - 0.5) * turn - 1e-9 or active["turns"] >= full_circle + 2
        else:
            side_turns = int(active.get("cue_turns") or math.ceil(self.settings.side_rad / turn - 1e-9))
            done = (abs(normalize_angle(active["target"] - yaw)) <= 0.5 * turn + 1e-6
                    or active["turns"] >= side_turns + 2)
        if not done:
            return self._turn(obs)
        self.stats["completed"] += 1
        self.stats["gain_m2"] += float(active["gain_m2"])
        self._log(obs, "glance_complete", kind=active["kind"], turns=active["turns"],
                  swept_deg=round(math.degrees(active["swept"]), 1), actions=int(obs.step) - active["started"])
        if active["kind"] == FULL:
            self._record_rotation(obs, world)
        else:
            self._remember_look(obs, active)
        self._finish_pause(active["turns"])
        self.active = None
        self._last_glance_step = int(obs.step)
        return None

    def _record_rotation(self, obs, world=None):
        """A completed full glance IS a completed look-around: it goes into the scan ledger.

        The ledger is the one memory of "this room is finished", and until
        2026-10-13 only the warm-up and a room scan wrote to it: the Darden
        recording ``ec8d17aeaaaa`` turned a full glance at action 61 inside
        the room it was walking into, then walked on to that room's vantage
        point and turned the same circle again at action 94 -- 33 actions
        for nothing. Recorded here, the room the agent stands in is finished
        the action the glance ends, and the loop's visit in force ends with it.
        """
        p = self.policy
        ledger = getattr(p, "scans", None)
        graph = getattr(p, "graph", None)
        if world is None:
            world = getattr(p, "last_world", None)
        if ledger is None or not hasattr(ledger, "record"):
            return
        room = None
        if graph is not None and world is not None and callable(getattr(graph, "room_at", None)):
            pid = graph.room_at(world, (float(obs.pose.x), float(obs.pose.y)))
            room = graph.registry.rooms.get(pid) if pid is not None else None
        ledger.record(obs, room, source="glance")
        self.stats["rotations_recorded"] += 1

    def _turn(self, obs):
        active = self.active
        p = self.policy
        turn = float(p.episode.action_spec.turn_angle_rad)
        yaw = float(obs.pose.yaw)
        if active["kind"] == FULL:
            step = active["direction"] * turn
        else:
            error = normalize_angle(active["target"] - yaw)
            step = max(-turn, min(turn, error))
        active["turns"] += 1
        self.stats["turns"] += 1
        p._action_owner = "glance"
        info = {"kind": "glance", "glance": active["kind"], "turn": active["turns"],
                "gain_m2": round(float(active["gain_m2"]), 2), "xy": list(active["xy"])}
        if active.get("cue"):
            info["cue"] = active["cue"]
        return NavigationCommand.hold(final_yaw=float(normalize_angle(yaw + step)), info=info)

    def _finish_pause(self, turns):
        memory = getattr(self.policy, "route_memory", None)
        if memory is not None and turns > 0 and hasattr(memory, "resume_after_pause"):
            memory.resume_after_pause(int(turns))

    # -- the record -----------------------------------------------------------
    def _log(self, obs, event, **fields):
        self.events.append(dict(step=int(obs.step), floor_id=getattr(getattr(self.policy, "mapping", None), "floor_id", None),
                                event=event, **fields))

    def state(self):
        """The glance in force (or the one planned) for the HUD and the per-step record."""
        plan = self.plan_in_force
        return {"active": None if self.active is None else {k: v for k, v in self.active.items()
                                                             if k not in ("last_yaw", "start_yaw", "xy_exact")},
                "planned": None if plan is None else {"kind": plan.kind, "gain_m2": round(plan.gain_m2, 2),
                                                      "actions": plan.actions,
                                                      "xy": [round(v, 2) for v in plan.candidate.xy],
                                                      "along_m": round(plan.candidate.along_m, 2)}}

    def diagnostics(self):
        return {"settings": asdict(self.settings), "stats": dict(self.stats), "events": list(self.events),
                "state": self.state()}
