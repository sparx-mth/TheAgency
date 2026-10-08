"""Episode-local target takeover: no transition back to global exploration."""
from __future__ import annotations

from dataclasses import dataclass, replace
import math

from sparx_agency.core.common.types import normalize_angle
from sparx_agency.core.planning.objnav.action_converter.action_choice import pitch_action, turn_action
from sparx_agency.core.planning.objnav.types.command import NavigationCommand
from sparx_agency.tasks.planning.objnav_benchmark_runtime.methods.approach_history import ApproachHistory
from sparx_agency.tasks.planning.objnav_benchmark_runtime.methods.perception import clipped_box, observed_objects
from sparx_agency.tasks.planning.objnav_benchmark_runtime.methods.perception_cycle import contradicted_by_map
from sparx_agency.tasks.planning.objnav_benchmark_runtime.methods.sightlines import unknown_around
from sparx_agency.tasks.planning.objnav_benchmark_runtime.methods.support_surface import (
    SUPPORT_SURFACE_CLASSES, is_support_surface, supports)
from sparx_agency.tasks.planning.objnav_benchmark_runtime.methods.target_backoff import BackoffManeuver
from sparx_agency.tasks.planning.objnav_benchmark_runtime.methods.target_context import suspect
from sparx_agency.tasks.planning.objnav_benchmark_runtime.methods.target_memory import TargetMemory3D
from sparx_agency.tasks.planning.objnav_benchmark_runtime.methods.target_path import TargetApproachPath


@dataclass(frozen=True)
class TargetClosingSettings:
    confidence: float = 0.50
    # A detector's confidence in ONE object swings with the view. The Ranchester
    # couch, 3.8 m from the stair foot, read 0.68 at one heading, 0.42 one
    # 30-degree turn later with the same object centred in the frame, and 0.52
    # clipped at the next heading. The lock needs two CONSECUTIVE frames, and
    # the agent's own centring turn broke the run every time: twenty-four frames
    # with the couch in view, two twelve-step verifications released. A
    # takeover STARTS on ``confidence``; once a candidate is active, a box of
    # the target's class that projects onto its anchor is a re-sighting and
    # counts from this lower threshold -- the tracking convention: detect
    # high, associate low. A weak box that lands anywhere else is nothing.
    track_confidence: float = 0.30
    confirmation_frames: int = 2
    association_radius_m: float = 0.50
    # Within this range the camera pitch FOLLOWS THE TARGET'S ELEVATION -- the
    # close-range band of the dynamic pitch controller (``_pitch``, ``_elevation``,
    # since 2026-10-07): a target below the camera (a toilet 0.4 m high drops under a
    # level camera's lower edge from about 1.4 m in) is looked DOWN at, one above it
    # (a wall-mounted television, the top of a wardrobe) is looked UP at, by at least
    # one tilt step; beyond the band the approach is level. The name is historical:
    # until 2026-10-07 only the look-down existed, and a hardcoded LOOK_DOWN drove
    # high targets out of the frame.
    look_down_distance_m: float = 1.30
    # A target within this height of the camera is level with it: neither LOOK is
    # forced, and the verification frame falls back to whichever view keeps the box
    # inside the frame.
    elevation_band_m: float = 0.15
    terminal_distance_m: float = 1.00
    range_tolerance_m: float = 0.05
    refine_distance_m: float = 0.15
    max_verify_steps: int = 12
    max_reacquire_steps: int = 24
    max_closing_steps: int = 160
    # A box at or above this confidence starts a takeover past the rejection memory
    # of UNVERIFIED and map-contradicted releases and the release cooldown (since
    # 2026-10-07): the memory stops the same far flicker from restarting the
    # takeover, and a 0.95 toilet in plain view at 2.4 m is not a flicker -- the
    # Allensville toilet run refused nine such frames (actions 45-53) because a lock
    # the map had wrongly outvoted was remembered 2 m around it. A spot released by a
    # FAILED INSPECTION (the closing looked from close and saw nothing) or an approach
    # that stalled is not bypassed -- the detector's word from far does not beat a look
    # from near -- and neither is a boxed-in spot, where the agent itself is the
    # problem. Border clipping, incoherent depth and the map's contradiction stand.
    # 1.0 disables the override.
    override_confidence: float = 0.80
    # The semantic sanity check (``target_context.suspect``): a candidate in a room
    # whose STRONG type excludes the target, or whose measured surface height
    # contradicts its class (a counter top read as a bed), counts
    # ``context_penalty`` of its confidence toward the start threshold and needs
    # ``context_confirmation_frames`` consecutive frames from at least two
    # viewpoints ``context_baseline_m`` apart to lock. Allensville/2 and
    # Newfields/2 of the 5x3 benchmark STOPped on a kitchen counter at 0.55 m after
    # two frames from one spot. ``context_check=False`` is the former behaviour.
    context_check: bool = True
    context_penalty: float = 0.70
    context_confirmation_frames: int = 4
    context_baseline_m: float = 0.30
    # In VERIFY a fresh, aligned candidate too close for a step used to be held --
    # and a satisfied hold is idle, which the headless agent turns into a TURN that
    # moves the box out of the frame (Allensville toilet, action 77: centred at
    # 1.18 m, TURN_LEFT, lost). The verification frame is chosen to keep the box
    # whole: a LOOK_DOWN for a low target (the user's close-range policy), else the
    # one LOOK or TURN whose predicted shift leaves the box inside the frame.
    verify_keep_in_frame: bool = True
    # A box touching the image border is a partial view: its depth centroid is
    # unreliable and the clipped end of a bed reads as a sofa. Such a box may
    # not START a takeover; once locked the approach tolerates spill-over.
    border_margin_px: int = 8
    # An unlocked candidate is unverified evidence. Releasing it back to
    # exploration (remembering the spot) keeps the no-return contract for
    # locked targets only; False restores the historical episode-ending error.
    release_unverified: bool = True
    rejection_radius_m: float = 1.00
    # The rejection memory stops the SAME far flicker from restarting the takeover
    # from the same place; a candidate seen from within this range is new
    # evidence worth two frames whatever was released there before (Ranchester
    # attempt 3: the couch released at 3.8 m from the stair head was never
    # re-verified from 1 m, and the episode was lost on that).
    rejection_min_range_m: float = 2.00
    # A long object seen from far shows a different part of itself in every
    # frame, and its visible centroid moves more than half a metre between two
    # of them. The association radius grows by this much per metre of range
    # beyond two metres (3.8 m -> 0.77 m, 5 m -> 0.95 m); within two metres it
    # is the plain radius.
    association_range_gain: float = 0.15
    # After a release, no new takeover may START on a far candidate for this
    # many actions: six twelve-step verifications from one spot three metres
    # from the couch (Ranchester attempt 4) were the same failure repeated,
    # and the search has to MOVE before it is worth two more frames. A close
    # view (within ``rejection_min_range_m``) is exempt, as always.
    release_cooldown_actions: int = 15
    # The terminal inspection can deadlock on a big object at close range: the
    # heading where the detector fires shows a box clipped at the image edge
    # whose centre says "turn", and after the turn a wall of upholstery shows
    # no box at all (Ranchester couch, actions 184-196: twelve such cycles).
    # When the inspection budget runs out but the LOCKED target was seen fresh
    # within the terminal range while the agent stood here, STOP rather than
    # end the episode. False restores the historical method error.
    stop_on_exhausted_inspection: bool = True
    # When the inspection budget runs out and the LOCKED target was NOT seen
    # fresh from here at all, the lock was wrong -- a sofa from four metres
    # that is nothing from one (Ranchester same-storey run, 2026-10-04: the
    # episode ended as an agent error with the real couch 2.85 m away). The
    # candidate is released like an unverified one, its anchor remembered with
    # twice the rejection radius, and the search resumes. False restores the
    # historical episode-ending error.
    release_on_failed_inspection: bool = True
    failed_inspection_radius_factor: float = 2.0
    # A LOCKED target with no safe path used to wait for A* for ever: the agent
    # that spawned beside a bed with the plant in view 4.4 m away (Hanson
    # 2026-10-05) spun 159 actions on "no safe target path" because the floor
    # under the camera's blind radius was unknown and unknown is impassable.
    # After ``footing_after_steps`` such actions the closing asks the camera
    # for a FOOTING sweep (one LOOK_DOWN, a circle, one LOOK_UP: the floor
    # around the feet becomes known, and a path either appears or is shown
    # not to exist from here); if ``release_after_footing_steps`` more
    # pathless actions follow, the lock is released WITHOUT a rejection -- the
    # target was never disproved, the spot was -- and the search moves on,
    # refusing a new takeover on a far candidate until the agent has left the
    # spot (``boxed_in_radius_m``). ``footing_sweep=False`` drops the sweep
    # alone (a protocol without LOOK actions never has it); the release after
    # the bound stands either way.
    footing_sweep: bool = True
    footing_after_steps: int = 2
    release_after_footing_steps: int = 6
    boxed_in_radius_m: float = 1.0
    # The sweep is asked for only where the floor around the feet is still unknown:
    # at least this share of the cells within ``footing_radius_m`` of the agent.
    footing_radius_m: float = 1.2
    footing_unknown_fraction: float = 0.25
    # An approach that goes nowhere -- the agent's position has not left a circle of
    # ``approach_stall_m`` in ``approach_stall_actions`` consecutive CLOSE actions --
    # releases the lock with a rejection. A* finds a path every action; the follower
    # pushes into something the map does not show (Leonardo 2026-10-05: 160 actions
    # of MOVE_FORWARD against an unseen obstacle 4.6 m from a real sofa, until the
    # closing bound raised an agent error). 0 disables the clock.
    approach_stall_actions: int = 20
    approach_stall_m: float = 0.20
    # THE 3-D MEMORY (``target_memory.TargetMemory3D``, since 2026-10-08). The target's
    # XYZ centroid is saved the moment a takeover starts and fused, precision-weighted,
    # with every frame that projects onto it: an observation from ``r`` metres enters
    # with sigma ``memory_noise_floor_m + memory_noise_per_m * r`` (depth noise and the
    # visible-part shift of a big object both grow with range), so the near frames of
    # the approach dominate and the estimate's standard error shrinks frame by frame.
    # Everything below reads that memory: the elevation the pitch follows, the range the
    # terminal test and the history STOP measure, the spot a support surface must sit at.
    memory_noise_floor_m: float = 0.03
    memory_noise_per_m: float = 0.03
    # EXTENDED APPROACH HISTORY (``approach_history.ApproachHistory``). A target the agent
    # tracked and walked toward over at least ``approach_history_m`` of path, closing at
    # least ``approach_history_closed_m`` of range, is known: inside the terminal range
    # a cropped box, a detector drop or a budget run down is not a reason to release it.
    # A fresh in-range sighting STOPs whatever its alignment, and after
    # ``history_stop_after_missing`` consecutive terminal frames without a fresh box the
    # closing completes on the 3-D memory and STOPs -- the one deliberate exception to
    # "no STOP from remembered range alone", earned by the path. 0 path disables it.
    approach_history_m: float = 3.0
    approach_history_closed_m: float = 1.0
    history_stop_after_missing: int = 3
    # SUDDEN CLOSE PROXIMITY (``target_backoff.BackoffManeuver``). A candidate first seen
    # within ``sudden_proximity_m`` with no approach history -- the agent turned and there
    # it was -- is not closed on from that single viewpoint. Before the lock the agent
    # backs ``backoff_steps`` forward steps into space it knows to be clear (its own
    # trail, the observed grid, the floor guard, A*, the NavMesh when bound), turns back to
    # face the remembered coordinates and re-verifies from there; the whole manoeuvre is
    # bounded by ``backoff_max_actions`` (a retreat straight back is six 30-degree turns,
    # three steps and six turns back). With no clear space known it is skipped and
    # recorded, and verification proceeds in place as before. ``backoff=False`` disables it.
    backoff: bool = True
    sudden_proximity_m: float = 1.30
    backoff_steps: int = 3
    backoff_max_actions: int = 20
    # SUPPORT-SURFACE RESILIENCE (``support_surface``). At close range a mounted or
    # supported object leaves the frame before its support does (a television on a
    # dresser becomes the dresser). A box of a ``support_surface_classes`` label whose
    # centroid lies under the memory's footprint and at or below its height is spatial
    # evidence that the tracked thing is still there: the inspection is not released as
    # "saw nothing", the map's vote for the surface is not a contradiction, and once the
    # inspection budget is spent with that evidence fresh and the memory inside the
    # terminal range, the closing STOPs. False restores the former releases.
    support_surface_resilience: bool = True
    support_surface_classes: tuple = SUPPORT_SURFACE_CLASSES

    def __post_init__(self):
        if isinstance(self.confidence, bool) or not math.isfinite(self.confidence) or not 0 < self.confidence <= 1:
            raise ValueError("Target confidence must be in (0, 1]")
        if (isinstance(self.track_confidence, bool) or not math.isfinite(self.track_confidence)
                or not 0 < self.track_confidence <= self.confidence):
            raise ValueError("track_confidence must be in (0, confidence]")
        for key in ("association_radius_m", "look_down_distance_m", "terminal_distance_m", "range_tolerance_m",
                    "refine_distance_m", "rejection_radius_m", "rejection_min_range_m", "failed_inspection_radius_factor",
                    "elevation_band_m"):
            value = getattr(self, key)
            if isinstance(value, bool) or not math.isfinite(value) or value <= 0:
                raise ValueError("%s must be positive and finite" % key)
        if isinstance(self.association_range_gain, bool) or not math.isfinite(self.association_range_gain) or self.association_range_gain < 0:
            raise ValueError("association_range_gain must be finite and non-negative")
        if type(self.release_cooldown_actions) is not int or self.release_cooldown_actions < 0:
            raise ValueError("release_cooldown_actions must be a non-negative int")
        if self.terminal_distance_m + self.range_tolerance_m > self.look_down_distance_m:
            raise ValueError("Terminal inspection must be within the look-down distance")
        for key in ("confirmation_frames", "max_verify_steps", "max_reacquire_steps", "max_closing_steps"):
            if type(getattr(self, key)) is not int or getattr(self, key) < (2 if key == "confirmation_frames" else 1):
                raise ValueError("Invalid target-closing action/frame bound: %s" % key)
        if type(self.border_margin_px) is not int or self.border_margin_px < 0:
            raise ValueError("border_margin_px must be a non-negative int")
        for key in ("release_unverified", "stop_on_exhausted_inspection", "release_on_failed_inspection", "footing_sweep"):
            if type(getattr(self, key)) is not bool:
                raise ValueError("%s must be a bool" % key)
        for key in ("footing_after_steps", "release_after_footing_steps"):
            if type(getattr(self, key)) is not int or getattr(self, key) < 1:
                raise ValueError("%s must be a positive int" % key)
        if type(self.approach_stall_actions) is not int or self.approach_stall_actions < 0:
            raise ValueError("approach_stall_actions must be a non-negative int")
        if isinstance(self.approach_stall_m, bool) or not math.isfinite(self.approach_stall_m) or self.approach_stall_m <= 0:
            raise ValueError("approach_stall_m must be positive and finite")
        for key in ("boxed_in_radius_m", "footing_radius_m"):
            value = getattr(self, key)
            if isinstance(value, bool) or not math.isfinite(value) or value <= 0:
                raise ValueError("%s must be positive and finite" % key)
        if (isinstance(self.footing_unknown_fraction, bool) or not math.isfinite(self.footing_unknown_fraction)
                or not 0 <= self.footing_unknown_fraction <= 1):
            raise ValueError("footing_unknown_fraction must lie in [0, 1]")
        if (isinstance(self.override_confidence, bool) or not math.isfinite(self.override_confidence)
                or not self.confidence <= self.override_confidence <= 1):
            raise ValueError("override_confidence must lie in [confidence, 1]")
        for key in ("context_check", "verify_keep_in_frame"):
            if type(getattr(self, key)) is not bool:
                raise ValueError("%s must be a bool" % key)
        if (isinstance(self.context_penalty, bool) or not math.isfinite(self.context_penalty)
                or not 0 < self.context_penalty <= 1):
            raise ValueError("context_penalty must lie in (0, 1]")
        if type(self.context_confirmation_frames) is not int or self.context_confirmation_frames < self.confirmation_frames:
            raise ValueError("context_confirmation_frames must be an int of at least confirmation_frames")
        if isinstance(self.context_baseline_m, bool) or not math.isfinite(self.context_baseline_m) or self.context_baseline_m < 0:
            raise ValueError("context_baseline_m must be finite and non-negative")
        for key in ("memory_noise_floor_m", "sudden_proximity_m"):
            value = getattr(self, key)
            if isinstance(value, bool) or not math.isfinite(value) or value <= 0:
                raise ValueError("%s must be positive and finite" % key)
        for key in ("memory_noise_per_m", "approach_history_m", "approach_history_closed_m"):
            value = getattr(self, key)
            if isinstance(value, bool) or not math.isfinite(value) or value < 0:
                raise ValueError("%s must be finite and non-negative" % key)
        for key in ("history_stop_after_missing", "backoff_steps", "backoff_max_actions"):
            if type(getattr(self, key)) is not int or getattr(self, key) < 1:
                raise ValueError("%s must be a positive int" % key)
        for key in ("backoff", "support_surface_resilience"):
            if type(getattr(self, key)) is not bool:
                raise ValueError("%s must be a bool" % key)
        if (not isinstance(self.support_surface_classes, (tuple, list)) or not self.support_surface_classes
                or not all(isinstance(name, str) and name.strip() for name in self.support_surface_classes)):
            raise ValueError("support_surface_classes must be a non-empty sequence of labels")


class TargetClosing:
    """Own detection verification, path, yaw and pitch until STOP or episode error.

    Only consecutive, coherent 3D observations of the same object count. Alias
    duplicates were removed by PerceptionCycle. No evaluator goal/DTG is read.
    The first confident candidate WITH a depth projection suspends exploration
    before any lock; an unverified candidate is released back to exploration
    (``release_unverified``), and so is a locked one whose terminal inspection
    never saw it from close (``release_on_failed_inspection``); a locked target
    seen fresh at terminal range is STOPped on, never given back.
    """

    def __init__(self, policy):
        self.policy, self.settings = policy, policy.settings.target_closing
        self.path = TargetApproachPath(policy)
        self.rejected = []                 # (anchor xyz, floor id, action, radius m, why) per released candidate
        self.releases = 0
        self.inspection_releases = 0
        self.inspection_resumptions = 0    # inspections ended to step closer: seen fresh, measured out of range
        self.stall_releases = 0            # locks released because the approach went nowhere
        self.map_releases = 0              # locks the map outvoted after the fact
        self.last_release = None           # why the last candidate was released
        self.released_step = None          # the action of the last release; the cooldown counts from it
        self.border_rejections = 0
        self.map_rejections = 0
        self.weak_resightings = 0          # frames that counted on ``track_confidence`` alone
        self.footing_sweeps = 0            # footing sweeps asked for by a lock with no safe path
        self.boxed_releases = 0            # locks released because no path existed from where the agent stood
        self.boxed_in = []                 # (robot xy, floor id, action) per boxed-in release
        self.overrides = 0                 # takeovers started past the memory by ``override_confidence``
        self.context_rejections = 0        # confident boxes the context penalty kept under the start threshold
        self.suspect_locks_held = 0        # frames a suspect candidate had the frames but not the viewpoints
        self.same_spot_rejections = 0      # near candidates refused because they were released from this very spot
        self.blind_frames = 0              # frames the detector was in back-off: no evidence, clocks paused
        self.backoffs = 0                  # backing manoeuvres made for a candidate seen suddenly at close range
        self.backoff_skips = 0             # ... and those skipped for want of clear space behind the agent
        self.support_sightings = 0         # frames a support-surface box stood consistent with the 3-D memory
        self.support_holds = 0             # inspection releases withheld on that evidence
        self.support_stops = 0             # STOPs on consistent support-surface tracking at terminal range
        self.history_holds = 0             # releases withheld inside the terminal range by the approach history
        self.history_stops = 0             # STOPs the approach history completed without a fresh aligned box
        self._clear()

    def _clear(self):
        """Per-candidate state; rejection memory and counters survive a release."""
        self.active = self.locked = False
        self.phase = "SEARCH"
        self.xyz = self.anchor = self.box = None
        self.label = None
        self.count = 0
        self.started = self.last_seen = self.processed = -1
        self.floor_id = None
        self.failure = None
        self.path = TargetApproachPath(self.policy)
        self.observed_xyz = None
        self.observed_near_m = None        # near-edge range of the fresh observation, when measured
        self.inspection_started = None
        self.inspection_index = 0
        self.inspection_sighting = None    # the last action of the inspection that saw the target in terminal range
        self.occluded_path_steps = 0
        self.verification_yaw = 0.0
        self.no_path_steps = 0             # consecutive actions the LOCKED target had no safe path
        self.footing_done = False          # a footing sweep was asked for by this lock
        self.pathless_after_footing = 0    # pathless actions since the sweep ended
        self.close_trail = []              # (x, y) per consecutive CLOSE action, for the stall clock
        self.suspect = None                # why the context check doubts this candidate (``target_context``)
        self.sightings = []                # (x, y) of the agent per counted consecutive frame
        self.memory = None                 # TargetMemory3D: the fused XYZ centroid; ``xyz`` mirrors it
        self.history = None                # ApproachHistory: the path walked since the takeover began
        self.backoff = None                # BackoffManeuver for a candidate seen suddenly at close range
        self.support_seen = -1             # last step a support-surface box stood consistent with the memory
        self.support_xyz = None
        self.terminal_missing = 0          # consecutive terminal-range frames without a fresh target box

    def _clipped(self, box, intrinsics):
        return clipped_box(box, intrinsics, self.settings.border_margin_px)

    #: Releases whose spot a box at ``override_confidence`` may start a takeover inside of.
    OVERRIDABLE_RELEASES = ("unverified", "contradicted by the map")

    def _rejected_nearby(self, xyz, floor_id, override=False, here=None):
        """Whether a released candidate's anchor covers ``xyz`` on ``floor_id``.

        With ``here`` (the agent's position) only the releases made from
        within ``boxed_in_radius_m`` of it count: the test for a NEAR
        candidate, which is new evidence from a new spot but the same
        evidence from the spot it was already released at (the review of
        2026-10-07 reproduced twelve takeovers of one chair from one spot).
        """
        for entry in self.rejected:
            spot, floor, _, radius = entry[:4]
            why = entry[4] if len(entry) > 4 else "unverified"     # a 4-tuple is a legacy (unverified) entry
            if floor != floor_id or math.dist(spot[:2], xyz[:2]) > radius:
                continue
            if here is not None:
                stood = entry[5] if len(entry) > 5 else None
                if stood is None or math.dist(stood, here) > self.settings.boxed_in_radius_m:
                    continue
            if override and why in self.OVERRIDABLE_RELEASES:
                continue
            return True
        return False

    def rejected_near(self, xy, floor_id):
        """Whether a released candidate's anchor lies within ``rejection_radius_m`` of ``xy`` on ``floor_id``.

        For the search loop: a confirmed landmark the takeover already
        verified and rejected at is not worth another look.
        """
        return self._rejected_nearby((float(xy[0]), float(xy[1]), 0.0), floor_id)

    def _candidates(self, obs):
        """Confident in-frame boxes, and those with a coherent 3-D position.

        Before a lock the box must also clear the image border and lie away
        from every spot already released as unverified on this floor --
        unless it reaches ``override_confidence`` (the unverified and
        map-contradicted memory, the cooldown and a boxed-in spot are
        bypassed; a failed inspection's spot is not). A box the context
        check doubts (``target_context.suspect``) counts
        ``context_penalty`` of its confidence toward the start threshold.
        While a candidate is active, a box of the target's class down to
        ``track_confidence`` is projected too, and counts if -- and only if
        -- it lands on the candidate's anchor: a re-sighting, not a new
        object.
        """
        p, s = self.policy, self.settings
        k = obs.camera.intrinsics
        tracking = self.active and self.anchor is not None
        threshold = s.track_confidence if tracking else s.confidence
        boxes = [d for d in p.perception.detections if p.target.accepts(d.cls)
                 and d.conf >= threshold
                 and 0 <= d.xyxy[0] < d.xyxy[2] <= k.width
                 and 0 <= d.xyxy[1] < d.xyxy[3] <= k.height]
        if not self.locked:
            kept = [d for d in boxes if not self._clipped(d.xyxy, k)]
            self.border_rejections += sum(1 for d in boxes if d.conf >= s.confidence) - sum(
                1 for d in kept if d.conf >= s.confidence)
            boxes = kept
        projected, kept = [], []
        for detection in boxes:
            rows = []
            strong = detection.conf >= s.confidence
            values = list(observed_objects(obs, (detection,), threshold, rows))
            if not values:
                if strong:
                    kept.append(detection)
                continue
            label, xyz = values[0]
            range_m = math.dist((obs.pose.x, obs.pose.y), xyz[:2])
            radius = s.association_radius_m + s.association_range_gain * max(0.0, range_m - 2.0)
            associated = (self.anchor is not None and p.mapping.floor_id == self.floor_id
                          and math.dist(self.anchor, xyz) <= radius)
            if not strong and not associated:
                continue                                       # a weak box is only ever a re-sighting
            override = s.override_confidence < 1.0 and detection.conf >= s.override_confidence
            if not self.locked and range_m > s.rejection_min_range_m:
                cooling = (not self.active and self.released_step is not None
                           and obs.step - self.released_step < s.release_cooldown_actions)
                if self._rejected_nearby(xyz, p.mapping.floor_id, override=override):
                    continue
                if cooling and not override:
                    continue
                if self._boxed_in_here(obs):
                    continue                                   # no path from HERE: the spot, not the box, is the problem
                if override and not associated and not self.active and (cooling or self._rejected_nearby(xyz, p.mapping.floor_id)):
                    self.overrides += 1
            elif not self.locked and not self.active:
                # A close view is new evidence -- from a NEW spot. From the spot the same
                # candidate was already released at, or given up at for want of a path, it
                # is the same evidence, and the takeover/release cycle it bought was unbounded
                # (twelve takeovers of one chair in eighty actions, review of 2026-10-07).
                here = (obs.pose.x, obs.pose.y)
                if self._rejected_nearby(xyz, p.mapping.floor_id, override=override, here=here) or self._boxed_in_here(obs):
                    self.same_spot_rejections += 1
                    continue
            if not self.locked and contradicted_by_map(p, label, xyz, rows[-1].get("radius_m")) is not None:
                self.map_rejections += 1                       # a confirmed non-target object stands here
                continue
            doubt = self._suspect(obs, label, xyz) if (s.context_check and not self.locked) else None
            if doubt is not None and not associated and detection.conf * s.context_penalty < s.confidence:
                # Confident enough on its own, not where it stands: no takeover on this box.
                self.context_rejections += 1
                kept.append(detection)
                continue
            kept.append(detection)
            if not obs.pose.z - 0.3 <= xyz[2] <= obs.pose.z + 2.2:
                continue
            if self.anchor is not None and not associated:
                continue
            if not strong:
                self.weak_resightings += 1
            projected.append((detection, tuple(float(v) for v in xyz), rows[-1].get("range_near_m"), doubt))
        return kept, projected

    def _suspect(self, obs, label, xyz):
        """Why the context check doubts a box of ``label`` at ``xyz`` (``target_context.suspect``), or None."""
        p = self.policy
        info, objects = {}, ()
        graph = getattr(p, "graph", None)
        world = getattr(p, "last_world", None)
        if graph is not None and world is not None and getattr(graph, "labels", None) is not None:
            pid = graph.room_near(world, (float(xyz[0]), float(xyz[1])), 0.6)
            if pid is not None:
                info = graph.label_info(pid) or {}
                objects = graph.objects_in(pid)
        return suspect(p.target, label, xyz, obs.pose.z, label=info.get("label"),
                       strength=info.get("strength"), objects=objects)

    def blind(self):
        """Whether this frame carried no detector evidence (the service failed or is in back-off)."""
        evidence = getattr(getattr(self.policy, "perception", None), "detector_evidence", None) or {}
        return bool(evidence.get("skipped") or evidence.get("failed"))

    def observe(self, obs):
        if obs.step == self.processed:
            return
        self.processed = obs.step
        p, s = self.policy, self.settings
        if self.active and self.blind():
            # No evidence either way: the consecutive-frame chain is carried across the
            # frame, nothing is counted and no clock advances (``plan`` pauses them). Before
            # 2026-10-07 a detector back-off read as "target not seen": a correct lock was
            # released as "the inspection saw nothing" with a 2 m rejection around it.
            self.blind_frames += 1
            if self.last_seen == obs.step - 1:
                self.last_seen = obs.step
            return
        boxes, projected = self._candidates(obs)
        # Only a candidate with a coherent 3-D projection starts a takeover: a confident
        # box without depth support has nothing the next frame can be checked against,
        # and the twelve-step verifications it bought (Ranchester attempts 4 and 5:
        # "waiting for valid target depth" from 3.8 m) released every time.
        if not self.active and projected:
            self.active, self.started, self.phase = True, obs.step, "VERIFY"
            self.verification_yaw = obs.pose.yaw
            p.route_memory.clear("target_takeover")
            p._route = p._goal = None
            p.peek.cancel(obs, "target_takeover", restore=False)
            p.camera_control.inspection = None
            glances = getattr(p, "glances", None)
            if glances is not None and glances.active is not None:
                # A cue glance toward the box is often what un-clipped it; left in
                # force it resumed, stale, after the release (turns toward a yaw the
                # route no longer has).
                glances.abort(obs, "target_takeover")
        if not self.active:
            return
        here = (float(obs.pose.x), float(obs.pose.y))
        if projected:
            detection, xyz, near, doubt = max(projected, key=lambda item: item[0].conf)
            self.count = self.count + 1 if obs.step == self.last_seen + 1 else 1
            self.sightings = (self.sightings if obs.step == self.last_seen + 1 else []) + [here]
            self.observed_xyz, self.observed_near_m = xyz, (None if near is None else float(near))
            range_m = math.dist(here, xyz[:2])
            if self.anchor is None:
                # The 3-D memory and the approach ledger start with the takeover; a candidate
                # that appears at arm's length with no history is re-verified from farther back.
                self.anchor, self.floor_id = xyz, p.mapping.floor_id
                self.memory = TargetMemory3D(xyz, range_m, obs.step, s.memory_noise_floor_m, s.memory_noise_per_m)
                self.history = ApproachHistory(here, range_m, obs.step)
                if s.backoff and range_m < s.sudden_proximity_m:
                    self.backoff = BackoffManeuver(p, s.backoff_steps, s.backoff_max_actions)
            else:
                self.memory.update(xyz, range_m, obs.step)
            self.xyz = self.memory.xyz
            self.box, self.label, self.last_seen = detection.xyxy, detection.cls, obs.step
            if doubt is not None and not self.locked:
                self.suspect = doubt
            if self.backoff is None or not (self.backoff.state == "pending" or self.backoff.active):
                self.locked |= self._confirmed()
            p._target_xy, p._target_step, p._target_floor_id = self.xyz[:2], obs.step, self.floor_id
        else:
            if not self.locked:
                self.count = 0
                self.sightings = []
            self._support_resighting(obs)
        if self.history is not None and self.xyz is not None:
            self.history.record(here, math.dist(here, self.xyz[:2]), obs.step)
        if self.backoff is not None and (self.backoff.state == "pending" or self.backoff.active):
            self.started += 1                  # the manoeuvre's actions are not verification actions: the clock waits
        if (not self.locked and s.release_unverified
                and obs.step - self.started >= s.max_verify_steps):
            self._release(obs)
            return
        if self.locked and s.release_on_failed_inspection and self.anchor is not None:
            landmark = contradicted_by_map(p, self.label or "", self.anchor)
            if landmark is not None and self._is_support(landmark.class_name):
                landmark = None                # the surface the target stands on: the same spot, not another object
            if landmark is not None:
                # The map outvoted the lock: the object at the anchor has been confirmed as
                # something else since (the sofa from four metres is the bed from two).
                self.map_releases += 1
                self._release(obs, radius_factor=s.failed_inspection_radius_factor, why="contradicted by the map")

    def _confirmed(self):
        """Whether the consecutive frames so far lock the candidate.

        ``confirmation_frames`` frames as before; a candidate the context
        check doubts (``suspect``) needs ``context_confirmation_frames`` of
        them, taken from at least two spots ``context_baseline_m`` apart --
        a turn in place is not a second viewpoint, and the Allensville
        counter was locked on two frames from one spot.
        """
        s = self.settings
        if self.suspect is None or not s.context_check:
            return self.count >= s.confirmation_frames
        if self.count < s.context_confirmation_frames:
            return False
        spots = self.sightings[-self.count:]
        baseline = max((math.dist(a, b) for a in spots for b in spots), default=0.0)
        if baseline + 1e-9 < s.context_baseline_m:
            self.suspect_locks_held += 1
            return False
        return True

    def _is_support(self, label):
        return self.settings.support_surface_resilience and is_support_surface(label, self.settings.support_surface_classes)

    def _support_resighting(self, obs):
        """Record a support-surface box standing where the memory holds the target, on a frame without the target's own class.

        The television on the dresser becomes the dresser from one metre: a
        box of a supporting class, projected to a coherent centroid under
        the memory's footprint and at or below its height, is spatial
        evidence that the tracked thing is still there (``support_surface``).
        It never counts toward the lock or the terminal test on its own.
        """
        p, s = self.policy, self.settings
        if not s.support_surface_resilience or not self.active or self.memory is None:
            return
        k = obs.camera.intrinsics
        boxes = [d for d in p.perception.detections if self._is_support(d.cls) and d.conf >= s.track_confidence
                 and 0 <= d.xyxy[0] < d.xyxy[2] <= k.width and 0 <= d.xyxy[1] < d.xyxy[3] <= k.height]
        if not boxes:
            return
        here = (obs.pose.x, obs.pose.y)
        for label, xyz in observed_objects(obs, boxes, s.track_confidence):
            range_m = math.dist(here, self.xyz[:2])
            radius = s.association_radius_m + s.association_range_gain * max(0.0, range_m - 2.0)
            if supports(self.xyz, xyz, radius):
                self.support_sightings += 1
                self.support_seen, self.support_xyz = obs.step, tuple(float(v) for v in xyz)
                return

    def _history_qualifies(self):
        """Whether this candidate was tracked and approached over an EXTENDED path (``approach_history_m``)."""
        s = self.settings
        return (self.history is not None and s.approach_history_m > 0
                and self.history.qualifies(s.approach_history_m, s.approach_history_closed_m))

    def _boxed_in_here(self, obs):
        """Whether the agent still stands where a lock was released for want of a path (``boxed_in_radius_m``)."""
        here = (obs.pose.x, obs.pose.y)
        floor = self.policy.mapping.floor_id
        return any(f == floor and math.dist(xy, here) <= self.settings.boxed_in_radius_m for xy, f, _ in self.boxed_in)

    def _release(self, obs, radius_factor=1.0, why="unverified", remember=True):
        """Give a candidate back to exploration and remember the spot (``radius_factor`` times the rejection radius).

        ``remember=False`` leaves the anchor out of the rejection memory: the
        candidate was not disproved, the place the agent stood was.
        """
        p = self.policy
        if self.anchor is not None and remember:
            self.rejected.append((self.anchor, self.floor_id, obs.step,
                                  float(radius_factor) * self.settings.rejection_radius_m, str(why),
                                  (float(obs.pose.x), float(obs.pose.y))))
        self.releases += 1
        self.released_step = int(obs.step)
        self.last_release = why
        p._target_xy = p._target_id = p._target_floor_id = None
        p._route = p._goal = None
        p.route_memory.clear("target_released")
        self._clear()
        self.processed = obs.step
        self.phase = "RELEASED"

    def fail(self, reason):
        self.phase, self.failure = "FAILED", reason
        # Expected method failure, not a converter/harness invariant violation:
        # the evaluator records it as failure, with any forced STOP distinguished.
        raise RuntimeError("Target closing failed (exploration remains locked): " + reason)

    def _elevation(self, obs, distance=None):
        """Where the target stands against the camera: ``+1`` above, ``-1`` below, ``0`` level.

        From the measured 3-D centroid when there is one (its height against
        the camera's, within ``elevation_band_m`` is level); else from the box
        centre's bearing above the horizon -- the optical ray's elevation
        corrected by the camera's own pitch -- read at ``distance`` when
        known, otherwise against half a tilt step. The one input of the
        dynamic pitch controller: LOOK_DOWN below, LOOK_UP above, never a
        fixed direction (a hardcoded look-down drove a wall-mounted
        television out of the frame).
        """
        s = self.settings
        camera_z = obs.pose.z + obs.camera.height_m
        if self.xyz is not None:
            dz = float(self.xyz[2]) - camera_z
            return 0 if abs(dz) <= s.elevation_band_m else (1 if dz > 0 else -1)
        if self.box is None:
            return 0
        k = obs.camera.intrinsics
        v = (self.box[1] + self.box[3]) / 2
        angle = math.atan2(k.cy - v, k.fy) - float(obs.pose.camera_pitch)      # elevation above the horizon
        if distance is not None and math.isfinite(distance) and distance > 0:
            dz = math.tan(angle) * distance
            return 0 if abs(dz) <= s.elevation_band_m else (1 if dz > 0 else -1)
        band = 0.5 * float(self.policy.episode.action_spec.tilt_angle_rad or 0.0)
        return 0 if abs(angle) <= band else (1 if angle > 0 else -1)

    def _pitch(self, obs, distance):
        actions = self.policy.episode.action_spec
        if not actions.has_camera_tilt:
            return None
        # The pitch is the target's measured height's, never its label's: "a potted
        # plant is low" sent the camera 30 degrees down at a plant standing in a metre-
        # tall planter, where its foliage never projected, for 24 actions (Hanson
        # 2026-10-05); a plant on the floor is low by this geometry anyway. Positive
        # is down: a target below the camera gives a positive angle, one above it a
        # negative one -- LOOK_UP. Inside ``look_down_distance_m`` the elevation
        # commits the camera to at least one tilt in its direction, so the terminal
        # frames hold a low toilet or a high television whole.
        camera_z = obs.pose.z + obs.camera.height_m
        desired = math.atan2(camera_z - self.xyz[2], max(distance, 1e-6))
        if distance < self.settings.look_down_distance_m:
            elevation = self._elevation(obs, distance)
            if elevation < 0:
                desired = max(actions.tilt_angle_rad, desired)
            elif elevation > 0:
                desired = min(-actions.tilt_angle_rad, desired)
        desired = round(desired / actions.tilt_angle_rad) * actions.tilt_angle_rad
        if actions.min_pitch_rad is not None:
            desired = max(actions.min_pitch_rad, desired)
        if actions.max_pitch_rad is not None:
            desired = min(actions.max_pitch_rad, desired)
        return float(desired)

    def _scan(self, obs, pitch, reason):
        """Face the candidate while it is in view -- and step towards it -- or sweep when it is not.

        The lock needs two CONSECUTIVE frames of the same object. The former
        pattern stepped through (0, +30, 0, -30 deg) around the bearing on
        every action, so a candidate three metres away sat at the frame's
        edge on every other frame, was refused as clipped, and the count
        reset -- six twelve-step verifications from one spot in Ranchester
        attempt 4. A fresh box is faced (its own centre, as the inspection
        does); once faced, and farther than the terminal range, the agent
        takes one step towards it -- the next frame shows it larger and its
        depth more coherent, and a verification that cannot be had from here
        is had from closer. The sweep is for a candidate that has dropped
        out of view.
        """
        self.phase = "REACQUIRE" if self.locked else "VERIFY"
        actions = self.policy.episode.action_spec
        offset = actions.turn_angle_rad
        bearing = (math.atan2(self.xyz[1] - obs.pose.y, self.xyz[0] - obs.pose.x)
                   if self.xyz is not None else self.verification_yaw)
        fresh = self.last_seen == obs.step and self.box is not None
        if fresh:
            self.inspection_index = 0
            error = self._bbox_error(obs)
            info = {"kind": "target_closing", "reason": reason, "target_visible": True}
            if self.suspect is not None:
                info["suspect"] = self.suspect
            distance = math.dist((obs.pose.x, obs.pose.y), self.xyz[:2]) if self.xyz is not None else float("inf")
            aligned = abs(error) <= offset + 1e-6
            # A suspect candidate needs two viewpoints ``context_baseline_m`` apart, and a
            # turn in place is not one: let it step inside the terminal range (never onto
            # the object) rather than hold at 1.2 m where it could never lock (review of
            # 2026-10-07: 52 frames held, five releases, no move).
            nearest = self.settings.terminal_distance_m if self.suspect is None else 0.6 * self.settings.terminal_distance_m
            if (not self.locked and aligned
                    and distance > nearest + actions.forward_step_m):
                # Two steps ahead: a one-step waypoint sits inside the converter's arrival
                # tolerance and yields no action at all. One MOVE_FORWARD results either way.
                # Within one turn of the centre the box stays in the frame after a step,
                # and a step is what makes the next frame a consecutive one; the centring
                # turn moved the box across the image and the detector dropped it on the
                # other side -- four times in a row at a chair 3.5 m off (Hanson 2026-10-05).
                # Never for a LOCKED target without a safe path: that one waits for A*.
                reach = 2.0 * actions.forward_step_m
                ahead = (obs.pose.x + reach * math.cos(obs.pose.yaw), obs.pose.y + reach * math.sin(obs.pose.yaw))
                if pitch is None and actions.has_camera_tilt and distance <= self.settings.look_down_distance_m:
                    # The dynamic pitch controller during close verification: the step toward a
                    # low target looks down, toward a high one looks up (the memory's elevation).
                    pitch = self._pitch(obs, distance)
                return NavigationCommand.follow([(obs.pose.x, obs.pose.y), ahead], camera_pitch=pitch,
                                                info=dict(info, verify_step="towards the candidate",
                                                          elevation=self._elevation(obs, distance)))
            if self.settings.verify_keep_in_frame and turn_action(error, offset) is None:
                # The hold below would be satisfied as it stands -- idle -- and an idle
                # result is executed as a turn. Choose the frame instead. For a LOCKED target
                # without a path too: the idle turn and the centring turn back alternated
                # for the whole pathless wait (233 reversals in one Darden episode).
                view = self._verification_view(obs, distance)
                if view is not None:
                    return view
            return NavigationCommand.hold(camera_pitch=pitch, final_yaw=normalize_angle(obs.pose.yaw + error), info=info)
        angles = (0, 1, 0, -1)
        yaw = normalize_angle(bearing + angles[self.inspection_index % 4] * offset)
        if turn_action(normalize_angle(yaw - obs.pose.yaw), offset) is None:
            self.inspection_index += 1
            yaw = normalize_angle(bearing + angles[self.inspection_index % 4] * offset)
        return NavigationCommand.hold(camera_pitch=pitch,
                                      final_yaw=yaw,
                                      info={"kind": "target_closing", "reason": reason, "target_visible": False})

    def _backoff_command(self, obs, world, distance):
        """The backing manoeuvre's action for a candidate seen suddenly at close range, or None.

        Begun on the first action after the takeover (the retreat spot needs
        the world), it owns the action until the agent stands back and faces
        the remembered target; then the consecutive-frame count is reset, so
        the lock needs fresh frames from the wider perspective. A manoeuvre
        with no clear space is skipped and recorded; verification proceeds in
        place as before.
        """
        manoeuvre = self.backoff
        if manoeuvre is None or manoeuvre.state == "done":
            return None
        p, s = self.policy, self.settings
        if manoeuvre.state == "pending":
            if manoeuvre.begin(obs, world, self.xyz):
                self.backoffs += 1
            else:
                self.backoff_skips += 1
                return None
        pitch = None
        if p.episode.action_spec.has_camera_tilt:
            pitch = self._pitch(obs, distance) if distance <= s.look_down_distance_m else 0.0
        command = manoeuvre.command(obs, world, self.xyz, pitch)
        if command is not None:
            self.phase = "BACK_OFF"
            return replace(command, info=dict(command.info, backoff=manoeuvre.diagnostics(),
                                              target_visible=self.last_seen == obs.step))
        self.count, self.sightings = 0, []          # re-verify from here: fresh consecutive frames
        return None

    def _verification_view(self, obs, distance):
        """The next frame for a fresh, aligned candidate too close to step toward: one that keeps its box whole.

        A satisfied hold is idle, and the headless agent's idle action is a
        turn, which moved the centred Allensville toilet out of the frame
        at 1.18 m. **The pitch follows the target's elevation**
        (:meth:`_elevation`): a target below the camera is looked DOWN at,
        one above it -- a wall-mounted television, the top of a wardrobe --
        is looked UP at; a box cut by the bottom or the top edge says the
        same thing; never a fixed direction. Level with the camera, the LOOK
        whose predicted shift (``fy * tan(tilt)`` pixels) leaves the box
        inside the frame margins, down before up (the floor is nearer than
        the ceiling); failing both, the TURN whose shift (``fx * tan(turn)``)
        keeps it, toward the box's side before away. None when no view keeps
        it.
        """
        actions, s = self.policy.episode.action_spec, self.settings
        k = obs.camera.intrinsics
        x1, y1, x2, y2 = self.box
        margin = s.border_margin_px
        info = {"kind": "target_closing", "reason": "verification frame that keeps the box whole",
                "target_visible": True}
        if self.suspect is not None:
            info["suspect"] = self.suspect
        pitch_now = float(obs.pose.camera_pitch)
        tilt = float(actions.tilt_angle_rad) if actions.has_camera_tilt else None
        top_cut, bottom_cut = y1 < margin, y2 > k.height - margin

        def pitch_ok(value):
            if actions.min_pitch_rad is not None and value < actions.min_pitch_rad - 1e-9:
                return False
            return actions.max_pitch_rad is None or value <= actions.max_pitch_rad + 1e-9

        def fits(du, dv):
            return (x1 + du >= margin and x2 + du <= k.width - margin
                    and y1 + dv >= margin and y2 + dv <= k.height - margin)

        def look(sign, step):
            """One tilt step, ``+1`` down / ``-1`` up: when the limits allow it and the box stays in (or is cut on that side)."""
            value = pitch_now + sign * tilt
            dv = -sign * k.fy * math.tan(tilt)                 # LOOK_DOWN moves the image content UP
            cut = bottom_cut if sign > 0 else top_cut
            if pitch_ok(value) and (cut or fits(0.0, dv)):
                return NavigationCommand.hold(camera_pitch=value, final_yaw=obs.pose.yaw,
                                              info=dict(info, verify_step=step, elevation=elevation))
            return None

        elevation = self._elevation(obs, distance) if tilt is not None else 0
        if tilt is not None:
            if elevation < 0 or (elevation == 0 and bottom_cut):
                view = look(+1, "look down at a low target")
                if view is not None:
                    return view
            elif elevation > 0 or (elevation == 0 and top_cut):
                view = look(-1, "look up at a high target")
                if view is not None:
                    return view
            for sign, step in ((+1, "look down"), (-1, "look up")):
                view = look(sign, step)
                if view is not None:
                    return view
        du = k.fx * math.tan(actions.turn_angle_rad)
        toward_left = (x1 + x2) / 2 < k.cx                  # the box is left of centre: a LEFT turn moves it right
        for turn_left in ((True, False) if toward_left else (False, True)):
            if fits(du if turn_left else -du, 0.0):
                yaw = normalize_angle(obs.pose.yaw + (actions.turn_angle_rad if turn_left else -actions.turn_angle_rad))
                return NavigationCommand.hold(camera_pitch=None if tilt is None else pitch_now, final_yaw=yaw,
                                              info=dict(info, verify_step="turn %s" % ("left" if turn_left else "right")))
        return None

    def plan(self, obs, world):
        p, s = self.policy, self.settings
        if self.failure:
            self.fail(self.failure)
        if self.blind():
            return self._blind_frame(obs, world)
        if obs.step - self.started >= s.max_closing_steps:
            self.fail("closing action bound reached")
        if not self.locked and not s.release_unverified and obs.step - self.started >= s.max_verify_steps:
            self.fail("candidate never received consecutive depth-consistent confirmations")
        if self.xyz is None:
            return self._scan(obs, None, "waiting for valid target depth")
        if p.mapping.floor_id != self.floor_id:
            self.fail("target floor changed")
        distance = math.dist((obs.pose.x, obs.pose.y), self.xyz[:2])
        if not self.locked:
            command = self._backoff_command(obs, world, distance)
            if command is not None:
                return command
            return self._scan(obs, None, "consecutive depth-consistent frames required")
        command = self._continue_footing(obs, world, distance)
        if command is not None:
            return command
        measured = self._fresh_measured(obs)
        out_of_range = measured is not None and measured > s.terminal_distance_m + s.range_tolerance_m
        if self.inspection_started is not None and self.inspection_sighting is None and out_of_range:
            # Seen fresh from here, aligned or not, but the measured surface lies beyond
            # the terminal range: the filtered estimate entered the inspection early (an
            # older, nearer reading averaged in). Standing still would spend the budget
            # on a target in plain view and release it as "saw nothing"; step closer.
            self.inspection_started = None
            self.inspection_resumptions += 1
        if self.inspection_started is not None or (distance <= s.terminal_distance_m and not out_of_range):
            command = self._inspect(obs, distance)
            if command is None:
                # The lock was wrong: released, the spot remembered. This action is one
                # turn in place -- the first frame of the resumed search -- and the next
                # action is the search's own; the closing never explores.
                turn = p.episode.action_spec.turn_angle_rad
                return NavigationCommand.hold(
                    final_yaw=normalize_angle(obs.pose.yaw + turn),
                    info={"kind": "target_released", "phase": "RELEASED",
                          "reason": "the inspection from %.1f m saw nothing; the search resumes" % distance})
            return command
        # During transit A* owns heading. A bbox outside the FOV must not replace
        # a collision-qualified path with a centering hold or a reacquisition spin.
        command = self.path.command(self, obs, world)
        if command is None:
            return self._no_path(obs, world, distance)
        self.no_path_steps = 0
        self.pathless_after_footing = 0
        if self._approach_stalled(obs):
            self.stall_releases += 1
            self._release(obs, why="approach stalled: no progress in %d actions" % s.approach_stall_actions)
            turn = p.episode.action_spec.turn_angle_rad
            return NavigationCommand.hold(
                final_yaw=normalize_angle(obs.pose.yaw + turn),
                info={"kind": "target_released", "phase": "RELEASED",
                      "reason": "the approach from %.1f m went nowhere for %d actions; the search resumes"
                      % (distance, s.approach_stall_actions)})
        visible = self.last_seen == obs.step
        self.phase = "CLOSE" if visible else "CLOSE_OCCLUDED"
        self.occluded_path_steps += int(not visible)
        # Level during transit; inside ``look_down_distance_m`` the pitch follows the
        # target's elevation (``_pitch``: down at a toilet, up at a wall-mounted
        # television) so the terminal frames hold it whole -- the close-range policy,
        # in both directions (since 2026-10-07; before, the approach forced level and
        # the inspection alone tilted).
        pitch = None
        if p.episode.action_spec.has_camera_tilt:
            pitch = self._pitch(obs, distance) if distance <= s.look_down_distance_m else 0.0
        return replace(command, camera_pitch=pitch,
                       info=dict(command.info, target_confirmed=True, persistent_lock=True,
                                 target_visible=visible, range_m=distance, phase=self.phase))

    def _blind_frame(self, obs, world):
        """The action for a frame without detector evidence: the clocks stand still, the approach may go on.

        Every bound that counts frames moves one action later; a locked
        target with a path is walked toward (A* needs no detections), an
        unlocked candidate is faced where it was last seen.
        """
        self.started += 1
        if self.inspection_started is not None:
            self.inspection_started += 1
        info = {"kind": "target_closing", "reason": "detector frame missing: no evidence, clocks paused",
                "target_visible": False, "blind": True}
        if self.locked and self.xyz is not None:
            command = self.path.command(self, obs, world)
            if command is not None:
                self.phase = "CLOSE_OCCLUDED"
                return replace(command, info=dict(command.info, **info, persistent_lock=True))
        bearing = (math.atan2(self.xyz[1] - obs.pose.y, self.xyz[0] - obs.pose.x)
                   if self.xyz is not None else self.verification_yaw)
        return NavigationCommand.hold(final_yaw=normalize_angle(bearing), info=info)

    def _no_path(self, obs, world, distance):
        """A LOCKED target A* cannot reach from here: look at the floor around, then give the spot up.

        First the footing sweep (``footing_sweep``; the camera controller
        owns it and its turns come back here one per action, tagged
        ``FOOTING``): the blind disk under the camera is the usual reason a
        path does not exist from where the agent has not moved, so the
        sweep is asked for only while ``footing_unknown_fraction`` of the
        cells within ``footing_radius_m`` are still unknown. Then, if the
        sweep bought no path either -- or there was nothing to sweep -- the
        lock is released without a rejection after
        ``release_after_footing_steps`` pathless actions, the spot is
        remembered as boxed in, and the search -- which can walk -- carries
        on; a new takeover from within ``boxed_in_radius_m`` of here is
        refused. Between those, the centring hold as before: the target
        stays in view.
        """
        p, s = self.policy, self.settings
        camera = p.camera_control
        self.no_path_steps += 1
        if (s.footing_sweep and not self.footing_done and p.episode.action_spec.has_camera_tilt
                and self.no_path_steps >= s.footing_after_steps):
            self.footing_done = True
            blind = unknown_around(world, (obs.pose.x, obs.pose.y), s.footing_radius_m) >= s.footing_unknown_fraction
            if blind and camera.begin_inspection(obs, p.mapping.floor_id, reason="footing"):
                self.footing_sweeps += 1
                sweep = camera.inspection_command(obs)
                if sweep is not None:
                    return self._footing_command(obs, sweep)
        if self.footing_done or not s.footing_sweep or not p.episode.action_spec.has_camera_tilt:
            self.pathless_after_footing += 1
            if self.pathless_after_footing >= s.release_after_footing_steps:
                self.boxed_releases += 1
                self.boxed_in.append(((float(obs.pose.x), float(obs.pose.y)), p.mapping.floor_id, int(obs.step)))
                self._release(obs, why="no safe path from here", remember=False)
                turn = p.episode.action_spec.turn_angle_rad
                return NavigationCommand.hold(
                    final_yaw=normalize_angle(obs.pose.yaw + turn),
                    info={"kind": "target_released", "phase": "RELEASED",
                          "reason": "no safe path to the target %.1f m away from here; the search moves on" % distance})
        return self._scan(obs, None, "no safe target path; exploration remains suspended")

    def _continue_footing(self, obs, world, distance):
        """The footing sweep in force, one action at a time -- cut short the moment a path exists.

        The sweep was asked for because A* had no path from here; its turns
        come back through here (``FOOTING``), and once the floor it mapped
        connects the agent to the target's standoff the rest of the circle
        is not worth its actions: the camera is told to restore (one
        LOOK_UP) and the approach begins level. Without this the sweep's
        pitch stood for the whole approach and the first inspection (Hanson
        2026-10-05, actions 5-63: 30 degrees down, the plant's foliage never
        projected, the lock released as "saw nothing").
        """
        camera = self.policy.camera_control
        if not camera.footing:
            return None
        sweep = camera.inspection_command(obs)
        if sweep is None:
            return None
        if (not camera.inspection["restoring"] and self.inspection_started is None
                and distance > self.settings.terminal_distance_m
                and self.path.command(self, obs, world) is not None):
            camera.inspection["restoring"] = True
            sweep = camera.inspection_command(obs)
            if sweep is None:
                return None
        return self._footing_command(obs, sweep)

    def refuses_far_candidate(self, obs, xy):
        """Whether a target seen at ``xy`` is one the takeover would not start on from here.

        Far candidates only (beyond ``rejection_min_range_m``; a close view is
        new evidence): one within the rejection memory, one seen during the
        cooldown after a release, or one seen from a spot the lock was given
        up at for want of a path. The legacy target evidence asks the same
        question, so the search does not walk after what the takeover has
        just refused (Hanson 2026-10-05, actions 18-26: nine actions toward
        a released chair, and the warm-up begun again where they ended).
        """
        s = self.settings
        if self.locked:
            return False
        range_m = math.dist((obs.pose.x, obs.pose.y), (float(xy[0]), float(xy[1])))
        if range_m <= s.rejection_min_range_m:
            return False
        cooling = (not self.active and self.released_step is not None
                   and obs.step - self.released_step < s.release_cooldown_actions)
        return (cooling or self._rejected_nearby((float(xy[0]), float(xy[1]), 0.0), self.policy.mapping.floor_id)
                or self._boxed_in_here(obs))

    def _footing_command(self, obs, sweep):
        self.phase = "FOOTING"
        return replace(sweep, info=dict(sweep.info, kind="target_closing", phase=self.phase, persistent_lock=True,
                                        target_visible=self.last_seen == obs.step,
                                        reason="no safe target path; mapping the floor around the feet"))

    def _bbox_error(self, obs):
        # Optical x points right whereas positive ENU yaw turns left. At a
        # pitched camera the vertical box coordinate also affects body forward.
        k = obs.camera.intrinsics
        u, v = (self.box[0] + self.box[2]) / 2, (self.box[1] + self.box[3]) / 2
        forward = math.cos(obs.pose.camera_pitch) - (v - k.cy) / k.fy * math.sin(obs.pose.camera_pitch)
        error = math.atan2(-(u - k.cx) / k.fx, forward)
        return error

    def _inspect(self, obs, distance):
        """Stationary pitch/yaw inspection; never STOP from remembered range alone.

        Three facts about a big object at terminal range shape this:

        * its box is clipped at an image edge, so the box centre is not the
          object's centre -- a fresh box that SPANS the image centre column
          is aligned, whatever its centre says (centring it turned the
          Ranchester agent off the couch twelve times in a row);
        * its centroid is not its range -- the terminal test reads the near
          edge of the freshly measured surface beside the centroid, which is
          what the benchmark's success radius is measured to;
        * the pitch its height predicts is where to look for it when it is
          NOT in view; a fresh, aligned sighting within range STOPs at
          whatever pitch it came at;
        * when the inspection budget runs out after the target WAS seen
          fresh in terminal range from this very spot, a STOP is the right
          verdict and an error is not (``stop_on_exhausted_inspection``);
          when it runs out and the target was NOT seen from here at all, the
          lock was wrong and the candidate is released
          (``release_on_failed_inspection``; None is returned and the
          caller hands the action to the search).
        """
        actions, s = self.policy.episode.action_spec, self.settings
        k = obs.camera.intrinsics
        if self.inspection_started is None:
            self.inspection_started, self.inspection_index = obs.step, 0
            self.inspection_sighting = None
        self.phase = "INSPECT"
        pitch = self._pitch(obs, distance)
        bearing = math.atan2(self.xyz[1] - obs.pose.y, self.xyz[0] - obs.pose.x)
        fresh = self.last_seen == obs.step
        spans_centre = fresh and self.box is not None and self.box[0] <= k.cx <= self.box[2]
        error = 0.0 if spans_centre else self._bbox_error(obs) if fresh else normalize_angle(bearing - obs.pose.yaw)
        yaw = normalize_angle(obs.pose.yaw + error)
        turning = not spans_centre and turn_action(error, actions.turn_angle_rad) is not None
        tilting = pitch is not None and pitch_action(obs.pose.camera_pitch, pitch, actions.tilt_angle_rad,
                                                     actions.min_pitch_rad, actions.max_pitch_rad) is not None
        limit = s.terminal_distance_m + s.range_tolerance_m
        measured = self._fresh_measured(obs)
        if measured is not None and measured <= limit:
            self.inspection_sighting = obs.step
        history = self._history_qualifies()
        support = s.support_surface_resilience and self.inspection_started <= self.support_seen
        self.terminal_missing = 0 if fresh else self.terminal_missing + 1
        info = {"kind": "target_closing", "target_confirmed": True, "persistent_lock": True,
                "target_visible": fresh, "range_m": distance, "measured_m": measured, "bbox_yaw_error_rad": error,
                "box_spans_centre": spans_centre, "phase": self.phase, "approach_history": history,
                "support_surface_fresh": self.support_seen == obs.step,
                "memory_sigma_m": None if self.memory is None else self.memory.sigma_m}
        # The pitch the geometry predicts is where to LOOK for the target, not a
        # condition on having seen it: the Hanson toilet projected only at 60 degrees
        # down where the prediction said 30, and twenty actions of LOOK_UP/LOOK_DOWN
        # followed a fresh, centred, in-range sighting before the budget STOPped.
        if fresh and not turning and measured <= limit:
            self.phase = "STOP"
            return NavigationCommand.stop_here(info=dict(info, phase="STOP", reason="fresh terminal target confirmation"))
        if history and fresh and measured <= limit:
            # Tracked and approached over metres: a cropped box whose centre says "turn" is
            # still the object, and the benchmark measures range, not heading.
            self.phase = "STOP"
            self.history_stops += 1
            return NavigationCommand.stop_here(info=dict(
                info, phase="STOP", reason="fresh in-range sighting after a %.1f m approach; alignment not required"
                % self.history.path_m))
        if (history and not fresh and distance <= limit
                and self.terminal_missing >= s.history_stop_after_missing):
            # The trajectory is complete and the detector has dropped the box for a few
            # frames: the 3-D memory refined over the approach is what we STOP on.
            self.phase = "STOP"
            self.history_stops += 1
            return NavigationCommand.stop_here(info=dict(
                info, phase="STOP", reason="approach history: target tracked over %.1f m, the 3-D memory stands %.2f m "
                "away (sigma %.2f m) and %d frames passed without a fresh box"
                % (self.history.path_m, distance, self.memory.sigma_m, self.terminal_missing)))
        views = self._inspection_views(pitch)
        blank = (s.release_on_failed_inspection and self.inspection_sighting is None
                 and self.last_seen < self.inspection_started and self.inspection_index >= len(views))
        exhausted = obs.step - self.inspection_started >= s.max_reacquire_steps
        if (blank or exhausted) and self.inspection_sighting is None and (history or support):
            if distance <= limit and (exhausted or history):
                # Not "saw nothing": the approach brought us here, or the surface the target
                # stands on is still measured at the memory's spot. Complete and STOP.
                self.phase = "STOP"
                if history:
                    self.history_stops += 1
                else:
                    self.support_stops += 1
                return NavigationCommand.stop_here(info=dict(
                    info, phase="STOP", reason=("approach history: %.1f m walked to this spot; the 3-D memory stands"
                                                % self.history.path_m) if history else
                    "support surface consistent with the 3-D memory at terminal range; inspection budget spent"))
            if history and distance > limit:
                # Too far for a memory STOP: step closer along the trajectory instead of releasing.
                self.history_holds += 1
                self.inspection_started = None
                self.inspection_resumptions += 1
                return NavigationCommand.hold(camera_pitch=None, final_yaw=yaw, info=dict(
                    info, reason="approach history: resuming the approach from %.2f m instead of releasing" % distance))
            if support and not exhausted:
                self.support_holds += 1                      # keep looking: the dynamic pitch brings the object back
                blank = False
        if exhausted or blank:
            if s.stop_on_exhausted_inspection and self.inspection_sighting is not None:
                self.phase = "STOP"
                return NavigationCommand.stop_here(info=dict(
                    info, phase="STOP", reason="inspection exhausted; target seen fresh in terminal range at action %d"
                    % self.inspection_sighting))
            if s.release_on_failed_inspection:
                # Every view tried once and nothing seen (``blank``), or the budget spent
                # without an in-range sighting: the lock was wrong.
                self.inspection_releases += 1
                self._release(obs, radius_factor=s.failed_inspection_radius_factor,
                              why="inspection saw nothing" + (" in any view" if blank else ""))
                return None
            self.fail("terminal visual confirmation unavailable")
        if not fresh or (not turning and not tilting):
            yaw, pitch = self._inspection_view(obs, bearing, pitch)
        elif turning:
            # The converter tilts before it faces: asking for the predicted pitch here
            # would LOOK away from a target in view and never turn (LOOK_DOWN/LOOK_UP
            # for the whole budget). Centre it at the pitch it was seen at.
            pitch = None
        return NavigationCommand.hold(camera_pitch=pitch, final_yaw=yaw, info=info)

    def _approach_stalled(self, obs):
        """Whether the last ``approach_stall_actions`` CLOSE actions left the agent inside a circle of ``approach_stall_m``."""
        s = self.settings
        if s.approach_stall_actions <= 0:
            return False
        self.close_trail.append((float(obs.pose.x), float(obs.pose.y)))
        if len(self.close_trail) <= s.approach_stall_actions:
            return False
        del self.close_trail[:-s.approach_stall_actions - 1]
        x0, y0 = self.close_trail[0]
        return all(math.dist((x, y), (x0, y0)) <= s.approach_stall_m for x, y in self.close_trail[1:])

    def _fresh_measured(self, obs):
        """The freshly measured range to the target this action -- the near edge of its surface -- or None when not seen."""
        if self.last_seen != obs.step or self.observed_xyz is None:
            return None
        measured = math.dist((obs.pose.x, obs.pose.y), self.observed_xyz[:2])
        if self.observed_near_m is not None:
            measured = min(measured, self.observed_near_m)
        return measured

    def _inspection_views(self, pitch):
        """The bounded yaw/pitch views an inspection cycles through, centred on the stored bearing."""
        actions = self.policy.episode.action_spec
        pitches = [pitch]
        if pitch is not None:
            for value in (pitch - actions.tilt_angle_rad, pitch + actions.tilt_angle_rad):
                if (actions.min_pitch_rad is None or value >= actions.min_pitch_rad) and (actions.max_pitch_rad is None or value <= actions.max_pitch_rad):
                    pitches.append(value)
        return [(offset, value) for offset in (0, 1, -1) for value in pitches]

    def _inspection_view(self, obs, bearing, pitch):
        """Try bounded up/down views centred on the stored target bearing."""
        actions = self.policy.episode.action_spec
        views = [(normalize_angle(bearing + offset * actions.turn_angle_rad), value)
                 for offset, value in self._inspection_views(pitch)]
        yaw, value = views[self.inspection_index % len(views)]
        aligned = turn_action(normalize_angle(yaw - obs.pose.yaw), actions.turn_angle_rad) is None
        pitched = value is None or pitch_action(obs.pose.camera_pitch, value, actions.tilt_angle_rad,
                                                actions.min_pitch_rad, actions.max_pitch_rad) is None
        if aligned and pitched:
            self.inspection_index += 1
            yaw, value = views[self.inspection_index % len(views)]
        return yaw, value

    def diagnostics(self):
        return {"active": self.active, "locked": self.locked, "phase": self.phase,
                "persistent_lock": self.locked, "visible": self.last_seen == self.processed,
                "navmesh_goal_xyz": self.path.goal_xyz, "goal_refinements": self.path.refinements,
                "occluded_path_steps": self.occluded_path_steps, "inspection_started_step": self.inspection_started,
                "inspection_sighting_step": self.inspection_sighting, "observed_near_m": self.observed_near_m,
                "consecutive_frames": self.count, "started_step": self.started,
                "last_seen_step": self.last_seen, "xyz": self.xyz, "bbox": self.box,
                "floor_id": self.floor_id, "label": self.label, "failure": self.failure,
                "releases": self.releases, "inspection_releases": self.inspection_releases,
                "inspection_resumptions": self.inspection_resumptions, "map_releases": self.map_releases,
                "stall_releases": self.stall_releases,
                "last_release": self.last_release, "border_rejections": self.border_rejections,
                "map_rejections": self.map_rejections, "weak_resightings": self.weak_resightings,
                "no_path_steps": self.no_path_steps, "footing_sweeps": self.footing_sweeps,
                "boxed_releases": self.boxed_releases,
                "overrides": self.overrides, "context_rejections": self.context_rejections,
                "suspect": self.suspect, "suspect_locks_held": self.suspect_locks_held,
                "same_spot_rejections": self.same_spot_rejections, "blind_frames": self.blind_frames,
                "memory": None if self.memory is None else self.memory.diagnostics(),
                "approach": None if self.history is None else dict(self.history.diagnostics(),
                                                                   qualifies=self._history_qualifies()),
                "backoff": None if self.backoff is None else self.backoff.diagnostics(),
                "backoffs": self.backoffs, "backoff_skips": self.backoff_skips,
                "support_sightings": self.support_sightings, "support_seen_step": self.support_seen,
                "support_xyz": self.support_xyz, "support_holds": self.support_holds, "support_stops": self.support_stops,
                "history_holds": self.history_holds, "history_stops": self.history_stops,
                "terminal_missing": self.terminal_missing,
                "boxed_in": [{"xy": [round(v, 2) for v in xy], "floor_id": floor, "step": step}
                             for xy, floor, step in self.boxed_in],
                "rejected": [{"xyz": list(entry[0]), "floor_id": entry[1], "step": entry[2], "radius_m": entry[3],
                              "why": entry[4] if len(entry) > 4 else "unverified",
                              "from_xy": None if len(entry) < 6 else [round(v, 2) for v in entry[5]]}
                             for entry in self.rejected]}
