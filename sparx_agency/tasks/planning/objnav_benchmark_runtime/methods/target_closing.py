"""Episode-local target takeover: no transition back to global exploration."""
from __future__ import annotations

from dataclasses import dataclass, replace
import math

from sparx_agency.core.common.types import normalize_angle
from sparx_agency.core.planning.objnav.action_converter.action_choice import pitch_action, turn_action
from sparx_agency.core.planning.objnav.types.command import NavigationCommand
from sparx_agency.tasks.planning.objnav_benchmark_runtime.methods.perception import clipped_box, observed_objects
from sparx_agency.tasks.planning.objnav_benchmark_runtime.methods.perception_cycle import contradicted_by_map
from sparx_agency.tasks.planning.objnav_benchmark_runtime.methods.sightlines import unknown_around
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
    look_down_distance_m: float = 1.20
    terminal_distance_m: float = 1.00
    range_tolerance_m: float = 0.05
    refine_distance_m: float = 0.15
    max_verify_steps: int = 12
    max_reacquire_steps: int = 24
    max_closing_steps: int = 160
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

    def __post_init__(self):
        if isinstance(self.confidence, bool) or not math.isfinite(self.confidence) or not 0 < self.confidence <= 1:
            raise ValueError("Target confidence must be in (0, 1]")
        if (isinstance(self.track_confidence, bool) or not math.isfinite(self.track_confidence)
                or not 0 < self.track_confidence <= self.confidence):
            raise ValueError("track_confidence must be in (0, confidence]")
        for key in ("association_radius_m", "look_down_distance_m", "terminal_distance_m", "range_tolerance_m",
                    "refine_distance_m", "rejection_radius_m", "rejection_min_range_m", "failed_inspection_radius_factor"):
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
        self.rejected = []                 # (anchor xyz, floor id, action, radius m) per released candidate
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

    def _clipped(self, box, intrinsics):
        return clipped_box(box, intrinsics, self.settings.border_margin_px)

    def _rejected_nearby(self, xyz, floor_id):
        return any(floor == floor_id and math.dist(spot[:2], xyz[:2]) <= radius
                   for spot, floor, _, radius in self.rejected)

    def rejected_near(self, xy, floor_id):
        """Whether a released candidate's anchor lies within ``rejection_radius_m`` of ``xy`` on ``floor_id``.

        For the search loop: a confirmed landmark the takeover already
        verified and rejected at is not worth another look.
        """
        return self._rejected_nearby((float(xy[0]), float(xy[1]), 0.0), floor_id)

    def _candidates(self, obs):
        """Confident in-frame boxes, and those with a coherent 3-D position.

        Before a lock the box must also clear the image border and lie away
        from every spot already released as unverified on this floor. While
        a candidate is active, a box of the target's class down to
        ``track_confidence`` is projected too, and counts if -- and only if --
        it lands on the candidate's anchor: a re-sighting, not a new object.
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
            if not self.locked and range_m > s.rejection_min_range_m:
                cooling = (not self.active and self.released_step is not None
                           and obs.step - self.released_step < s.release_cooldown_actions)
                if cooling or self._rejected_nearby(xyz, p.mapping.floor_id) or self._boxed_in_here(obs):
                    continue
            if not self.locked and contradicted_by_map(p, label, xyz, rows[-1].get("radius_m")) is not None:
                self.map_rejections += 1                       # a confirmed non-target object stands here
                continue
            kept.append(detection)
            if not obs.pose.z - 0.3 <= xyz[2] <= obs.pose.z + 2.2:
                continue
            if self.anchor is not None and not associated:
                continue
            if not strong:
                self.weak_resightings += 1
            projected.append((detection, tuple(float(v) for v in xyz), rows[-1].get("range_near_m")))
        return kept, projected

    def observe(self, obs):
        if obs.step == self.processed:
            return
        self.processed = obs.step
        p, s = self.policy, self.settings
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
        if projected:
            detection, xyz, near = max(projected, key=lambda item: item[0].conf)
            self.count = self.count + 1 if obs.step == self.last_seen + 1 else 1
            self.observed_xyz, self.observed_near_m = xyz, (None if near is None else float(near))
            self.xyz = tuple((old + new) / 2 for old, new in zip(self.xyz, xyz)) if self.locked else xyz
            self.box, self.label, self.last_seen = detection.xyxy, detection.cls, obs.step
            if self.anchor is None:
                self.anchor, self.floor_id = xyz, p.mapping.floor_id
            self.locked |= self.count >= s.confirmation_frames
            p._target_xy, p._target_step, p._target_floor_id = self.xyz[:2], obs.step, self.floor_id
        elif not self.locked:
            self.count = 0
        if (not self.locked and s.release_unverified
                and obs.step - self.started >= s.max_verify_steps):
            self._release(obs)
            return
        if (self.locked and s.release_on_failed_inspection and self.anchor is not None
                and contradicted_by_map(p, self.label or "", self.anchor) is not None):
            # The map outvoted the lock: the object at the anchor has been confirmed as
            # something else since (the sofa from four metres is the bed from two).
            self.map_releases += 1
            self._release(obs, radius_factor=s.failed_inspection_radius_factor, why="contradicted by the map")

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
                                  float(radius_factor) * self.settings.rejection_radius_m))
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

    def _pitch(self, obs, distance):
        actions = self.policy.episode.action_spec
        if not actions.has_camera_tilt:
            return None
        # The pitch is the target's measured height's, never its label's: "a potted
        # plant is low" sent the camera 30 degrees down at a plant standing in a metre-
        # tall planter, where its foliage never projected, for 24 actions (Hanson
        # 2026-10-05); a plant on the floor is low by this geometry anyway.
        camera_z = obs.pose.z + obs.camera.height_m
        desired = math.atan2(camera_z - self.xyz[2], max(distance, 1e-6))
        low = self.xyz[2] < camera_z - 0.15
        if low and distance < self.settings.look_down_distance_m:
            desired = max(actions.tilt_angle_rad, desired)
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
            distance = math.dist((obs.pose.x, obs.pose.y), self.xyz[:2]) if self.xyz is not None else float("inf")
            if (not self.locked and abs(error) <= offset + 1e-6
                    and distance > self.settings.terminal_distance_m + actions.forward_step_m):
                # Two steps ahead: a one-step waypoint sits inside the converter's arrival
                # tolerance and yields no action at all. One MOVE_FORWARD results either way.
                # Within one turn of the centre the box stays in the frame after a step,
                # and a step is what makes the next frame a consecutive one; the centring
                # turn moved the box across the image and the detector dropped it on the
                # other side -- four times in a row at a chair 3.5 m off (Hanson 2026-10-05).
                # Never for a LOCKED target without a safe path: that one waits for A*.
                reach = 2.0 * actions.forward_step_m
                ahead = (obs.pose.x + reach * math.cos(obs.pose.yaw), obs.pose.y + reach * math.sin(obs.pose.yaw))
                return NavigationCommand.follow([(obs.pose.x, obs.pose.y), ahead], camera_pitch=pitch,
                                                info=dict(info, verify_step="towards the candidate"))
            return NavigationCommand.hold(camera_pitch=pitch, final_yaw=normalize_angle(obs.pose.yaw + error), info=info)
        angles = (0, 1, 0, -1)
        yaw = normalize_angle(bearing + angles[self.inspection_index % 4] * offset)
        if turn_action(normalize_angle(yaw - obs.pose.yaw), offset) is None:
            self.inspection_index += 1
            yaw = normalize_angle(bearing + angles[self.inspection_index % 4] * offset)
        return NavigationCommand.hold(camera_pitch=pitch,
                                      final_yaw=yaw,
                                      info={"kind": "target_closing", "reason": reason, "target_visible": False})

    def plan(self, obs, world):
        p, s = self.policy, self.settings
        if self.failure:
            self.fail(self.failure)
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
        return replace(command, camera_pitch=0.0 if p.episode.action_spec.has_camera_tilt else None,
                       info=dict(command.info, target_confirmed=True, persistent_lock=True,
                                 target_visible=visible, range_m=distance, phase=self.phase))

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
        info = {"kind": "target_closing", "target_confirmed": True, "persistent_lock": True,
                "target_visible": fresh, "range_m": distance, "measured_m": measured, "bbox_yaw_error_rad": error,
                "box_spans_centre": spans_centre, "phase": self.phase}
        # The pitch the geometry predicts is where to LOOK for the target, not a
        # condition on having seen it: the Hanson toilet projected only at 60 degrees
        # down where the prediction said 30, and twenty actions of LOOK_UP/LOOK_DOWN
        # followed a fresh, centred, in-range sighting before the budget STOPped.
        if fresh and not turning and measured <= limit:
            self.phase = "STOP"
            return NavigationCommand.stop_here(info=dict(info, phase="STOP", reason="fresh terminal target confirmation"))
        views = self._inspection_views(pitch)
        blank = (s.release_on_failed_inspection and self.inspection_sighting is None
                 and self.last_seen < self.inspection_started and self.inspection_index >= len(views))
        if obs.step - self.inspection_started >= s.max_reacquire_steps or blank:
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
                "boxed_in": [{"xy": [round(v, 2) for v in xy], "floor_id": floor, "step": step}
                             for xy, floor, step in self.boxed_in],
                "rejected": [{"xyz": list(spot), "floor_id": floor, "step": step, "radius_m": radius}
                             for spot, floor, step, radius in self.rejected]}
