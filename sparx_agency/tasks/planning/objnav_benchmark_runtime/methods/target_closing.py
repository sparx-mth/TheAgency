"""Episode-local target takeover: no transition back to global exploration."""
from __future__ import annotations

from dataclasses import dataclass, replace
import math

from sparx_agency.core.common.types import normalize_angle
from sparx_agency.core.planning.objnav.action_converter.action_choice import pitch_action, turn_action
from sparx_agency.core.planning.objnav.types.command import NavigationCommand
from sparx_agency.tasks.planning.objnav_benchmark_runtime.methods.perception import clipped_box, observed_objects
from sparx_agency.tasks.planning.objnav_benchmark_runtime.methods.perception_cycle import contradicted_by_map
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
        for key in ("release_unverified", "stop_on_exhausted_inspection", "release_on_failed_inspection"):
            if type(getattr(self, key)) is not bool:
                raise ValueError("%s must be a bool" % key)


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
        self.map_releases = 0              # locks the map outvoted after the fact
        self.last_release = None           # why the last candidate was released
        self.released_step = None          # the action of the last release; the cooldown counts from it
        self.border_rejections = 0
        self.map_rejections = 0
        self.weak_resightings = 0          # frames that counted on ``track_confidence`` alone
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
                if cooling or self._rejected_nearby(xyz, p.mapping.floor_id):
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

    def _release(self, obs, radius_factor=1.0, why="unverified"):
        """Give a candidate back to exploration and remember the spot (``radius_factor`` times the rejection radius)."""
        p = self.policy
        if self.anchor is not None:
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
        desired = math.atan2(obs.pose.z + obs.camera.height_m - self.xyz[2], max(distance, 1e-6))
        low = self.label in ("toilet", "potted plant") or self.xyz[2] < obs.pose.z + obs.camera.height_m - 0.15
        if low and distance < self.settings.look_down_distance_m:
            desired = max(actions.tilt_angle_rad, math.atan2(
                obs.pose.z + obs.camera.height_m - self.xyz[2], max(distance, 1e-6)))
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
            if (not self.locked and turn_action(error, offset) is None
                    and distance > self.settings.terminal_distance_m + actions.forward_step_m):
                # Two steps ahead: a one-step waypoint sits inside the converter's arrival
                # tolerance and yields no action at all. One MOVE_FORWARD results either way.
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
        if self.inspection_started is not None or distance <= s.terminal_distance_m:
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
            return self._scan(obs, None, "no safe target path; exploration remains suspended")
        visible = self.last_seen == obs.step
        self.phase = "CLOSE" if visible else "CLOSE_OCCLUDED"
        self.occluded_path_steps += int(not visible)
        return replace(command, camera_pitch=0.0 if p.episode.action_spec.has_camera_tilt else None,
                       info=dict(command.info, target_confirmed=True, persistent_lock=True,
                                 target_visible=visible, range_m=distance, phase=self.phase))

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
        measured = None
        if fresh:
            measured = math.dist((obs.pose.x, obs.pose.y), self.observed_xyz[:2])
            if self.observed_near_m is not None:
                measured = min(measured, self.observed_near_m)
            if measured <= limit:
                self.inspection_sighting = obs.step
        info = {"kind": "target_closing", "target_confirmed": True, "persistent_lock": True,
                "target_visible": fresh, "range_m": distance, "measured_m": measured, "bbox_yaw_error_rad": error,
                "box_spans_centre": spans_centre, "phase": self.phase}
        if fresh and not turning and not tilting and measured <= limit:
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
        return NavigationCommand.hold(camera_pitch=pitch, final_yaw=yaw, info=info)

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
                "map_releases": self.map_releases,
                "last_release": self.last_release, "border_rejections": self.border_rejections,
                "map_rejections": self.map_rejections, "weak_resightings": self.weak_resightings,
                "rejected": [{"xyz": list(spot), "floor_id": floor, "step": step, "radius_m": radius}
                             for spot, floor, step, radius in self.rejected]}
