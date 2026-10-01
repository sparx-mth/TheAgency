"""Episode-local target takeover: no transition back to global exploration."""
from __future__ import annotations

from dataclasses import dataclass, replace
import math

from sparx_agency.core.common.types import normalize_angle
from sparx_agency.core.planning.objnav.action_converter.action_choice import pitch_action, turn_action
from sparx_agency.core.planning.objnav.types.command import NavigationCommand
from sparx_agency.tasks.planning.objnav_benchmark_runtime.methods.perception import observed_objects
from sparx_agency.tasks.planning.objnav_benchmark_runtime.methods.target_path import TargetApproachPath


@dataclass(frozen=True)
class TargetClosingSettings:
    confidence: float = 0.50
    confirmation_frames: int = 2
    association_radius_m: float = 0.50
    look_down_distance_m: float = 1.20
    terminal_distance_m: float = 1.00
    range_tolerance_m: float = 0.05
    refine_distance_m: float = 0.15
    max_verify_steps: int = 12
    max_reacquire_steps: int = 24
    max_closing_steps: int = 160

    def __post_init__(self):
        if isinstance(self.confidence, bool) or not math.isfinite(self.confidence) or not 0 < self.confidence <= 1:
            raise ValueError("Target confidence must be in (0, 1]")
        for key in ("association_radius_m", "look_down_distance_m", "terminal_distance_m", "range_tolerance_m", "refine_distance_m"):
            value = getattr(self, key)
            if isinstance(value, bool) or not math.isfinite(value) or value <= 0:
                raise ValueError("%s must be positive and finite" % key)
        if self.terminal_distance_m + self.range_tolerance_m > self.look_down_distance_m:
            raise ValueError("Terminal inspection must be within the look-down distance")
        for key in ("confirmation_frames", "max_verify_steps", "max_reacquire_steps", "max_closing_steps"):
            if type(getattr(self, key)) is not int or getattr(self, key) < (2 if key == "confirmation_frames" else 1):
                raise ValueError("Invalid target-closing action/frame bound: %s" % key)


class TargetClosing:
    """Own detection verification, path, yaw and pitch until STOP or episode error.

    Only consecutive, coherent 3D observations of the same object count. Alias
    duplicates were removed by PerceptionCycle. No evaluator goal/DTG is read.
    The first confident candidate suspends exploration even before lock; a failed
    verification ends the episode rather than relaxing the no-return contract.
    """

    def __init__(self, policy):
        self.policy, self.settings = policy, policy.settings.target_closing
        self.active = self.locked = False
        self.phase = "SEARCH"
        self.xyz = self.anchor = self.box = None
        self.label = None
        self.count = 0
        self.started = self.last_seen = self.processed = -1
        self.floor_id = None
        self.failure = None
        self.path = TargetApproachPath(policy)
        self.observed_xyz = None
        self.inspection_started = None
        self.inspection_index = 0
        self.occluded_path_steps = 0
        self.verification_yaw = 0.0

    def observe(self, obs):
        if obs.step == self.processed:
            return
        self.processed = obs.step
        p, s = self.policy, self.settings
        k = obs.camera.intrinsics
        candidates = [d for d in p.perception.detections if p.target.accepts(d.cls)
                      and d.conf >= s.confidence
                      and 0 <= d.xyxy[0] < d.xyxy[2] <= k.width
                      and 0 <= d.xyxy[1] < d.xyxy[3] <= k.height]
        if not self.active and candidates:
            self.active, self.started, self.phase = True, obs.step, "VERIFY"
            self.verification_yaw = obs.pose.yaw
            p.route_memory.clear("target_takeover")
            p._route = p._goal = None
            p.peek.cancel(obs, "target_takeover", restore=False)
            p.camera_control.inspection = None
        if not self.active:
            return
        projected = []
        for detection in candidates:
            values = list(observed_objects(obs, (detection,), s.confidence))
            if not values:
                continue
            label, xyz = values[0]
            if not obs.pose.z - 0.3 <= xyz[2] <= obs.pose.z + 2.2:
                continue
            if self.anchor is not None and (p.mapping.floor_id != self.floor_id
                    or math.dist(self.anchor, xyz) > s.association_radius_m):
                continue
            projected.append((detection, tuple(float(v) for v in xyz)))
        if not projected:
            if not self.locked:
                self.count = 0
            return
        detection, xyz = max(projected, key=lambda pair: pair[0].conf)
        self.count = self.count + 1 if obs.step == self.last_seen + 1 else 1
        self.observed_xyz = xyz
        self.xyz = tuple((old + new) / 2 for old, new in zip(self.xyz, xyz)) if self.locked else xyz
        self.box, self.label, self.last_seen = detection.xyxy, detection.cls, obs.step
        if self.anchor is None:
            self.anchor, self.floor_id = xyz, p.mapping.floor_id
        self.locked |= self.count >= s.confirmation_frames
        p._target_xy, p._target_step, p._target_floor_id = self.xyz[:2], obs.step, self.floor_id

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
        self.phase = "REACQUIRE" if self.locked else "VERIFY"
        # Only before verification or when no safe path exists. A known target
        # bearing anchors the scan; never integrate offsets from current yaw.
        offset = self.policy.episode.action_spec.turn_angle_rad
        bearing = (math.atan2(self.xyz[1] - obs.pose.y, self.xyz[0] - obs.pose.x)
                   if self.xyz is not None else self.verification_yaw)
        angles = (0, 1, 0, -1)
        yaw = normalize_angle(bearing + angles[self.inspection_index % 4] * offset)
        if turn_action(normalize_angle(yaw - obs.pose.yaw), offset) is None:
            self.inspection_index += 1
            yaw = normalize_angle(bearing + angles[self.inspection_index % 4] * offset)
        return NavigationCommand.hold(camera_pitch=pitch,
                                      final_yaw=yaw,
                                      info={"kind": "target_closing", "reason": reason})

    def plan(self, obs, world):
        p, s = self.policy, self.settings
        if self.failure:
            self.fail(self.failure)
        if obs.step - self.started >= s.max_closing_steps:
            self.fail("closing action bound reached")
        if not self.locked and obs.step - self.started >= s.max_verify_steps:
            self.fail("candidate never received consecutive depth-consistent confirmations")
        if self.xyz is None:
            return self._scan(obs, None, "waiting for valid target depth")
        if p.mapping.floor_id != self.floor_id:
            self.fail("target floor changed")
        distance = math.dist((obs.pose.x, obs.pose.y), self.xyz[:2])
        if not self.locked:
            return self._scan(obs, None, "consecutive depth-consistent frames required")
        if self.inspection_started is not None or distance <= s.terminal_distance_m:
            return self._inspect(obs, distance)
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
        """Stationary pitch/yaw inspection; never STOP from remembered range alone."""
        actions, s = self.policy.episode.action_spec, self.settings
        if self.inspection_started is None:
            self.inspection_started, self.inspection_index = obs.step, 0
        self.phase = "INSPECT"
        if obs.step - self.inspection_started >= s.max_reacquire_steps:
            self.fail("terminal visual confirmation unavailable")
        pitch = self._pitch(obs, distance)
        bearing = math.atan2(self.xyz[1] - obs.pose.y, self.xyz[0] - obs.pose.x)
        fresh = self.last_seen == obs.step
        error = self._bbox_error(obs) if fresh else normalize_angle(bearing - obs.pose.yaw)
        yaw = normalize_angle(obs.pose.yaw + error)
        turning = turn_action(error, actions.turn_angle_rad) is not None
        tilting = pitch is not None and pitch_action(obs.pose.camera_pitch, pitch, actions.tilt_angle_rad,
                                                     actions.min_pitch_rad, actions.max_pitch_rad) is not None
        info = {"kind": "target_closing", "target_confirmed": True, "persistent_lock": True,
                "target_visible": fresh, "range_m": distance, "bbox_yaw_error_rad": error, "phase": self.phase}
        limit = s.terminal_distance_m + s.range_tolerance_m
        if fresh and not turning and not tilting and distance <= limit:
            measured = math.dist((obs.pose.x, obs.pose.y), self.observed_xyz[:2])
            if measured <= limit:
                self.phase = "STOP"
                return NavigationCommand.stop_here(info=dict(info, phase="STOP", reason="fresh terminal target confirmation"))
        if not fresh or (not turning and not tilting):
            yaw, pitch = self._inspection_view(obs, bearing, pitch)
        return NavigationCommand.hold(camera_pitch=pitch, final_yaw=yaw, info=info)

    def _inspection_view(self, obs, bearing, pitch):
        """Try bounded up/down views centred on the stored target bearing."""
        actions = self.policy.episode.action_spec
        pitches = [pitch]
        if pitch is not None:
            for value in (pitch - actions.tilt_angle_rad, pitch + actions.tilt_angle_rad):
                if (actions.min_pitch_rad is None or value >= actions.min_pitch_rad) and (actions.max_pitch_rad is None or value <= actions.max_pitch_rad):
                    pitches.append(value)
        views = [(normalize_angle(bearing + offset * actions.turn_angle_rad), value)
                 for offset in (0, 1, -1) for value in pitches]
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
                "consecutive_frames": self.count, "started_step": self.started,
                "last_seen_step": self.last_seen, "xyz": self.xyz, "bbox": self.box,
                "floor_id": self.floor_id, "label": self.label, "failure": self.failure}
