"""Continuous raw perception, conservative semantic fusion, and fresh target evidence."""
from __future__ import annotations

from collections import Counter
from copy import deepcopy
import math
import time
import numpy as np

from sparx_agency.tasks.planning.objnav_benchmark_runtime.methods.doors import DOOR_LABELS
from sparx_agency.tasks.planning.objnav_benchmark_runtime.methods.object_evidence import deduplicate_detections
from sparx_agency.tasks.planning.objnav_benchmark_runtime.methods.perception import observed_objects, STAIR_LABELS


class PerceptionCycle:
    def __init__(self, policy):
        self.policy = policy
        self.step = None
        self.raw = self.detections = ()
        self.projections = []
        self.counts = Counter()
        self.detector_evidence = {}

    def observe(self, obs):
        """Exactly one synchronous prediction for this RGB/depth/pose tuple."""
        if self.step == obs.step:
            return
        p = self.policy
        started = time.monotonic()
        self.raw = tuple(p.detector.detect(obs.rgb))
        self.detector_evidence = deepcopy(getattr(p.detector, "last_diagnostics", {}))
        p.telemetry.latencies["detector_http"].append((time.monotonic() - started) * 1000)
        self.step = obs.step
        self.detections = tuple(deduplicate_detections(self.raw))
        p._duplicates_removed += len(self.raw) - len(self.detections)
        self.projections = []
        self.counts["raw_frames"] += 1
        if p.building:
            p.building.semantic_stair_hint = any(d.cls in STAIR_LABELS and d.conf >= p.settings.detection_confidence for d in self.raw)

    def fuse(self, obs):
        p = self.policy
        self.observe(obs)
        if obs.step - p._target_step > p.target_settings.max_unseen_steps:
            p._target_id = p._target_xy = p._target_floor_id = None
        committed = p.building is not None and p.building.committed
        normal = p.camera_control.normal_view(obs.pose)
        allowed = not committed and normal
        if allowed:
            p.doors.update(obs, self.raw)
        confirmed = False
        used = set()
        self.projections = []
        for label, xyz in observed_objects(obs, self.detections, p.settings.detection_confidence, self.projections):
            row = self.projections[-1]
            if label in DOOR_LABELS or label in STAIR_LABELS:
                row["status"] = "terrain_or_portal_context"
                continue
            reason = "transition_in_progress" if committed else "inspection_view" if not normal else self._floor_support(xyz)
            if reason is not None:
                row["status"] = reason
                continue
            # The existing nearest/radius map remains the association owner.
            # Reject a drift chain or inconsistent height BEFORE mutating it.
            nearby = [lm for lm in p.landmarks.all_landmarks() if lm.class_name == label and math.dist(lm.xy, xyz[:2]) <= 0.70]
            nearest = min(nearby, key=lambda lm: math.dist(lm.xy, xyz[:2])) if nearby else None
            anchor = p._object_geometry.get(nearest.id) if nearest else None
            if anchor is not None and (math.dist(anchor[:2], xyz[:2]) > 0.5 or abs(anchor[2] - xyz[2]) > 0.35):
                row["status"] = "inconsistent_3d_association"
                continue
            if nearest is not None and nearest.id in used:
                row["status"] = "same_frame_association"
                continue
            landmark = p.landmarks.observe(label, tuple(float(v) for v in xyz[:2]), frame_id=obs.step)
            p._object_geometry.setdefault(landmark.id, tuple(float(v) for v in xyz))
            used.add(landmark.id)
            row.update(status="fused", floor_id=p.mapping.floor_id, landmark_id=landmark.id)
            if p.target.accepts(label) and not p.target_evidence.is_suppressed(landmark.id, obs.step):
                supported = p.target_evidence.observe(landmark, obs.pose, obs.step)
                if p._target_id in (None, landmark.id):
                    p._target_id, p._target_xy, p._target_step = landmark.id, landmark.xy, obs.step
                    p._target_floor_id = p.mapping.floor_id
                    confirmed |= supported
        self.counts.update(row["status"] for row in self.projections)
        return bool(confirmed and allowed and p._target_floor_id == p.mapping.floor_id)

    def _floor_support(self, xyz):
        p = self.policy
        if p.building is None:
            return None
        height = p.mapping._anchor
        if not height - 0.30 <= xyz[2] <= height + 2.2:
            return "wrong_floor_height"
        terrain = p.building.terrain
        near = [value[0] for (x, y, _), value in terrain.samples.items()
                if value[2] and math.hypot((x + 0.5) * terrain.resolution - xyz[0],
                                          (y + 0.5) * terrain.resolution - xyz[1]) <= 0.75
                and value[0] <= xyz[2] + 0.1]
        if len(near) >= 6:
            floor = min(near)
            return None if height - 0.40 <= floor <= height + 0.18 else "unsupported_floor_association"
        # A frontal object can occlude the floor itself. Previously observed
        # free cells near its surface support membership; unknown is not free.
        world = p.mapping.worlds[p.mapping.floor_id]
        x, y = world.world_to_grid(*xyz[:2])
        r = int(math.ceil(0.35 / world.resolution))
        if not world.in_bounds(x, y):
            return "outside_observed_map"
        patch = world.grid[max(0, y-r):y+r+1, max(0, x-r):x+r+1]
        return None if np.count_nonzero(patch == world.values.free) >= 3 else "unobserved_floor_support"

    def diagnostics(self):
        return {"observation_step": self.step, "counts": dict(self.counts),
                "projections": list(self.projections), "detector_evidence": self.detector_evidence}

